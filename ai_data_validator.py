import pandas as pd
import pyodbc
from google.cloud import bigquery
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import argparse
import decimal
from openai import AzureOpenAI
import time
import re

VERSION = "2.0.0"

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_migration_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIDataValidator:
    def __init__(self, config_file: str = 'config.json'):
        """
        Inicializa el validador con capacidades de IA
        
        Args:
            config_file: Archivo de configuración JSON
        """
        self.config = self._load_config(config_file)
        self.netezza_conn = None
        self.bigquery_client = None
        self.azure_client = None
        self.validation_results = []
        self.ai_generated_queries = {}  # Cache de queries generadas
        
    def _load_config(self, config_file: str) -> dict:
        """Carga configuración desde archivo JSON"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Archivo de configuración {config_file} no encontrado")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Configuración por defecto"""
        return {
            "netezza": {
                "server": "your_netezza_server",
                "database": "your_database",
                "username": "your_username",
                "password": "your_password",
                "port": 5480
            },
            "bigquery": {
                "project_id": "your_project_id"
            },
            "azure_openai": {
                "endpoint": "https://your-resource.openai.azure.com/",
                "api_key": "your_api_key",
                "api_version": "2024-02-15-preview",
                "deployment_name": "gpt-4"
            },
            "validation": {
                "tolerance_percent": 0.5,
                "max_rows_comparison": 10000,
                "retry_attempts": 3
            }
        }
    
    def connect_netezza(self):
        """Establece conexión con Netezza"""
        try:
            netezza_config = self.config['netezza']
            connection_string = (
                f"DRIVER={{NetezzaSQL}};"
                f"SERVER={netezza_config['server']};"
                f"PORT={netezza_config['port']};"
                f"DATABASE={netezza_config['database']};"
                f"UID={netezza_config['username']};"
                f"PWD={netezza_config['password']};"
                f"Timeout=60;"
            )
            self.netezza_conn = pyodbc.connect(connection_string)
            logger.info("Conexión a Netezza establecida")
        except Exception as e:
            logger.error(f"Error conectando a Netezza: {str(e)}")
            raise
    
    def connect_bigquery(self):
        """Establece conexión con BigQuery"""
        try:
            bigquery_config = self.config['bigquery']
            self.bigquery_client = bigquery.Client(
                project=bigquery_config['project_id']
            )
            logger.info("Conexión a BigQuery establecida")
        except Exception as e:
            logger.error(f"Error conectando a BigQuery: {str(e)}")
            raise
    
    def connect_azure_openai(self):
        """Establece conexión con Azure OpenAI"""
        try:
            azure_config = self.config['azure_openai']
            self.azure_client = AzureOpenAI(
                azure_endpoint=azure_config['endpoint'],
                api_key=azure_config['api_key'],
                api_version=azure_config['api_version']
            )
            logger.info("Conexión a Azure OpenAI establecida")
        except Exception as e:
            logger.error(f"Error conectando a Azure OpenAI: {str(e)}")
            raise
    
    def get_table_schema(self, source: str, database: str, schema: str, table: str) -> List[Dict]:
        """
        Obtiene el esquema de una tabla
        
        Args:
            source: 'netezza' o 'bigquery'
            database: Nombre de la base de datos (solo Netezza)
            schema: Nombre del esquema/dataset
            table: Nombre de la tabla
            
        Returns:
            Lista de diccionarios con información de columnas
        """
        if source == 'netezza':
            return self._get_netezza_schema(database, schema, table)
        else:
            return self._get_bigquery_schema(schema, table)
    
    def _get_netezza_schema(self, database: str, schema: str, table: str) -> List[Dict]:
        """Obtiene esquema de Netezza"""
        query = f"""
            SELECT 
                ATTNAME as column_name,
                FORMAT_TYPE as data_type,
                CASE WHEN ATTNOTNULL THEN 'NO' ELSE 'YES' END as is_nullable
            FROM {database}.INFORMATION_SCHEMA._V_RELATION_COLUMN
            WHERE NAME = ? 
            AND SCHEMA = ?
            ORDER BY ATTNUM
        """
        try:
            cursor = self.netezza_conn.cursor()
            cursor.execute(query, (table.upper(), schema.upper()))
            columns = cursor.fetchall()
            
            schema_info = []
            for col in columns:
                schema_info.append({
                    'column_name': col[0].lower(),
                    'data_type': col[1],
                    'is_nullable': col[2]
                })
            return schema_info
        except Exception as e:
            logger.error(f"Error obteniendo esquema Netezza {database}.{schema}.{table}: {str(e)}")
            return []
    
    def _get_bigquery_schema(self, dataset: str, table: str) -> List[Dict]:
        """Obtiene esquema de BigQuery"""
        try:
            table_ref = self.bigquery_client.dataset(dataset).table(table)
            table_obj = self.bigquery_client.get_table(table_ref)
            
            schema_info = []
            for field in table_obj.schema:
                schema_info.append({
                    'column_name': field.name.lower(),
                    'data_type': field.field_type,
                    'is_nullable': 'YES' if field.mode == 'NULLABLE' else 'NO'
                })
            return schema_info
        except Exception as e:
            logger.error(f"Error obteniendo esquema BigQuery {dataset}.{table}: {str(e)}")
            return []
    
    def _classify_columns(self, schema: List[Dict]) -> Dict[str, List[str]]:
        """
        Clasifica columnas por tipo para facilitar agregaciones
        
        Returns:
            Dict con listas de columnas: numeric, date, temporal, dimension, key
        """
        numeric_types = {'INTEGER', 'INT', 'INT64', 'BIGINT', 'SMALLINT', 
                        'NUMERIC', 'DECIMAL', 'FLOAT', 'FLOAT64', 'DOUBLE PRECISION', 'REAL'}
        date_types = {'DATE', 'TIMESTAMP', 'DATETIME'}
        
        result = {
            'numeric': [],
            'date': [],
            'temporal': [],  # Campos que contienen fecha/tiempo en el nombre
            'dimension': [],
            'key': []
        }
        
        for col in schema:
            col_name = col['column_name'].lower()
            col_type = col['data_type'].upper().split('(')[0]
            
            # Clasificar por tipo de dato
            if col_type in numeric_types:
                # Excluir IDs y claves
                if not (col_name.endswith(('_key', '_id', '_key_1', '_id_1')) or 
                       col_name.startswith('id_')):
                    result['numeric'].append(col_name)
            
            if col_type in date_types:
                result['date'].append(col_name)
            
            # Detectar campos temporales por nombre
            temporal_keywords = ['fecha', 'date', 'time', 'periodo', 'period', 
                                'mes', 'month', 'ano', 'year', 'dia', 'day', 
                                'semana', 'week', 'trimestre', 'quarter']
            if any(keyword in col_name for keyword in temporal_keywords):
                result['temporal'].append(col_name)
            
            # Claves
            if col_name.endswith(('_key', '_id', '_key_1', '_id_1')) or col_name.startswith('id_'):
                result['key'].append(col_name)
            
            # Dimensiones potenciales (no clave, no numérico)
            if (col_type not in numeric_types and 
                col_type not in date_types and
                not (col_name.endswith(('_key', '_id')) or col_name.startswith('id_'))):
                result['dimension'].append(col_name)
        
        return result
    
    def _build_ai_prompt(self, nz_schema: List[Dict], bq_schema: List[Dict],
                        nz_table_full: str, bq_table_full: str, 
                        filtros: str = "") -> str:
        """
        Construye el prompt para Azure OpenAI
        
        Args:
            nz_schema: Esquema de Netezza
            bq_schema: Esquema de BigQuery
            nz_table_full: Nombre completo tabla Netezza (database.schema.table)
            bq_table_full: Nombre completo tabla BigQuery (project.dataset.table)
            filtros: Filtros adicionales
            
        Returns:
            Prompt formateado
        """
        # Clasificar columnas
        nz_classified = self._classify_columns(nz_schema)
        bq_classified = self._classify_columns(bq_schema)
        
        # Construir contexto de esquema
        schema_context = {
            "netezza": {
                "table": nz_table_full,
                "numeric_fields": nz_classified['numeric'],
                "date_fields": nz_classified['date'],
                "temporal_fields": nz_classified['temporal'],
                "dimension_fields": nz_classified['dimension'][:10],  # Limitar
                "key_fields": nz_classified['key']
            },
            "bigquery": {
                "table": bq_table_full,
                "numeric_fields": bq_classified['numeric'],
                "date_fields": bq_classified['date'],
                "temporal_fields": bq_classified['temporal'],
                "dimension_fields": bq_classified['dimension'][:10],
                "key_fields": bq_classified['key']
            }
        }
        
        prompt = f"""Eres un experto en SQL y migración de bases de datos. Tu tarea es generar queries de comparación entre una tabla en Netezza y su equivalente en BigQuery.

**CONTEXTO DE LAS TABLAS:**
```json
{json.dumps(schema_context, indent=2, ensure_ascii=False)}
```

**FILTROS ADICIONALES:** {filtros if filtros else "Ninguno"}

**INSTRUCCIONES:**
1. Genera queries de comparación que incluyan:
   - Agrupación por dimensiones temporales (fecha, semana, mes, año) si existen campos DATE/TIMESTAMP
   - Si no hay campos temporales, agrupa por dimensiones categóricas relevantes (máximo 2)
   - SUM() para todos los campos numéricos
   - COUNT(DISTINCT campo) para campos de dimensión que no sean claves
   - COUNT(*) para conteo total de registros

2. Las queries deben ser comparables entre sí (mismos campos, mismo orden)

3. Sintaxis específica:
   - Netezza: Usa funciones estándar SQL, TO_CHAR para formateo de fechas
   - BigQuery: Usa EXTRACT, FORMAT_DATE para fechas

4. Limita los resultados a máximo 1000 filas con ORDER BY descendente por el primer agregado

5. Si hay filtros, aplícalos en el WHERE de ambas queries

**FORMATO DE SALIDA (JSON estricto):**
```json
{{
  "grouping_strategy": "descripción de la estrategia de agrupación elegida",
  "temporal_granularity": "day|week|month|year|none",
  "netezza_query": "query completa para Netezza",
  "bigquery_query": "query completa para BigQuery",
  "comparison_fields": [
    {{
      "field_name": "nombre del campo",
      "aggregation": "SUM|COUNT|COUNT_DISTINCT",
      "data_type": "numeric|dimension"
    }}
  ],
  "group_by_fields": ["lista de campos de agrupación"]
}}
```

Genera SOLO el JSON, sin explicaciones adicionales."""
        
        return prompt
    
    def _call_azure_openai(self, prompt: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Llama a Azure OpenAI con reintentos
        
        Args:
            prompt: Prompt a enviar
            max_retries: Número máximo de reintentos
            
        Returns:
            Respuesta parseada como dict o None si falla
        """
        for attempt in range(max_retries):
            try:
                response = self.azure_client.chat.completions.create(
                    model=self.config['azure_openai']['deployment_name'],
                    messages=[
                        {"role": "system", "content": "Eres un experto en SQL y análisis de datos. Respondes SOLO con JSON válido."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Baja temperatura para respuestas deterministas
                    max_tokens=2000
                )
                
                content = response.choices[0].message.content.strip()
                
                # Limpiar markdown si existe
                content = re.sub(r'^```json\s*', '', content)
                content = re.sub(r'\s*```$', '', content)
                
                # Parsear JSON
                result = json.loads(content)
                logger.info(f"Azure OpenAI respondió exitosamente (intento {attempt + 1})")
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Error parseando JSON (intento {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Backoff exponencial
                else:
                    logger.error(f"Falló parseo JSON después de {max_retries} intentos")
                    return None
                    
            except Exception as e:
                logger.error(f"Error llamando Azure OpenAI (intento {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None
        
        return None
    
    def _execute_comparison_queries(self, nz_query: str, bq_query: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ejecuta las queries de comparación en ambas bases
        
        Returns:
            Tuple de (DataFrame Netezza, DataFrame BigQuery)
        """
        # Ejecutar en Netezza
        try:
            cursor = self.netezza_conn.cursor()
            cursor.execute(nz_query)
            nz_data = cursor.fetchall()
            nz_columns = [desc[0].lower() for desc in cursor.description]
            
            if nz_data:
                nz_data_converted = [tuple(row) for row in nz_data]
                df_nz = pd.DataFrame(nz_data_converted, columns=nz_columns)
            else:
                df_nz = pd.DataFrame(columns=nz_columns)
                
            logger.info(f"Query Netezza ejecutada: {len(df_nz)} filas")
        except Exception as e:
            logger.error(f"Error ejecutando query Netezza: {str(e)}")
            df_nz = pd.DataFrame()
        
        # Ejecutar en BigQuery
        try:
            bq_job = self.bigquery_client.query(bq_query)
            bq_data = list(bq_job.result())
            
            if bq_data:
                bq_columns = [key.lower() for key in bq_data[0].keys()]
                df_bq = pd.DataFrame([dict(row) for row in bq_data], columns=bq_columns)
            else:
                df_bq = pd.DataFrame()
                
            logger.info(f"Query BigQuery ejecutada: {len(df_bq)} filas")
        except Exception as e:
            logger.error(f"Error ejecutando query BigQuery: {str(e)}")
            df_bq = pd.DataFrame()
        
        return df_nz, df_bq
    
    def _compare_dataframes(self, df_nz: pd.DataFrame, df_bq: pd.DataFrame,
                           group_by_fields: List[str], 
                           comparison_fields: List[Dict]) -> pd.DataFrame:
        """
        Compara los DataFrames y calcula diferencias
        
        Returns:
            DataFrame con comparación y columna 'match' por cada campo
        """
        if df_nz.empty and df_bq.empty:
            return pd.DataFrame({'error': ['Ambas tablas vacías']})
        
        # Normalizar nombres de columnas
        df_nz.columns = df_nz.columns.str.lower()
        df_bq.columns = df_bq.columns.str.lower()
        
        # Merge por campos de agrupación
        if group_by_fields:
            group_by_lower = [f.lower() for f in group_by_fields]
            df_merge = pd.merge(
                df_nz, df_bq, 
                on=group_by_lower, 
                how='outer', 
                suffixes=('_nz', '_bq')
            )
        else:
            # Sin agrupación, comparación directa
            df_merge = pd.DataFrame()
            for field_info in comparison_fields:
                field = field_info['field_name'].lower()
                df_merge[f'{field}_nz'] = df_nz[field] if field in df_nz.columns else None
                df_merge[f'{field}_bq'] = df_bq[field] if field in df_bq.columns else None
        
        # Calcular diferencias y matches
        tolerance = self.config['validation']['tolerance_percent'] / 100.0
        
        for field_info in comparison_fields:
            field = field_info['field_name'].lower()
            col_nz = f'{field}_nz'
            col_bq = f'{field}_bq'
            
            if col_nz in df_merge.columns and col_bq in df_merge.columns:
                # Convertir a numérico
                df_merge[col_nz] = pd.to_numeric(df_merge[col_nz], errors='coerce')
                df_merge[col_bq] = pd.to_numeric(df_merge[col_bq], errors='coerce')
                
                # Calcular diferencia porcentual
                df_merge[f'{field}_diff_pct'] = df_merge.apply(
                    lambda row: (
                        abs(row[col_nz] - row[col_bq]) / abs(row[col_nz]) 
                        if pd.notna(row[col_nz]) and pd.notna(row[col_bq]) and row[col_nz] != 0
                        else (0.0 if row[col_nz] == row[col_bq] else None)
                    ),
                    axis=1
                )
                
                # Match considerando tolerancia
                df_merge[f'{field}_match'] = df_merge[f'{field}_diff_pct'].apply(
                    lambda x: x <= tolerance if pd.notna(x) else False
                )
        
        return df_merge
    
    def validate_with_ai(self, excel_file: str, sheet_name: str = 'validaciones'):
        """
        Proceso principal de validación usando IA
        
        Args:
            excel_file: Archivo Excel con mapeo de tablas
            sheet_name: Nombre de la hoja
        """
        try:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df = df.drop_duplicates(subset=['netezza_database', 'netezza_schema', 
                                           'netezza_table', 'bigquery_dataset', 
                                           'bigquery_table'])
            
            logger.info(f"Iniciando validación con IA de {len(df)} tablas")
            
            for idx, row in df.iterrows():
                try:
                    nz_database = row['netezza_database']
                    nz_schema = row['netezza_schema']
                    nz_table = row['netezza_table']
                    bq_project = row.get('bigquery_project', self.config['bigquery']['project_id'])
                    bq_dataset = row['bigquery_dataset']
                    bq_table = row['bigquery_table']
                    filtros = row.get('filtros', '')
                    
                    logger.info(f"[{idx+1}/{len(df)}] Procesando {nz_database}.{nz_schema}.{nz_table}")
                    
                    # Obtener esquemas
                    nz_schema_info = self.get_table_schema('netezza', nz_database, nz_schema, nz_table)
                    bq_schema_info = self.get_table_schema('bigquery', bq_project, bq_dataset, bq_table)
                    
                    if not nz_schema_info or not bq_schema_info:
                        logger.warning(f"Esquema no disponible para {nz_table}")
                        continue
                    
                    # Construir nombres completos
                    nz_table_full = f"{nz_database}.{nz_schema}.{nz_table}"
                    bq_table_full = f"`{bq_project}.{bq_dataset}.{bq_table}`"
                    
                    # Crear cache key
                    cache_key = f"{nz_table_full}_{bq_table_full}_{filtros}"
                    
                    # Verificar cache
                    if cache_key in self.ai_generated_queries:
                        logger.info(f"Usando query cacheada para {nz_table}")
                        ai_response = self.ai_generated_queries[cache_key]
                    else:
                        # Generar prompt y llamar IA
                        prompt = self._build_ai_prompt(
                            nz_schema_info, bq_schema_info,
                            nz_table_full, bq_table_full, filtros
                        )
                        
                        ai_response = self._call_azure_openai(prompt)
                        
                        if not ai_response:
                            logger.error(f"No se pudo generar query para {nz_table}")
                            continue
                        
                        # Guardar en cache
                        self.ai_generated_queries[cache_key] = ai_response
                    
                    # Ejecutar queries
                    nz_query = ai_response['netezza_query']
                    bq_query = ai_response['bigquery_query']
                    
                    df_nz, df_bq = self._execute_comparison_queries(nz_query, bq_query)
                    
                    # Comparar resultados
                    df_comparison = self._compare_dataframes(
                        df_nz, df_bq,
                        ai_response.get('group_by_fields', []),
                        ai_response.get('comparison_fields', [])
                    )
                    
                    # Calcular métricas de cuadratura
                    match_cols = [col for col in df_comparison.columns if col.endswith('_match')]
                    total_match = all(df_comparison[match_cols].all()) if match_cols else False
                    
                    result = {
                        'netezza_table': nz_table_full,
                        'bigquery_table': bq_table_full,
                        'ai_strategy': ai_response.get('grouping_strategy', ''),
                        'temporal_granularity': ai_response.get('temporal_granularity', ''),
                        'netezza_query': nz_query,
                        'bigquery_query': bq_query,
                        'comparison_df': df_comparison,
                        'total_rows_nz': len(df_nz),
                        'total_rows_bq': len(df_bq),
                        'compared_rows': len(df_comparison),
                        'total_match': total_match,
                        'match_percentage': (
                            df_comparison[match_cols].all(axis=1).sum() / len(df_comparison) * 100
                            if len(df_comparison) > 0 and match_cols else 0
                        )
                    }
                    
                    self.validation_results.append(result)
                    
                    logger.info(f"✓ {nz_table}: Match {result['match_percentage']:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error procesando tabla {idx}: {str(e)}")
                    continue
            
            logger.info("Validación con IA completada")
            
        except Exception as e:
            logger.error(f"Error en validación con IA: {str(e)}")
            raise
    
    def generate_report(self, excel_report: str = 'ai_validation_report.xlsx'):
        """
        Genera reporte Excel con resultados
        
        Args:
            excel_report: Nombre del archivo Excel de salida
        """
        try:
            with pd.ExcelWriter(excel_report, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                # Formatos
                header_format = workbook.add_format({
                    'bold': True, 
                    'bg_color': '#4472C4',
                    'font_color': 'white'
                })
                match_format = workbook.add_format({'bg_color': '#C6EFCE'})
                mismatch_format = workbook.add_format({'bg_color': '#FFC7CE'})
                
                # Hoja resumen
                summary_data = []
                for result in self.validation_results:
                    summary_data.append({
                        'Tabla Netezza': result['netezza_table'],
                        'Tabla BigQuery': result['bigquery_table'],
                        'Estrategia IA': result['ai_strategy'],
                        'Granularidad': result['temporal_granularity'],
                        'Filas NZ': result['total_rows_nz'],
                        'Filas BQ': result['total_rows_bq'],
                        'Filas Comparadas': result['compared_rows'],
                        'Match %': f"{result['match_percentage']:.2f}%",
                        'Cuadra': '✓' if result['total_match'] else '✗'
                    })
                
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Aplicar formatos al resumen
                worksheet = writer.sheets['Resumen']
                for col_num, value in enumerate(df_summary.columns.values):
                    worksheet.write(0, col_num, value, header_format)
                
                # Colorear filas según match
                for idx, row in df_summary.iterrows():
                    if row['Cuadra'] == '✓':
                        worksheet.set_row(idx + 1, None, match_format)
                    else:
                        worksheet.set_row(idx + 1, None, mismatch_format)
                
                # Hojas de detalle por tabla
                for idx, result in enumerate(self.validation_results):
                    sheet_name = f"Det_{idx+1}"[:31]  # Límite Excel
                    
                    df_detail = result['comparison_df']
                    df_detail.to_excel(writer, sheet_name=sheet_name, index=False, startrow=5)
                    
                    worksheet = writer.sheets[sheet_name]
                    
                    # Metadata
                    worksheet.write(0, 0, 'Tabla Netezza:', header_format)
                    worksheet.write(0, 1, result['netezza_table'])
                    worksheet.write(1, 0, 'Tabla BigQuery:', header_format)
                    worksheet.write(1, 1, result['bigquery_table'])
                    worksheet.write(2, 0, 'Estrategia:', header_format)
                    worksheet.write(2, 1, result['ai_strategy'])
                    worksheet.write(3, 0, 'Query Netezza:', header_format)
                    worksheet.write(3, 1, result['netezza_query'])
                    worksheet.write(4, 0, 'Query BigQuery:', header_format)
                    worksheet.write(4, 1, result['bigquery_query'])
                    
                    # Headers
                    for col_num, value in enumerate(df_detail.columns.values):
                        worksheet.write(5, col_num, value, header_format)
                    
                    # Colorear filas con mismatch
                    match_cols = [col for col in df_detail.columns if col.endswith('_match')]
                    for row_idx in range(len(df_detail)):
                        if match_cols:
                            row_matches = all(df_detail.iloc[row_idx][col] for col in match_cols)
                            if not row_matches:
                                worksheet.set_row(row_idx + 6, None, mismatch_format)
                
                # Hoja de queries generadas (para auditoría)
                queries_data = []
                for result in self.validation_results:
                    queries_data.append({
                        'Tabla': result['netezza_table'],
                        'Estrategia': result['ai_strategy'],
                        'Query Netezza': result['netezza_query'],
                        'Query BigQuery': result['bigquery_query']
                    })
                
                df_queries = pd.DataFrame(queries_data)
                df_queries.to_excel(writer, sheet_name='Queries IA', index=False)
                
                # Ajustar anchos de columna
                for sheet_name in writer.sheets:
                    worksheet = writer.sheets[sheet_name]
                    for i, col in enumerate(df_summary.columns if sheet_name == 'Resumen' else df_queries.columns):
                        max_length = max(
                            df_summary[col].astype(str).map(len).max() if sheet_name == 'Resumen' else df_queries[col].astype(str).map(len).max(),
                            len(col)
                        )
                        worksheet.set_column(i, i, min(max_length + 2, 50))
            
            logger.info(f"Reporte generado: {excel_report}")
            
            # Guardar cache de queries
            cache_file = 'ai_queries_cache.json'
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.ai_generated_queries, f, indent=2, ensure_ascii=False)
            logger.info(f"Cache de queries guardado: {cache_file}")
            
        except Exception as e:
            logger.error(f"Error generando reporte: {str(e)}")
            raise
    
    def close_connections(self):
        """Cierra las conexiones"""
        if self.netezza_conn:
            self.netezza_conn.close()
            logger.info("Conexión Netezza cerrada")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Validador AI-Powered: Netezza -> BigQuery"
    )
    parser.add_argument(
        '-m', '--mapping', 
        default='cuadratura.xlsx',
        help='Archivo Excel con mapeo de tablas'
    )
    parser.add_argument(
        '-c', '--config',
        default='config.json',
        help='Archivo de configuración JSON'
    )
    parser.add_argument(
        '-x', '--excel_report',
        default='ai_validation_report.xlsx',
        help='Nombre del reporte Excel'
    )
    parser.add_argument(
        '--cache-only',
        action='store_true',
        help='Solo usar queries del cache, no llamar a IA'
    )
    
    args = parser.parse_args()
    
    try:
        validator = AIDataValidator(config_file=args.config)
        logger.info(f"Iniciando AI Data Validator v{VERSION}")
        
        # Conexiones
        validator.connect_netezza()
        validator.connect_bigquery()
        
        if not args.cache_only:
            validator.connect_azure_openai()
        else:
            logger.info("Modo cache: no se conectará a Azure OpenAI")
            # Cargar cache existente
            try:
                with open('ai_queries_cache.json', 'r', encoding='utf-8') as f:
                    validator.ai_generated_queries = json.load(f)
                logger.info(f"Cache cargado: {len(validator.ai_generated_queries)} queries")
            except FileNotFoundError:
                logger.warning("No se encontró archivo de cache")
        
        # Validación
        validator.validate_with_ai(args.mapping)
        
        # Generar reporte
        validator.generate_report(excel_report=args.excel_report)
        
        # Estadísticas finales
        total = len(validator.validation_results)
        matched = sum(1 for r in validator.validation_results if r['total_match'])
        logger.info(f"\n{'='*60}")
        logger.info(f"RESUMEN FINAL:")
        logger.info(f"  Total tablas procesadas: {total}")
        logger.info(f"  Tablas con cuadratura OK: {matched} ({matched/total*100:.1f}%)")
        logger.info(f"  Tablas con diferencias: {total - matched}")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Error en proceso principal: {str(e)}")
        raise
    
    finally:
        if 'validator' in locals():
            validator.close_connections()


if __name__ == "__main__":
    main()