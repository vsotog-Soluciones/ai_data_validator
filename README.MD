# AI Data Validator - Netezza to BigQuery

Herramienta de validacion automatizada con IA para comparar datos entre Netezza y BigQuery.

## Caracteristicas Principales

- **Generacion Inteligente de Queries**: Azure OpenAI analiza el esquema de cada tabla y genera queries de comparacion optimizadas
- **Agrupacion Automatica**: Detecta campos temporales (fecha, mes, año) y dimensiones para agrupaciones inteligentes
- **Comparacion Exhaustiva**: Suma campos numericos, cuenta distintos en dimensiones
- **Tolerancia Configurable**: Define umbrales de diferencia aceptables (default 0.5%)
- **Cache de Queries**: Reutiliza queries generadas para ejecuciones posteriores
- **Reportes Detallados**: Excel con resumen ejecutivo y hojas de detalle por tabla

## Requisitos Previos

```bash
pip install pandas pyodbc google-cloud-bigquery openai xlsxwriter openpyxl
```

### Configuracion de Conexiones

1. **Netezza**: Driver ODBC instalado y VPN activa
2. **BigQuery**: Autenticacion configurada via `gcloud auth application-default login`
3. **Azure OpenAI**: Recurso creado en Azure con deployment de GPT-4

## Estructura de Archivos

```
proyecto/
├── ai_data_validator.py          # Script principal
├── config.json                    # Configuracion de conexiones
├── cuadratura.xlsx               # Excel con mapeo de tablas
├── ai_validation_report.xlsx     # Reporte generado (salida)
├── ai_queries_cache.json         # Cache de queries IA (auto-generado)
└── ai_migration_validation.log   # Log de ejecucion
```

## Formato del Excel de Entrada

El archivo `cuadratura.xlsx` debe tener una hoja llamada **"validaciones"** con estas columnas:

| netezza_database | netezza_schema | netezza_table | bigquery_project | bigquery_dataset | bigquery_table | filtros |
|------------------|----------------|---------------|------------------|------------------|----------------|---------|
| DWH_PROD | VENTAS | FACT_VENTAS | mi-proyecto | analytics | fact_sales | fecha >= '2024-01-01' |
| DWH_PROD | VENTAS | DIM_PRODUCTO | mi-proyecto | analytics | dim_product | |

**Columnas opcionales:**
- `filtros`: Condiciones WHERE para aplicar en ambas queries

## Uso

### Validacion Completa con IA

```bash
python ai_data_validator.py --mapping cuadratura.xlsx --config config.json --excel_report resultados_validacion.xlsx
```

### Modo Solo Cache (sin llamar a IA)

Util para re-ejecutar con queries ya generadas:

```bash
python ai_data_validator.py --mapping cuadratura.xlsx --cache-only --excel_report resultados_cache.xlsx
```

## Como Funciona la IA

### 1. Analisis de Esquema
La IA recibe para cada tabla:
- Lista de campos numericos (suma)
- Lista de campos de fecha/timestamp (agrupacion temporal)
- Lista de dimensiones (count distinct)
- Campos clave (excluidos de agregaciones)

### 2. Decision de Estrategia
La IA decide automaticamente:
- **Granularidad temporal**: dia, semana, mes, año o ninguna
- **Dimensiones adicionales**: hasta 2 dimensiones categoricas relevantes
- **Agregaciones**: SUM para numericos, COUNT DISTINCT para dimensiones

### 3. Generacion de Queries Compatibles
Genera queries paralelas para ambas bases con:
- Sintaxis especifica de cada plataforma
- Mismos campos de agrupacion y agregacion
- Ordenamiento descendente por primer agregado
- Limite de 1000 filas para performance

### Ejemplo de Output de IA

```json
{
  "grouping_strategy": "Agrupacion por año y mes de fecha_venta",
  "temporal_granularity": "month",
  "netezza_query": "SELECT EXTRACT(YEAR FROM fecha_venta) as anio, EXTRACT(MONTH FROM fecha_venta) as mes, SUM(monto) as total_monto, COUNT(DISTINCT id_cliente) as clientes FROM DWH.VENTAS.FACT_VENTAS GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 1000",
  "bigquery_query": "SELECT EXTRACT(YEAR FROM fecha_venta) as anio, EXTRACT(MONTH FROM fecha_venta) as mes, SUM(monto) as total_monto, COUNT(DISTINCT id_cliente) as clientes FROM `proyecto.ventas.fact_ventas` GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 1000",
  "comparison_fields": [
    {"field_name": "total_monto", "aggregation": "SUM", "data_type": "numeric"},
    {"field_name": "clientes", "aggregation": "COUNT_DISTINCT", "data_type": "dimension"}
  ],
  "group_by_fields": ["anio", "mes"]
}
```

## Interpretacion del Reporte

### Hoja "Resumen"
- **Match %**: Porcentaje de filas donde todos los campos cuadran dentro de tolerancia
- **Cuadra**: checkmark si 100% de filas coinciden, X si hay diferencias
- Color verde: Tabla OK
- Color rojo: Tabla con diferencias

### Hojas "Det_N"
Detalle por tabla con:
- Metadata de la comparacion
- Queries ejecutadas (auditoria)
- Datos comparados lado a lado
- Columnas `*_diff_pct`: Diferencia porcentual
- Columnas `*_match`: True/False segun tolerancia
- Filas en rojo: Registros con diferencias

### Hoja "Queries IA"
Catalogo de todas las queries generadas para:
- Auditoria de logica
- Reutilizacion manual
- Debugging

## Configuracion Avanzada

### Ajustar Tolerancia

En `config.json`:
```json
"validation": {
  "tolerance_percent": 0.5,
  "max_rows_comparison": 10000,
  "retry_attempts": 3
}
```

### Personalizacion del Prompt IA

En el codigo, metodo `_build_ai_prompt()`, puedes modificar las instrucciones para:
- Priorizar ciertas dimensiones
- Excluir campos especificos
- Cambiar estrategias de agrupacion

### Optimizacion de Performance

1. **Usar filtros**: Reduce volumen de datos comparados
2. **Cache de queries**: Segunda ejecucion es instantanea
3. **Limitar tablas**: Procesa subconjuntos en el Excel

## Troubleshooting

### Error: "No se pudo generar query"
- Verifica conectividad a Azure OpenAI
- Revisa limites de rate en tu deployment
- Chequea formato del esquema de tabla

### Error: "Tabla no encontrada"
- Verifica nombres exactos (case-sensitive en Netezza)
- Confirma permisos de lectura
- Valida conexion VPN (Netezza)

### Diferencias Inesperadas
1. Revisa queries en hoja "Queries IA"
2. Ejecuta manualmente en cada base
3. Verifica tipos de datos (casting implicito)
4. Chequea filtros aplicados

### Performance Lenta
- Reduce `max_rows_comparison` en config
- Agrega filtros en Excel (ej: ultimos 6 meses)
- Usa `--cache-only` en re-ejecuciones

## Soporte

- Log detallado: `ai_migration_validation.log`
- Cache de queries: `ai_queries_cache.json` (compartir para analisis)

## Proximas Mejoras

- Validacion de valores nulos/duplicados
- Deteccion de outliers con IA
- Comparacion de distribuciones estadisticas
- Dashboard interactivo con resultados
- Alertas automaticas por Slack/Email

---

**Version**: 2.0.0  
**Autor**: Equipo de Migracion de Datos  
**Ultima actualizacion**: Noviembre 2024