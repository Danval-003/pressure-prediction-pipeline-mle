# Datos y Artefactos

## División de `train.csv`

Para evitar manejar un único CSV de ~440 MB, se creó el script
`scripts/split_train_csv.py` que divide `data/raw/train.csv` en fragmentos
`train_part_XXX.csv` (por defecto 300 000 filas cada uno).

Uso:

```bash
python scripts/split_train_csv.py \
  --source data/raw/train.csv \
  --dest data/raw \
  --rows-per-chunk 300000 \
  --prefix train_part
```

**Nota:** el script espera que `data/raw/train.csv` exista; genera
archivos numerados secuencialmente y no sobrescribe fragmentos presentes.

## Consumo en los paquetes

- `crispdm_data_preparation.load_raw_train_csv` detecta automáticamente
  si el argumento apunta a un archivo único o a un directorio, buscando
  primero `train_part_*.csv` y luego `train.csv`.
- El resto del pipeline (`get_preprocessed_dataframe`,
  `get_lstm_ready_xy`, `crispdm_evaluating`, `crispdm_deployment`) se
  beneficia de esta lógica sin cambios adicionales.
- La app de Streamlit (`crispdm_data_understanding.app`) hace lo mismo:
  concatena todos los fragmentos si están disponibles.

## Directorios clave

- `data/raw/`: contiene `train.csv` (opcional) y los fragmentos
  `train_part_*.csv`. Se copia dentro de la imagen Docker del pipeline
  para que funcione por defecto; en producción conviene montarlo como
  volumen.
- `models/`: destino sugerido para guardar `*.keras` desde la etapa de
  `Modeling`. Tanto el pipeline Docker como la imagen de servicio esperan
  encontrar aquí el modelo entrenado.
- `artifacts/`: carpeta donde el pipeline deja reportes (ej.
  `pipeline_report.json`) y predicciones (`deployment_predictions.csv`).

Todos estos directorios se pueden montar como volúmenes en los contenedores:

```bash
docker run --rm \
  -v "$(pwd)/data/raw":/data \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/artifacts":/artifacts \
  docker.io/<usuario>/personal-asa-crispdm:latest \
  --data-path /data \
  --model-path /models/lstm_docker.keras \
  --artifacts-dir /artifacts
```

## Datos sintéticos de respaldo

`docker/run_crispdm_pipeline.py` genera un conjunto sintético pequeño
cuando no encuentra archivos en `--data-path`. Esto permite ejecutar el
pipeline (p. ej. en CI) sin montar datos reales, pero los resultados son
solo demostrativos. Ajusta `--sample-breaths` para controlar su tamaño.
