# Imagen Docker del Pipeline (crispdm)

La imagen principal (`Dockerfile`) empaqueta todo el flujo CRISP-DM:

- Ejecuta el pipeline end-to-end (Data Preparation → Modeling →
  Evaluation → Deployment) y genera artefactos.
- Puede servir la app de Streamlit para la etapa de Data Understanding.

## Contenido

- Python 3.11 slim.
- Instalación editable de todos los paquetes CRISP-DM desde `packages/`.
- Scripts:
  - `docker/run_crispdm_pipeline.py`
  - `docker/entrypoint.sh`
- Datos incluidos: los fragmentos `data/raw/train_part_*.csv`.
- Volúmenes declarados: `/data`, `/models`, `/artifacts`.

## Modos de ejecución (`SERVICE`)

| Valor        | Descripción                                                     |
|--------------|-----------------------------------------------------------------|
| `pipeline`   | (default) Corre el pipeline completo.                           |
| `streamlit`  | Levanta la app `data_understanding_app.py` en el puerto 8501.   |
| `bash`/otro  | Ejecuta comandos arbitrarios dentro del contenedor.             |

### Pipeline

```bash
docker run --rm \
  -e SERVICE=pipeline \
  -e DATA_PATH=/data \
  -v "$(pwd)/data/raw":/data \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/artifacts":/artifacts \
  docker.io/<usuario>/personal-asa-crispdm:latest \
  --data-path /data \
  --model-path /models/lstm_docker.keras \
  --artifacts-dir /artifacts \
  --max-breaths 1500 \
  --epochs 15
```

Parámetros relevantes:

- `--data-path`: archivo o carpeta con los CSV. Si no existe, el script
  genera datos sintéticos (`--sample-breaths`).
- `--max-breaths`: limita la cantidad de respiraciones para mantener el
  consumo de RAM bajo (por defecto 1500).
- `--deploy-breaths`: número de respiraciones (completas) usadas para
  generar predicciones de despliegue (default 3).
- `--model-path` y `--artifacts-dir`: dónde guardar el modelo `.keras`
  y los artefactos finales.

### Streamlit

```bash
docker run --rm -e SERVICE=streamlit -p 8501:8501 \
  docker.io/<usuario>/personal-asa-crispdm:latest
```

Monte `-v "$(pwd)/data/raw":/data` y configure `DATA_PATH=/data` si
quiere usar un dataset distinto al incluido.

## CI/CD

`.github/workflows/docker.yml`:

- Se ejecuta en `push` a `main`, `workflow_dispatch` y tags
  `crispdm-image-v*`.
- Usa Buildx + Docker Hub (`personal-asa-crispdm`).
- Requiere los secretos `DOCKERHUB_USERNAME` y `DOCKERHUB_TOKEN`.

## Nota de rendimiento

La imagen está pensada para entornos de demostración o pipelines
controlados. Ajusta `--max-breaths`, `--epochs`, `--batch-size` y las
opciones de TensorFlow (variables `TF_CPP_MIN_LOG_LEVEL`,
`OMP_NUM_THREADS`) según el hardware disponible.
