# Personal_asa – CRISP-DM Packages

Este repositorio ahora está organizado como un **monorepo de paquetes**
Python, uno por cada etapa aplicable del ciclo CRISP-DM:

| Etapa | Distribución | Módulo |
| --- | --- | --- |
| Data Preparation | `crispdm-data-preparation` | `crispdm_data_preparation` |
| Data Understanding | `crispdm-data-understanding` | `crispdm_data_understanding` |
| Modeling | `crispdm-modeling` | `crispdm_modeling` |
| Evaluation | `crispdm-evaluating` | `crispdm_evaluating` |
| Deployment | `crispdm-deployment` | `crispdm_deployment` |

Cada paquete vive dentro de `packages/<etapa>` con un `pyproject.toml`
propio y un árbol `src/` listo para publicarse.

## Uso local

Para evitar instalaciones repetidas durante el desarrollo se añadió
`local_dev.py`, que inserta todos los `packages/*/src` en `sys.path`.
Los archivos de compatibilidad (`data_preparation.py`, `modeling.py`,
etc.) importan este módulo automáticamente, por lo que puedes seguir
ejecutándolos como scripts si estás dentro del repositorio.

Si prefieres instalarlos de forma editable:

```bash
pip install -e packages/data_preparation
pip install -e packages/data_understanding[app]
pip install -e packages/modeling
pip install -e packages/evaluating
pip install -e packages/deployment
```

## CI/CD

- `.github/workflows/ci.yml` construye **todos** los paquetes en cada
  `push` a `main` y en cada `pull_request`.
- `.github/workflows/release.yml` publica automáticamente el paquete
  correspondiente cuando se empuja un tag con el formato:

  ```
  data-preparation-vMAJOR.MINOR.PATCH
  data-understanding-vMAJOR.MINOR.PATCH
  modeling-vMAJOR.MINOR.PATCH
  evaluating-vMAJOR.MINOR.PATCH
  deployment-vMAJOR.MINOR.PATCH
  ```

  El workflow valida que la versión declarada en `pyproject.toml`
  coincida con la del tag, construye el paquete, sube los artefactos y
  adjunta los `.whl`/`.tar.gz` a un release en GitHub.

- `.github/workflows/docker.yml` construye y publica la imagen Docker en
  Docker Hub (`docker.io/<usuario>/personal-asa-crispdm`) cada vez que se empuja a
  `main` o se crea un tag `crispdm-image-v*`.

## Flujo de versionamiento

1. Actualiza la versión en `pyproject.toml` del paquete deseado.
2. Crea el tag correspondiente, por ejemplo:

   ```bash
   git tag data-preparation-v0.1.0
   git push origin data-preparation-v0.1.0
   ```

3. El workflow `release` generará automáticamente el release con los
   artefactos listos para descargarse desde GitHub.

## Docker

El `Dockerfile` empaqueta todo el flujo CRISP-DM junto con una ruta para
levantar la app de Streamlit:

- `docker/run_crispdm_pipeline.py` ejecuta las etapas de preparación,
  modelado, evaluación y genera predicciones de despliegue (`Deployment`).
  Se puede limitar el número de respiraciones (`--max-breaths`) para
  mantener bajo el consumo de memoria/RAM. Si la ruta indicada en
  `--data-path` no existe, se genera automáticamente un dataset sintético
  muy pequeño (unos cuantos `breath_id`) para permitir pruebas rápidas.
- `docker/entrypoint.sh` permite elegir el modo de ejecución vía la
  variable `SERVICE`:
  - `SERVICE=pipeline` (por defecto): corre el pipeline end-to-end.
  - `SERVICE=streamlit`: expone la app de Data Understanding en `:8501`.

Ejemplos:

```bash
# Pipeline (monta datos, modelos y artefactos desde el host)
docker run --rm \
  -e SERVICE=pipeline \
  -e DATA_PATH=/data/train_part_001.csv \
  -v "$(pwd)/data/raw":/data \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/artifacts":/artifacts \
  docker.io/<usuario>/personal-asa-crispdm:latest \
  --data-path /data/train_part_001.csv \
  --model-path /models/lstm_docker.keras \
  --artifacts-dir /artifacts \
  --max-breaths 1500

# Streamlit (EDA)
docker run --rm -e SERVICE=streamlit -p 8501:8501 \
  docker.io/<usuario>/personal-asa-crispdm:latest
```

> Nota: la imagen no incluye el dataset para evitar tamaños enormes. Es
> necesario montar la carpeta `data/raw` (con `train.csv` o los
> `train_part_*.csv`) dentro del contenedor.
