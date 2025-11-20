# Imagen Docker de Servicio (crispdm-serve)

`Dockerfile.serve` define una imagen enfocada únicamente en servir el
modelo entrenado mediante una API FastAPI.

## Contenido

- Python 3.11 slim + paquetes mínimos:
  - `crispdm_data_preparation`
  - `crispdm_deployment`
  - `fastapi`, `uvicorn`
- Script principal: `docker/serve_api.py`.
- Entry point: `docker/entrypoint_serve.sh` (lanza Uvicorn).
- Volúmenes recomendados: `/models` (modelo `.keras`) y `/data`
  (fragmentos `train_part_*.csv` para reconstruir el `scaler`).

## Variables de entorno

| Variable              | Descripción                                             | Default                  |
|-----------------------|---------------------------------------------------------|--------------------------|
| `MODEL_PATH`          | Ruta al `.keras` entrenado.                             | `/models/lstm_model.keras` |
| `DATA_PATH`           | Ruta al archivo/directorio con los datos usados para    | `/app/data/raw`          |
|                       | reconstruir `scaler`/`features`.                        |                          |
| `DEPLOY_BREATH_LIMIT` | Número máximo de respiraciones por request.             | `512`                    |
| `HOST` / `PORT`       | Parámetros de Uvicorn.                                  | `0.0.0.0` / `8000`       |

## Endpoints

- `GET /health`: devuelve estado, ruta del modelo y de los datos.
- `POST /predict`: recibe un arreglo `records` con filas estilo Kaggle
  (`breath_id`, `R`, `C`, `time_step`, `u_in`, `u_out`, etc.).

Ejemplo de request:

```json
{
  "records": [
    {"breath_id": 1001, "R": 20, "C": 50, "time_step": 0.0, "u_in": 0.0, "u_out": 0},
    {"breath_id": 1001, "R": 20, "C": 50, "time_step": 0.033, "u_in": 5.0, "u_out": 0}
  ]
}
```

Response:

```json
{
  "predictions": [
    {"breath_id": 1001, "time_step": 0.0, "predicted_pressure": 6.12},
    {"breath_id": 1001, "time_step": 0.033, "predicted_pressure": 6.37},
    ...
  ],
  "breaths": 1
}
```

Se valida que todas las respiraciones estén completas (80 registros) y
que el número total no supere `DEPLOY_BREATH_LIMIT`.

## Uso

```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/models/lstm_model.keras \
  -e DATA_PATH=/data \
  -v "$(pwd)/models":/models \
  -v "$(pwd)/data/raw":/data \
  docker.io/<usuario>/personal-asa-crispdm-serve:latest
```

## CI/CD

`.github/workflows/docker-serve.yml`:

- Construye `Dockerfile.serve` y publica en
  `docker.io/<usuario>/personal-asa-crispdm-serve`.
- Se ejecuta en `push` a `main`, `workflow_dispatch` y tags
  `crispdm-serve-v*`.
- Usa los mismos secretos de Docker Hub que la imagen del pipeline.
