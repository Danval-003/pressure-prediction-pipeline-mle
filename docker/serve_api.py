"""FastAPI service that exposes the trained LSTM model for inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model
from joblib import load as joblib_load

from crispdm_deployment import (
    preprocess_for_inference,
    predict_pressures,
    build_predictions_dataframe,
)

# Paths configurables por variables de entorno
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models/lstm_model.keras"))
SCALER_PATH = Path(os.environ.get("SCALER_PATH", "/models/scaler.joblib"))
FEATURES_PATH = Path(os.environ.get("FEATURES_PATH", "/models/features.json"))

# Límite de respiraciones por request (para proteger memoria)
DEPLOY_BREATH_LIMIT = int(os.environ.get("DEPLOY_BREATH_LIMIT", "512"))


def _load_artifacts():
    """Carga modelo, scaler y features desde disco."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. "
            "Monta la carpeta 'models' con el archivo .keras entrenado."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el scaler en {SCALER_PATH}. "
            "Asegúrate de que el pipeline haya guardado 'scaler.joblib'."
        )

    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de features en {FEATURES_PATH}. "
            "Asegúrate de que el pipeline haya guardado 'features.json'."
        )

    print(f"[serve] Cargando modelo desde {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    print(f"[serve] Cargando scaler desde {SCALER_PATH}")
    scaler = joblib_load(SCALER_PATH)

    print(f"[serve] Cargando lista de features desde {FEATURES_PATH}")
    features_text = FEATURES_PATH.read_text(encoding="utf-8")
    features = json.loads(features_text)

    if not isinstance(features, list):
        raise ValueError(
            f"El archivo {FEATURES_PATH} no contiene una lista de features válida."
        )

    print(f"[serve] Artefactos listos. Número de features: {len(features)}")
    return model, scaler, features


# Cargamos una sola vez, al levantar el proceso
MODEL, SCALER, FEATURES = _load_artifacts()

app = FastAPI(
    title="CRISP-DM Ventilator Pressure - Deployment API",
    version="0.2.0",
)


class Measurement(BaseModel):
    breath_id: int = Field(..., description="Identificador de la respiración")
    R: int
    C: int
    time_step: float
    u_in: float
    u_out: int
    id: Optional[str] = Field(default=None, description="ID opcional de la fila")
    pressure: Optional[float] = Field(
        default=None,
        description="Presión observada (opcional, se ignora para inferencia)",
    )


class PredictionRequest(BaseModel):
    records: List[Measurement]


@app.get("/health")
def health() -> dict:
    """Endpoint simple para verificar que el servicio está vivo."""
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "features_path": str(FEATURES_PATH),
        "deploy_breath_limit": DEPLOY_BREATH_LIMIT,
        "num_features": len(FEATURES),
    }


@app.post("/predict")
def predict(req: PredictionRequest) -> dict:
    if not req.records:
        raise HTTPException(status_code=400, detail="No se recibieron registros.")

    # Convertir a DataFrame
    df = pd.DataFrame([record.dict() for record in req.records])

    # Validar número de respiraciones
    if "breath_id" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="Falta la columna 'breath_id' en los registros.",
        )

    breath_ids = df["breath_id"].nunique()
    if breath_ids > DEPLOY_BREATH_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Se recibieron {breath_ids} respiraciones; "
                f"el máximo permitido es {DEPLOY_BREATH_LIMIT}."
            ),
        )

    try:
        # Mismo preprocesamiento que en entrenamiento, pero usando el scaler ya entrenado
        X_input, breath_matrix, time_matrix = preprocess_for_inference(
            df, SCALER, FEATURES
        )
        preds = predict_pressures(MODEL, X_input)
        pred_df = build_predictions_dataframe(preds, breath_matrix, time_matrix)
    except Exception as exc:  # noqa: BLE001
        # En producción podrías loggear `exc` con más detalle
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "predictions": pred_df.to_dict(orient="records"),
        "breaths": int(len(pred_df) / 80),
    }
