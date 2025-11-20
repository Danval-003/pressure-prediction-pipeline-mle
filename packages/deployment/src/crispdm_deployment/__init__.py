# deployment_service.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Ajusta el import si usas un nombre distinto para tu módulo/paquete
# de preparación de datos. En este repositorio usamos
# ``crispdm_data_preparation`` para mantener consistencia.
from crispdm_data_preparation import get_lstm_ready_xy, add_features

TIMESTEPS_PER_BREATH = 80
DEFAULT_MODEL_PATH = Path("models/lstm_model.keras")


# ======================================================================
# Carga de artefactos (modelo, scaler, features)
# ======================================================================

def load_artifacts(
    model_path: Path | str = DEFAULT_MODEL_PATH,
) -> Tuple[tf.keras.Model, Any, List[str]]:
    """
    Carga el modelo LSTM entrenado y reconstruye el scaler + features
    usando la misma función de preparación de datos que en training.

    NOTA: get_lstm_ready_xy recalcula X,y sobre el train.csv completo.
    Lo usamos solo una vez al arrancar el servicio, para recuperar:
      - scaler (RobustScaler fit en training)
      - features (orden de columnas usadas)

    Parameters
    ----------
    model_path : Path | str
        Ruta al archivo .keras (modelo entrenado).

    Returns
    -------
    model : tf.keras.Model
    scaler : objeto de sklearn (RobustScaler)
    features : list[str]
        Nombres de columnas usadas como features, en el mismo orden
        que en entrenamiento.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "Asegúrate de haber corrido la etapa de Modeling."
        )

    print(f"[deploy] Cargando modelo desde: {model_path}")
    model = load_model(model_path)

    # Solo usamos scaler y features; X,y no los necesitamos aquí
    print("[deploy] Reconstruyendo scaler y features desde get_lstm_ready_xy()...")
    _, _, scaler, features = get_lstm_ready_xy()

    print(f"[deploy] Se recuperaron {len(features)} features del pipeline de data prep.")
    return model, scaler, features


# ======================================================================
# Preprocesamiento para inferencia
# ======================================================================

def preprocess_for_inference(
    df_raw: pd.DataFrame,
    scaler,
    features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aplica el MISMO preprocesamiento que en entrenamiento a un
    DataFrame nuevo con forma Kaggle-like:

      columnas esperadas al menos:
        ['breath_id', 'R', 'C', 'time_step', 'u_in', 'u_out']
      (no necesita 'id' ni 'pressure')

    Pasos:
      1) Copia y asegura tipos de R y C.
      2) Aplica add_features(df) para generar lags, diffs, cumsum, etc.
      3) fillna(0)
      4) Map de R y C a {0,1,2}
      5) Ordena por ['breath_id', 'time_step']
      6) Extrae breath_id y time_step en forma (n_breaths, 80)
      7) Subselecciona columnas 'features' y aplica scaler.transform(...)
      8) Reshape a (n_breaths, 80, n_features)

    Returns
    -------
    X_input : np.ndarray
        Tensores listos para el modelo, shape (n_breaths, 80, n_features).
    breath_ids : np.ndarray
        Matriz (n_breaths, 80) con los breath_id correspondientes.
    time_steps : np.ndarray
        Matriz (n_breaths, 80) con los time_step correspondientes.
    """
    required_cols = {"breath_id", "R", "C", "time_step", "u_in", "u_out"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas en df_raw: {missing}. "
            "Se esperaban al menos: "
            "['breath_id', 'R', 'C', 'time_step', 'u_in', 'u_out']"
        )

    df = df_raw.copy()

    # Asegurar tipos consistentes
    df["R"] = df["R"].astype(int)
    df["C"] = df["C"].astype(int)

    # Si existe 'id', la ignoramos; no se usa en el modelo
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Feature engineering (misma función que en data prep)
    df = add_features(df)

    # NaNs -> 0
    df.fillna(0, inplace=True)

    # Encodings de R y C (mismos mapeos que en entrenamiento)
    df["R"] = df["R"].map({5: 0, 20: 1, 50: 2}).astype(int)
    df["C"] = df["C"].map({10: 0, 20: 1, 50: 2}).astype(int)

    # Ordenar las filas igual que en training
    df = df.sort_values(["breath_id", "time_step"]).reset_index(drop=True)

    # Extraer breath_id y time_step para reconstruir la salida
    breath_ids = df["breath_id"].to_numpy()
    time_steps = df["time_step"].to_numpy()

    if len(breath_ids) % TIMESTEPS_PER_BREATH != 0:
        raise ValueError(
            f"El número total de filas ({len(breath_ids)}) no es múltiplo de "
            f"{TIMESTEPS_PER_BREATH}. Asegúrate de enviar respiraciones completas."
        )

    n_breaths = len(breath_ids) // TIMESTEPS_PER_BREATH

    breath_ids = breath_ids.reshape(n_breaths, TIMESTEPS_PER_BREATH)
    time_steps = time_steps.reshape(n_breaths, TIMESTEPS_PER_BREATH)

    # Seleccionar features en el mismo orden que en entrenamiento
    for col in features:
        if col not in df.columns:
            raise ValueError(
                f"La columna de feature '{col}' no está presente en el DataFrame "
                "preprocesado. Asegúrate de que el pipeline de features sea consistente."
            )

    feature_df = df[features].copy()

    # Aplicar scaler entrenado
    X_scaled = scaler.transform(feature_df)  # (n_rows, n_features)
    n_rows, n_features = X_scaled.shape

    if n_rows != n_breaths * TIMESTEPS_PER_BREATH:
        raise ValueError(
            "La cantidad de filas después de escalar no coincide con "
            "n_breaths * TIMESTEPS_PER_BREATH."
        )

    X_input = X_scaled.reshape(n_breaths, TIMESTEPS_PER_BREATH, n_features)

    return X_input, breath_ids, time_steps


# ======================================================================
# Predicción
# ======================================================================

def predict_pressures(
    model: tf.keras.Model,
    X_input: np.ndarray,
) -> np.ndarray:
    """
    Genera predicciones de presión para cada respiración.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo LSTM cargado.
    X_input : np.ndarray
        Tensor de entrada (n_breaths, 80, n_features).

    Returns
    -------
    y_pred : np.ndarray
        Predicciones de presión con shape (n_breaths, 80).
    """
    y_pred = model.predict(X_input, verbose=0)
    y_pred = np.squeeze(y_pred)  # (n_breaths, 80)
    return y_pred


def build_predictions_dataframe(
    y_pred: np.ndarray,
    breath_ids: np.ndarray,
    time_steps: np.ndarray,
) -> pd.DataFrame:
    """
    Reconstruye un DataFrame con las predicciones asociadas
    a cada (breath_id, time_step).

    Parameters
    ----------
    y_pred : np.ndarray
        Predicciones (n_breaths, 80).
    breath_ids : np.ndarray
        IDs (n_breaths, 80).
    time_steps : np.ndarray
        time_step (n_breaths, 80).

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame con columnas:
          ['breath_id', 'time_step', 'predicted_pressure']
    """
    if y_pred.shape != breath_ids.shape or y_pred.shape != time_steps.shape:
        raise ValueError(
            "Shapes incompatibles entre y_pred, breath_ids y time_steps."
        )

    df_out = pd.DataFrame(
        {
            "breath_id": breath_ids.flatten(),
            "time_step": time_steps.flatten(),
            "predicted_pressure": y_pred.flatten(),
        }
    )
    return df_out
