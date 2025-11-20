from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

from crispdm_data_preparation import get_lstm_ready_xy
from crispdm_modeling import train_val_split



# ======================================================================
# Parámetros “de negocio” para la etapa de Evaluation
# ======================================================================

# Umbral de RMSE promedio global a partir del cual consideramos
# que el modelo es razonablemente útil (ajústalo a tu criterio).
BUSINESS_RMSE_THRESHOLD = 0.7

# Umbral máximo de diferencia de RMSE entre grupos de R o C
# antes de considerarlo un posible sesgo fuerte.
GROUP_RMSE_GAP_THRESHOLD = 0.3

# Ruta por defecto del modelo entrenado (salvado en la etapa de Modeling)
DEFAULT_MODEL_PATH = Path("models/lstm_model.keras")


# ======================================================================
# Carga de modelo y datos de evaluación
# ======================================================================

def load_trained_model(model_path: Path = DEFAULT_MODEL_PATH) -> tf.keras.Model:
    """
    Carga un modelo Keras ya entrenado desde disco.

    Parameters
    ----------
    model_path : Path
        Ruta al archivo .keras o .h5 con los pesos entrenados.

    Returns
    -------
    model : tf.keras.Model
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en: {model_path}. "
            f"Asegúrate de haber corrido la etapa de Modeling y guardado el modelo."
        )
    print(f"[eval] Cargando modelo entrenado desde: {model_path}")
    model = load_model(model_path)
    return model


def load_eval_data(
    train_size: float = 0.80,
    val_size: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Reconstruye el conjunto de datos y genera el split train/val
    exactamente como en la etapa de Modeling.

    Además regresa, para el set de validación, los códigos de R y C
    sin escalar para poder detectar posibles sesgos por tipo de pulmón.

    Returns
    -------
    X_val : np.ndarray
        Features de validación con shape (n_val, 80, n_features).
    y_val : np.ndarray
        Targets de validación con shape (n_val, 80).
    RC_val : np.ndarray
        Matriz (n_val, 2) con [R_code, C_code] de cada respiración.
        Los códigos son los mapeos 0,1,2 usados en el preprocesado.
    features : list[str]
        Lista de nombres de features usada en X.
    """
    print("[eval] Cargando datos preprocesados con get_lstm_ready_xy()...")
    X_all, y_all, scaler, features = get_lstm_ready_xy()
    print(f"[eval] X_all shape: {X_all.shape}, y_all shape: {y_all.shape}")
    print(f"[eval] Número de features: {len(features)}")

    # Split igual que en Modeling
    X_train, X_val = train_val_split(X_all, train_size=train_size, val_size=val_size)
    y_train, y_val = train_val_split(y_all, train_size=train_size, val_size=val_size)

    print(f"[eval] X_train: {X_train.shape}, X_val: {X_val.shape}")
    print(f"[eval] y_train: {y_train.shape}, y_val: {y_val.shape}")

    # ------------------------------------------------------------------
    # Reconstruir R y C sin escalar para cada respiración
    # ------------------------------------------------------------------
    # Tomamos solo el primer time_step de cada respiración (R y C son constantes
    # en los 80 pasos) y deshacemos el escalado con el RobustScaler.
    if "R" not in features or "C" not in features:
        raise ValueError("Las columnas 'R' y 'C' no están presentes en 'features'.")

    idx_R = features.index("R")
    idx_C = features.index("C")

    # X_all[:, 0, :]  -> (n_breaths, n_features) con el primer time_step de cada respiración
    X0_scaled = X_all[:, 0, :]
    X0_unscaled = scaler.inverse_transform(X0_scaled)

    # Nos quedamos solo con R y C (ya mapeados a {0,1,2} en la etapa de preparación)
    RC_all = X0_unscaled[:, [idx_R, idx_C]]

    # Hacemos el mismo split train/val para RC_all, para alinear con X_val e y_val
    RC_train, RC_val = train_val_split(RC_all, train_size=train_size, val_size=val_size)

    return X_val, y_val, RC_val, features


# ======================================================================
# Métricas de evaluación
# ======================================================================

def evaluate_predictions(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Calcula métricas globales de evaluación para el modelo LSTM.

    - MSE y RMSE global (todas las respiraciones, todos los time steps)
    - MAE global
    - RMSE por respiración

    Returns
    -------
    core_metrics : dict
        Diccionario con mse, rmse, mae, rmse_mean, rmse_std.
    rmse_per_breath : np.ndarray
        Vector (n_val,) con el RMSE por respiración.
    """
    print("[eval] Generando predicciones sobre el set de validación...")
    y_pred = model.predict(X_val, verbose=0)
    y_pred = np.squeeze(y_pred)  # (n_val, 80)

    if y_pred.shape != y_val.shape:
        raise ValueError(
            f"Shape de predicciones {y_pred.shape} no coincide con y_val {y_val.shape}"
        )

    # Aplanamos para métricas globales
    y_true_flat = y_val.flatten()
    y_pred_flat = y_pred.flatten()

    mse = float(mean_squared_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))

    # RMSE por respiración (eje 1 = time_step)
    rmse_per_breath = np.sqrt(np.mean((y_pred - y_val) ** 2, axis=1))
    rmse_mean = float(rmse_per_breath.mean())
    rmse_std = float(rmse_per_breath.std())

    core_metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
    }

    return core_metrics, rmse_per_breath


def compute_group_rmse_by_R_C(
    rmse_per_breath: np.ndarray,
    RC_val: np.ndarray,
) -> Dict[str, Dict[int, float]]:
    """
    Calcula el RMSE promedio por grupo de R y C (códigos 0,1,2).

    Parameters
    ----------
    rmse_per_breath : np.ndarray
        Vector (n_val,) con RMSE para cada respiración.
    RC_val : np.ndarray
        Matriz (n_val, 2) con [R_code, C_code] para cada respiración.

    Returns
    -------
    group_rmse : dict
        {
          "R": {0: rmse_R0, 1: rmse_R1, 2: rmse_R2},
          "C": {0: rmse_C0, 1: rmse_C1, 2: rmse_C2},
        }
    """
    if RC_val.shape[1] != 2:
        raise ValueError("RC_val debe tener shape (n_val, 2) con columnas [R, C].")

    R_codes = RC_val[:, 0].astype(int)
    C_codes = RC_val[:, 1].astype(int)

    group_rmse_R: Dict[int, float] = {}
    group_rmse_C: Dict[int, float] = {}

    for r in sorted(np.unique(R_codes)):
        mask = R_codes == r
        if mask.any():
            group_rmse_R[r] = float(rmse_per_breath[mask].mean())

    for c in sorted(np.unique(C_codes)):
        mask = C_codes == c
        if mask.any():
            group_rmse_C[c] = float(rmse_per_breath[mask].mean())

    group_rmse = {"R": group_rmse_R, "C": group_rmse_C}
    return group_rmse


# ======================================================================
# Evaluación contra criterios de negocio
# ======================================================================

def check_business_criteria(
    core_metrics: Dict[str, float],
    group_rmse: Dict[str, Dict[int, float]],
    rmse_threshold: float = BUSINESS_RMSE_THRESHOLD,
    group_gap_threshold: float = GROUP_RMSE_GAP_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evalúa si el modelo cumple criterios mínimos de negocio:

    1) Desempeño global suficiente (RMSE por debajo de cierto umbral).
    2) Ausencia de sesgos fuertes entre grupos de R y C
       (que representen tipos de pulmón distintos).

    Returns
    -------
    result : dict
        {
          "meets_rmse_target": bool,
          "max_gap_R": float,
          "max_gap_C": float,
          "has_strong_bias": bool,
          "overall_recommendation": str,
        }
    """
    rmse = core_metrics["rmse"]
    meets_rmse_target = rmse <= rmse_threshold

    # Gap de RMSE entre grupos de R y C
    gap_R = 0.0
    gap_C = 0.0

    if group_rmse["R"]:
        values_R = list(group_rmse["R"].values())
        gap_R = max(values_R) - min(values_R)

    if group_rmse["C"]:
        values_C = list(group_rmse["C"].values())
        gap_C = max(values_C) - min(values_C)

    has_strong_bias = (gap_R > group_gap_threshold) or (gap_C > group_gap_threshold)

    # Reglas simples para recomendación
    if meets_rmse_target and not has_strong_bias:
        overall = (
            "✅ El modelo cumple el criterio de desempeño global y no muestra "
            "sesgos fuertes entre tipos de pulmón. Se recomienda para un despliegue piloto "
            "en entorno controlado."
        )
    elif meets_rmse_target and has_strong_bias:
        overall = (
            "⚠️ El modelo tiene buen RMSE global, pero muestra diferencias importantes "
            "entre grupos de R/C. Podría usarse en pruebas, pero **NO** debería "
            "desplegarse sin antes mitigar estos sesgos."
        )
    else:
        overall = (
            "❌ El modelo no alcanza el criterio mínimo de RMSE global. No se recomienda "
            "su despliegue en producción en el estado actual."
        )

    return {
        "meets_rmse_target": meets_rmse_target,
        "max_gap_R": float(gap_R),
        "max_gap_C": float(gap_C),
        "has_strong_bias": has_strong_bias,
        "overall_recommendation": overall,
    }


# ======================================================================
# Script principal: etapa de Evaluation (CRISP-DM)
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Etapa de Evaluation (CRISP-DM) para el modelo LSTM de "
            "Ventilator Pressure Prediction."
        )
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Ruta al archivo .keras del modelo entrenado.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.80,
        help="Proporción de datos para entrenamiento (resto es validación).",
    )
    args = parser.parse_args()

    model_path = Path(args.model_path)

    # 1) Cargar modelo entrenado
    model = load_trained_model(model_path)

    # 2) Reconstruir datos de evaluación (validación)
    X_val, y_val, RC_val, features = load_eval_data(
        train_size=args.train_size,
        val_size=1.0 - args.train_size,
    )

    # 3) Métricas globales
    core_metrics, rmse_per_breath = evaluate_predictions(model, X_val, y_val)

    print("\n[eval] Métricas globales en el set de validación:")
    print(f"  - MSE  : {core_metrics['mse']:.4f}")
    print(f"  - RMSE : {core_metrics['rmse']:.4f}")
    print(f"  - MAE  : {core_metrics['mae']:.4f}")
    print(f"  - RMSE por respiración (media): {core_metrics['rmse_mean']:.4f}")
    print(f"  - RMSE por respiración (std)  : {core_metrics['rmse_std']:.4f}")

    # 4) RMSE por grupo de R y C (para detectar sesgos por tipo de pulmón)
    group_rmse = compute_group_rmse_by_R_C(rmse_per_breath, RC_val)

    print("\n[eval] RMSE promedio por grupo de R (códigos 0,1,2):")
    for r_code, rmse_r in group_rmse["R"].items():
        print(f"  - R = {r_code}: RMSE = {rmse_r:.4f}")

    print("\n[eval] RMSE promedio por grupo de C (códigos 0,1,2):")
    for c_code, rmse_c in group_rmse["C"].items():
        print(f"  - C = {c_code}: RMSE = {rmse_c:.4f}")

    # 5) Evaluar criterios de negocio y riesgos
    business_eval = check_business_criteria(core_metrics, group_rmse)

    print("\n[eval] Evaluación respecto a criterios de negocio:")
    print(f"  - ¿Cumple umbral de RMSE global (≤ {BUSINESS_RMSE_THRESHOLD})? "
          f"{'Sí' if business_eval['meets_rmse_target'] else 'No'}")
    print(f"  - Diferencia máxima de RMSE entre grupos de R: "
          f"{business_eval['max_gap_R']:.4f}")
    print(f"  - Diferencia máxima de RMSE entre grupos de C: "
          f"{business_eval['max_gap_C']:.4f}")
    print(f"  - ¿Sesgo fuerte por grupos de R/C (> {GROUP_RMSE_GAP_THRESHOLD})? "
          f"{'Sí' if business_eval['has_strong_bias'] else 'No'}")

    print("\n[eval] Recomendación global:")
    print(business_eval["overall_recommendation"])
