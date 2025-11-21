"""Execute the CRISP-DM pipeline end-to-end in a resource-aware mode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from crispdm_data_preparation import (
    TIMESTEPS_PER_BREATH,
    get_lstm_ready_xy,
    load_raw_train_csv,
)
from crispdm_modeling import (
    build_lstm_model,
    train_val_split,
    train_lstm_model,
)
from crispdm_evaluating import (
    evaluate_predictions,
    compute_group_rmse_by_R_C,
    check_business_criteria,
)
from crispdm_deployment import (
    preprocess_for_inference,
    predict_pressures,
    build_predictions_dataframe,
)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            if isinstance(key, (str, int, float, bool)) or key is None:
                new_key = key
            elif isinstance(key, (np.integer,)):
                new_key = int(key)
            elif isinstance(key, (np.floating,)):
                new_key = float(key)
            else:
                new_key = str(key)
            sanitized[new_key] = _sanitize_for_json(value)
        return sanitized
    if isinstance(obj, list):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return _json_default(obj)
    return obj


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _limit_breaths(X: np.ndarray, y: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or X.shape[0] <= limit:
        return X, y
    print(
        f"[pipeline] Limiting dataset to the first {limit} breaths "
        f"instead of {X.shape[0]} for a lighter run."
    )
    return X[:limit], y[:limit]


def _extract_rc_codes(X: np.ndarray, scaler, features: list[str], train_size: float) -> tuple[np.ndarray, np.ndarray]:
    n_breaths = X.shape[0]
    n_features = X.shape[2]
    first_step_scaled = X[:, 0, :].reshape(n_breaths, n_features)
    first_step_unscaled = scaler.inverse_transform(first_step_scaled)
    idx_R = features.index("R")
    idx_C = features.index("C")
    rc_all = np.rint(first_step_unscaled[:, [idx_R, idx_C]]).astype(int)
    val_size = 1.0 - train_size
    rc_train, rc_val = train_val_split(rc_all, train_size=train_size, val_size=val_size)
    return rc_train, rc_val


def _prepare_deployment_sample(raw_df: pd.DataFrame, breaths: int) -> pd.DataFrame:
    unique_ids = raw_df["breath_id"].unique()
    selected = unique_ids[:breaths]
    subset = raw_df[raw_df["breath_id"].isin(selected)].copy()
    if subset.empty:
        raise ValueError(
            "No rows selected for deployment inference. "
            "Check that the dataset contains enough breaths."
        )
    return subset


def _generate_sample_dataset(target: Path, breaths: int) -> None:
    print(f"[pipeline] Generating sample dataset at {target} ({breaths} breaths).")
    rows = []
    idx = 0
    breath_ids = range(breaths)
    resistances = [5, 20, 50]
    compliances = [10, 20, 50]

    for b in breath_ids:
        R = resistances[b % len(resistances)]
        C = compliances[b % len(compliances)]
        base_pressure = 5 + (b * 0.5)
        for step in range(TIMESTEPS_PER_BREATH):
            time_step = step * 0.033
            u_in = max(0.0, min(100.0, (step / TIMESTEPS_PER_BREATH) * 80))
            u_out = 1 if step > TIMESTEPS_PER_BREATH // 2 else 0
            pressure = base_pressure + (u_in * 0.05) - (u_out * 0.5)
            rows.append(
                {
                    "id": f"id_{idx}",
                    "breath_id": b,
                    "R": R,
                    "C": C,
                    "time_step": time_step,
                    "u_in": u_in,
                    "u_out": u_out,
                    "pressure": pressure,
                }
            )
            idx += 1

    df = pd.DataFrame(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def _ensure_data_source(path: Path, sample_breaths: int) -> Path:
    """
    Garantiza que exista una fuente de datos utilizable.

    Estrategia:
      1. Si existe dataset local (archivo o carpeta con train.csv / train_part_*.csv), se usa.
      2. Si NO existe dataset local pero está configurado GITHUB_DATA_BASE_URL,
         se asume que crispdm_data_preparation se encargará de descargar los datos
         remotos (no se genera nada sintético).
      3. Si no hay datos locales NI remotos configurados, se genera un dataset
         sintético liviano para poder ejecutar la tubería.
    """

    github_base = "https://raw.githubusercontent.com/Danval-003/pressure-prediction-pipeline-mle/refs/heads/main/data/raw"

    def has_data_dir(directory: Path) -> bool:
        if not directory.exists():
            return False
        if list(directory.glob("train_part_*.csv")):
            return True
        if (directory / "train.csv").exists():
            return True
        return False

    # 1) Datos locales existentes
    if path.exists():
        if path.is_file():
            return path
        if path.is_dir() and has_data_dir(path):
            return path

    # 2) No hay datos locales, pero sí fuente remota -> la lógica de descarga
    #    (load_raw_train_csv / get_lstm_ready_xy) se encargará usando la ruta.
    if github_base:
        print(
            "[pipeline] No local data found at "
            f"{path}, but GITHUB_DATA_BASE_URL is set. "
            "Remote data will be used when loading."
        )
        return path

    # 3) Sin datos locales ni remotos: generar dataset sintético pequeño
    print(
        "[pipeline] No local data found and GITHUB_DATA_BASE_URL is not set. "
        "Generating a small synthetic dataset to proceed."
    )

    # Si el path termina con extensión, asumimos archivo
    if path.suffix:
        target_file = path
    else:
        target_file = path / "train_part_sample.csv"

    _generate_sample_dataset(target_file, breaths=sample_breaths)

    # Si originalmente se apuntaba a un directorio, devolvemos el directorio
    if not path.suffix:
        return target_file.parent
    return target_file


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = Path(args.data_path)
    source_path = _ensure_data_source(data_path, args.sample_breaths)
    model_path = Path(args.model_path)
    artifacts_dir = _ensure_dir(Path(args.artifacts_dir))
    _ensure_dir(model_path.parent)

    print(f"[pipeline] Loading data from {source_path} ...")
    X, y, scaler, features = get_lstm_ready_xy(csv_path=source_path)
    X, y = _limit_breaths(X, y, args.max_breaths)

    train_size = args.train_size
    val_size = 1.0 - train_size

    X_train, X_val = train_val_split(X, train_size=train_size, val_size=val_size)
    y_train, y_val = train_val_split(y, train_size=train_size, val_size=val_size)

    print("[pipeline] Building LSTM model ...")
    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]), learning_rate=args.learning_rate)

    print("[pipeline] Training model ...")
    model, history = train_lstm_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        checkpoint_path=model_path,
        verbose=1,
    )
    model.save(model_path)

    print("[pipeline] Evaluating model ...")
    metrics, rmse_per_breath = evaluate_predictions(model, X_val, y_val)
    _, rc_val = _extract_rc_codes(X, scaler, features, train_size=train_size)
    group_rmse = compute_group_rmse_by_R_C(rmse_per_breath, rc_val)
    business_eval = check_business_criteria(metrics, group_rmse)

    print("[pipeline] Running deployment inference sample ...")
    raw_df = load_raw_train_csv(csv_path=source_path)
    deploy_df = _prepare_deployment_sample(raw_df, args.deploy_breaths)
    deployed_model = load_model(model_path)
    X_input, breath_ids, time_steps = preprocess_for_inference(deploy_df, scaler, features)
    preds = predict_pressures(deployed_model, X_input)
    pred_df = build_predictions_dataframe(preds, breath_ids, time_steps)
    predictions_path = artifacts_dir / "deployment_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)

    summary = {
        "data_path": str(data_path),
        "breaths_used": int(X.shape[0]),
        "train_size": train_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "metrics": metrics,
        "group_rmse": group_rmse,
        "business_evaluation": business_eval,
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
    }
    serializable_summary = _sanitize_for_json(summary)
    summary_json = json.dumps(serializable_summary, indent=2)
    (artifacts_dir / "pipeline_report.json").write_text(summary_json, encoding="utf-8")

    print("[pipeline] Completed. Summary:")
    print(summary_json)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=str,
        default="/app/data/raw",
        help=(
            "Ruta al archivo o carpeta con el dataset. "
            "Si no existe pero está configurado GITHUB_DATA_BASE_URL, "
            "se usarán datos remotos. Si no hay datos ni remotos ni locales, "
            "se generará un dataset sintético pequeño."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/lstm_docker.keras",
        help="Ruta donde se guardará el modelo entrenado.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directorio donde se escriben reportes y predicciones.",
    )
    parser.add_argument(
        "--max-breaths",
        type=int,
        default=1500,
        help="Máximo de respiraciones a usar para entrenar/evaluar (controla RAM).",
    )
    parser.add_argument(
        "--deploy-breaths",
        type=int,
        default=3,
        help="Cantidad de respiraciones para generar predicciones de despliegue.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Porción usada para entrenamiento (resto para validación).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Número máximo de épocas para entrenar.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Tamaño de batch para el entrenamiento.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Paciencia para EarlyStopping.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate para el optimizador Adam.",
    )
    parser.add_argument(
        "--sample-breaths",
        type=int,
        default=5,
        help="Respiraciones sintéticas a generar si no hay dataset disponible.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
