"""Execute the CRISP-DM pipeline end-to-end in a resource-aware mode."""

from __future__ import annotations

import argparse
import json
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


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _limit_breaths(X: np.ndarray, y: np.ndarray, limit: int | None) -> tuple[np.ndarray, np.ndarray]:
    if limit is None or X.shape[0] <= limit:
        return X, y
    print(f"[pipeline] Limiting dataset to the first {limit} breaths "
          f"instead of {X.shape[0]} for a lighter run.")
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
    if subset.empty():
        raise ValueError("No rows selected for deployment inference. "
                         "Check that the dataset contains enough breaths.")
    return subset


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    artifacts_dir = _ensure_dir(Path(args.artifacts_dir))
    _ensure_dir(model_path.parent)

    print(f"[pipeline] Loading data from {data_path} ...")
    X, y, scaler, features = get_lstm_ready_xy(csv_path=data_path)
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
    raw_df = load_raw_train_csv(csv_path=data_path)
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
    (artifacts_dir / "pipeline_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[pipeline] Completed. Summary:")
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data/train_part_001.csv",
        help="Ruta al archivo o carpeta con el dataset (por defecto /data/train_part_001.csv).",
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
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
