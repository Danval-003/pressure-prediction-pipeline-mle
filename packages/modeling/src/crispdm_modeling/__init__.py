from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ======================================================================
# Utilidades de división train / valid (como en el cuadernito)
# ======================================================================

def train_val_split(
    data_array: np.ndarray,
    train_size: float = 0.80,
    val_size: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divide un array en subconjuntos de entrenamiento y validación
    sin mezclar el orden (como en el notebook original).

    Parameters
    ----------
    data_array : np.ndarray
        Datos a dividir (por ejemplo X o y).
    train_size : float
        Proporción para el conjunto de entrenamiento.
    val_size : float
        Proporción para el conjunto de validación.

    Returns
    -------
    (train_array, val_array)
    """
    n = len(data_array)
    if not np.isclose(train_size + val_size, 1.0):
        raise ValueError("train_size + val_size debe ser igual a 1.0")

    train_end = int(train_size * n)
    val_end = train_end + int(val_size * n)

    train_array = data_array[:train_end]
    val_array = data_array[train_end:val_end]

    return train_array, val_array


# ======================================================================
# Construcción del modelo LSTM (mismo estilo que el notebook)
# ======================================================================

def build_lstm_model(
    input_shape: Tuple[int, int],
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """
    Crea y compila un modelo LSTM para predicción de presión.

    Arquitectura basada en:
      - 3 capas LSTM apiladas: 128, 64, 32 unidades
      - Dropout + BatchNormalization entre capas
      - Capa densa final de salida (1) para regresión por time step.

    Parameters
    ----------
    input_shape : (timesteps, n_features)
        Shape esperada de X (por ejemplo: (80, 39)).
    learning_rate : float
        Tasa de aprendizaje del optimizador Adam.

    Returns
    -------
    model : tf.keras.Model
        Modelo LSTM compilado con loss MSE.
    """
    timesteps, n_features = input_shape

    model = Sequential()
    # Primera capa LSTM
    model.add(
        LSTM(
            units=128,
            input_shape=(timesteps, n_features),
            return_sequences=True,
        )
    )
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Segunda capa LSTM
    model.add(LSTM(units=64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Tercera capa LSTM
    model.add(LSTM(units=32, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # Capa de salida para regresión (una presión por time step)
    model.add(Dense(1))

    # Compilación
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss="mse", metrics=["mse"])

    return model


# ======================================================================
# Entrenamiento del modelo
# ======================================================================

def train_lstm_model(
    model: tf.keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 200,
    batch_size: int = 32,
    patience: int = 35,
    checkpoint_path: str | None = "models/lstm_model.keras",
    verbose: int = 1,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Entrena el modelo LSTM con EarlyStopping y, opcionalmente,
    guarda el mejor modelo en disco.

    Parameters
    ----------
    model : tf.keras.Model
        Modelo ya construido y compilado.
    X_train, y_train : np.ndarray
        Datos de entrenamiento (shapes típicos: (N, 80, F) y (N, 80)).
    X_val, y_val : np.ndarray
        Datos de validación.
    epochs : int
        Número máximo de épocas.
    batch_size : int
        Tamaño de batch.
    patience : int
        Paciencia para EarlyStopping sobre val_loss.
    checkpoint_path : str | None
        Ruta donde guardar el mejor modelo (.keras). Si None,
        no se guarda checkpoint.
    verbose : int
        Verbosidad de Keras .fit (0, 1 o 2).

    Returns
    -------
    model : tf.keras.Model
        Modelo con los mejores pesos (por EarlyStopping).
    history : tf.keras.callbacks.History
        Historial de entrenamiento.
    """
    callbacks = []

    if patience is not None and patience > 0:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            mode="min",
            restore_best_weights=True,
        )
        callbacks.append(early_stopping)

    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path)
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
        )
        callbacks.append(model_checkpoint)

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=verbose,
    )

    return model, history


# ======================================================================
# Evaluación del modelo
# ======================================================================

def evaluate_lstm_model(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Dict[str, Any]:
    """
    Evalúa el modelo LSTM sobre el set de validación.

    Calcula:
      - val_loss y val_mse desde model.evaluate
      - RMSE promedio por respiración (sobre la secuencia de 80 pasos)

    Parameters
    ----------
    model : tf.keras.Model
        Modelo ya entrenado.
    X_val, y_val : np.ndarray
        Set de validación.

    Returns
    -------
    metrics : dict
        Diccionario con métricas de evaluación.
    """
    # Pérdida y MSE desde Keras
    eval_results = model.evaluate(X_val, y_val, verbose=0)
    # model.metrics_names típicamente: ['loss', 'mse']
    loss = float(eval_results[0])
    mse = float(eval_results[1]) if len(eval_results) > 1 else loss

    # Predicciones
    y_pred = model.predict(X_val, verbose=0)
    # Squeezear para que el shape sea (N, 80)
    y_pred = np.squeeze(y_pred)

    # Asegurar shapes compatibles
    if y_pred.shape != y_val.shape:
        raise ValueError(
            f"Shape de predicciones {y_pred.shape} no coincide con y_val {y_val.shape}"
        )

    # RMSE por respiración (secuencia)
    rmse_per_breath = np.sqrt(np.mean((y_pred - y_val) ** 2, axis=1))
    rmse_mean = float(rmse_per_breath.mean())
    rmse_std = float(rmse_per_breath.std())

    metrics = {
        "val_loss": loss,
        "val_mse": mse,
        "val_rmse_mean": rmse_mean,
        "val_rmse_std": rmse_std,
    }
    return metrics

