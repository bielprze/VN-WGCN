#!python
#!/usr/bin/env python3
"""
simple_lstm.py

Train and evaluate a simple LSTM forecasting model on traffic speed data.
"""
import os
import random
import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm

import helper
from gcnnmodel import calculate_route_distance, preprocess


# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_FOLDER = Path("2024-03-01_35")
GEOJSON_PATH = Path("sensors_location.geojson")
CHECKPOINTS_ROOT = Path("weights_lstm")

TRAIN_SIZE = 0.5
VAL_SIZE = 0.2
INPUT_SEQ_LEN = 12
FORECAST_HORIZONS = [1, 2, 3]
SEEDS = list(range(10))

# Model hyperparameters
LSTM_UNITS = 200
DENSE_UNITS = 200
LEARNING_RATE = 1e-4
EPOCHS = 5
BATCH_SIZE = 128
VERBOSE = 1


# ─── Utilities ─────────────────────────────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dirs(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def generate_dataset(
    data: np.ndarray,
    input_seq_len: int,
    forecast_horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input/output arrays for LSTM: X shape (samples, input_seq_len, features),
    y shape (samples, features * forecast_horizon).
    """
    total_steps, num_features = data.shape
    samples = total_steps - input_seq_len - forecast_horizon + 1
    X = np.zeros((samples, input_seq_len, num_features), dtype=float)
    y = np.zeros((samples, num_features * forecast_horizon), dtype=float)

    for i in range(samples):
        X[i] = data[i : i + input_seq_len]
        for h in range(forecast_horizon):
            y[i, h * num_features : (h + 1) * num_features] = data[
                i + input_seq_len + h
            ]
    return X, y


def build_lstm_model(
    input_seq_len: int,
    num_features: int,
    forecast_horizon: int,
) -> keras.Model:
    """
    Build and compile a simple LSTM forecasting model.
    """
    n_outputs = num_features * forecast_horizon
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(input_seq_len, num_features)),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(n_outputs),
    ])
    optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])
    return model


def evaluate_model_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_dir: Path,
) -> None:
    """
    Train the LSTM model and save checkpoints and logs.
    """
    ensure_dirs(save_dir)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(save_dir / "model.{epoch:02d}-{val_loss:.4f}.h5"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        keras.callbacks.CSVLogger(
            filename=str(save_dir / "training_log.csv"),
            separator=",",
            append=False,
        ),
    ]
    model = build_lstm_model(INPUT_SEQ_LEN, X_train.shape[2], y_train.shape[1] // X_train.shape[2])
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=VERBOSE,
    )


def main():
    setup_logging()
    ensure_dirs(CHECKPOINTS_ROOT)

    # Load and preprocess data
    sensors, full_data, _ = helper.load_data_and_combine_with_geojson(
        str(DATA_FOLDER) + "/", str(GEOJSON_PATH)
    )
    _, _ = calculate_route_distance(sensors)  # distances not used in simple LSTM
    train_arr, val_arr, _, scaler = preprocess(full_data, TRAIN_SIZE, VAL_SIZE)

    for horizon in tqdm(FORECAST_HORIZONS, desc="Horizons"):
        for seed in tqdm(SEEDS, desc="Seeds", leave=False):
            set_seed(seed)
            X_train, y_train = generate_dataset(train_arr, INPUT_SEQ_LEN, horizon)
            X_val, y_val = generate_dataset(val_arr, INPUT_SEQ_LEN, horizon)

            save_dir = CHECKPOINTS_ROOT / f"h{horizon}_s{seed}"
            logging.info("Training LSTM: horizon=%d seed=%d", horizon, seed)
            evaluate_model_lstm(X_train, y_train, X_val, y_val, save_dir)


if __name__ == "__main__":
    main()
