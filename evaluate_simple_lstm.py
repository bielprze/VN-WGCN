#!python
#!/usr/bin/env python3
"""
evaluate_simple_lstm.py

Evaluate pretrained simple LSTM models on test data and compute summary metrics.
"""
import logging
import random
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
import csv

import helper
from simple_lstm import generate_dataset, build_lstm_model, CHECKPOINTS_ROOT
from gcnnmodel import preprocess

# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_FOLDER = Path("2024-03-01_35")
GEOJSON_PATH = Path("sensors_location.geojson")
EVAL_OUTPUT = Path("evaluation_simple_lstm.csv")

TRAIN_SIZE = 0.5
VAL_SIZE = 0.2
INPUT_SEQ_LEN = 12
FORECAST_HORIZONS = [1, 2, 3]
SEEDS = list(range(10))

# Logging setup
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# Deterministic runs
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Invert scaling on multi-horizon predictions
def invert_scaled(
    y_scaled: np.ndarray,
    scaler,
    num_features: int,
    horizon: int,
) -> np.ndarray:
    """
    y_scaled shape: (samples, num_features*horizon)
    returns y_inv same shape
    """
    samples = y_scaled.shape[0]
    reshaped = y_scaled.reshape((samples * horizon, num_features))
    inv = scaler.inverse_transform(reshaped)
    return inv.reshape((samples, num_features * horizon))

# Compute metrics across samples

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Returns mean and sd for MSE, RMSE, MAE, and MRE.
    """
    samples = y_true.shape[0]
    feats = y_true.shape[1]

    mse_list, mae_list, rmse_list, mre_list = [], [], [], []

    for i in range(samples):
        true = y_true[i]
        pred = y_pred[i]
        err = pred - true
        mse = np.mean(err ** 2)
        mae = np.mean(np.abs(err))
        rmse = np.sqrt(mse)
        nonzero = true > 1e-6
        mre = np.mean(np.abs(err[nonzero] / true[nonzero])) if np.any(nonzero) else 0.0

        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(rmse)
        mre_list.append(mre)

    return (
        np.mean(mse_list), np.std(mse_list),
        np.mean(rmse_list), np.std(rmse_list),
        np.mean(mae_list), np.std(mae_list),
        np.mean(mre_list), np.std(mre_list),
    )

# Main evaluation loop
def main():
    setup_logging()
    # Write CSV header
    headers = [
        "horizon", "seed",
        "mean_mse", "sd_mse",
        "mean_rmse", "sd_rmse",
        "mean_mae", "sd_mae",
        "mean_mre", "sd_mre",
    ]
    with EVAL_OUTPUT.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)

    # Load full data and scaler
    sensors, full_data, _ = helper.load_data_and_combine_with_geojson(
        str(DATA_FOLDER) + "/", str(GEOJSON_PATH)
    )
    _, _, test_arr, scaler = preprocess(full_data, TRAIN_SIZE, VAL_SIZE)
    num_features = test_arr.shape[1]

    for horizon in FORECAST_HORIZONS:
        for seed in SEEDS:
            set_seed(seed)
            # build model and load weights
            model = build_lstm_model(INPUT_SEQ_LEN, num_features, horizon)
            ckpt_dir = CHECKPOINTS_ROOT / f"h{horizon}_s{seed}"
            latest = max(ckpt_dir.glob("model*.h5"), key=lambda p: p.stat().st_mtime)
            model.load_weights(str(latest))
            logging.info("Loaded weights: %s", latest)

            # prepare test inputs
            X_test, y_test = generate_dataset(test_arr, INPUT_SEQ_LEN, horizon)
            # predict and invert scaling
            y_pred_scaled = model.predict(X_test)
            y_true_inv = invert_scaled(y_test, scaler, num_features, horizon)
            y_pred_inv = invert_scaled(y_pred_scaled, scaler, num_features, horizon)

            # compute and save metrics
            metrics = calculate_metrics(y_true_inv, y_pred_inv)
            row = [horizon, seed] + [f"{m:.4f}" for m in metrics]
            with EVAL_OUTPUT.open("a", newline="") as csvfile:
                csv.writer(csvfile).writerow(row)
            logging.info("Eval h=%d seed=%d → mse=%.4f rmse=%.4f", horizon, seed, metrics[0], metrics[2])

if __name__ == "__main__":
    main()
