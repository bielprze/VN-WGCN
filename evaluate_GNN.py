#!python
#!/usr/bin/env python3
"""
evaluate_GNN.py

Evaluate trained Spatio-Temporal GCN models:
 - S-TGCN when USE_DELAUNAY=False
 - VNWS-TGCN when USE_DELAUNAY=True
"""

import os
import random
import logging
import statistics
from pathlib import Path
from typing import Tuple, List
import yaml

import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras.losses import MeanSquaredError
from keras.optimizers import RMSprop

import helper
from gcnnmodel import (
    GraphInfo,
    LSTMGC,
    compute_adjacency_matrix,
    calculate_adjacency_matrix_Delaunay,
    calculate_route_distance,
    preprocess,
    create_tf_dataset,
)

# ─── Configuration ──────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Data & paths
FOLDER_TO_LOAD = Path(cfg["folder_to_load"])
GEOJSON_PATH   = Path(cfg["geojson_path"])
SAVE_FOLDER    = Path(cfg["save_folder"])
OUTPUT_CSV     = Path(cfg["output_csv"])

# Data split
TRAIN_SIZE = cfg["train_size"]
VAL_SIZE   = cfg["val_size"]

# Graph parameters
SIGMA2  = cfg["sigma2"]
EPSILON = cfg["epsilon"]

# Delaunay settings
USE_DELAUNAY       = cfg["use_delaunay"]
MAX_DEPTH          = cfg["max_depth"]
ADJ_SCALER_OPTIONS = cfg["adj_scaler_options"]

# Model / dataset parameters
IN_FEAT           = cfg["in_feat"]
OUT_FEAT          = cfg["out_feat"]
LSTM_UNITS        = cfg["lstm_units"]
GRAPH_CONV_PARAMS = cfg["graph_conv_params"]
INPUT_SEQ_LEN     = cfg["input_seq_len"]
MULTI_HORIZON     = cfg.get("multi_horizon", False)

# Evaluation loop settings
FORECAST_HORIZONS = cfg["forecast_horizons"]
SEEDS             = cfg["seeds"]

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


def ensure_dirs():
    SAVE_FOLDER.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load raw data and compute route distances & sensor keys."""
    sensors, full_data, all_keys = helper.load_data_and_combine_with_geojson(
        str(FOLDER_TO_LOAD) + "/", str(GEOJSON_PATH)
    )
    route_distances, all_points = calculate_route_distance(sensors)
    logging.info("Loaded route_distances shape=%s", route_distances.shape)
    logging.info("Loaded speeds_array    shape=%s", full_data.shape)
    sensor_keys = list(sensors.keys())
    return route_distances, all_points, full_data, sensor_keys, all_keys


def make_datasets(
    speeds: np.ndarray, seed: int, horizon: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split data, normalize, and build TF datasets."""
    train_arr, val_arr, test_arr, scaler = preprocess(speeds, TRAIN_SIZE, VAL_SIZE)
    logging.info(
        "Split data: train=%s, val=%s, test=%s",
        train_arr.shape, val_arr.shape, test_arr.shape
    )

    train_ds = create_tf_dataset(
        train_arr, INPUT_SEQ_LEN, horizon, batch_size=train_arr.shape[0] if False else 64, seed=seed
    )
    val_ds = create_tf_dataset(
        val_arr, INPUT_SEQ_LEN, horizon, batch_size=64, seed=seed
    )
    test_ds = create_tf_dataset(
        test_arr,
        INPUT_SEQ_LEN,
        horizon,
        batch_size=test_arr.shape[0],
        shuffle=False,
        multi_horizon=MULTI_HORIZON,
        seed=seed,
    )
    return train_ds, val_ds, test_ds, scaler


def build_graph(
    route_distances: np.ndarray,
    all_points: np.ndarray,
    use_delaunay: bool,
    max_depth: int,
    adj_scaler: int,
) -> GraphInfo:
    """Compute adjacency matrix and wrap into GraphInfo."""
    if use_delaunay:
        adj = calculate_adjacency_matrix_Delaunay(all_points, max_depth, adj_scaler)
    else:
        adj = compute_adjacency_matrix(route_distances, SIGMA2, EPSILON)

    idx, nbr = np.where(adj > 0.0)
    weights = adj[idx, nbr]
    graph = GraphInfo(
        edges=(idx.tolist(), nbr.tolist(), weights.tolist()),
        num_nodes=adj.shape[0],
    )
    logging.info("Graph built: %d nodes, %d edges", graph.num_nodes, len(idx))
    return graph


def build_model(graph: GraphInfo, horizon: int) -> Model:
    """Instantiate and compile the trained model architecture."""
    stgcn = LSTMGC(
        IN_FEAT,
        OUT_FEAT,
        LSTM_UNITS,
        INPUT_SEQ_LEN,
        horizon,
        graph,
        GRAPH_CONV_PARAMS,
        seed=None,
    )
    inputs = layers.Input(shape=(INPUT_SEQ_LEN, graph.num_nodes, IN_FEAT))
    outputs = stgcn(inputs)
    model = Model(inputs, outputs)
    model.compile(optimizer=RMSprop(2e-4), loss=MeanSquaredError())
    return model


def evaluate_model(
    model: Model,
    test_ds: tf.data.Dataset,
    scaler,
    sensor_keys: List[str],
    all_keys: List[str],
) -> Tuple[float, float, float, float]:
    """
    Run a single-batch prediction, invert scaling, and compute
    mean MSE, RMSE, MAE, and MRE across sensors.
    """
    x, y_true = next(test_ds.as_numpy_iterator())
    y_pred = model.predict(x)

    # reshape to [batch, sensors]
    x_last = x[:, -1, :, 0]
    y_true_flat = y_true[:, 0, :]
    y_pred_flat = y_pred[:, 0, :]

    # inverse scale
    x_inv = scaler.inverse_transform(x_last)
    y_true_inv = scaler.inverse_transform(y_true_flat)
    y_pred_inv = scaler.inverse_transform(y_pred_flat)

    mse_list, rmse_list, mae_list, mre_list = [], [], [], []

    for pred, true in zip(y_pred_inv, y_true_inv):
        errors = pred - true
        mse = np.mean(errors ** 2)
        mae = np.mean(np.abs(errors))

        nonzero_mask = true > 1e-6
        mre = np.mean(np.abs(errors[nonzero_mask] / true[nonzero_mask])) if np.any(nonzero_mask) else 0.0

        mse_list.append(mse)
        mae_list.append(mae)
        rmse_list.append(np.sqrt(mse))
        mre_list.append(mre)

    return (
        statistics.mean(mse_list),
        statistics.stdev(mse_list),
        statistics.mean(rmse_list),
        statistics.stdev(rmse_list),
        statistics.mean(mae_list),
        statistics.stdev(mae_list),
        statistics.mean(mre_list),
        statistics.stdev(mre_list),
    )


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    ensure_dirs()

    # Write CSV header
    header = [
        "horizon", "seed", "use_Delaunay", "max_depth", "adj_scaler",
        "mean_mse", "sd_mse", "mean_rmse", "sd_rmse",
        "mean_mae", "sd_mae", "mean_mre", "sd_mre",
    ]
    with OUTPUT_CSV.open("w", newline="") as f:
        f.write(",".join(header) + "\n")

    route_distances, all_points, speeds, sensor_keys, all_keys = load_data()

    for horizon in FORECAST_HORIZONS:
        for seed in SEEDS:
            set_seed(seed)
            _, _, test_ds, scaler = make_datasets(speeds, seed, horizon)

            for adj_scaler in ADJ_SCALER_OPTIONS:
                graph = build_graph(
                    route_distances, all_points, USE_DELAUNAY, MAX_DEPTH, adj_scaler
                )
                model = build_model(graph, horizon)

                # load trained weights
                weight_file = (
                    SAVE_FOLDER
                    / f"darmstadt_h{horizon}_s{seed}_"
                    f"{'D' if USE_DELAUNAY else 'S'}_"
                    f"d{MAX_DEPTH}_a{adj_scaler}.h5"
                )
                model.load_weights(str(weight_file))
                logging.info("Loaded weights from %s", weight_file)

                # evaluate
                metrics = evaluate_model(model, test_ds, scaler, sensor_keys, all_keys)
                row = [horizon, seed, USE_DELAUNAY, MAX_DEPTH, adj_scaler] + list(metrics)

                with OUTPUT_CSV.open("a", newline="") as f:
                    f.write(",".join(str(v) for v in row) + "\n")

                logging.info(
                    "Eval done: horizon=%d, seed=%d, scaler=%d → mse=%.4f, rmse=%.4f",
                    horizon, seed, adj_scaler, metrics[0], metrics[2]
                )


if __name__ == "__main__":
    main()