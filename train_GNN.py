#!python
#!/usr/bin/env python3
"""
train_GNN.py

Spatio-Temporal Graph Convolutional Network training pipeline:
 - S-TGCN when use_Delaunay=False
 - VNWS-TGCN when use_Delaunay=True
"""

import os
import random
import logging
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from keras import layers, Model
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.optimizers import RMSprop
from tqdm import tqdm

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

# Data & paths
FOLDER_TO_LOAD = "2024-03-01_35/"
GEOJSON_PATH = "sensors_location.geojson"
SAVE_FOLDER = "weights"

# Train/val/test split (fractions)
TRAIN_SIZE = 0.5
VAL_SIZE = 0.2

# Graph parameters
SIGMA2 = 0.1
EPSILON = 0.95

# Delaunay settings
USE_DELAUNAY = True
MAX_DEPTH = 5
ADJ_SCALER_OPTIONS = [0, 1, 2]

# Model / data parameters
IN_FEAT = 1
OUT_FEAT = 10
LSTM_UNITS = 64
GRAPH_CONV_PARAMS = {
    "aggregation_type": "mean",
    "combination_type": "concat",
    "activation": None,
}

# Training loop settings
EPOCHS = 40
BATCH_SIZE = 64
INPUT_SEQ_LEN = 12
FORECAST_HORIZONS = [1, 2, 3]
SEEDS = list(range(10))


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
    os.makedirs(SAVE_FOLDER, exist_ok=True)


def load_data() -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """Load raw speeds and compute distances/points."""
    sensors, full_data, _ = helper.load_data_and_combine_with_geojson(
        FOLDER_TO_LOAD, GEOJSON_PATH
    )
    route_distances, all_points = calculate_route_distance(sensors)
    logging.info("route_distances shape=%s", route_distances.shape)
    logging.info("speeds_array    shape=%s", full_data.shape)
    return route_distances, all_points, full_data


def make_datasets(
    speeds: np.ndarray, seed: int, forecast_horizon: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split, normalize, and wrap in TF datasets."""
    train_arr, val_arr, test_arr, _ = preprocess(speeds, TRAIN_SIZE, VAL_SIZE)
    logging.info("train size=%s, val size=%s, test size=%s",
                 train_arr.shape, val_arr.shape, test_arr.shape)

    train_ds = create_tf_dataset(
        train_arr, INPUT_SEQ_LEN, forecast_horizon, BATCH_SIZE, seed=seed
    )
    val_ds = create_tf_dataset(
        val_arr, INPUT_SEQ_LEN, forecast_horizon, BATCH_SIZE, seed=seed
    )
    test_ds = create_tf_dataset(
        test_arr,
        INPUT_SEQ_LEN,
        forecast_horizon,
        batch_size=test_arr.shape[0],
        shuffle=False,
        seed=seed,
    )
    return train_ds, val_ds, test_ds


def build_graph(
    route_distances: np.ndarray,
    all_points: np.ndarray,
    use_delaunay: bool,
    max_depth: int,
    adj_scaler: int,
) -> GraphInfo:
    """Compute adjacency and pack into GraphInfo."""
    if use_delaunay:
        adj = calculate_adjacency_matrix_Delaunay(all_points, max_depth, adj_scaler)
    else:
        adj = compute_adjacency_matrix(route_distances, SIGMA2, EPSILON)

    # only edges with weight > 0
    idx, nbr = np.where(adj > 0.0)
    weights = adj[idx, nbr]
    graph = GraphInfo(
        edges=(idx.tolist(), nbr.tolist(), weights.tolist()),
        num_nodes=adj.shape[0],
    )
    logging.info("Graph: %d nodes, %d edges", graph.num_nodes, len(idx))
    return graph


def build_model(graph: GraphInfo, forecast_horizon: int) -> Model:
    """Instantiate and compile the ST-GCN model."""
    stgcn = LSTMGC(
        IN_FEAT,
        OUT_FEAT,
        LSTM_UNITS,
        INPUT_SEQ_LEN,
        forecast_horizon,
        graph,
        GRAPH_CONV_PARAMS,
        seed=None,  # seed already set globally
    )
    inputs = layers.Input(shape=(INPUT_SEQ_LEN, graph.num_nodes, IN_FEAT))
    outputs = stgcn(inputs)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=RMSprop(learning_rate=2e-4),
        loss=MeanSquaredError(),
    )
    return model


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    ensure_dirs()

    route_distances, all_points, speeds = load_data()

    for horizon in tqdm(FORECAST_HORIZONS, desc="Horizons"):
        for seed in tqdm(SEEDS, desc="Seeds", leave=False):
            set_seed(seed)
            train_ds, val_ds, test_ds = make_datasets(speeds, seed, horizon)

            for adj_scaler in ADJ_SCALER_OPTIONS:
                graph = build_graph(
                    route_distances, all_points, USE_DELAUNAY, MAX_DEPTH, adj_scaler
                )
                model = build_model(graph, horizon)

                logging.info(
                    "Training: seed=%d, horizon=%d, adj_scaler=%d",
                    seed, horizon, adj_scaler
                )
                model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=EPOCHS,
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=2,
                )

                filename = f"darmstadt_h{horizon}_s{seed}_" \
                           f"{'D' if USE_DELAUNAY else 'S'}_" \
                           f"d{MAX_DEPTH}_a{adj_scaler}.h5"
                save_path = os.path.join(SAVE_FOLDER, filename)
                model.save_weights(save_path)
                logging.info("Saved weights to %s", save_path)


if __name__ == "__main__":
    main()