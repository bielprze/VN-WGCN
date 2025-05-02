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
import yaml

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
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

# Data & paths
FOLDER_TO_LOAD     = cfg["folder_to_load"]
GEOJSON_PATH       = cfg["geojson_path"]
SAVE_FOLDER        = cfg["save_folder"]

# Train/val/test split (fractions)
TRAIN_SIZE         = cfg["train_size"]
VAL_SIZE           = cfg["val_size"]

# Graph parameters
SIGMA2             = cfg["sigma2"]
EPSILON            = cfg["epsilon"]

# Delaunay settings
USE_DELAUNAY       = cfg["use_delaunay"]
MAX_DEPTH          = cfg["max_depth"]
ADJ_SCALER_OPTIONS = cfg["adj_scaler_options"]

# Model / data parameters
IN_FEAT            = cfg["in_feat"]
OUT_FEAT           = cfg["out_feat"]
LSTM_UNITS         = cfg["lstm_units"]
GRAPH_CONV_PARAMS  = cfg["graph_conv_params"]

# Training loop settings
EPOCHS             = cfg["epochs"]
BATCH_SIZE         = cfg["batch_size"]
INPUT_SEQ_LEN      = cfg["input_seq_len"]
FORECAST_HORIZONS  = cfg["forecast_horizons"]
SEEDS              = cfg["seeds"]

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

def load_data() -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Load raw vehicle counts and compute distances/points."""
    sensors, counts_array, _ = helper.load_data_and_combine_with_geojson(
        FOLDER_TO_LOAD, GEOJSON_PATH
    )
    route_distances, all_points = calculate_route_distance(sensors)
    logging.info("route_distances shape=%s", route_distances.shape)
    logging.info("vehicle_counts_array shape=%s", counts_array.shape)
    return route_distances, all_points, counts_array

def make_datasets(
    counts: np.ndarray, seed: int, forecast_horizon: int
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Split, normalize, and wrap in TF datasets."""
    train_arr, val_arr, test_arr, _ = preprocess(counts, TRAIN_SIZE, VAL_SIZE)
    logging.info(
        "train size=%s, val size=%s, test size=%s",
        train_arr.shape, val_arr.shape, test_arr.shape
    )

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

    route_distances, all_points, counts = load_data()

    for horizon in tqdm(FORECAST_HORIZONS, desc="Horizons"):
        for seed in tqdm(SEEDS, desc="Seeds", leave=False):
            set_seed(seed)
            train_ds, val_ds, _ = make_datasets(counts, seed, horizon)

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

                filename = (
                    f"darmstadt_h{horizon}_s{seed}_"
                    f"{'D' if USE_DELAUNAY else 'S'}_"
                    f"d{MAX_DEPTH}_a{adj_scaler}.h5"
                )
                save_path = os.path.join(SAVE_FOLDER, filename)
                model.save_weights(save_path)
                logging.info("Saved weights to %s", save_path)


if __name__ == "__main__":
    main()
