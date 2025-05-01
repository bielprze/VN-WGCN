"""
helper.py

Utility functions for loading and preprocessing sensor and GeoJSON data.
"""

import json
import os
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
from pyproj import Transformer


def load_and_normalize_geojson(
    file_path: str, width: int = 256, height: int = 256
) -> Dict[str, Tuple[float, float]]:
    """
    Load a GeoJSON file and normalize the point coordinates to fit within
    a width x height bitmap.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    coords = [feat["geometry"]["coordinates"] for feat in data.get("features", [])]
    if not coords:
        return {}

    xs, ys = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    normalized: Dict[str, Tuple[float, float]] = {}
    for feat in data.get("features", []):
        sensor_id = feat["properties"].get("sensor_id")
        lon, lat = feat["geometry"]["coordinates"]

        norm_x = ((lon - min_x) / (max_x - min_x) * width) if max_x != min_x else width / 2
        norm_y = ((lat - min_y) / (max_y - min_y) * height) if max_y != min_y else height / 2

        normalized[sensor_id] = (norm_x, norm_y)

    return normalized


def load_data_and_combine_with_geojson(
    folder_to_load: str,
    geojson_path: str
) -> Tuple[Dict[str, Tuple[float, float]], np.ndarray, List[str]]:
    """
    Load sensor locations (projected), time series data, filter out
    zero-only sensors, and assemble into a full data matrix.

    Returns:
      - dict of sensor_id → (x, y)
      - 2D numpy array of shape (time_steps, num_sensors)
      - list of all originally found sensor_ids
    """
    sensor_points = _load_geojson_recalculate(geojson_path)
    sensor_values, max_len = _load_sensor_data(folder_to_load)

    all_keys = list(sensor_values.keys())
    clean_points = _clean_keys_remove_zero(
        sensor_points,
        all_keys,
        sensor_values,
        exclude_keys=['A173']
    )
    kept_keys = list(clean_points.keys())

    full_data = np.zeros((max_len, len(kept_keys)), dtype=float)
    for idx, key in enumerate(kept_keys):
        vals = sensor_values.get(key, [])
        n = len(vals)
        full_data[:n, idx] = vals

    return clean_points, full_data, all_keys


def _load_geojson_recalculate(file_path: str) -> Dict[str, Tuple[float, float]]:
    """
    Internal: load a GeoJSON file and project coordinates from
    EPSG:4326 to EPSG:2100 (Greek Grid).
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2100", always_xy=True)
    points: Dict[str, Tuple[float, float]] = {}

    for feat in data.get("features", []):
        lon, lat = feat["geometry"]["coordinates"]
        sensor_id = feat["properties"].get("sensor_id")
        x, y = transformer.transform(lon, lat)
        points[sensor_id] = (x, y)

    return points


def _load_sensor_data(folder_path: str) -> Tuple[Dict[str, List[int]], int]:
    """
    Internal: read all CSVs in a folder, extract the 'Total' column
    as int lists, return mapping sensor_id → values and the maximum series length.
    """
    data: Dict[str, List[int]] = {}
    max_len = 0

    for fname in os.listdir(folder_path):
        sensor_id, ext = os.path.splitext(fname)
        if sensor_id.startswith('Alle0LSA'):
            continue

        full_path = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(full_path)
            if 'Total' in df.columns:
                vals = df['Total'].dropna().astype(int).tolist()
                data[sensor_id] = vals
                max_len = max(max_len, len(vals))
            else:
                print(f"Warning: 'Total' column not found in {fname}")
        except Exception as e:
            print(f"Error processing {fname}: {e}")

    return data, max_len


def _clean_keys(
    sensors: Dict[str, Tuple[float, float]],
    keys: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Internal: keep only sensors whose IDs appear in the provided list.
    """
    return {k: sensors[k] for k in keys if k in sensors}


def _clean_keys_remove_zero(
    sensors: Dict[str, Tuple[float, float]],
    keys: List[str],
    sensor_values: Dict[str, List[float]],
    exclude_keys: List[str]
) -> Dict[str, Tuple[float, float]]:
    """
    Internal: remove sensors that are in exclude_keys or whose entire
    value series is zero.
    """
    clean: Dict[str, Tuple[float, float]] = {}

    for key in keys:
        if key in exclude_keys or key not in sensors:
            continue
        vals = sensor_values.get(key, [])
        if any(abs(v) > 1e-6 for v in vals):
            clean[key] = sensors[key]

    return clean