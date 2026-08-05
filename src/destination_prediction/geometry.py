"""Geographic projection, distance, prefix, and path-resampling operations."""

from __future__ import annotations

import math

import numpy as np


EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arctan2(
        np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a))
    )


def local_xy(lon, lat, lon0: float, lat0: float) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    x = EARTH_RADIUS_M * np.radians(lon - lon0) * math.cos(math.radians(lat0))
    y = EARTH_RADIUS_M * np.radians(lat - lat0)
    return np.column_stack([x, y])


def prefix_cutoff(length: int, ratio: float) -> int:
    """Return the ties-to-even point-count cutoff used in every experiment."""
    return min(length, max(2, int(round((length - 1) * ratio)) + 1))


def resample_path(seq: list[list[float]] | np.ndarray, n_points: int) -> np.ndarray:
    arr = np.asarray(seq, dtype=float)
    if len(arr) == 1:
        return np.repeat(arr, n_points, axis=0)
    segment = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment)])
    if cumulative[-1] <= 0:
        return np.repeat(arr[:1], n_points, axis=0)
    targets = np.linspace(0.0, cumulative[-1], n_points)
    x = np.interp(targets, cumulative, arr[:, 0])
    y = np.interp(targets, cumulative, arr[:, 1])
    return np.column_stack([x, y])
