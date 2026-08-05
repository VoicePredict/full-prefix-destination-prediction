"""Probabilistic grid-pattern retrieval and its matched last-state ablation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from destination_prediction.catalogue import endpoint_cluster_map
from destination_prediction.geometry import prefix_cutoff, resample_path


def resampled_cells(
    seq: list[list[float]], ratio: float, cell_size: float, length: int
) -> np.ndarray:
    prefix = np.asarray(seq, dtype=float)[: prefix_cutoff(len(seq), ratio)]
    cells = np.floor(prefix / float(cell_size))
    keep = np.r_[True, np.any(cells[1:] != cells[:-1], axis=1)]
    return resample_path(cells[keep], length)


def deterministic_descending_order(scores: np.ndarray) -> np.ndarray:
    """Reproduce the frozen NumPy ordering on deterministic training rows."""
    return np.argsort(-np.asarray(scores, dtype=float), kind="quicksort")


def aligned_squared_distance(
    patterns: np.ndarray, query: np.ndarray, use_last_state_only: bool = False
) -> np.ndarray:
    """Matched distance; only the aligned states included in the mean differ."""
    differences = np.square(patterns - query[None, :, :]).sum(axis=2)
    if use_last_state_only:
        return differences[:, -1]
    return differences.mean(axis=1)


def grid_pattern_predictions(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    catalogue: pd.DataFrame,
    ratios: list[float],
    cell_sizes: list[float],
    sigmas: list[float],
    topks: list[int],
    alignment_states: int,
    family: str = "probabilistic_grid_pattern_retrieval",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_by_user = {
        int(user_id): group.reset_index(drop=True)
        for user_id, group in train.groupby("user_id", sort=True)
    }
    cluster_map = endpoint_cluster_map(train, catalogue)
    centers_by_user = {
        int(user_id): group.set_index("center_id")
        for user_id, group in catalogue.groupby("user_id", sort=True)
    }
    for ratio in ratios:
        for cell_size in cell_sizes:
            patterns = {
                user_id: np.stack(
                    [
                        resampled_cells(seq, ratio, cell_size, alignment_states)
                        for seq in group["local_seq"]
                    ]
                )
                for user_id, group in train_by_user.items()
            }
            for test_row in evaluate.itertuples(index=False):
                user_id = int(test_row.user_id)
                query = resampled_cells(
                    test_row.local_seq, ratio, cell_size, alignment_states
                )
                squared = aligned_squared_distance(patterns[user_id], query)
                personal = train_by_user[user_id]
                candidate_clusters = np.asarray(
                    [
                        cluster_map[(user_id, str(trip_id))]
                        for trip_id in personal["trip_id"]
                    ],
                    dtype=int,
                )
                for sigma in sigmas:
                    start = time.perf_counter()
                    scores = -0.5 * squared / max(float(sigma) ** 2, 1e-12)
                    scores = scores - float(np.max(scores))
                    posterior = np.exp(scores)
                    posterior /= float(posterior.sum())
                    order = deterministic_descending_order(posterior)
                    for topk in topks:
                        selected_idx = order[: min(int(topk), len(order))]
                        mass: dict[int, float] = {}
                        for index in selected_idx:
                            cluster_id = int(candidate_clusters[index])
                            mass[cluster_id] = mass.get(cluster_id, 0.0) + float(
                                posterior[index]
                            )
                        selected_cluster = sorted(
                            mass, key=lambda cluster_id: (-mass[cluster_id], cluster_id)
                        )[0]
                        center = centers_by_user[user_id].loc[selected_cluster]
                        rows.append(
                            {
                                "family": family,
                                "config_id": (
                                    f"cell{int(cell_size)}_sigma{sigma:g}_top{int(topk)}"
                                ),
                                "cell_size_m": float(cell_size),
                                "emission_sigma_cells": float(sigma),
                                "posterior_top_k": int(topk),
                                "ratio": float(ratio),
                                "user_id": user_id,
                                "trip_id": str(test_row.trip_id),
                                "native_pred_lon": float(center.medoid_lon),
                                "native_pred_lat": float(center.medoid_lat),
                                "posterior_mass": float(mass[selected_cluster]),
                                "query_runtime_ms": (time.perf_counter() - start) * 1000.0,
                            }
                        )
    return pd.DataFrame(rows)
