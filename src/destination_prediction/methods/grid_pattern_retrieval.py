"""Probabilistic grid-pattern retrieval and its matched last-state ablation."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from destination_prediction.catalogue import endpoint_catalogue_assignment
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


def candidate_order(
    squared_distances: np.ndarray,
    posterior: np.ndarray,
    trip_ids: np.ndarray,
    tie_rule: str,
) -> np.ndarray:
    """Order candidates under an explicit historical or canonical tie rule.

    ``legacy`` reproduces the historical NumPy quicksort over posterior values;
    tied positions therefore depend on the historical training-row layout.
    ``trajectory_id`` orders by increasing distance and then lexicographically
    by immutable source trajectory identifier.
    """
    if tie_rule == "legacy":
        return deterministic_descending_order(posterior)
    if tie_rule == "trajectory_id":
        return np.lexsort(
            (
                np.asarray(trip_ids, dtype=str),
                np.asarray(squared_distances, dtype=float),
            )
        )
    raise ValueError(f"Unknown grid candidate tie rule: {tie_rule}")


def gaussian_posterior(
    squared_distances: np.ndarray, sigma_cells: float
) -> np.ndarray:
    """Normalize the Gaussian weights defined for all personal candidates."""
    scores = -0.5 * np.asarray(squared_distances, dtype=float) / max(
        float(sigma_cells) ** 2, 1e-12
    )
    scores -= float(np.max(scores))
    posterior = np.exp(scores)
    return posterior / float(posterior.sum())


def selected_candidate_indices(
    squared_distances: np.ndarray,
    posterior: np.ndarray,
    trip_ids: np.ndarray,
    top_k: int,
    tie_rule: str,
    distance_tolerance: float = 1e-12,
) -> np.ndarray:
    """Select the top-k pool under one explicit equal-distance rule."""
    limit = min(int(top_k), len(squared_distances))
    if limit < 1:
        raise ValueError("At least one grid candidate is required")
    if tie_rule in {"legacy", "trajectory_id"}:
        return candidate_order(
            squared_distances, posterior, trip_ids, tie_rule
        )[:limit]
    if tie_rule == "include_boundary":
        order = candidate_order(
            squared_distances, posterior, trip_ids, "trajectory_id"
        )
        ordered_distances = np.asarray(squared_distances, dtype=float)[order]
        boundary = float(ordered_distances[limit - 1])
        included = (ordered_distances < boundary) | np.isclose(
            ordered_distances,
            boundary,
            rtol=0.0,
            atol=float(distance_tolerance),
        )
        return order[included]
    raise ValueError(f"Unknown grid candidate tie rule: {tie_rule}")


def emit_destination_region(
    squared_distances: np.ndarray,
    trip_ids: np.ndarray,
    candidate_regions: np.ndarray,
    sigma_cells: float,
    top_k: int,
    tie_rule: str,
    distance_tolerance: float = 1e-12,
) -> tuple[int, float, int, bool]:
    """Apply the shared kernel, pool, region aggregation, and tie rules."""
    posterior = gaussian_posterior(squared_distances, sigma_cells)
    selected = selected_candidate_indices(
        squared_distances,
        posterior,
        trip_ids,
        top_k,
        tie_rule,
        distance_tolerance,
    )
    mass: dict[int, float] = {}
    for position in selected:
        region = int(candidate_regions[position])
        mass[region] = mass.get(region, 0.0) + float(posterior[position])
    emitted = int(sorted(mass, key=lambda region: (-mass[region], region))[0])
    largest = mass[emitted]
    mass_tie = sum(value == largest for value in mass.values()) > 1
    return emitted, float(largest), int(len(selected)), bool(mass_tie)


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
    tie_rule: str = "legacy",
) -> pd.DataFrame:
    """Predict destinations with an explicit candidate tie rule.

    The default remains ``legacy`` so historical discovery results are not
    changed silently. Callers can request ``trajectory_id`` for row-order-
    invariant lexicographic tie handling.
    """
    if tie_rule not in {"legacy", "trajectory_id"}:
        raise ValueError(f"Unknown grid candidate tie rule: {tie_rule}")
    rows: list[dict[str, object]] = []
    train_by_user = {
        int(user_id): group.reset_index(drop=True)
        for user_id, group in train.groupby("user_id", sort=True)
    }
    cluster_map = endpoint_catalogue_assignment(train, catalogue)
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
                trip_ids = personal["trip_id"].astype(str).to_numpy()
                candidate_clusters = np.asarray(
                    [
                        cluster_map[(user_id, str(trip_id))]
                        for trip_id in personal["trip_id"]
                    ],
                    dtype=int,
                )
                for sigma in sigmas:
                    start = time.perf_counter()
                    for topk in topks:
                        selected_cluster, selected_mass, _pool_size, _mass_tie = (
                            emit_destination_region(
                                squared,
                                trip_ids,
                                candidate_clusters,
                                sigma,
                                topk,
                                tie_rule,
                            )
                        )
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
                                "tie_rule": str(tie_rule),
                                "ratio": float(ratio),
                                "user_id": user_id,
                                "trip_id": str(test_row.trip_id),
                                "native_pred_lon": float(center.medoid_lon),
                                "native_pred_lat": float(center.medoid_lat),
                                "posterior_mass": selected_mass,
                                "query_runtime_ms": (time.perf_counter() - start) * 1000.0,
                            }
                        )
    return pd.DataFrame(rows)
