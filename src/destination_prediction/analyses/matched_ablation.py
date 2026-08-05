#!/usr/bin/env python3
"""Matched last-state grid ablation and shared hierarchical bootstrap."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from destination_prediction.context import RunContext
from destination_prediction.methods.grid_pattern_retrieval import resampled_cells
from destination_prediction.protocol import EvaluationProtocol


CONTEXT = RunContext.from_environment()
EXP_DIR = CONTEXT.run_directory
CONFIG = CONTEXT.config
SOURCE = CONTEXT.dependency_directory(CONFIG["source_experiment"])
OUTPUT = EXP_DIR / "outputs"


def load_source_protocol():
    return EvaluationProtocol(
        CONTEXT.dependency_context(CONFIG["source_experiment"])
    )


PROTOCOL = load_source_protocol()


def emit_region(
    squared_distance: np.ndarray,
    candidate_regions: np.ndarray,
    sigma: float,
    top_k: int,
) -> tuple[int, float]:
    """Apply the frozen Gaussian, top-k, aggregation, and tie rules."""
    scores = -0.5 * squared_distance / max(float(sigma) ** 2, 1e-12)
    scores -= float(np.max(scores))
    posterior = np.exp(scores)
    posterior /= float(posterior.sum())
    order = np.argsort(-posterior)
    selected = order[: min(int(top_k), len(order))]
    mass: dict[int, float] = {}
    for position in selected:
        region = int(candidate_regions[position])
        mass[region] = mass.get(region, 0.0) + float(posterior[position])
    emitted = sorted(mass, key=lambda region: (-mass[region], region))[0]
    return emitted, float(mass[emitted])


def matched_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    marked = PROTOCOL.load_marked()
    train, test = PROTOCOL.partitions(marked, "outer")
    origins = PROTOCOL.user_origins(train)
    train = PROTOCOL.add_local_sequences(train, origins)
    test = PROTOCOL.add_local_sequences(test, origins)
    catalog = PROTOCOL.build_catalog(
        train, float(PROTOCOL.CONFIG["task"]["primary_dbscan_eps_m"])
    )
    cluster_map = PROTOCOL.endpoint_cluster_map(train, catalog)
    train_by_user = {
        int(user): frame.reset_index(drop=True)
        for user, frame in train.groupby("user_id", sort=True)
    }
    centers_by_user = {
        int(user): frame.set_index("center_id")
        for user, frame in catalog.groupby("user_id", sort=True)
    }
    cfg = CONFIG["frozen_grid_configuration"]
    cell_size = float(cfg["cell_size_m"])
    sigma = float(cfg["emission_sigma_cells"])
    top_k = int(cfg["posterior_top_k"])
    states = int(cfg["alignment_states"])
    rows: list[dict[str, object]] = []

    for ratio in map(float, CONFIG["observation_ratios"]):
        patterns = {
            user: np.stack(
                [
                    resampled_cells(seq, ratio, cell_size, states)
                    for seq in frame["local_seq"]
                ]
            )
            for user, frame in train_by_user.items()
        }
        candidate_regions = {
            user: np.asarray(
                [
                    cluster_map[(user, str(trip_id))]
                    for trip_id in frame["trip_id"]
                ],
                dtype=int,
            )
            for user, frame in train_by_user.items()
        }
        for test_row in test.itertuples(index=False):
            user = int(test_row.user_id)
            query = resampled_cells(
                test_row.local_seq, ratio, cell_size, states
            )
            difference = patterns[user] - query[None, :, :]
            squared_full = np.square(difference).sum(axis=2).mean(axis=1)
            squared_last = np.square(difference[:, -1, :]).sum(axis=1)
            for method, squared in [
                (str(CONFIG["methods"]["full"]), squared_full),
                (str(CONFIG["methods"]["ablation"]), squared_last),
            ]:
                region, mass = emit_region(
                    squared, candidate_regions[user], sigma, top_k
                )
                center = centers_by_user[user].loc[region]
                rows.append(
                    {
                        "family": "matched_grid_ablation",
                        "config_id": method,
                        "method": method,
                        "ratio": ratio,
                        "user_id": user,
                        "trip_id": str(test_row.trip_id),
                        "native_pred_lon": float(center.medoid_lon),
                        "native_pred_lat": float(center.medoid_lat),
                        "predicted_region": region,
                        "posterior_mass": mass,
                        "cell_size_m": cell_size,
                        "emission_sigma_cells": sigma,
                        "posterior_top_k": top_k,
                        "alignment_states": states,
                    }
                )

    native = pd.DataFrame(rows)
    evaluated = PROTOCOL.evaluate_native_predictions(native, test, catalog)
    return evaluated, catalog


def reproduction_audit(evaluated: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(SOURCE / "outputs/non_neural/test_predictions.csv")
    source = source[
        source["family"] == "probabilistic_grid_pattern_retrieval"
    ].copy()
    source["trip_id"] = source["trip_id"].astype(str)
    full = evaluated[
        evaluated["method"] == str(CONFIG["methods"]["full"])
    ].copy()
    full["trip_id"] = full["trip_id"].astype(str)
    keys = ["user_id", "trip_id", "ratio"]
    audit = full[keys + ["pred_center_id"]].merge(
        source[keys + ["pred_center_id"]],
        on=keys,
        how="outer",
        suffixes=("_rerun", "_source"),
        indicator=True,
        validate="one_to_one",
    )
    audit["exact_region_match"] = (
        (audit["_merge"] == "both")
        & (audit["pred_center_id_rerun"] == audit["pred_center_id_source"])
    )
    return audit


def comparison_cases(evaluated: pd.DataFrame) -> pd.DataFrame:
    matched = evaluated[
        [
            "method",
            "ratio",
            "user_id",
            "trip_id",
            "hit_r90_all",
            "destination_error_m",
            "within_200m",
            "within_1000m",
        ]
    ].copy()
    source = pd.read_csv(
        SOURCE / "outputs/final/common_predictions_all_sensitivities.csv"
    )
    source = source[
        (source["catalog_eps_m"] == 200.0)
        & source["method"].isin(
            [
                str(CONFIG["methods"]["secondary_full_prefix"]),
                str(CONFIG["methods"]["secondary_current_position"]),
            ]
        )
    ][
        [
            "method",
            "ratio",
            "user_id",
            "trip_id",
            "hit_r90_all",
            "destination_error_m",
            "within_200m",
            "within_1000m",
        ]
    ].copy()
    cases = pd.concat([matched, source], ignore_index=True)
    cases["trip_id"] = cases["trip_id"].astype(str)
    cases["hit_r90_all"] = cases["hit_r90_all"].astype(float)
    return cases.sort_values(
        ["method", "ratio", "user_id", "trip_id"]
    ).reset_index(drop=True)


def method_metrics(cases: pd.DataFrame) -> pd.DataFrame:
    per_user = (
        cases.groupby(["method", "ratio", "user_id"], as_index=False)
        .agg(
            hit_r90=("hit_r90_all", "mean"),
            d200=("within_200m", "mean"),
            d1km=("within_1000m", "mean"),
            median_error_m=("destination_error_m", "median"),
            trajectories=("trip_id", "size"),
        )
    )
    return (
        per_user.groupby(["method", "ratio"], as_index=False)
        .agg(
            hit_r90=("hit_r90", "mean"),
            d200=("d200", "mean"),
            d1km=("d1km", "mean"),
            mean_user_median_error_m=("median_error_m", "mean"),
            users=("user_id", "size"),
            trajectories=("trajectories", "sum"),
        )
        .sort_values(["method", "ratio"])
        .reset_index(drop=True)
    )


def save_resample_plan(
    pivot: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], pd.DataFrame]:
    users = np.asarray(
        sorted(pivot.index.get_level_values("user_id").unique()), dtype=np.int16
    )
    arrays: list[np.ndarray] = []
    index_rows: list[dict[str, object]] = []
    for user_position, user in enumerate(users):
        frame = pivot.xs(int(user), level="user_id").sort_index()
        arrays.append(frame.to_numpy(dtype=float))
        for within_position, trip_id in enumerate(frame.index.astype(str)):
            index_rows.append(
                {
                    "user_position": user_position,
                    "within_user_position": within_position,
                    "user_id": int(user),
                    "trip_id": str(trip_id),
                }
            )
    lengths = np.asarray([len(array) for array in arrays], dtype=np.uint16)
    replicates = int(CONFIG["bootstrap"]["replicates"])
    rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]))
    user_draws = rng.integers(
        0, len(users), size=(replicates, len(users)), dtype=np.uint8
    )
    draw_lengths = lengths[user_draws].reshape(-1).astype(np.int64)
    offsets = np.empty(len(draw_lengths) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(draw_lengths, out=offsets[1:])
    within_draws = np.empty(int(offsets[-1]), dtype=np.uint16)
    for draw_index, length in enumerate(draw_lengths):
        start, end = int(offsets[draw_index]), int(offsets[draw_index + 1])
        within_draws[start:end] = rng.integers(
            0, int(length), size=int(length), dtype=np.uint16
        )
    np.savez_compressed(
        OUTPUT / "bootstrap_resample_ids.npz",
        users=users,
        user_draw_positions=user_draws,
        within_user_draw_positions=within_draws,
        within_user_draw_offsets=offsets,
        trajectories_per_user=lengths,
    )
    case_index = pd.DataFrame(index_rows)
    case_index.to_csv(OUTPUT / "bootstrap_case_index.csv", index=False)
    return user_draws, within_draws, offsets, arrays, case_index


def hierarchical_bootstrap(cases: pd.DataFrame) -> pd.DataFrame:
    pivot = cases.pivot(
        index=["user_id", "trip_id"],
        columns=["method", "ratio"],
        values="hit_r90_all",
    ).sort_index()
    if pivot.isna().any().any():
        raise RuntimeError("Comparison matrix is incomplete.")
    columns = list(pivot.columns)
    column_index = {key: position for position, key in enumerate(columns)}
    user_draws, within_draws, offsets, arrays, _ = save_resample_plan(pivot)
    estimates = np.empty(
        (len(user_draws), len(columns)), dtype=np.float64
    )
    slots = user_draws.shape[1]
    for replicate in range(len(user_draws)):
        aggregate = np.zeros(len(columns), dtype=float)
        for slot in range(slots):
            draw_index = replicate * slots + slot
            start = int(offsets[draw_index])
            end = int(offsets[draw_index + 1])
            user_position = int(user_draws[replicate, slot])
            aggregate += arrays[user_position][
                within_draws[start:end]
            ].mean(axis=0)
        estimates[replicate] = aggregate / slots

    observed = np.mean(
        np.stack([array.mean(axis=0) for array in arrays]), axis=0
    )
    quantities: dict[str, tuple[float, np.ndarray, str]] = {}
    ratios = list(map(float, CONFIG["observation_ratios"]))
    methods = list(dict.fromkeys(cases["method"].astype(str)))
    for method in methods:
        per_ratio = []
        observed_ratio = []
        for ratio in ratios:
            position = column_index[(method, ratio)]
            values = estimates[:, position] * 100.0
            value = float(observed[position] * 100.0)
            quantities[f"rate|{method}|{ratio:.2f}"] = (
                value,
                values,
                "percent",
            )
            per_ratio.append(values)
            observed_ratio.append(value)
        quantities[f"mean_rate|{method}"] = (
            float(np.mean(observed_ratio)),
            np.mean(np.column_stack(per_ratio), axis=1),
            "percent",
        )

    pairs = [
        (
            str(CONFIG["methods"]["full"]),
            str(CONFIG["methods"]["ablation"]),
        ),
        (
            str(CONFIG["methods"]["secondary_full_prefix"]),
            str(CONFIG["methods"]["secondary_current_position"]),
        ),
    ]
    for first, second in pairs:
        differences = []
        observed_differences = []
        for ratio in ratios:
            first_position = column_index[(first, ratio)]
            second_position = column_index[(second, ratio)]
            values = (
                estimates[:, first_position] - estimates[:, second_position]
            ) * 100.0
            value = float(
                (observed[first_position] - observed[second_position]) * 100.0
            )
            quantities[f"difference|{first}|{second}|{ratio:.2f}"] = (
                value,
                values,
                "percentage_points",
            )
            differences.append(values)
            observed_differences.append(value)
        quantities[f"mean_difference|{first}|{second}"] = (
            float(np.mean(observed_differences)),
            np.mean(np.column_stack(differences), axis=1),
            "percentage_points",
        )
        quantities[f"stage_interaction|{first}|{second}|75minus25"] = (
            float(observed_differences[-1] - observed_differences[0]),
            differences[-1] - differences[0],
            "percentage_points",
        )

    rows: list[dict[str, object]] = []
    for name, (value, values, unit) in quantities.items():
        low, high = np.quantile(values, [0.025, 0.975])
        low_half, high_half = np.quantile(
            values[: len(values) // 2], [0.025, 0.975]
        )
        rows.append(
            {
                "quantity": name,
                "unit": unit,
                "observed": value,
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_low_first_5000": float(low_half),
                "ci_high_first_5000": float(high_half),
                "max_endpoint_change_5000_to_10000": float(
                    max(abs(low - low_half), abs(high - high_half))
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    evaluated, catalog = matched_predictions()
    evaluated.to_csv(OUTPUT / "matched_grid_predictions.csv", index=False)
    catalog.to_csv(OUTPUT / "training_destination_catalog.csv", index=False)
    audit = reproduction_audit(evaluated)
    audit.to_csv(OUTPUT / "full_grid_reproduction_audit.csv", index=False)
    cases = comparison_cases(evaluated)
    cases.to_csv(OUTPUT / "comparison_cases.csv", index=False)
    metrics = method_metrics(cases)
    metrics.to_csv(OUTPUT / "method_metrics.csv", index=False)
    bootstrap = hierarchical_bootstrap(cases)
    bootstrap.to_csv(OUTPUT / "paired_bootstrap.csv", index=False)
    summary = {
        "status": "completed",
        "users": int(cases["user_id"].nunique()),
        "test_trajectories": int(
            cases[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ),
        "ratios": sorted(map(float, cases["ratio"].unique())),
        "methods": sorted(cases["method"].unique().tolist()),
        "full_grid_predictions_reproduced": int(
            audit["exact_region_match"].sum()
        ),
        "full_grid_predictions_expected": int(len(audit)),
        "bootstrap_replicates": int(CONFIG["bootstrap"]["replicates"]),
        "bootstrap_seed": int(CONFIG["bootstrap"]["seed"]),
        "exact_resample_ids_saved": True,
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
