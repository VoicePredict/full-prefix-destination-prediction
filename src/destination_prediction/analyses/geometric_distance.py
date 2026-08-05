#!/usr/bin/env python3
"""Controlled distance-function robustness for personal prefix retrieval."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext
from destination_prediction.protocol import EvaluationProtocol


CONTEXT = RunContext.from_environment()
EXP_DIR = CONTEXT.run_directory
CONFIG = CONTEXT.config
SOURCE = CONTEXT.dependency_directory(CONFIG["source_experiment"])
OUTPUT = EXP_DIR / "outputs"


def source_protocol():
    return EvaluationProtocol(
        CONTEXT.dependency_context(CONFIG["source_experiment"])
    )


def pairwise_pointwise(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.linalg.norm(train - query[None, :, :], axis=2).mean(axis=1)


def pairwise_dtw_mean(train: np.ndarray, query: np.ndarray) -> np.ndarray:
    count, length, _ = train.shape
    costs = np.linalg.norm(
        train[:, :, None, :] - query[None, None, :, :], axis=3
    )
    accumulated = np.full(
        (count, length + 1, length + 1), np.inf, dtype=np.float64
    )
    path_lengths = np.zeros(
        (count, length + 1, length + 1), dtype=np.int16
    )
    accumulated[:, 0, 0] = 0.0
    for train_index in range(1, length + 1):
        for query_index in range(1, length + 1):
            predecessors = np.stack(
                [
                    accumulated[:, train_index - 1, query_index - 1],
                    accumulated[:, train_index - 1, query_index],
                    accumulated[:, train_index, query_index - 1],
                ],
                axis=1,
            )
            choice = np.argmin(predecessors, axis=1)
            selected_cost = np.take_along_axis(
                predecessors, choice[:, None], axis=1
            )[:, 0]
            predecessor_lengths = np.stack(
                [
                    path_lengths[:, train_index - 1, query_index - 1],
                    path_lengths[:, train_index - 1, query_index],
                    path_lengths[:, train_index, query_index - 1],
                ],
                axis=1,
            )
            selected_length = np.take_along_axis(
                predecessor_lengths, choice[:, None], axis=1
            )[:, 0]
            accumulated[:, train_index, query_index] = (
                costs[:, train_index - 1, query_index - 1] + selected_cost
            )
            path_lengths[:, train_index, query_index] = selected_length + 1
    return accumulated[:, length, length] / np.maximum(
        path_lengths[:, length, length], 1
    )


def pairwise_discrete_frechet(
    train: np.ndarray, query: np.ndarray
) -> np.ndarray:
    count, length, _ = train.shape
    costs = np.linalg.norm(
        train[:, :, None, :] - query[None, None, :, :], axis=3
    )
    coupling = np.empty((count, length, length), dtype=np.float64)
    coupling[:, 0, 0] = costs[:, 0, 0]
    for train_index in range(1, length):
        coupling[:, train_index, 0] = np.maximum(
            coupling[:, train_index - 1, 0], costs[:, train_index, 0]
        )
    for query_index in range(1, length):
        coupling[:, 0, query_index] = np.maximum(
            coupling[:, 0, query_index - 1], costs[:, 0, query_index]
        )
    for train_index in range(1, length):
        for query_index in range(1, length):
            predecessor = np.minimum.reduce(
                [
                    coupling[:, train_index - 1, query_index],
                    coupling[:, train_index - 1, query_index - 1],
                    coupling[:, train_index, query_index - 1],
                ]
            )
            coupling[:, train_index, query_index] = np.maximum(
                predecessor, costs[:, train_index, query_index]
            )
    return coupling[:, -1, -1]


DISTANCES = {
    "aligned_pointwise_mean": pairwise_pointwise,
    "dtw_mean_path_cost": pairwise_dtw_mean,
    "discrete_frechet": pairwise_discrete_frechet,
}


def retrieval_predictions(protocol, train, test) -> pd.DataFrame:
    ratios = list(map(float, CONFIG["observation_ratios"]))
    resample_points = int(CONFIG["resample_points"])
    rows = []
    for ratio in ratios:
        for user_id, train_user in train.groupby("user_id", sort=True):
            user_id = int(user_id)
            train_user = train_user.reset_index(drop=True)
            test_user = test[test["user_id"] == user_id].reset_index(drop=True)
            train_prefixes = np.stack(
                [
                    protocol.resample_path(
                        np.asarray(sequence, dtype=float)[
                            : protocol.prefix_cutoff(len(sequence), ratio)
                        ],
                        resample_points,
                    )
                    for sequence in train_user["local_seq"]
                ]
            )
            test_prefixes = np.stack(
                [
                    protocol.resample_path(
                        np.asarray(sequence, dtype=float)[
                            : protocol.prefix_cutoff(len(sequence), ratio)
                        ],
                        resample_points,
                    )
                    for sequence in test_user["local_seq"]
                ]
            )
            for test_index, test_row in enumerate(
                test_user.itertuples(index=False)
            ):
                query = test_prefixes[test_index]
                for method_name in CONFIG["distance_variants"]:
                    start = time.perf_counter()
                    distances = DISTANCES[method_name](train_prefixes, query)
                    analogue_index = int(np.argmin(distances))
                    runtime_ms = (time.perf_counter() - start) * 1000.0
                    analogue = train_user.iloc[analogue_index]
                    lon, lat = analogue["wgs_seq"][-1]
                    rows.append(
                        {
                            "family": "geometric_distance_robustness",
                            "method": method_name,
                            "ratio": ratio,
                            "user_id": user_id,
                            "trip_id": str(test_row.trip_id),
                            "native_pred_lon": float(lon),
                            "native_pred_lat": float(lat),
                            "analogue_trip_id": str(analogue["trip_id"]),
                            "nearest_distance": float(
                                distances[analogue_index]
                            ),
                            "query_runtime_ms": runtime_ms,
                            "resample_points": resample_points,
                        }
                    )
    return pd.DataFrame(rows)


def load_current_reference() -> pd.DataFrame:
    token = int(round(float(CONFIG["support_quantile"]) * 100))
    path = SOURCE / "outputs/final/common_predictions_all_sensitivities.csv"
    frame = pd.read_csv(
        path,
        usecols=[
            "method",
            "ratio",
            "user_id",
            "trip_id",
            "catalog_eps_m",
            f"hit_r{token}_all",
            "within_200m",
            "within_1000m",
            "destination_error_m",
        ],
    )
    frame = frame[
        (frame["method"] == CONFIG["reference_method"])
        & np.isclose(frame["catalog_eps_m"], float(CONFIG["catalog_eps_m"]))
    ].copy()
    frame["trip_id"] = frame["trip_id"].astype(str)
    return frame.rename(
        columns={
            f"hit_r{token}_all": "hit_r90",
        }
    )[
        [
            "method",
            "ratio",
            "user_id",
            "trip_id",
            "hit_r90",
            "within_200m",
            "within_1000m",
            "destination_error_m",
        ]
    ]


def method_summary(all_cases: pd.DataFrame) -> pd.DataFrame:
    per_user = (
        all_cases.groupby(["method", "ratio", "user_id"], as_index=False)
        .agg(
            hit_r90=("hit_r90", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            median_error_m=("destination_error_m", "median"),
            cases=("trip_id", "nunique"),
        )
    )
    per_ratio = (
        per_user.groupby(["method", "ratio"], as_index=False)
        .agg(
            user_macro_hit_r90=("hit_r90", "mean"),
            user_macro_within_200m=("within_200m", "mean"),
            user_macro_within_1000m=("within_1000m", "mean"),
            mean_user_median_error_m=("median_error_m", "mean"),
            users=("user_id", "nunique"),
            cases=("cases", "sum"),
        )
    )
    mean = (
        per_user.groupby(["method", "user_id"], as_index=False)
        .agg(
            hit_r90=("hit_r90", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            median_error_m=("median_error_m", "mean"),
            cases=("cases", "sum"),
        )
        .groupby("method", as_index=False)
        .agg(
            user_macro_hit_r90=("hit_r90", "mean"),
            user_macro_within_200m=("within_200m", "mean"),
            user_macro_within_1000m=("within_1000m", "mean"),
            mean_user_median_error_m=("median_error_m", "mean"),
            users=("user_id", "nunique"),
            cases=("cases", "sum"),
        )
    )
    mean["ratio"] = "mean"
    return pd.concat([per_ratio, mean], ignore_index=True)


def bootstrap_contrasts(all_cases: pd.DataFrame) -> pd.DataFrame:
    reference = str(CONFIG["reference_method"])
    methods = list(CONFIG["distance_variants"])
    replicates = int(CONFIG["bootstrap"]["replicates"])
    rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]))
    pivot = all_cases.pivot(
        index=["user_id", "trip_id", "ratio"],
        columns="method",
        values="hit_r90",
    ).dropna().astype(float)
    users = np.asarray(
        sorted(pivot.index.get_level_values("user_id").unique()), dtype=int
    )
    by_user = {
        int(user): pivot.xs(user, level="user_id")
        for user in users
    }

    def bootstrap_user_trip_effects(
        effects_by_user: dict[int, np.ndarray],
    ) -> tuple[float, float, float, float]:
        observed = float(
            np.mean(
                [
                    np.asarray(effects_by_user[int(user)], dtype=float).mean()
                    for user in users
                ]
            )
        )
        estimates = np.empty(replicates, dtype=float)
        batch_size = 500
        for start in range(0, replicates, batch_size):
            count = min(batch_size, replicates - start)
            sampled_users = rng.choice(
                users, size=(count, len(users)), replace=True
            )
            user_means = np.empty((count, len(users)), dtype=float)
            for slot in range(len(users)):
                selected = sampled_users[:, slot]
                for user in users:
                    mask = selected == user
                    occurrences = int(mask.sum())
                    if occurrences == 0:
                        continue
                    values = np.asarray(
                        effects_by_user[int(user)], dtype=float
                    )
                    sampled_positions = rng.integers(
                        0,
                        len(values),
                        size=(occurrences, len(values)),
                    )
                    user_means[mask, slot] = values[
                        sampled_positions
                    ].mean(axis=1)
            estimates[start : start + count] = user_means.mean(axis=1)
        low, high = np.quantile(estimates, [0.025, 0.975])
        return observed, float(estimates.mean()), float(low), float(high)

    quantities = [
        ("ratio", ratio) for ratio in map(float, CONFIG["observation_ratios"])
    ] + [("mean_across_ratios", "mean"), ("difference_in_change", "75_minus_25")]
    rows = []
    for method in methods:
        for quantity, value in quantities:
            effects_by_user = {}
            for user in users:
                user_frame = by_user[int(user)]
                difference = (
                    user_frame[method] - user_frame[reference]
                ).unstack("ratio")
                if quantity == "ratio":
                    effects = difference[float(value)].to_numpy(dtype=float)
                elif quantity == "mean_across_ratios":
                    effects = difference.mean(axis=1).to_numpy(dtype=float)
                else:
                    effects = (
                        difference[0.75] - difference[0.25]
                    ).to_numpy(dtype=float)
                effects_by_user[int(user)] = effects
            observed, bootstrap_mean, low, high = bootstrap_user_trip_effects(
                effects_by_user
            )
            rows.append(
                {
                    "method": method,
                    "reference_method": reference,
                    "quantity": quantity,
                    "ratio_or_change": value,
                    "observed_difference": observed,
                    "bootstrap_mean": bootstrap_mean,
                    "ci_low": low,
                    "ci_high": high,
                    "users": int(len(users)),
                }
            )
    return pd.DataFrame(rows)


def runtime_summary(native: pd.DataFrame) -> pd.DataFrame:
    return (
        native.groupby(["method", "ratio"], as_index=False)
        .agg(
            median_query_runtime_ms=("query_runtime_ms", "median"),
            p90_query_runtime_ms=(
                "query_runtime_ms",
                lambda values: float(np.quantile(values, 0.9)),
            ),
            mean_query_runtime_ms=("query_runtime_ms", "mean"),
            queries=("trip_id", "size"),
        )
    )


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol = source_protocol()
    marked = protocol.load_marked()
    train, test = protocol.partitions(marked, "outer")
    origins = protocol.user_origins(train)
    train = protocol.add_local_sequences(train, origins)
    test = protocol.add_local_sequences(test, origins)
    catalog = protocol.build_catalog(train, float(CONFIG["catalog_eps_m"]))
    native = retrieval_predictions(protocol, train, test)
    evaluated = protocol.evaluate_native_predictions(native, test, catalog)
    token = int(round(float(CONFIG["support_quantile"]) * 100))
    evaluated = evaluated.rename(
        columns={f"hit_r{token}_all": "hit_r90"}
    )
    evaluated.to_csv(OUTPUT / "geometric_predictions.csv", index=False)
    current = load_current_reference()
    columns = [
        "method",
        "ratio",
        "user_id",
        "trip_id",
        "hit_r90",
        "within_200m",
        "within_1000m",
        "destination_error_m",
    ]
    all_cases = pd.concat(
        [evaluated[columns], current[columns]], ignore_index=True
    )
    all_cases.to_csv(OUTPUT / "comparison_cases.csv", index=False)
    method_summary(all_cases).to_csv(
        OUTPUT / "method_metrics.csv", index=False
    )
    bootstrap_contrasts(all_cases).to_csv(
        OUTPUT / "paired_bootstrap.csv", index=False
    )
    runtime_summary(native).to_csv(
        OUTPUT / "distance_runtime_diagnostic.csv", index=False
    )
    source_pointwise = pd.read_csv(
        SOURCE / "outputs/non_neural/test_predictions.csv",
        usecols=[
            "family",
            "config_id",
            "ratio",
            "user_id",
            "trip_id",
            "analogue_trip_id",
        ],
        dtype={
            "trip_id": "string",
            "analogue_trip_id": "string",
        },
    )
    source_pointwise = source_pointwise[
        (source_pointwise["family"] == "geometric_retrieval")
        & (source_pointwise["config_id"] == "points_16")
    ].copy()
    source_pointwise["trip_id"] = source_pointwise["trip_id"].astype(str)
    source_pointwise["analogue_trip_id"] = source_pointwise[
        "analogue_trip_id"
    ].astype(str)
    rerun = native[native["method"] == "aligned_pointwise_mean"][
        ["ratio", "user_id", "trip_id", "analogue_trip_id"]
    ]
    audit = rerun.merge(
        source_pointwise,
        on=["ratio", "user_id", "trip_id"],
        suffixes=("_rerun", "_source"),
        validate="one_to_one",
    )
    audit["analogue_match"] = (
        audit["analogue_trip_id_rerun"]
        == audit["analogue_trip_id_source"]
    )
    audit.to_csv(OUTPUT / "pointwise_reproduction_audit.csv", index=False)
    summary = {
        "status": "completed",
        "users": int(test["user_id"].nunique()),
        "test_trajectories": int(len(test)),
        "prediction_rows": int(len(evaluated)),
        "distance_variants": CONFIG["distance_variants"],
        "pointwise_analogue_matches": int(audit["analogue_match"].sum()),
        "pointwise_cases": int(len(audit)),
        "test_outcomes_used_for_configuration": False,
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
