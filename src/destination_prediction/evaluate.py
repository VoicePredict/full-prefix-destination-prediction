#!/usr/bin/env python3
"""Create complete metrics, sensitivity tables, seed summaries, and paired intervals."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from destination_prediction.catalogue import build_catalogue
from destination_prediction.context import RunContext
from destination_prediction.data import load_marked, partitions
from destination_prediction.metrics import evaluate_native_predictions


PRINCIPAL = [
    "geometric_retrieval",
    "probabilistic_grid_pattern_retrieval",
    "tsmini",
    "bigru",
]
BASELINES = [
    "most_frequent_personal_destination",
    "nearest_known_destination_to_current_point",
    "last_observed_point",
    "current_position_only_personal_retrieval",
]
METHODS = [*PRINCIPAL, *BASELINES]


def normalize_input(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "family",
        "config_id",
        "ratio",
        "user_id",
        "trip_id",
        "native_pred_lon",
        "native_pred_lat",
    ]
    optional = ["seed", "query_runtime_ms", "analogue_trip_id"]
    out = frame[[*columns, *[column for column in optional if column in frame.columns]]].copy()
    if "seed" not in out:
        out["seed"] = -1
    if "query_runtime_ms" not in out:
        out["query_runtime_ms"] = np.nan
    mapping = {
        "geometric_retrieval": "geometric_retrieval",
        "probabilistic_grid_pattern_retrieval": "probabilistic_grid_pattern_retrieval",
        "tsmini_retrieval": "tsmini",
        "bigru_destination_classifier": "bigru",
    }
    out["method"] = out["family"].map(mapping).fillna(out["config_id"])
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").fillna(-1).astype(int)
    out["trip_id"] = out["trip_id"].astype(str)
    return out


def load_predictions(run_directory: Path) -> pd.DataFrame:
    frames = [
        normalize_input(pd.read_csv(run_directory / "outputs/non_neural/test_predictions.csv")),
        normalize_input(pd.read_csv(run_directory / "outputs/bigru/test_predictions_by_seed.csv")),
        normalize_input(pd.read_csv(run_directory / "outputs/tsmini/test_predictions_by_seed.csv")),
    ]
    return pd.concat(frames, ignore_index=True)


def metric_tables(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group = ["catalog_eps_m", "method", "seed", "ratio"]
    trip = (
        evaluated.groupby(group, as_index=False)
        .agg(
            n=("trip_id", "size"),
            coverage_r80=("covered_r80", "mean"),
            coverage_r90=("covered_r90", "mean"),
            coverage_r95=("covered_r95", "mean"),
            hit_r80=("hit_r80_all", "mean"),
            hit_r90=("hit_r90_all", "mean"),
            hit_r95=("hit_r95_all", "mean"),
            hit_fixed_200m=("hit_fixed_200m_all", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            median_error_m=("destination_error_m", "median"),
            p90_error_m=("destination_error_m", lambda values: values.quantile(0.90)),
            native_within_200m=("native_within_200m", "mean"),
            native_within_1000m=("native_within_1000m", "mean"),
            native_median_error_m=("native_destination_error_m", "median"),
            native_p90_error_m=("native_destination_error_m", lambda values: values.quantile(0.90)),
            median_query_runtime_ms=("query_runtime_ms", "median"),
        )
    )
    per_user = (
        evaluated.groupby([*group, "user_id"], as_index=False)
        .agg(
            n=("trip_id", "size"),
            coverage_r80=("covered_r80", "mean"),
            coverage_r90=("covered_r90", "mean"),
            coverage_r95=("covered_r95", "mean"),
            hit_r80=("hit_r80_all", "mean"),
            hit_r90=("hit_r90_all", "mean"),
            hit_r95=("hit_r95_all", "mean"),
            hit_fixed_200m=("hit_fixed_200m_all", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            median_error_m=("destination_error_m", "median"),
            p90_error_m=("destination_error_m", lambda values: values.quantile(0.90)),
            native_within_200m=("native_within_200m", "mean"),
            native_within_1000m=("native_within_1000m", "mean"),
            native_median_error_m=("native_destination_error_m", "median"),
            native_p90_error_m=("native_destination_error_m", lambda values: values.quantile(0.90)),
            median_query_runtime_ms=("query_runtime_ms", "median"),
        )
    )
    macro = (
        per_user.groupby(group, as_index=False)
        .agg(
            users=("user_id", "size"),
            trips=("n", "sum"),
            coverage_r80=("coverage_r80", "mean"),
            coverage_r90=("coverage_r90", "mean"),
            coverage_r95=("coverage_r95", "mean"),
            hit_r80=("hit_r80", "mean"),
            hit_r90=("hit_r90", "mean"),
            hit_r95=("hit_r95", "mean"),
            hit_fixed_200m=("hit_fixed_200m", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            median_error_m=("median_error_m", "mean"),
            p90_error_m=("p90_error_m", "mean"),
            native_within_200m=("native_within_200m", "mean"),
            native_within_1000m=("native_within_1000m", "mean"),
            native_median_error_m=("native_median_error_m", "mean"),
            native_p90_error_m=("native_p90_error_m", "mean"),
            median_query_runtime_ms=("median_query_runtime_ms", "median"),
        )
    )
    return trip, per_user, macro


def seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in metrics.columns
        if column not in {"catalog_eps_m", "method", "seed", "ratio", "users", "trips"}
    ]
    rows = []
    for keys, group in metrics.groupby(["catalog_eps_m", "method", "ratio"], sort=True):
        row = {"catalog_eps_m": keys[0], "method": keys[1], "ratio": keys[2], "seeds": len(group)}
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def case_seed_mean(evaluated: pd.DataFrame, primary_eps_m: float) -> pd.DataFrame:
    primary = evaluated[
        (evaluated["catalog_eps_m"] == float(primary_eps_m))
        & evaluated["method"].isin(METHODS)
    ].copy()
    return (
        primary.groupby(["method", "ratio", "user_id", "trip_id"], as_index=False)
        .agg(hit=("hit_r90_all", "mean"))
    )


def hierarchical_bootstrap(
    cases: pd.DataFrame, ratios: list[float], replicates: int, seed: int
) -> tuple[pd.DataFrame, dict[str, float]]:
    pivot = cases.pivot(
        index=["user_id", "trip_id"], columns=["method", "ratio"], values="hit"
    ).sort_index()
    users = np.asarray(
        sorted(pivot.index.get_level_values("user_id").unique()),
        dtype=int,
    )
    columns = list(pivot.columns)
    column_index = {key: index for index, key in enumerate(columns)}
    by_user = {
        int(user): pivot.xs(user, level="user_id").to_numpy(dtype=float)
        for user in users
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, len(columns)), dtype=float)
    batch_size = 1000
    for start in range(0, replicates, batch_size):
        count = min(batch_size, replicates - start)
        sampled_users = rng.choice(
            users, size=(count, len(users)), replace=True
        )
        user_means = np.empty(
            (count, len(users), len(columns)), dtype=float
        )
        for slot in range(len(users)):
            selected_users = sampled_users[:, slot]
            for user in users:
                mask = selected_users == user
                occurrences = int(mask.sum())
                if occurrences == 0:
                    continue
                values = by_user[int(user)]
                sampled_positions = rng.integers(
                    0,
                    len(values),
                    size=(occurrences, len(values)),
                )
                user_means[mask, slot, :] = values[
                    sampled_positions
                ].mean(axis=1)
        estimates[start : start + count] = user_means.mean(axis=1)

    ratios = list(map(float, ratios))
    quantities: dict[str, np.ndarray] = {}
    for method in METHODS:
        per_ratio = []
        for ratio in ratios:
            values = estimates[:, column_index[(method, ratio)]]
            quantities[f"rate|{method}|{ratio:.2f}"] = values
            per_ratio.append(values)
        quantities[f"primary_mean_rate|{method}"] = np.mean(
            np.column_stack(per_ratio), axis=1
        )
        quantities[f"change75_25|{method}"] = (
            estimates[:, column_index[(method, 0.75)]]
            - estimates[:, column_index[(method, 0.25)]]
        )
    for first, second in itertools.combinations(METHODS, 2):
        per_ratio_differences = []
        for ratio in ratios:
            values = (
                estimates[:, column_index[(first, ratio)]]
                - estimates[:, column_index[(second, ratio)]]
            )
            quantities[f"difference|{first}|{second}|{ratio:.2f}"] = values
            per_ratio_differences.append(values)
        quantities[f"primary_mean_difference|{first}|{second}"] = np.mean(
            np.column_stack(per_ratio_differences), axis=1
        )
        quantities[f"interaction75_25|{first}|{second}"] = (
            (
                estimates[:, column_index[(first, 0.75)]]
                - estimates[:, column_index[(first, 0.25)]]
            )
            - (
                estimates[:, column_index[(second, 0.75)]]
                - estimates[:, column_index[(second, 0.25)]]
            )
        )
    rows = []
    maximum_shift = 0.0
    for name, values in quantities.items():
        low, high = np.quantile(values, [0.025, 0.975])
        low_half, high_half = np.quantile(values[: replicates // 2], [0.025, 0.975])
        maximum_shift = max(maximum_shift, abs(low - low_half), abs(high - high_half))
        rows.append(
            {
                "quantity": name,
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_low_first_half": float(low_half),
                "ci_high_first_half": float(high_half),
            }
        )
    return pd.DataFrame(rows), {"maximum_endpoint_shift_5000_vs_10000": maximum_shift}


def run(context: RunContext) -> dict[str, object]:
    config = context.config
    output = context.run_directory / "outputs" / "final"
    output.mkdir(parents=True, exist_ok=True)
    marked = load_marked(context.run_directory)
    train, test = partitions(marked, "outer")
    native = load_predictions(context.run_directory)
    quality_cfg = config["evaluation"]["quality_sensitivity"]
    quality_by_key = {
        (int(row.user_id), str(row.trip_id)): bool(
            float(row.max_recording_gap_s) <= float(quality_cfg["maximum_recording_gap_seconds"])
            and float(row.max_implied_speed_mps) <= float(quality_cfg["maximum_implied_ground_speed_mps"])
            and (
                not bool(quality_cfg["require_strictly_increasing_timestamps"])
                or int(row.duplicate_or_nonpositive_time_steps) == 0
            )
        )
        for row in test.itertuples(index=False)
    }
    evaluated_frames = []
    catalogs = []
    for eps_m in map(float, config["evaluation"]["dbscan_eps_sensitivity_m"]):
        catalog = build_catalogue(
            train,
            eps_m,
            list(map(float, config["evaluation"]["support_quantiles"])),
            int(config["task"]["dbscan_min_samples"]),
        )
        catalogs.append(catalog)
        evaluated = evaluate_native_predictions(
            native,
            test,
            catalog,
            list(map(float, config["evaluation"]["support_quantiles"])),
            list(map(float, config["evaluation"]["fixed_radius_m"])),
            list(map(float, config["evaluation"]["point_thresholds_m"])),
        )
        evaluated["catalog_eps_m"] = eps_m
        evaluated["quality_sensitivity_eligible"] = [
            quality_by_key[(int(user_id), str(trip_id))]
            for user_id, trip_id in zip(evaluated["user_id"], evaluated["trip_id"])
        ]
        evaluated_frames.append(evaluated)
    evaluated = pd.concat(evaluated_frames, ignore_index=True)
    pd.concat(catalogs, ignore_index=True).to_csv(
        output / "outer_training_destination_catalogs.csv", index=False
    )
    evaluated.to_csv(output / "common_predictions_all_sensitivities.csv", index=False)
    trip, per_user, macro = metric_tables(evaluated)
    trip.to_csv(output / "metrics_trip_weighted_by_seed.csv", index=False)
    per_user.to_csv(output / "metrics_by_user_seed.csv", index=False)
    macro.to_csv(output / "metrics_user_macro_by_seed.csv", index=False)
    seed_summary(trip).to_csv(output / "metrics_trip_weighted_seed_summary.csv", index=False)
    seed_summary(macro).to_csv(output / "metrics_user_macro_seed_summary.csv", index=False)
    quality_primary = evaluated[
        (evaluated["catalog_eps_m"] == float(config["task"]["primary_dbscan_eps_m"]))
        & evaluated["quality_sensitivity_eligible"]
    ].copy()
    quality_trip, quality_by_user, quality_macro = metric_tables(quality_primary)
    quality_trip.to_csv(output / "quality_sensitivity_trip_weighted_by_seed.csv", index=False)
    quality_by_user.to_csv(output / "quality_sensitivity_by_user_seed.csv", index=False)
    seed_summary(quality_macro).to_csv(
        output / "quality_sensitivity_user_macro_seed_summary.csv", index=False
    )

    bootstrap_config = config["evaluation"]["bootstrap"]
    intervals, convergence = hierarchical_bootstrap(
        case_seed_mean(evaluated, float(config["task"]["primary_dbscan_eps_m"])),
        list(map(float, config["task"]["observation_ratios"])),
        int(bootstrap_config["replicates"]),
        int(bootstrap_config["seed"]),
    )
    intervals.to_csv(output / "paired_hierarchical_bootstrap.csv", index=False)
    audit = {
        "primary_estimand": "user-macro Hit@R90 with uncovered cases counted as errors",
        "primary_contrast": (
            "paired difference in mean user-macro Hit@R90 across the four "
            "prespecified observation ratios"
        ),
        "principal_methods": PRINCIPAL,
        "baseline_methods": BASELINES,
        "observation_ratios": config["task"]["observation_ratios"],
        "catalog_eps_sensitivity_m": config["evaluation"]["dbscan_eps_sensitivity_m"],
        "support_quantiles": config["evaluation"]["support_quantiles"],
        "neural_seed_handling": "metrics are reported per seed and as equal-weight seed mean and SD; case bootstrap uses per-case seed means",
        "bootstrap": config["evaluation"]["bootstrap"],
        "bootstrap_convergence": convergence,
        "test_trips": int(len(test)),
        "quality_sensitivity_test_trips": int(
            sum(quality_by_key.values())
        ),
        "quality_sensitivity_definition": quality_cfg,
        "users": int(test["user_id"].nunique()),
    }
    (output / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    audit = run(RunContext.from_environment())
    print(json.dumps(audit, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
