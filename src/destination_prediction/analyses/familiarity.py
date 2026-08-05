#!/usr/bin/env python3
"""Analyse destination familiarity and train-prefix analogue availability."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext, implementation_constants
from destination_prediction.data import add_local_sequences, load_marked, partitions, user_origins
from destination_prediction.geometry import prefix_cutoff, resample_path


CONTEXT = RunContext.from_environment()
CONFIG = CONTEXT.config
EXP_DIR = CONTEXT.run_directory


PRIMARY_EPS_M = float(CONFIG["task"]["primary_dbscan_eps_m"])
PRIMARY_SUPPORT_QUANTILE = float(CONFIG["task"]["primary_support_quantile"])
PRIMARY_SUPPORT_TOKEN = int(round(100 * PRIMARY_SUPPORT_QUANTILE))
ANALOGUE_QUANTILES = tuple(
    float(value) for value in CONFIG["evaluation"]["support_quantiles"]
)
ANALOGUE_RESAMPLE_POINTS = int(
    implementation_constants(CONFIG, "tsmini_inspired_retrieval")[
        "target_resample_points"
    ]
)
BOOTSTRAP_REPLICATES = int(CONFIG["evaluation"]["bootstrap"]["replicates"])
BOOTSTRAP_SEED = int(CONFIG["evaluation"]["bootstrap"]["seed"])


def load_case_predictions() -> pd.DataFrame:
    path = EXP_DIR / "outputs/final/common_predictions_all_sensitivities.csv"
    columns = [
        "method",
        "ratio",
        "user_id",
        "trip_id",
        "catalog_eps_m",
        f"covered_r{PRIMARY_SUPPORT_TOKEN}",
        f"true_center_id_r{PRIMARY_SUPPORT_TOKEN}",
        f"hit_r{PRIMARY_SUPPORT_TOKEN}_all",
        "within_200m",
        "within_1000m",
        "destination_error_m",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[frame["catalog_eps_m"] == PRIMARY_EPS_M].copy()
    frame["trip_id"] = frame["trip_id"].astype(str)
    return (
        frame.groupby(["method", "ratio", "user_id", "trip_id"], as_index=False)
        .agg(
            covered_r90=(f"covered_r{PRIMARY_SUPPORT_TOKEN}", "first"),
            true_center_id_r90=(f"true_center_id_r{PRIMARY_SUPPORT_TOKEN}", "first"),
            hit_r90=(f"hit_r{PRIMARY_SUPPORT_TOKEN}_all", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            destination_error_m=("destination_error_m", "mean"),
        )
    )


def destination_popularity_cases(case_predictions: pd.DataFrame) -> pd.DataFrame:
    cases = case_predictions[
        [
            "user_id",
            "trip_id",
            "covered_r90",
            "true_center_id_r90",
        ]
    ].drop_duplicates()
    catalog = pd.read_csv(
        EXP_DIR / "outputs/final/outer_training_destination_catalogs.csv"
    )
    catalog = catalog[catalog["catalog_eps_m"] == PRIMARY_EPS_M].copy()
    catalog["training_destinations"] = catalog.groupby("user_id")["support"].transform(
        "sum"
    )
    catalog["maximum_personal_support"] = catalog.groupby("user_id")[
        "support"
    ].transform("max")
    catalog["training_destination_share"] = (
        catalog["support"] / catalog["training_destinations"]
    )
    catalog["is_modal_destination"] = (
        catalog["support"] == catalog["maximum_personal_support"]
    )
    lookup = catalog[
        [
            "user_id",
            "center_id",
            "support",
            "training_destinations",
            "training_destination_share",
            "is_modal_destination",
            f"radius_q{PRIMARY_SUPPORT_TOKEN}_m",
        ]
    ].rename(columns={"center_id": "true_center_id_r90"})
    cases = cases.merge(
        lookup,
        on=["user_id", "true_center_id_r90"],
        how="left",
        validate="many_to_one",
    )
    cases["destination_popularity_stratum"] = "outside_training_r90"
    covered = cases["covered_r90"].astype(bool)
    modal = cases["is_modal_destination"].eq(True)
    cases.loc[
        covered & modal,
        "destination_popularity_stratum",
    ] = "popular_modal"
    cases.loc[
        covered
        & ~modal
        & cases["support"].ge(2),
        "destination_popularity_stratum",
    ] = "other_repeated"
    cases.loc[
        covered
        & ~modal
        & cases["support"].eq(1),
        "destination_popularity_stratum",
    ] = "known_singleton"
    return cases.sort_values(["user_id", "trip_id"]).reset_index(drop=True)


def resampled_prefixes(frame: pd.DataFrame, ratio: float) -> np.ndarray:
    return np.stack(
        [
            resample_path(
                np.asarray(sequence, dtype=float)[
                    : prefix_cutoff(len(sequence), ratio)
                ],
                ANALOGUE_RESAMPLE_POINTS,
            )
            for sequence in frame["local_seq"]
        ]
    )


def mean_path_distance_matrix(
    first: np.ndarray, second: np.ndarray, chunk_size: int = 64
) -> np.ndarray:
    output = np.empty((len(first), len(second)), dtype=np.float64)
    for start in range(0, len(first), chunk_size):
        stop = min(start + chunk_size, len(first))
        output[start:stop] = np.linalg.norm(
            first[start:stop, None, :, :] - second[None, :, :, :],
            axis=3,
        ).mean(axis=2)
    return output


def prefix_analogue_cases() -> pd.DataFrame:
    marked = load_marked(EXP_DIR)
    train, test = partitions(marked, "outer")
    origins = user_origins(train)
    train = add_local_sequences(train, origins)
    test = add_local_sequences(test, origins)
    rows: list[dict[str, object]] = []
    for ratio in map(float, CONFIG["task"]["observation_ratios"]):
        for user_id, train_user in train.groupby("user_id", sort=True):
            test_user = test[test["user_id"] == int(user_id)].reset_index(drop=True)
            train_user = train_user.reset_index(drop=True)
            train_prefixes = resampled_prefixes(train_user, ratio)
            test_prefixes = resampled_prefixes(test_user, ratio)
            train_pairwise = mean_path_distance_matrix(
                train_prefixes, train_prefixes
            )
            np.fill_diagonal(train_pairwise, np.inf)
            train_loo_nearest = train_pairwise.min(axis=1)
            test_nearest = mean_path_distance_matrix(
                test_prefixes, train_prefixes
            ).min(axis=1)
            thresholds = {
                int(round(100 * quantile)): float(
                    np.quantile(train_loo_nearest, quantile)
                )
                for quantile in ANALOGUE_QUANTILES
            }
            for position, test_row in enumerate(test_user.itertuples(index=False)):
                row: dict[str, object] = {
                    "ratio": ratio,
                    "user_id": int(user_id),
                    "trip_id": str(test_row.trip_id),
                    "nearest_train_prefix_distance_m": float(test_nearest[position]),
                    "analogue_resample_points": ANALOGUE_RESAMPLE_POINTS,
                }
                for token, threshold in thresholds.items():
                    row[f"train_loo_q{token}_m"] = threshold
                    row[f"no_train_analogue_q{token}"] = bool(
                        test_nearest[position] > threshold
                    )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["ratio", "user_id", "trip_id"]
    ).reset_index(drop=True)


def aggregate_metrics(
    frame: pd.DataFrame,
    grouping_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(
        [*grouping_columns, "method", "ratio"], sort=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(
            zip([*grouping_columns, "method", "ratio"], keys)
        )
        per_user = (
            group.groupby("user_id", as_index=False)
            .agg(
                hit_r90=("hit_r90", "mean"),
                within_200m=("within_200m", "mean"),
                within_1000m=("within_1000m", "mean"),
                median_error_m=("destination_error_m", "median"),
                trips=("trip_id", "size"),
            )
        )
        rows.append(
            {
                **key_values,
                "users": int(per_user["user_id"].nunique()),
                "trips": int(group[["user_id", "trip_id"]].drop_duplicates().shape[0]),
                "user_macro_hit_r90": float(per_user["hit_r90"].mean()),
                "trip_weighted_hit_r90": float(group["hit_r90"].mean()),
                "user_macro_within_200m": float(per_user["within_200m"].mean()),
                "user_macro_within_1000m": float(per_user["within_1000m"].mean()),
                "mean_user_median_error_m": float(per_user["median_error_m"].mean()),
            }
        )
    return pd.DataFrame(rows)


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    analysis: str,
    stratum: str,
    ratio: float,
) -> pd.DataFrame:
    pivot = frame.pivot(
        index=["user_id", "trip_id"],
        columns="method",
        values="hit_r90",
    ).sort_index()
    pivot = pivot.dropna(axis=0)
    methods = list(pivot.columns)
    users = np.asarray(
        sorted(pivot.index.get_level_values("user_id").unique()),
        dtype=int,
    )
    if len(users) < 2 or len(pivot) < 2:
        return pd.DataFrame()
    by_user = {
        int(user): pivot.xs(user, level="user_id").to_numpy(dtype=float)
        for user in users
    }
    estimates = np.empty((BOOTSTRAP_REPLICATES, len(methods)), dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    batch_size = 1000
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        count = min(batch_size, BOOTSTRAP_REPLICATES - start)
        sampled_users = rng.choice(
            users, size=(count, len(users)), replace=True
        )
        user_means = np.empty(
            (count, len(users), len(methods)), dtype=float
        )
        for slot in range(len(users)):
            selected = sampled_users[:, slot]
            for user in users:
                mask = selected == user
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
    rows: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        values = estimates[:, method_index]
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "analysis": analysis,
                "stratum": stratum,
                "ratio": ratio,
                "quantity": "rate",
                "method": method,
                "reference_method": "",
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "users": int(len(users)),
                "trips": int(len(pivot)),
            }
        )
    for first_index, second_index in itertools.combinations(
        range(len(methods)), 2
    ):
        values = estimates[:, first_index] - estimates[:, second_index]
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "analysis": analysis,
                "stratum": stratum,
                "ratio": ratio,
                "quantity": "difference",
                "method": methods[first_index],
                "reference_method": methods[second_index],
                "bootstrap_mean": float(values.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "users": int(len(users)),
                "trips": int(len(pivot)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_all_strata(
    frame: pd.DataFrame,
    analysis: str,
    stratum_column: str,
) -> pd.DataFrame:
    outputs = []
    for (stratum, ratio), group in frame.groupby(
        [stratum_column, "ratio"], sort=True
    ):
        result = hierarchical_bootstrap(
            group,
            analysis=analysis,
            stratum=str(stratum),
            ratio=float(ratio),
        )
        if not result.empty:
            outputs.append(result)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def main() -> int:
    output = EXP_DIR / "outputs/analysis"
    output.mkdir(parents=True, exist_ok=True)
    predictions = load_case_predictions()

    popularity_cases = destination_popularity_cases(predictions)
    popularity_cases.to_csv(
        output / "destination_popularity_cases.csv", index=False
    )
    popularity_evaluated = predictions.merge(
        popularity_cases[
            [
                "user_id",
                "trip_id",
                "destination_popularity_stratum",
                "support",
                "training_destination_share",
            ]
        ],
        on=["user_id", "trip_id"],
        validate="many_to_one",
    )
    popularity_metrics = aggregate_metrics(
        popularity_evaluated, ["destination_popularity_stratum"]
    )
    popularity_metrics.to_csv(
        output / "destination_popularity_metrics.csv", index=False
    )

    analogue_cases = prefix_analogue_cases()
    analogue_cases.to_csv(output / "prefix_analogue_cases.csv", index=False)
    analogue_evaluated = predictions.merge(
        analogue_cases,
        on=["ratio", "user_id", "trip_id"],
        validate="many_to_one",
    )
    analogue_metric_frames = []
    for quantile in ANALOGUE_QUANTILES:
        token = int(round(100 * quantile))
        current = analogue_evaluated.copy()
        current["analogue_availability"] = np.where(
            current[f"no_train_analogue_q{token}"],
            "no_close_train_analogue",
            "train_analogue_present",
        )
        metrics = aggregate_metrics(current, ["analogue_availability"])
        metrics.insert(0, "analogue_threshold_quantile", quantile)
        analogue_metric_frames.append(metrics)
    analogue_metrics = pd.concat(analogue_metric_frames, ignore_index=True)
    analogue_metrics.to_csv(
        output / "prefix_analogue_metrics.csv", index=False
    )

    primary_token = PRIMARY_SUPPORT_TOKEN
    primary_analogue = analogue_evaluated.copy()
    primary_analogue["analogue_availability"] = np.where(
        primary_analogue[f"no_train_analogue_q{primary_token}"],
        "no_close_train_analogue",
        "train_analogue_present",
    )
    bootstrap = pd.concat(
        [
            bootstrap_all_strata(
                popularity_evaluated,
                analysis="destination_popularity",
                stratum_column="destination_popularity_stratum",
            ),
            bootstrap_all_strata(
                primary_analogue,
                analysis="prefix_analogue_q90",
                stratum_column="analogue_availability",
            ),
        ],
        ignore_index=True,
    )
    bootstrap.to_csv(output / "stratified_bootstrap.csv", index=False)

    joint = (
        popularity_cases[
            ["user_id", "trip_id", "destination_popularity_stratum"]
        ]
        .merge(
            analogue_cases[
                [
                    "ratio",
                    "user_id",
                    "trip_id",
                    f"no_train_analogue_q{primary_token}",
                ]
            ],
            on=["user_id", "trip_id"],
            validate="one_to_many",
        )
        .assign(
            analogue_availability=lambda frame: np.where(
                frame[f"no_train_analogue_q{primary_token}"],
                "no_close_train_analogue",
                "train_analogue_present",
            )
        )
        .groupby(
            [
                "ratio",
                "destination_popularity_stratum",
                "analogue_availability",
            ],
            as_index=False,
        )
        .agg(
            trips=("trip_id", "size"),
            users=("user_id", "nunique"),
        )
    )
    joint.to_csv(
        output / "destination_popularity_by_prefix_analogue.csv",
        index=False,
    )

    summary = {
        "status": "completed",
        "primary_destination_popularity_definition": (
            "A test destination is popular_modal when it lies inside a "
            "training-only personal R90 support region whose cluster has the "
            "maximum outer-training endpoint support for that user. Tied "
            "maxima are retained."
        ),
        "destination_popularity_strata": [
            "popular_modal",
            "other_repeated",
            "known_singleton",
            "outside_training_r90",
        ],
        "prefix_analogue_definition": (
            "For each user and observation ratio, 32-point arc-length "
            "prefix distance is compared with the training leave-one-out "
            "nearest-neighbour distribution. No close train analogue means "
            "that the test nearest-neighbour distance exceeds the selected "
            "training-only quantile."
        ),
        "analogue_resample_points": ANALOGUE_RESAMPLE_POINTS,
        "analogue_threshold_quantiles": list(ANALOGUE_QUANTILES),
        "primary_analogue_threshold_quantile": PRIMARY_SUPPORT_QUANTILE,
        "bootstrap": {
            "design": "paired hierarchical user-then-trip bootstrap within stratum",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "test_trips": int(
            popularity_cases[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ),
        "ratios": [float(value) for value in CONFIG["task"]["observation_ratios"]],
        "destination_popularity_counts": {
            str(key): int(value)
            for key, value in popularity_cases[
                "destination_popularity_stratum"
            ].value_counts().sort_index().items()
        },
        "no_close_train_analogue_counts_q90": {
            f"{float(ratio):.2f}": int(group[f"no_train_analogue_q{PRIMARY_SUPPORT_TOKEN}"].sum())
            for ratio, group in analogue_cases.groupby("ratio", sort=True)
        },
    }
    (output / "familiarity_analogue_analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
