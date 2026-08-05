#!/usr/bin/env python3
"""EDBT diagnostic analyses using frozen discovery and held-out outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext
from destination_prediction.protocol import EvaluationProtocol


CONTEXT = RunContext.from_environment()
EXP_DIR = CONTEXT.run_directory
CONFIG = CONTEXT.config
OUTPUT = EXP_DIR / "outputs"
CASES = OUTPUT / "cases"
TABLES = OUTPUT / "tables"
CHECKS = OUTPUT / "checks"


def source_path(cohort: str) -> Path:
    return CONTEXT.dependency_directory(CONFIG["source_experiments"][cohort])


def cohort_protocol(cohort: str):
    return EvaluationProtocol(
        CONTEXT.dependency_context(CONFIG["source_experiments"][cohort])
    )


def prepare_cohort(cohort: str):
    protocol = cohort_protocol(cohort)
    marked = protocol.load_marked()
    train, test = protocol.partitions(marked, "outer")
    origins = protocol.user_origins(train)
    train = protocol.add_local_sequences(train, origins)
    test = protocol.add_local_sequences(test, origins)
    catalog = protocol.build_catalog(train, float(CONFIG["catalog_eps_m"]))
    cluster_map = protocol.endpoint_cluster_map(train, catalog)
    return protocol, train, test, catalog, cluster_map


def destination_ambiguity_cases(
    cohort: str,
    protocol,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cluster_map: dict[tuple[int, str], int],
) -> pd.DataFrame:
    ratios = list(map(float, CONFIG["observation_ratios"]))
    neighbours = [
        int(CONFIG["ambiguity"]["primary_neighbors"]),
        *map(int, CONFIG["ambiguity"]["sensitivity_neighbors"]),
    ]
    neighbours = sorted(set(neighbours))
    maximum_k = max(neighbours)
    rows: list[dict[str, object]] = []
    for ratio in ratios:
        for user_id, train_user in train.groupby("user_id", sort=True):
            user_id = int(user_id)
            train_user = train_user.reset_index(drop=True)
            test_user = test[test["user_id"] == user_id].reset_index(drop=True)
            train_positions = np.asarray(
                [
                    sequence[
                        protocol.prefix_cutoff(len(sequence), ratio) - 1
                    ]
                    for sequence in train_user["local_seq"]
                ],
                dtype=float,
            )
            test_positions = np.asarray(
                [
                    sequence[
                        protocol.prefix_cutoff(len(sequence), ratio) - 1
                    ]
                    for sequence in test_user["local_seq"]
                ],
                dtype=float,
            )
            destination_labels = np.asarray(
                [
                    cluster_map[(user_id, str(trip_id))]
                    for trip_id in train_user["trip_id"]
                ],
                dtype=int,
            )
            distances = np.linalg.norm(
                test_positions[:, None, :] - train_positions[None, :, :],
                axis=2,
            )
            order = np.argsort(distances, axis=1)[:, :maximum_k]
            for position, test_row in enumerate(test_user.itertuples(index=False)):
                for requested_k in neighbours:
                    effective_k = min(requested_k, len(train_user))
                    labels = destination_labels[order[position, :effective_k]]
                    _, counts = np.unique(labels, return_counts=True)
                    probabilities = counts.astype(float) / float(effective_k)
                    probabilities = np.sort(probabilities)[::-1]
                    entropy = float(
                        -np.sum(probabilities * np.log(probabilities))
                    )
                    top1 = float(probabilities[0])
                    top2 = float(probabilities[1]) if len(probabilities) > 1 else 0.0
                    rows.append(
                        {
                            "cohort": cohort,
                            "ratio": ratio,
                            "user_id": user_id,
                            "trip_id": str(test_row.trip_id),
                            "requested_k": requested_k,
                            "effective_k": effective_k,
                            "distinct_destinations": int(len(counts)),
                            "entropy_nats": entropy,
                            "effective_destinations": float(math.exp(entropy)),
                            "top1_share": top1,
                            "top2_share": top2,
                            "top1_top2_margin": top1 - top2,
                            "nearest_training_position_m": float(
                                distances[position, order[position, 0]]
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["requested_k", "ratio", "user_id", "trip_id"]
    ).reset_index(drop=True)


def ambiguity_thresholds(discovery: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (requested_k, ratio), group in discovery.groupby(
        ["requested_k", "ratio"], sort=True
    ):
        low, high = group["entropy_nats"].quantile([1 / 3, 2 / 3]).to_numpy()
        rows.append(
            {
                "requested_k": int(requested_k),
                "ratio": float(ratio),
                "entropy_low_upper": float(low),
                "entropy_medium_upper": float(high),
                "discovery_cases": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def apply_ambiguity_bins(
    frame: pd.DataFrame, thresholds: pd.DataFrame
) -> pd.DataFrame:
    out = frame.merge(
        thresholds,
        on=["requested_k", "ratio"],
        how="left",
        validate="many_to_one",
    )
    out["ambiguity_stratum"] = "high"
    out.loc[
        out["entropy_nats"] <= out["entropy_medium_upper"],
        "ambiguity_stratum",
    ] = "medium"
    out.loc[
        out["entropy_nats"] <= out["entropy_low_upper"],
        "ambiguity_stratum",
    ] = "low"
    return out


def load_predictions() -> pd.DataFrame:
    path = (
        source_path("held_out")
        / "outputs/final/common_predictions_all_sensitivities.csv"
    )
    token = int(round(float(CONFIG["support_quantile"]) * 100))
    columns = [
        "method",
        "ratio",
        "user_id",
        "trip_id",
        "catalog_eps_m",
        f"covered_r{token}",
        f"hit_r{token}_all",
        "within_200m",
        "within_1000m",
        "destination_error_m",
        "native_destination_error_m",
    ]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame[
        np.isclose(frame["catalog_eps_m"], float(CONFIG["catalog_eps_m"]))
    ].copy()
    frame["trip_id"] = frame["trip_id"].astype(str)
    return (
        frame.groupby(["method", "ratio", "user_id", "trip_id"], as_index=False)
        .agg(
            covered_r90=(f"covered_r{token}", "first"),
            hit_r90=(f"hit_r{token}_all", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            destination_error_m=("destination_error_m", "mean"),
            native_destination_error_m=(
                "native_destination_error_m",
                "mean",
            ),
        )
    )


def user_macro_summary(
    frame: pd.DataFrame,
    strata: list[str],
    analysis: str,
) -> pd.DataFrame:
    keys = [*strata, "ratio", "method", "user_id"]
    per_user = (
        frame.groupby(keys, as_index=False)
        .agg(
            hit_r90=("hit_r90", "mean"),
            within_200m=("within_200m", "mean"),
            within_1000m=("within_1000m", "mean"),
            median_error_m=("destination_error_m", "median"),
            cases=("trip_id", "nunique"),
        )
    )
    summary = (
        per_user.groupby([*strata, "ratio", "method"], as_index=False)
        .agg(
            user_macro_hit_r90=("hit_r90", "mean"),
            user_macro_within_200m=("within_200m", "mean"),
            user_macro_within_1000m=("within_1000m", "mean"),
            mean_user_median_error_m=("median_error_m", "mean"),
            users=("user_id", "nunique"),
            user_case_sum=("cases", "sum"),
        )
    )
    summary.insert(0, "analysis", analysis)
    return summary


def paired_hierarchical_bootstrap(
    frame: pd.DataFrame,
    strata: dict[str, object],
) -> dict[str, object] | None:
    focal = CONFIG["focal_methods"]["full_prefix"]
    reference = CONFIG["focal_methods"]["current_only"]
    pivot = (
        frame[frame["method"].isin([focal, reference])]
        .pivot(
            index=["user_id", "trip_id"],
            columns="method",
            values="hit_r90",
        )
        .dropna()
    )
    users = np.asarray(
        sorted(pivot.index.get_level_values("user_id").unique()), dtype=int
    )
    if len(users) < 2:
        return None
    by_user = {
        int(user): pivot.xs(user, level="user_id")[
            [focal, reference]
        ].to_numpy(dtype=float)
        for user in users
    }
    replicates = int(CONFIG["bootstrap"]["replicates"])
    seed_offset = sum(ord(ch) for ch in json.dumps(strata, sort_keys=True))
    rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]) + seed_offset)
    estimates = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        means = []
        for user in sampled_users:
            values = by_user[int(user)]
            positions = rng.integers(0, len(values), size=len(values))
            sampled = values[positions]
            means.append(float((sampled[:, 0] - sampled[:, 1]).mean()))
        estimates[replicate] = float(np.mean(means))
    low, high = np.quantile(estimates, [0.025, 0.975])
    observed_per_user = (
        pivot[focal] - pivot[reference]
    ).groupby(level="user_id").mean()
    return {
        **strata,
        "method": focal,
        "reference_method": reference,
        "observed_difference": float(observed_per_user.mean()),
        "bootstrap_mean": float(estimates.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "users": int(len(users)),
        "cases": int(len(pivot)),
    }


def ambiguity_outputs(
    predictions: pd.DataFrame,
    held_out_ambiguity: pd.DataFrame,
) -> None:
    reported = set(CONFIG["reported_methods"])
    joined = predictions[
        predictions["method"].isin(reported)
    ].merge(
        held_out_ambiguity,
        on=["ratio", "user_id", "trip_id"],
        how="inner",
        validate="many_to_many",
    )
    joined.to_csv(CASES / "prediction_cases_with_ambiguity.csv", index=False)
    user_macro_summary(
        joined,
        ["requested_k", "ambiguity_stratum"],
        "location_ambiguity",
    ).to_csv(TABLES / "ambiguity_method_metrics.csv", index=False)
    distribution = (
        held_out_ambiguity.groupby(
            ["requested_k", "ratio", "ambiguity_stratum"], as_index=False
        )
        .agg(
            cases=("trip_id", "size"),
            users=("user_id", "nunique"),
            mean_entropy_nats=("entropy_nats", "mean"),
            mean_effective_destinations=("effective_destinations", "mean"),
            mean_top1_share=("top1_share", "mean"),
        )
    )
    distribution["case_share"] = distribution["cases"] / distribution.groupby(
        ["requested_k", "ratio"]
    )["cases"].transform("sum")
    distribution.to_csv(TABLES / "ambiguity_distribution.csv", index=False)
    primary_k = int(CONFIG["ambiguity"]["primary_neighbors"])
    bootstrap_rows = []
    primary = joined[joined["requested_k"] == primary_k]
    for (ratio, stratum), group in primary.groupby(
        ["ratio", "ambiguity_stratum"], sort=True
    ):
        result = paired_hierarchical_bootstrap(
            group,
            {
                "analysis": "location_ambiguity",
                "requested_k": primary_k,
                "ratio": float(ratio),
                "stratum": str(stratum),
            },
        )
        if result is not None:
            bootstrap_rows.append(result)
    pd.DataFrame(bootstrap_rows).to_csv(
        TABLES / "ambiguity_focal_bootstrap.csv", index=False
    )


def cohort_user_profiles(
    cohort: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    catalog: pd.DataFrame,
    ambiguity: pd.DataFrame,
) -> pd.DataFrame:
    support_token = int(round(float(CONFIG["support_quantile"]) * 100))
    rows = []
    analogue_path = (
        source_path(cohort) / "outputs/analysis/prefix_analogue_cases.csv"
    )
    analogue_column = f"no_train_analogue_q{support_token}"
    if analogue_path.exists():
        analogue = pd.read_csv(analogue_path)
    else:
        analogue = pd.DataFrame(
            columns=["user_id", "trip_id", analogue_column]
        )
    for user_id, train_user in train.groupby("user_id", sort=True):
        user_id = int(user_id)
        personal_catalog = catalog[catalog["user_id"] == user_id].copy()
        support = personal_catalog["support"].to_numpy(dtype=float)
        probabilities = support / support.sum()
        entropy = float(-np.sum(probabilities * np.log(probabilities)))
        user_ambiguity = ambiguity[
            (ambiguity["user_id"] == user_id)
            & (
                ambiguity["requested_k"]
                == int(CONFIG["ambiguity"]["primary_neighbors"])
            )
        ]
        user_analogue = analogue[analogue["user_id"] == user_id]
        no_analogue_share = (
            float(user_analogue[analogue_column].astype(bool).mean())
            if len(user_analogue)
            else float("nan")
        )
        rows.append(
            {
                "cohort": cohort,
                "user_id": user_id,
                "training_trajectories": int(len(train_user)),
                "test_trajectories": int(
                    (test["user_id"] == user_id).sum()
                ),
                "destination_regions": int(len(personal_catalog)),
                "destination_entropy_nats": entropy,
                "effective_destination_regions": float(math.exp(entropy)),
                "modal_destination_share": float(probabilities.max()),
                "singleton_region_share": float((support == 1).mean()),
                "singleton_training_endpoint_share": float(
                    support[support == 1].sum() / support.sum()
                ),
                "mean_location_ambiguity_entropy": float(
                    user_ambiguity["entropy_nats"].mean()
                ),
                "mean_effective_compatible_destinations": float(
                    user_ambiguity["effective_destinations"].mean()
                ),
                "no_prefix_analogue_q90_share": no_analogue_share,
            }
        )
    return pd.DataFrame(rows)


def discovery_history_thresholds(profiles: pd.DataFrame) -> dict[str, float]:
    feature = str(CONFIG["history_size"]["bin_feature"])
    low, high = profiles[feature].quantile([1 / 3, 2 / 3]).to_numpy()
    return {"low_upper": float(low), "medium_upper": float(high)}


def apply_history_bins(
    profiles: pd.DataFrame, thresholds: dict[str, float]
) -> pd.DataFrame:
    feature = str(CONFIG["history_size"]["bin_feature"])
    out = profiles.copy()
    out["history_size_stratum"] = "long"
    out.loc[
        out[feature] <= thresholds["medium_upper"], "history_size_stratum"
    ] = "medium"
    out.loc[
        out[feature] <= thresholds["low_upper"], "history_size_stratum"
    ] = "short"
    return out


def user_effects(predictions: pd.DataFrame) -> pd.DataFrame:
    focal = CONFIG["focal_methods"]["full_prefix"]
    reference = CONFIG["focal_methods"]["current_only"]
    pivot = predictions[
        predictions["method"].isin([focal, reference])
    ].pivot(
        index=["ratio", "user_id", "trip_id"],
        columns="method",
        values="hit_r90",
    )
    pivot["paired_effect"] = pivot[focal] - pivot[reference]
    per_ratio = (
        pivot.reset_index()
        .groupby(["user_id", "ratio"], as_index=False)
        .agg(
            full_prefix_hit_r90=(focal, "mean"),
            current_only_hit_r90=(reference, "mean"),
            paired_effect=("paired_effect", "mean"),
            cases=("trip_id", "nunique"),
        )
    )
    mean_rows = (
        per_ratio.groupby("user_id", as_index=False)
        .agg(
            full_prefix_hit_r90=("full_prefix_hit_r90", "mean"),
            current_only_hit_r90=("current_only_hit_r90", "mean"),
            paired_effect=("paired_effect", "mean"),
            cases=("cases", "sum"),
        )
    )
    mean_rows["ratio"] = "mean"
    return pd.concat([per_ratio, mean_rows], ignore_index=True)


def spearman_with_bootstrap(frame: pd.DataFrame) -> pd.DataFrame:
    features = [
        "training_trajectories",
        "destination_regions",
        "destination_entropy_nats",
        "effective_destination_regions",
        "modal_destination_share",
        "singleton_region_share",
        "singleton_training_endpoint_share",
        "mean_location_ambiguity_entropy",
        "mean_effective_compatible_destinations",
        "no_prefix_analogue_q90_share",
    ]
    rng = np.random.default_rng(int(CONFIG["bootstrap"]["seed"]))
    replicates = int(CONFIG["bootstrap"]["replicates"])
    rows = []
    for feature in features:
        values = frame[[feature, "paired_effect"]].dropna().to_numpy(dtype=float)
        observed = float(
            pd.Series(values[:, 0]).corr(
                pd.Series(values[:, 1]), method="spearman"
            )
        )
        estimates = []
        for _ in range(replicates):
            positions = rng.integers(0, len(values), size=len(values))
            sample = values[positions]
            estimate = pd.Series(sample[:, 0]).corr(
                pd.Series(sample[:, 1]), method="spearman"
            )
            if np.isfinite(estimate):
                estimates.append(float(estimate))
        low, high = np.quantile(estimates, [0.025, 0.975])
        rows.append(
            {
                "feature": feature,
                "spearman_rho": observed,
                "bootstrap_mean": float(np.mean(estimates)),
                "ci_low": float(low),
                "ci_high": float(high),
                "users": int(len(values)),
                "valid_bootstrap_replicates": int(len(estimates)),
            }
        )
    return pd.DataFrame(rows)


def heterogeneity_outputs(
    predictions: pd.DataFrame,
    discovery_profiles: pd.DataFrame,
    held_out_profiles: pd.DataFrame,
) -> None:
    thresholds = discovery_history_thresholds(discovery_profiles)
    (CHECKS / "history_size_thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n", encoding="utf-8"
    )
    profiles = apply_history_bins(held_out_profiles, thresholds)
    effects = user_effects(predictions)
    profiles.to_csv(CASES / "held_out_user_profiles.csv", index=False)
    effects.to_csv(CASES / "held_out_user_effects.csv", index=False)
    mean_effects = effects[effects["ratio"].astype(str) == "mean"].copy()
    joined = profiles.merge(
        mean_effects.drop(columns=["ratio"]),
        on="user_id",
        validate="one_to_one",
    )
    joined.to_csv(TABLES / "user_heterogeneity.csv", index=False)
    spearman_with_bootstrap(joined).to_csv(
        TABLES / "user_feature_effect_correlations.csv", index=False
    )
    prediction_history = predictions.merge(
        profiles[["user_id", "history_size_stratum"]],
        on="user_id",
        validate="many_to_one",
    )
    user_macro_summary(
        prediction_history,
        ["history_size_stratum"],
        "history_size",
    ).to_csv(TABLES / "history_size_method_metrics.csv", index=False)


def joint_familiarity_outputs(predictions: pd.DataFrame) -> None:
    base = source_path("held_out") / "outputs/analysis"
    popularity = pd.read_csv(base / "destination_popularity_cases.csv")
    popularity["trip_id"] = popularity["trip_id"].astype(str)
    analogue = pd.read_csv(base / "prefix_analogue_cases.csv")
    analogue["trip_id"] = analogue["trip_id"].astype(str)
    analogue = analogue[
        ["ratio", "user_id", "trip_id", "no_train_analogue_q90"]
    ]
    reported = set(CONFIG["reported_methods"])
    joined = (
        predictions[predictions["method"].isin(reported)]
        .merge(
            popularity[
                ["user_id", "trip_id", "destination_popularity_stratum"]
            ],
            on=["user_id", "trip_id"],
            validate="many_to_one",
        )
        .merge(
            analogue,
            on=["ratio", "user_id", "trip_id"],
            validate="many_to_one",
        )
    )
    joined["prefix_familiarity"] = np.where(
        joined["no_train_analogue_q90"].astype(bool),
        "no_close_analogue",
        "analogue_present",
    )
    user_macro_summary(
        joined,
        ["destination_popularity_stratum", "prefix_familiarity"],
        "joint_familiarity",
    ).to_csv(TABLES / "joint_familiarity_method_metrics.csv", index=False)
    counts = (
        joined[
            [
                "ratio",
                "user_id",
                "trip_id",
                "destination_popularity_stratum",
                "prefix_familiarity",
            ]
        ]
        .drop_duplicates()
        .groupby(
            [
                "ratio",
                "destination_popularity_stratum",
                "prefix_familiarity",
            ],
            as_index=False,
        )
        .agg(cases=("trip_id", "size"), users=("user_id", "nunique"))
    )
    counts.to_csv(TABLES / "joint_familiarity_case_counts.csv", index=False)


def point_coverage_outputs(predictions: pd.DataFrame) -> None:
    per_user = (
        predictions.groupby(["method", "ratio", "user_id"], as_index=False)
        .agg(
            coverage_r90=("covered_r90", "mean"),
            d200=("within_200m", "mean"),
            d1km=("within_1000m", "mean"),
            median_error_m=("destination_error_m", "median"),
            median_native_error_m=("native_destination_error_m", "median"),
            p90_error_m=(
                "destination_error_m",
                lambda values: float(np.quantile(values, 0.9)),
            ),
            p90_native_error_m=(
                "native_destination_error_m",
                lambda values: float(np.quantile(values, 0.9)),
            ),
        )
    )
    (
        per_user.groupby(["method", "ratio"], as_index=False)
        .agg(
            user_macro_coverage_r90=("coverage_r90", "mean"),
            user_macro_d200=("d200", "mean"),
            user_macro_d1km=("d1km", "mean"),
            mean_user_median_error_m=("median_error_m", "mean"),
            mean_user_native_median_error_m=(
                "median_native_error_m",
                "mean",
            ),
            mean_user_p90_error_m=("p90_error_m", "mean"),
            mean_user_native_p90_error_m=("p90_native_error_m", "mean"),
        )
    ).to_csv(TABLES / "point_error_and_coverage_metrics.csv", index=False)


def main() -> int:
    for directory in [CASES, TABLES, CHECKS]:
        directory.mkdir(parents=True, exist_ok=True)
    prepared = {}
    ambiguity = {}
    for cohort in ["discovery", "held_out"]:
        protocol, train, test, catalog, cluster_map = prepare_cohort(cohort)
        prepared[cohort] = (train, test, catalog)
        ambiguity[cohort] = destination_ambiguity_cases(
            cohort, protocol, train, test, cluster_map
        )
        ambiguity[cohort].to_csv(
            CASES / f"{cohort}_location_ambiguity_cases.csv", index=False
        )
    thresholds = ambiguity_thresholds(ambiguity["discovery"])
    thresholds.to_csv(CHECKS / "ambiguity_thresholds.csv", index=False)
    held_out_ambiguity = apply_ambiguity_bins(
        ambiguity["held_out"], thresholds
    )
    held_out_ambiguity.to_csv(
        CASES / "held_out_location_ambiguity_binned.csv", index=False
    )
    predictions = load_predictions()
    ambiguity_outputs(predictions, held_out_ambiguity)
    profiles = {}
    for cohort in ["discovery", "held_out"]:
        train, test, catalog = prepared[cohort]
        profiles[cohort] = cohort_user_profiles(
            cohort, train, test, catalog, ambiguity[cohort]
        )
    heterogeneity_outputs(
        predictions, profiles["discovery"], profiles["held_out"]
    )
    joint_familiarity_outputs(predictions)
    point_coverage_outputs(predictions)
    summary = {
        "status": "completed",
        "discovery_users": int(profiles["discovery"]["user_id"].nunique()),
        "held_out_users": int(profiles["held_out"]["user_id"].nunique()),
        "held_out_test_trajectories": int(
            predictions[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ),
        "observation_ratios": CONFIG["observation_ratios"],
        "primary_ambiguity_neighbors": CONFIG["ambiguity"][
            "primary_neighbors"
        ],
        "sensitivity_ambiguity_neighbors": CONFIG["ambiguity"][
            "sensitivity_neighbors"
        ],
        "bootstrap_replicates": CONFIG["bootstrap"]["replicates"],
        "test_outcomes_used_to_define_ambiguity": False,
        "held_out_outcomes_used_to_define_stratum_thresholds": False,
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
