#!/usr/bin/env python3
"""Fail an evaluation run unless protocol and output invariants are satisfied."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from destination_prediction.context import RunContext, parse_run_context


def _configure(context: RunContext) -> None:
    global CONTEXT, OUTPUT, CONFIG
    CONTEXT = context
    OUTPUT = context.run_directory / "outputs"
    CONFIG = context.config


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def unique_cases(path: Path, expected_rows: int, seeds: int) -> dict[str, int]:
    frame = pd.read_csv(path, dtype={"trip_id": str})
    require(len(frame) == expected_rows, f"{path}: {len(frame)} rows, expected {expected_rows}")
    keys = ["family", "config_id", "seed", "ratio", "user_id", "trip_id"]
    if seeds == 1 and "seed" not in frame:
        keys.remove("seed")
    require(not frame.duplicated(keys).any(), f"{path}: duplicate case rows")
    require(frame["native_pred_lon"].notna().all(), f"{path}: missing longitude")
    require(frame["native_pred_lat"].notna().all(), f"{path}: missing latitude")
    return {"rows": int(len(frame)), "unique_cases": int(frame[["user_id", "trip_id"]].drop_duplicates().shape[0])}


def validate(context: RunContext) -> int:
    _configure(context)
    context.require_fixed_configurations_match()
    preparation = json.loads(
        (OUTPUT / "preparation/preparation_summary.json").read_text(encoding="utf-8")
    )
    require(preparation["source_plt_files"] == 18670, "Unexpected GeoLife PLT file count")
    require(preparation["selected_user_count"] == int(CONFIG["dataset"]["cohort_users"]), "Cohort size mismatch")
    selected_users = set(map(int, preparation["selected_users"]))
    historical_users = set(map(int, CONFIG["dataset"]["exclude_user_ids"]))
    prespecified_users = set(map(int, CONFIG["dataset"].get("include_user_ids", [])))
    require(
        selected_users.isdisjoint(historical_users),
        "Previously used user entered the frozen evaluation cohort",
    )
    if prespecified_users:
        require(
            selected_users == prespecified_users,
            "Prepared users differ from the frozen evaluation cohort",
        )
        cohort_manifest = pd.read_csv(
            CONTEXT.repository_root / "reference" / "manifests" / "cohort_manifest.csv"
        )
        frozen_users = set(
            cohort_manifest.loc[
                cohort_manifest["evaluation_membership"].astype(bool), "user_id"
            ].astype(int)
        )
        require(
            selected_users == frozen_users,
            "Prepared users differ from the normalized cohort manifest",
        )
    else:
        require(
            len(selected_users) == int(CONFIG["dataset"]["cohort_users"]),
            "Expanded cohort is not the complete configured complement",
        )
    require(preparation["manual_segmentation"] is False, "Manual segmentation flag changed")
    require(preparation["row_level_mode_filtering"] is False, "Row-level mode filtering detected")
    require(preparation["source_file_retained_as_indivisible_unit"] is True, "PLT unit not preserved")

    manifest = pd.read_csv(OUTPUT / "preparation/partition_manifest.csv", dtype={"trip_id": str})
    require(not manifest.duplicated(["user_id", "trip_id"]).any(), "Duplicate trajectory IDs")
    require(set(manifest["partition"]) == {"fit", "validation", "test"}, "Incomplete partitions")
    for _, group in manifest.sort_values(["user_id", "chronological_index"]).groupby("user_id"):
        labels = group["partition"].tolist()
        encoded = [{"fit": 0, "validation": 1, "test": 2}[label] for label in labels]
        require(encoded == sorted(encoded), "Non-chronological partition order")

    test_trips = int((manifest["partition"] == "test").sum())
    ratios = len(CONFIG["task"]["observation_ratios"])
    seeds = len(CONFIG["selection"]["final_neural_seeds"])
    checks = {
        "non_neural": unique_cases(
            OUTPUT / "non_neural/test_predictions.csv",
            test_trips * ratios * 6,
            1,
        ),
        "recurrent": unique_cases(
            OUTPUT / "recurrent/test_predictions_by_seed.csv",
            test_trips * ratios * seeds,
            seeds,
        ),
        "tsmini": unique_cases(
            OUTPUT / "tsmini/test_predictions_by_seed.csv",
            test_trips * ratios * seeds,
            seeds,
        ),
    }

    expected_seeds = set(map(int, CONFIG["selection"]["final_neural_seeds"]))
    for family in ["recurrent", "tsmini"]:
        frame = pd.read_csv(OUTPUT / family / "test_predictions_by_seed.csv")
        require(set(map(int, frame["seed"].unique())) == expected_seeds, f"{family}: seed mismatch")

    train_trip_ids = set(
        zip(
            manifest.loc[manifest["partition"].isin(["fit", "validation"]), "user_id"].astype(int),
            manifest.loc[manifest["partition"].isin(["fit", "validation"]), "trip_id"].astype(str),
        )
    )
    catalog = pd.read_csv(OUTPUT / "non_neural/outer_training_destination_catalog.csv", dtype={"medoid_trip_id": str})
    medoids = set(zip(catalog["user_id"].astype(int), catalog["medoid_trip_id"].astype(str)))
    require(medoids.issubset(train_trip_ids), "Outer catalog contains non-training medoid")

    fixed_paths = {
        "geometric_retrieval": OUTPUT / "non_neural/geometric_selection/selected_config.json",
        "probabilistic_grid_pattern_retrieval": OUTPUT / "non_neural/grid_pattern_selection/selected_config.json",
        "recurrent": OUTPUT / "recurrent/selected_config.json",
        "tsmini": OUTPUT / "tsmini/selected_config.json",
    }
    for family, path in fixed_paths.items():
        selected = json.loads(path.read_text(encoding="utf-8"))
        require(selected.get("outer_test_read") is False, f"{path}: outer-test selection flag")
        require(
            selected.get("selection_source") == "frozen 30-user discovery validation",
            f"{path}: configuration was not marked as externally frozen",
        )
        for key, expected in CONFIG["selection"]["fixed_configs"][family].items():
            require(selected.get(key) == expected, f"{path}: frozen {key} changed")

    required_final = [
        "common_predictions_all_sensitivities.csv",
        "metrics_trip_weighted_seed_summary.csv",
        "metrics_user_macro_seed_summary.csv",
        "paired_hierarchical_bootstrap.csv",
        "quality_sensitivity_user_macro_seed_summary.csv",
        "evaluation_audit.json",
        "catalogue_assignment_audit.json",
    ]
    for name in required_final:
        require((OUTPUT / "final" / name).exists(), f"Missing final output: {name}")
    if prespecified_users:
        for name in [
            "destination_popularity_metrics.csv",
            "prefix_analogue_metrics.csv",
            "stratified_bootstrap.csv",
            "familiarity_analogue_analysis_summary.json",
        ]:
            require((OUTPUT / "analysis" / name).exists(), f"Missing analysis output: {name}")

    evaluation = json.loads(
        (OUTPUT / "final/evaluation_audit.json").read_text(encoding="utf-8")
    )
    require("primary_contrast" in evaluation, "Primary mean-across-ratios contrast missing")
    require(len(evaluation.get("baseline_methods", [])) == 4, "Baseline bootstrap scope incomplete")
    intervals = pd.read_csv(OUTPUT / "final/paired_hierarchical_bootstrap.csv")
    require(
        intervals["quantity"].str.startswith("primary_mean_rate|").sum() == 8,
        "Primary mean rates are incomplete",
    )
    require(
        intervals["quantity"].str.startswith("primary_mean_difference|").sum() == 28,
        "Primary pairwise contrasts are incomplete",
    )

    report = {
        "status": "passed",
        "users": int(manifest["user_id"].nunique()),
        "trips": int(len(manifest)),
        "test_trips": test_trips,
        "cohort_disjoint_from_configured_exclusions": True,
        "checks": checks,
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


def main() -> int:
    return validate(parse_run_context(description=__doc__))


if __name__ == "__main__":
    raise SystemExit(main())
