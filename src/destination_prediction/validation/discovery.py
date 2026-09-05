#!/usr/bin/env python3
"""Fail the discovery run unless protocol and output invariants are satisfied."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from destination_prediction.context import RunContext, parse_run_context


def _configure(context: RunContext) -> None:
    global OUTPUT, CONFIG
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
    preparation = json.loads(
        (OUTPUT / "preparation/preparation_summary.json").read_text(encoding="utf-8")
    )
    require(preparation["source_plt_files"] == 18670, "Unexpected GeoLife PLT file count")
    require(preparation["selected_user_count"] == int(CONFIG["dataset"]["cohort_users"]), "Cohort size mismatch")
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

    for path in [
        OUTPUT / "non_neural/geometric_selection/selected_config.json",
        OUTPUT / "non_neural/grid_pattern_selection/selected_config.json",
        OUTPUT / "recurrent/selected_config.json",
        OUTPUT / "tsmini/selected_config.json",
    ]:
        selected = json.loads(path.read_text(encoding="utf-8"))
        require(selected.get("outer_test_read") is False, f"{path}: outer-test selection flag")

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

    report = {
        "status": "passed",
        "users": int(manifest["user_id"].nunique()),
        "trips": int(len(manifest)),
        "test_trips": test_trips,
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
