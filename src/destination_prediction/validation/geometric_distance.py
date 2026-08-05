#!/usr/bin/env python3
"""Validate geometric-distance robustness outputs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext


CONTEXT = RunContext.from_environment()
EXP_DIR = CONTEXT.run_directory
OUTPUT = EXP_DIR / "outputs"
CONFIG = CONTEXT.config


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    config = load_config()
    summary = json.loads(
        (OUTPUT / "run_summary.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(OUTPUT / "geometric_predictions.csv")
    audit = pd.read_csv(OUTPUT / "pointwise_reproduction_audit.csv")
    bootstrap = pd.read_csv(OUTPUT / "paired_bootstrap.csv")
    expected = (
        1475
        * len(config["observation_ratios"])
        * len(config["distance_variants"])
    )
    require(len(predictions) == expected, "unexpected prediction row count")
    require(predictions["user_id"].nunique() == 37, "expected 37 users")
    require(
        set(predictions["method"]) == set(config["distance_variants"]),
        "distance variants differ from config",
    )
    require(
        predictions.groupby(["method", "ratio"]).size().eq(1475).all(),
        "each method-ratio must cover every test trajectory",
    )
    require(audit["analogue_match"].astype(bool).all(), "pointwise rerun mismatch")
    require(
        np.isfinite(
            predictions[
                ["destination_error_m", "query_runtime_ms", "nearest_distance"]
            ].to_numpy(dtype=float)
        ).all(),
        "non-finite values",
    )
    require(
        set(bootstrap["quantity"])
        == {"ratio", "mean_across_ratios", "difference_in_change"},
        "missing bootstrap quantities",
    )
    require(
        summary["test_outcomes_used_for_configuration"] is False,
        "test outcomes cannot configure distances",
    )
    report = {
        "status": "passed",
        "prediction_rows": int(len(predictions)),
        "pointwise_reproduction_cases": int(len(audit)),
        "bootstrap_rows": int(len(bootstrap)),
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
