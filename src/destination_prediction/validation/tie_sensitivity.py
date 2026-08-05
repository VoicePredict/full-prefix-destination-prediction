#!/usr/bin/env python3
"""Validate the matched-grid tie sensitivity outputs."""

from __future__ import annotations

import json

import pandas as pd

from destination_prediction.context import RunContext


EXP_DIR = RunContext.from_environment().run_directory
OUTPUT = EXP_DIR / "outputs"


def main() -> int:
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(OUTPUT / "tie_sensitivity_predictions.csv")
    ties = pd.read_csv(OUTPUT / "top10_boundary_tie_summary.csv")
    bootstrap = pd.read_csv(OUTPUT / "paired_tie_sensitivity_bootstrap.csv")
    audit = pd.read_csv(OUTPUT / "legacy_reproduction_audit.csv")
    checks = {
        "run_completed": summary["status"] == "completed",
        "users_37": int(summary["users"]) == 37,
        "trajectories_1475": int(summary["test_trajectories"]) == 1475,
        "prediction_rows": len(predictions) == 5 * 2 * 4 * 1475,
        "tie_summary_rows": len(ties) == 8,
        "bootstrap_rows": len(bootstrap) == 5 * 5,
        "legacy_audit_complete": len(audit) == 2 * 4 * 1475,
        "ordered_intervals": bool(
            (bootstrap["ci_low"] <= bootstrap["observed"]).all()
            and (bootstrap["observed"] <= bootstrap["ci_high"]).all()
        ),
        "saved_plan_reused": bool(summary["saved_resample_plan_reused"]),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    report = {
        "status": "passed",
        "checks": checks,
        "legacy_exact_matches": int(audit["exact_region_match"].sum()),
        "legacy_expected": int(len(audit)),
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "COMPLETE").touch()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
