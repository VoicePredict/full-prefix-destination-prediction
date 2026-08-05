#!/usr/bin/env python3
"""Validate the matched familiarity diagnostic."""

from __future__ import annotations

import json

import pandas as pd

from destination_prediction.context import RunContext


EXP_DIR = RunContext.from_environment().run_directory
OUTPUT = EXP_DIR / "outputs"


def main() -> int:
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    cases = pd.read_csv(OUTPUT / "matched_familiarity_cases.csv")
    metrics = pd.read_csv(OUTPUT / "matched_familiarity_metrics.csv")
    bootstrap = pd.read_csv(OUTPUT / "matched_familiarity_bootstrap.csv")
    checks = {
        "run_completed": summary["status"] == "completed",
        "users_37": int(summary["users"]) == 37,
        "trajectories_1475": int(summary["trajectories"]) == 1475,
        "complete_cases": len(cases) == 4 * 1475,
        "two_strata_four_ratios": len(metrics) == 8 and len(bootstrap) == 8,
        "labels_training_only": not bool(
            summary["held_out_outcomes_used_to_define_strata"]
        ),
        "ordered_intervals": bool(
            (bootstrap["ci_low"] <= bootstrap["observed_difference"]).all()
            and (
                bootstrap["observed_difference"] <= bootstrap["ci_high"]
            ).all()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    report = {"status": "passed", "checks": checks}
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "COMPLETE").touch()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
