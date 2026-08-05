#!/usr/bin/env python3
"""Validate deterministic matched-grid robustness outputs."""

from __future__ import annotations

import json

import pandas as pd

from destination_prediction.context import RunContext


EXP_DIR = RunContext.from_environment().run_directory
OUTPUT = EXP_DIR / "outputs"


def main() -> int:
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    held = pd.read_csv(OUTPUT / "held_out_predictions.csv")
    expanded = pd.read_csv(OUTPUT / "expanded46_predictions.csv")
    audit = pd.read_csv(OUTPUT / "canonical_point_count_audit.csv")
    robustness = pd.read_csv(OUTPUT / "matched_robustness_bootstrap.csv")
    cross = pd.read_csv(OUTPUT / "canonical_cross_method_bootstrap.csv")
    checks = {
        "run_completed": summary["status"] == "completed",
        "held_users_37": int(summary["held_out_users"]) == 37,
        "held_trajectories_1475": int(summary["held_out_trajectories"]) == 1475,
        "held_rows": len(held) == 3 * 2 * 4 * 1475,
        "expanded_users_46": int(summary["expanded_users"]) == 46,
        "expanded_trajectories_2888": int(summary["expanded_trajectories"]) == 2888,
        "expanded_rows": len(expanded) == 2 * 4 * 2888,
        "canonical_audit_exact": bool(audit["exact_region_match"].all())
        and len(audit) == 2 * 4 * 1475,
        "robustness_intervals_ordered": bool(
            (robustness["ci_low"] <= robustness["observed"]).all()
            and (robustness["observed"] <= robustness["ci_high"]).all()
        ),
        "cross_intervals_ordered": bool(
            (cross["ci_low"] <= cross["observed"]).all()
            and (cross["observed"] <= cross["ci_high"]).all()
        ),
        "deterministic_rule": bool(summary["deterministic_tie_rule"]),
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
