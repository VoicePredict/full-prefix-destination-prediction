#!/usr/bin/env python3
"""Validate deterministic matched-grid robustness outputs."""

from __future__ import annotations

import json

import pandas as pd

from destination_prediction.context import RunContext, parse_run_context


def validate(context: RunContext) -> int:
    output = context.run_directory / "outputs"
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    held = pd.read_csv(output / "held_out_predictions.csv")
    expanded = pd.read_csv(output / "expanded46_predictions.csv")
    dbscan_membership = pd.read_csv(output / "dbscan_membership_predictions.csv")
    assignment_cases = pd.read_csv(output / "catalogue_assignment_cases.csv")
    assignment_bootstrap = pd.read_csv(output / "catalogue_assignment_bootstrap.csv")
    assignment_changes = pd.read_csv(output / "catalogue_assignment_change_audit.csv")
    audit = pd.read_csv(output / "canonical_point_count_audit.csv")
    robustness = pd.read_csv(output / "matched_robustness_bootstrap.csv")
    cross_cases = pd.read_csv(output / "canonical_cross_method_cases.csv")
    cross = pd.read_csv(output / "canonical_cross_method_bootstrap.csv")
    checks = {
        "run_completed": summary["status"] == "completed",
        "held_users_37": int(summary["held_out_users"]) == 37,
        "held_trajectories_1475": int(summary["held_out_trajectories"]) == 1475,
        "held_rows": len(held) == 3 * 2 * 4 * 1475,
        "expanded_users_46": int(summary["expanded_users"]) == 46,
        "expanded_trajectories_2888": int(summary["expanded_trajectories"]) == 2888,
        "expanded_rows": len(expanded) == 2 * 4 * 2888,
        "dbscan_membership_rows": len(dbscan_membership) == 2 * 4 * 1475,
        "dbscan_membership_users_37": int(dbscan_membership["user_id"].nunique()) == 37,
        "dbscan_membership_trajectories_1475": int(
            dbscan_membership[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ) == 1475,
        "catalogue_assignment_rows_23600": len(assignment_cases) == 2 * 2 * 4 * 1475,
        "catalogue_assignment_rules": set(assignment_cases["assignment_rule"])
        == {"nearest_medoid", "dbscan_membership"},
        "catalogue_assignment_bootstrap_rows": len(assignment_bootstrap) == 32,
        "catalogue_assignment_change_rows": len(assignment_changes) == 8,
        "canonical_audit_exact": bool(audit["exact_region_match"].all())
        and len(audit) == 2 * 4 * 1475,
        "canonical_cross_method_cases": len(cross_cases) == 7 * 4 * 1475
        and int(cross_cases["user_id"].nunique()) == 37
        and int(
            cross_cases[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ) == 1475,
        "robustness_intervals_ordered": bool(
            (robustness["ci_low"] <= robustness["observed"]).all()
            and (robustness["observed"] <= robustness["ci_high"]).all()
        ),
        "cross_intervals_ordered": bool(
            (cross["ci_low"] <= cross["observed"]).all()
            and (cross["observed"] <= cross["ci_high"]).all()
        ),
        "deterministic_rule": bool(summary["deterministic_tie_rule"]),
        "dbscan_membership_summary":
        int(summary["dbscan_membership_sensitivity_users"]) == 37
        and int(summary["dbscan_membership_sensitivity_trajectories"]) == 1475,
        "dbscan_membership_analysis_present": bool(
            (robustness["analysis"] == "held_out_point_count_dbscan_membership").any()
        ),
        "catalogue_assignment_stage_direction_retained": bool(
            (
                assignment_bootstrap[
                    (assignment_bootstrap["assignment_rule"] == "dbscan_membership")
                    & (assignment_bootstrap["quantity"] == "difference_in_change")
                ]["ci_high"]
                < 0
            ).all()
        ),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    report = {"status": "passed", "checks": checks}
    (output / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "COMPLETE").touch()
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    return validate(parse_run_context(description=__doc__))


if __name__ == "__main__":
    raise SystemExit(main())
