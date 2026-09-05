#!/usr/bin/env python3
"""Validate deterministic matched-grid diagnostic outputs."""

from __future__ import annotations

import json

import pandas as pd

from destination_prediction.context import RunContext, parse_run_context


def validate(context: RunContext) -> int:
    output = context.run_directory / "outputs"
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    effects = pd.read_csv(output / "matched_user_effects.csv")
    ambiguity = pd.read_csv(output / "matched_ambiguity_bootstrap.csv")
    history = pd.read_csv(output / "matched_history_size_metrics.csv")
    correlations = pd.read_csv(output / "matched_user_feature_correlations.csv")
    checks = {
        "run_completed": summary["status"] == "completed",
        "users_37": int(summary["users"]) == 37,
        "effect_rows": len(effects) == 37 * 4,
        "ambiguity_rows": len(ambiguity) == 12,
        "history_rows": len(history) == 12,
        "correlation_rows": len(correlations) == 4,
        "deterministic_variant": summary["variant"] == "id_top10",
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
