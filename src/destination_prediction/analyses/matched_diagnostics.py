#!/usr/bin/env python3
"""Recompute matched diagnostics with deterministic equal-distance ordering."""

from __future__ import annotations

import json

import pandas as pd

from destination_prediction.analyses import diagnostic_statistics as statistics
from destination_prediction.context import RunContext


CONTEXT = RunContext.from_environment()
EXP_DIR = CONTEXT.run_directory
CONFIG = CONTEXT.config
PREDICTIONS = CONTEXT.dependency_outputs(CONFIG["prediction_source"])
DIAGNOSTIC = CONTEXT.dependency_outputs(CONFIG["diagnostic_source"])
OUTPUT = EXP_DIR / "outputs"
FULL = str(CONFIG["full_method"])
LAST = str(CONFIG["ablation_method"])

def paired_cases() -> pd.DataFrame:
    cases = pd.read_csv(
        PREDICTIONS / "tie_sensitivity_predictions.csv",
        dtype={"trip_id": str},
    )
    cases = cases[
        (cases["variant"] == str(CONFIG["variant"]))
        & cases["method"].isin([FULL, LAST])
    ].copy()
    wide = cases.pivot(
        index=["user_id", "trip_id", "ratio"],
        columns="method",
        values="hit_r90_all",
    ).reset_index()
    wide["paired_effect"] = wide[FULL].astype(float) - wide[LAST].astype(float)
    return wide


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    wide = paired_cases()
    effects = statistics.user_effects(wide, FULL, LAST)
    effects.to_csv(OUTPUT / "matched_user_effects.csv", index=False)

    ambiguity = pd.read_csv(
        DIAGNOSTIC / "cases/held_out_location_ambiguity_binned.csv"
    )
    ambiguity = ambiguity[ambiguity["requested_k"] == 10].copy()
    ambiguity_metrics = statistics.stratum_metrics(
        wide, ambiguity, "ambiguity_stratum", "ambiguity_k10", FULL, LAST
    )
    ambiguity_metrics.to_csv(
        OUTPUT / "matched_ambiguity_metrics.csv", index=False
    )
    bootstrap = CONFIG["bootstrap"]
    ambiguity_bootstrap = statistics.stratum_bootstrap(
        wide,
        ambiguity,
        "ambiguity_stratum",
        "ambiguity_k10",
        int(bootstrap["replicates"]),
        int(bootstrap["seed"]),
    )
    ambiguity_bootstrap.to_csv(
        OUTPUT / "matched_ambiguity_bootstrap.csv", index=False
    )

    profiles = pd.read_csv(
        DIAGNOSTIC / "cases/held_out_user_profiles.csv"
    )
    history_metrics = statistics.stratum_metrics(
        wide,
        profiles[["user_id", "history_size_stratum"]],
        "history_size_stratum",
        "history_size",
        FULL,
        LAST,
    )
    history_metrics.to_csv(
        OUTPUT / "matched_history_size_metrics.csv", index=False
    )
    correlations = statistics.correlations(
        effects, profiles, int(bootstrap["replicates"]), int(bootstrap["seed"])
    )
    correlations.to_csv(
        OUTPUT / "matched_user_feature_correlations.csv", index=False
    )
    summary = {
        "status": "completed",
        "users": int(effects["user_id"].nunique()),
        "ratios": sorted(map(float, effects["ratio"].unique())),
        "variant": str(CONFIG["variant"]),
        "ambiguity_rows": int(len(ambiguity_metrics)),
        "history_rows": int(len(history_metrics)),
        "correlations": int(len(correlations)),
        "interpretation": str(CONFIG["interpretation"]),
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
