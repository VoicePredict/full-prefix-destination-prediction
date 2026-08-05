#!/usr/bin/env python3
"""Validate the EDBT ambiguity and heterogeneity analysis outputs."""

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
        (OUTPUT / "analysis_summary.json").read_text(encoding="utf-8")
    )
    ambiguity = pd.read_csv(
        OUTPUT / "cases/held_out_location_ambiguity_binned.csv"
    )
    profiles = pd.read_csv(OUTPUT / "cases/held_out_user_profiles.csv")
    effects = pd.read_csv(OUTPUT / "tables/user_heterogeneity.csv")
    boot = pd.read_csv(OUTPUT / "tables/ambiguity_focal_bootstrap.csv")
    expected_cases = 1475 * len(config["observation_ratios"]) * (
        1 + len(config["ambiguity"]["sensitivity_neighbors"])
    )
    require(len(ambiguity) == expected_cases, "unexpected ambiguity row count")
    require(ambiguity["user_id"].nunique() == 37, "expected 37 held-out users")
    require(len(profiles) == 37, "expected one profile per held-out user")
    require(len(effects) == 37, "expected one mean effect per held-out user")
    require(
        set(ambiguity["requested_k"])
        == {
            config["ambiguity"]["primary_neighbors"],
            *config["ambiguity"]["sensitivity_neighbors"],
        },
        "neighbour sensitivity values differ from config",
    )
    require(
        set(ambiguity["ambiguity_stratum"]).issubset(
            {"low", "medium", "high"}
        ),
        "unexpected ambiguity stratum",
    )
    require(
        np.isfinite(
            ambiguity[
                [
                    "entropy_nats",
                    "effective_destinations",
                    "top1_share",
                    "top1_top2_margin",
                ]
            ].to_numpy(dtype=float)
        ).all(),
        "non-finite ambiguity values",
    )
    require((ambiguity["entropy_nats"] >= 0).all(), "negative entropy")
    require(
        ((ambiguity["top1_share"] >= 0) & (ambiguity["top1_share"] <= 1)).all(),
        "invalid top1 shares",
    )
    require(len(boot) > 0, "missing ambiguity bootstrap results")
    require(
        summary["test_outcomes_used_to_define_ambiguity"] is False,
        "ambiguity must use training labels only",
    )
    require(
        summary["held_out_outcomes_used_to_define_stratum_thresholds"] is False,
        "held-out outcomes must not define strata",
    )
    report = {
        "status": "passed",
        "ambiguity_rows": int(len(ambiguity)),
        "held_out_users": int(len(profiles)),
        "user_effect_rows": int(len(effects)),
        "bootstrap_rows": int(len(boot)),
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
