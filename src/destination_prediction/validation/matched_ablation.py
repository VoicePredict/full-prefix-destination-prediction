#!/usr/bin/env python3
"""Validate matched-grid ablation outputs and bootstrap integrity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext


EXP_DIR = RunContext.from_environment().run_directory
OUTPUT = EXP_DIR / "outputs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    required = [
        "matched_grid_predictions.csv",
        "training_destination_catalog.csv",
        "full_grid_reproduction_audit.csv",
        "comparison_cases.csv",
        "method_metrics.csv",
        "paired_bootstrap.csv",
        "bootstrap_resample_ids.npz",
        "bootstrap_case_index.csv",
        "run_summary.json",
        "code_and_config_checksums.sha256",
    ]
    for name in required:
        require((OUTPUT / name).is_file(), f"Missing output: {name}")

    predictions = pd.read_csv(OUTPUT / "matched_grid_predictions.csv")
    cases = pd.read_csv(OUTPUT / "comparison_cases.csv")
    audit = pd.read_csv(OUTPUT / "full_grid_reproduction_audit.csv")
    bootstrap = pd.read_csv(OUTPUT / "paired_bootstrap.csv")
    case_index = pd.read_csv(OUTPUT / "bootstrap_case_index.csv")
    plan = np.load(OUTPUT / "bootstrap_resample_ids.npz")

    require(len(predictions) == 11800, "Expected two methods x 5,900 cases.")
    require(predictions["user_id"].nunique() == 37, "Expected 37 users.")
    require(predictions["ratio"].nunique() == 4, "Expected four ratios.")
    require(
        predictions.groupby("method").size().eq(5900).all(),
        "Matched methods must have 5,900 predictions each.",
    )
    require(len(audit) == 5900, "Full-grid audit must contain 5,900 cases.")
    require(
        audit["exact_region_match"].astype(bool).all(),
        "Full-grid predictions do not exactly reproduce the source run.",
    )
    require(len(cases) == 23600, "Expected four methods x 5,900 cases.")
    require(
        cases.groupby("method").size().eq(5900).all(),
        "Every comparison method must have 5,900 cases.",
    )
    require(
        not cases.duplicated(["method", "ratio", "user_id", "trip_id"]).any(),
        "Duplicate method-case rows found.",
    )
    require(len(bootstrap) == 32, "Unexpected bootstrap quantity count.")
    require(
        np.isfinite(
            bootstrap[
                ["observed", "bootstrap_mean", "ci_low", "ci_high"]
            ].to_numpy(dtype=float)
        ).all(),
        "Non-finite bootstrap output.",
    )
    require(
        (bootstrap["ci_low"] <= bootstrap["ci_high"]).all(),
        "Invalid bootstrap interval.",
    )

    user_draws = plan["user_draw_positions"]
    within_draws = plan["within_user_draw_positions"]
    offsets = plan["within_user_draw_offsets"]
    lengths = plan["trajectories_per_user"]
    require(user_draws.shape == (10000, 37), "Invalid user-draw shape.")
    require(len(lengths) == 37, "Invalid per-user length vector.")
    require(len(offsets) == 10000 * 37 + 1, "Invalid draw offsets.")
    require(int(offsets[-1]) == len(within_draws), "Draw offsets do not close.")
    require(len(case_index) == 1475, "Case index must contain 1,475 trajectories.")
    for flat_slot in np.linspace(
        0, user_draws.size - 1, num=1000, dtype=int
    ):
        replicate, slot = divmod(int(flat_slot), 37)
        user_position = int(user_draws[replicate, slot])
        start, end = int(offsets[flat_slot]), int(offsets[flat_slot + 1])
        require(end - start == int(lengths[user_position]), "Wrong inner draw length.")
        require(
            int(within_draws[start:end].max(initial=0))
            < int(lengths[user_position]),
            "Inner draw index out of range.",
        )

    focal = bootstrap[
        bootstrap["quantity"].str.startswith(
            "difference|grid_full_prefix|grid_last_state_only|"
        )
    ]
    require(len(focal) == 4, "Missing matched per-ratio contrasts.")
    require(
        (focal["observed"].abs() > 0).any(),
        "Matched ablation is unexpectedly identical at every ratio.",
    )
    report = {
        "status": "passed",
        "matched_prediction_rows": int(len(predictions)),
        "comparison_case_rows": int(len(cases)),
        "users": int(cases["user_id"].nunique()),
        "trajectories": int(case_index.shape[0]),
        "ratios": sorted(map(float, cases["ratio"].unique())),
        "full_grid_exact_matches": int(
            audit["exact_region_match"].astype(bool).sum()
        ),
        "bootstrap_rows": int(len(bootstrap)),
        "bootstrap_replicates": int(user_draws.shape[0]),
        "saved_within_user_draw_ids": int(len(within_draws)),
    }
    (OUTPUT / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
