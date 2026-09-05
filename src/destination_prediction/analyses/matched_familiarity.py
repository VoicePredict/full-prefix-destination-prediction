#!/usr/bin/env python3
"""Matched full-grid effects stratified by training-prefix familiarity."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext, parse_run_context


def _configure(context: RunContext) -> None:
    global CONFIG, PREDICTION_SOURCE, FAMILIARITY_SOURCE, OUTPUT, FULL, LAST
    CONFIG = context.config
    PREDICTION_SOURCE = context.dependency_outputs(CONFIG["prediction_source"])
    FAMILIARITY_SOURCE = (
        context.dependency_outputs(CONFIG["familiarity_source"]) / "analysis"
    )
    OUTPUT = context.run_directory / "outputs"
    FULL = str(CONFIG["full_method"])
    LAST = str(CONFIG["ablation_method"])


def joined_cases() -> pd.DataFrame:
    predictions = pd.read_csv(
        PREDICTION_SOURCE / "tie_sensitivity_predictions.csv",
        dtype={"trip_id": str},
    )
    predictions = predictions[
        (predictions["variant"] == str(CONFIG["variant"]))
        & predictions["method"].isin([FULL, LAST])
    ].copy()
    wide = predictions.pivot(
        index=["ratio", "user_id", "trip_id"],
        columns="method",
        values="hit_r90_all",
    ).reset_index()
    wide["paired_effect"] = wide[FULL].astype(float) - wide[LAST].astype(float)

    familiarity = pd.read_csv(
        FAMILIARITY_SOURCE / "prefix_analogue_cases.csv",
        dtype={"trip_id": str},
    )
    familiarity = familiarity[
        ["ratio", "user_id", "trip_id", "no_train_analogue_q90"]
    ].copy()
    joined = wide.merge(
        familiarity,
        on=["ratio", "user_id", "trip_id"],
        how="left",
        validate="one_to_one",
    )
    if joined["no_train_analogue_q90"].isna().any():
        raise RuntimeError("Missing training-prefix familiarity labels.")
    joined["prefix_familiarity"] = np.where(
        joined["no_train_analogue_q90"].astype(bool),
        "no_close_analogue",
        "analogue_present",
    )
    return joined


def metrics(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (stratum, ratio), frame in cases.groupby(
        ["prefix_familiarity", "ratio"], sort=True
    ):
        per_user = frame.groupby("user_id").agg(
            full=(FULL, "mean"),
            last=(LAST, "mean"),
            difference=("paired_effect", "mean"),
            cases=("trip_id", "size"),
        )
        rows.append(
            {
                "prefix_familiarity": str(stratum),
                "ratio": float(ratio),
                "full_grid_hit_r90": float(per_user["full"].mean()),
                "last_state_grid_hit_r90": float(per_user["last"].mean()),
                "paired_difference": float(per_user["difference"].mean()),
                "users": int(len(per_user)),
                "cases": int(per_user["cases"].sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap(cases: pd.DataFrame) -> pd.DataFrame:
    replicates = int(CONFIG["bootstrap"]["replicates"])
    seed = int(CONFIG["bootstrap"]["seed"])
    rows: list[dict[str, object]] = []
    grouped = list(cases.groupby(["prefix_familiarity", "ratio"], sort=True))
    for offset, ((stratum, ratio), frame) in enumerate(grouped):
        by_user = {
            int(user): group["paired_effect"].to_numpy(dtype=float)
            for user, group in frame.groupby("user_id", sort=True)
        }
        users = np.asarray(sorted(by_user), dtype=int)
        observed = float(
            np.mean([values.mean() for values in by_user.values()])
        )
        rng = np.random.default_rng(seed + offset)
        estimates = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            sampled_users = rng.choice(users, size=len(users), replace=True)
            user_means = []
            for user in sampled_users:
                values = by_user[int(user)]
                positions = rng.integers(0, len(values), size=len(values))
                user_means.append(float(values[positions].mean()))
            estimates[replicate] = float(np.mean(user_means))
        low, high = np.quantile(estimates, [0.025, 0.975])
        rows.append(
            {
                "prefix_familiarity": str(stratum),
                "ratio": float(ratio),
                "observed_difference": observed,
                "bootstrap_mean": float(estimates.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "users": int(len(users)),
                "cases": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def run(context: RunContext) -> int:
    _configure(context)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = joined_cases()
    cases.to_csv(OUTPUT / "matched_familiarity_cases.csv", index=False)
    metrics(cases).to_csv(OUTPUT / "matched_familiarity_metrics.csv", index=False)
    bootstrap(cases).to_csv(
        OUTPUT / "matched_familiarity_bootstrap.csv", index=False
    )
    summary = {
        "status": "completed",
        "users": int(cases["user_id"].nunique()),
        "trajectories": int(
            cases[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ),
        "ratios": sorted(map(float, cases["ratio"].unique())),
        "familiarity_definition_reused": True,
        "held_out_outcomes_used_to_define_strata": False,
        "interpretation": "post-hoc descriptive diagnostic",
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    return run(parse_run_context(description=__doc__))


if __name__ == "__main__":
    raise SystemExit(main())
