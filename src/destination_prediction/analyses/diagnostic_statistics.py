"""Reusable statistics for matched post-hoc diagnostic analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def user_effects(
    wide: pd.DataFrame, full_method: str, last_method: str
) -> pd.DataFrame:
    return (
        wide.groupby(["user_id", "ratio"], as_index=False)
        .agg(
            full_grid_hit_r90=(full_method, "mean"),
            last_state_grid_hit_r90=(last_method, "mean"),
            paired_effect=("paired_effect", "mean"),
            cases=("trip_id", "size"),
        )
        .sort_values(["user_id", "ratio"])
        .reset_index(drop=True)
    )


def _join_labels(
    wide: pd.DataFrame, labels: pd.DataFrame, label_column: str
) -> pd.DataFrame:
    labels = labels.copy()
    keys = ["user_id"]
    if "trip_id" in labels.columns:
        labels["trip_id"] = labels["trip_id"].astype(str)
        keys.append("trip_id")
    if "ratio" in labels.columns:
        keys.append("ratio")
    joined = wide.merge(
        labels[keys + [label_column]], on=keys, how="left", validate="many_to_one"
    )
    if joined[label_column].isna().any():
        raise RuntimeError(f"Missing {label_column} labels")
    return joined


def stratum_metrics(
    wide: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str,
    analysis: str,
    full_method: str,
    last_method: str,
) -> pd.DataFrame:
    joined = _join_labels(wide, labels, label_column)
    rows: list[dict[str, object]] = []
    for (stratum, ratio), frame in joined.groupby([label_column, "ratio"], sort=True):
        per_user = frame.groupby("user_id").agg(
            full=(full_method, "mean"),
            ablation=(last_method, "mean"),
            difference=("paired_effect", "mean"),
            cases=("trip_id", "size"),
        )
        rows.append(
            {
                "analysis": analysis,
                "stratum": str(stratum),
                "ratio": float(ratio),
                "full_grid_hit_r90": float(per_user["full"].mean()),
                "last_state_grid_hit_r90": float(per_user["ablation"].mean()),
                "paired_difference": float(per_user["difference"].mean()),
                "users": int(len(per_user)),
                "cases": int(per_user["cases"].sum()),
            }
        )
    return pd.DataFrame(rows)


def stratum_bootstrap(
    wide: pd.DataFrame,
    labels: pd.DataFrame,
    label_column: str,
    analysis: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    joined = _join_labels(wide, labels, label_column)
    rows: list[dict[str, object]] = []
    for offset, ((stratum, ratio), frame) in enumerate(
        joined.groupby([label_column, "ratio"], sort=True)
    ):
        by_user = {
            int(user): group["paired_effect"].to_numpy(dtype=float)
            for user, group in frame.groupby("user_id", sort=True)
        }
        users = np.asarray(sorted(by_user), dtype=int)
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
        per_user = frame.groupby("user_id")["paired_effect"].mean()
        low, high = np.quantile(estimates, [0.025, 0.975])
        rows.append(
            {
                "analysis": analysis,
                "stratum": str(stratum),
                "ratio": float(ratio),
                "observed_difference": float(per_user.mean()),
                "bootstrap_mean": float(estimates.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "users": int(len(users)),
                "cases": int(len(frame)),
            }
        )
    return pd.DataFrame(rows)


def correlations(
    effects: pd.DataFrame,
    profiles: pd.DataFrame,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    overall = effects.groupby("user_id", as_index=False)["paired_effect"].mean()
    joined = profiles.merge(overall, on="user_id", how="inner", validate="one_to_one")
    features = [
        "training_trajectories",
        "destination_regions",
        "destination_entropy_nats",
        "no_prefix_analogue_q90_share",
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for feature in features:
        x = joined[feature].to_numpy(dtype=float)
        y = joined["paired_effect"].to_numpy(dtype=float)
        observed = float(spearmanr(x, y).statistic)
        estimates = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            positions = rng.integers(0, len(joined), size=len(joined))
            estimates[replicate] = float(spearmanr(x[positions], y[positions]).statistic)
        estimates = estimates[np.isfinite(estimates)]
        low, high = np.quantile(estimates, [0.025, 0.975])
        rows.append(
            {
                "feature": feature,
                "spearman_rho": observed,
                "bootstrap_mean": float(estimates.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "users": int(len(joined)),
                "valid_bootstrap_replicates": int(len(estimates)),
            }
        )
    return pd.DataFrame(rows)
