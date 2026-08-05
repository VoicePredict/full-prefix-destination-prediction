"""Paired hierarchical bootstrap primitives used by paper analyses."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def paired_hierarchical_bootstrap(
    paired_cases: pd.DataFrame,
    value_columns: Sequence[str],
    replicates: int,
    seed: int,
    user_column: str = "user_id",
) -> np.ndarray:
    """Resample users, then paired cases within each sampled user.

    The same sampled rows are used for all ``value_columns``; vector-valued
    observations such as the four observation ratios therefore stay paired.
    Each sampled user contributes equally to a replicate estimate.
    """
    users = sorted(paired_cases[user_column].unique())
    groups = {
        user: paired_cases.loc[paired_cases[user_column] == user, value_columns]
        .to_numpy(dtype=float)
        for user in users
    }
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, len(value_columns)), dtype=float)
    for replicate in range(replicates):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        user_estimates = []
        for user in sampled_users:
            values = groups[user]
            positions = rng.integers(0, len(values), size=len(values))
            user_estimates.append(values[positions].mean(axis=0))
        output[replicate] = np.mean(user_estimates, axis=0)
    return output


def percentile_interval(
    estimates: np.ndarray, confidence_level: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    alpha = (1.0 - confidence_level) / 2.0
    return (
        np.quantile(estimates, alpha, axis=0),
        np.quantile(estimates, 1.0 - alpha, axis=0),
    )
