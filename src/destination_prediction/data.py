"""Prepared trajectory loading and chronological partition operations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from destination_prediction.geometry import local_xy


def load_marked(run_directory: Path) -> pd.DataFrame:
    preparation = run_directory / "outputs" / "preparation"
    trips = pd.read_pickle(preparation / "geolife_source_faithful_trips.pkl")
    manifest = pd.read_csv(
        preparation / "partition_manifest.csv",
        usecols=["user_id", "trip_id", "partition"],
    )
    trips["trip_id"] = trips["trip_id"].astype(str)
    manifest["trip_id"] = manifest["trip_id"].astype(str)
    marked = trips.merge(manifest, on=["user_id", "trip_id"], validate="one_to_one")
    return marked.sort_values(["user_id", "start_datetime", "trip_id"]).reset_index(drop=True)


def partitions(marked: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if stage == "inner":
        train_mask = marked["partition"] == "fit"
        eval_mask = marked["partition"] == "validation"
    elif stage == "outer":
        train_mask = marked["partition"].isin(["fit", "validation"])
        eval_mask = marked["partition"] == "test"
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return (
        marked.loc[train_mask].reset_index(drop=True),
        marked.loc[eval_mask].reset_index(drop=True),
    )


def user_origins(train: pd.DataFrame) -> dict[int, tuple[float, float]]:
    origins: dict[int, tuple[float, float]] = {}
    for user_id, group in train.groupby("user_id", sort=True):
        points = np.asarray([point for seq in group["wgs_seq"] for point in seq], dtype=float)
        origins[int(user_id)] = (float(np.median(points[:, 0])), float(np.median(points[:, 1])))
    return origins


def add_local_sequences(
    frame: pd.DataFrame, origins: dict[int, tuple[float, float]]
) -> pd.DataFrame:
    out = frame.copy()
    local_sequences = []
    for row in out.itertuples(index=False):
        lon0, lat0 = origins[int(row.user_id)]
        seq = np.asarray(row.wgs_seq, dtype=float)
        local_sequences.append(local_xy(seq[:, 0], seq[:, 1], lon0, lat0).tolist())
    out["local_seq"] = local_sequences
    return out
