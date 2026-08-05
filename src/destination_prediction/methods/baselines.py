"""Simple personalized destination baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from destination_prediction.catalogue import nearest_catalogue_center
from destination_prediction.geometry import prefix_cutoff


def baseline_predictions(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    catalogue: pd.DataFrame,
    ratios: list[float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_by_user = {
        int(user_id): group.reset_index(drop=True)
        for user_id, group in train.groupby("user_id", sort=True)
    }
    catalogues = {
        int(user_id): group.reset_index(drop=True)
        for user_id, group in catalogue.groupby("user_id", sort=True)
    }
    for ratio in ratios:
        train_current = {
            user_id: np.asarray(
                [seq[prefix_cutoff(len(seq), ratio) - 1] for seq in group["local_seq"]],
                dtype=float,
            )
            for user_id, group in train_by_user.items()
        }
        for test_row in evaluate.itertuples(index=False):
            user_id = int(test_row.user_id)
            cutoff = prefix_cutoff(len(test_row.wgs_seq), ratio)
            current_lon, current_lat = map(float, test_row.wgs_seq[cutoff - 1])
            current_local = np.asarray(test_row.local_seq[cutoff - 1], dtype=float)
            personal = train_by_user[user_id]
            centers = catalogues[user_id]
            frequent = centers.sort_values(
                ["support", "center_id"], ascending=[False, True]
            ).iloc[0]
            nearest, _ = nearest_catalogue_center(
                user_id, current_lon, current_lat, catalogue
            )
            distances = np.linalg.norm(train_current[user_id] - current_local, axis=1)
            analogue = personal.iloc[int(np.argmin(distances))]
            analogue_lon, analogue_lat = analogue["wgs_seq"][-1]
            candidates = [
                (
                    "most_frequent_personal_destination",
                    float(frequent.medoid_lon),
                    float(frequent.medoid_lat),
                ),
                (
                    "nearest_known_destination_to_current_point",
                    float(nearest.medoid_lon),
                    float(nearest.medoid_lat),
                ),
                ("last_observed_point", current_lon, current_lat),
                (
                    "current_position_only_personal_retrieval",
                    float(analogue_lon),
                    float(analogue_lat),
                ),
            ]
            for method, lon, lat in candidates:
                rows.append(
                    {
                        "family": "baseline",
                        "config_id": method,
                        "ratio": float(ratio),
                        "user_id": user_id,
                        "trip_id": str(test_row.trip_id),
                        "native_pred_lon": lon,
                        "native_pred_lat": lat,
                        "query_runtime_ms": 0.0,
                    }
                )
    return pd.DataFrame(rows)
