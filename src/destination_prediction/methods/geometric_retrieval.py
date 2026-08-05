"""Full-prefix geometric nearest-trajectory retrieval."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from destination_prediction.geometry import prefix_cutoff, resample_path


def geometric_predictions(
    train: pd.DataFrame,
    evaluate: pd.DataFrame,
    ratios: list[float],
    points_grid: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    train_by_user = {
        int(user_id): group.reset_index(drop=True)
        for user_id, group in train.groupby("user_id", sort=True)
    }
    for ratio in ratios:
        for n_points in points_grid:
            prepared = {
                user_id: np.stack(
                    [
                        resample_path(
                            np.asarray(seq)[: prefix_cutoff(len(seq), ratio)], n_points
                        )
                        for seq in group["local_seq"]
                    ]
                )
                for user_id, group in train_by_user.items()
            }
            for test_row in evaluate.itertuples(index=False):
                user_id = int(test_row.user_id)
                start = time.perf_counter()
                query = resample_path(
                    np.asarray(test_row.local_seq)[
                        : prefix_cutoff(len(test_row.local_seq), ratio)
                    ],
                    n_points,
                )
                distances = np.linalg.norm(
                    prepared[user_id] - query[None, :, :], axis=2
                ).mean(axis=1)
                analogue = train_by_user[user_id].iloc[int(np.argmin(distances))]
                lon, lat = analogue["wgs_seq"][-1]
                rows.append(
                    {
                        "family": "geometric_retrieval",
                        "config_id": f"points_{n_points}",
                        "resample_points": int(n_points),
                        "ratio": float(ratio),
                        "user_id": user_id,
                        "trip_id": str(test_row.trip_id),
                        "native_pred_lon": float(lon),
                        "native_pred_lat": float(lat),
                        "analogue_trip_id": str(analogue["trip_id"]),
                        "query_runtime_ms": (time.perf_counter() - start) * 1000.0,
                    }
                )
    return pd.DataFrame(rows)
