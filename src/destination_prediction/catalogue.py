"""Training-only personal destination catalogue construction and lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from destination_prediction.geometry import local_xy


def build_catalogue(
    train: pd.DataFrame,
    eps_m: float,
    support_quantiles: list[float],
    min_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for user_id, group in train.groupby("user_id", sort=True):
        endpoints = np.asarray([seq[-1] for seq in group["wgs_seq"]], dtype=float)
        trip_ids = group["trip_id"].astype(str).to_numpy()
        lon0 = float(np.median(endpoints[:, 0]))
        lat0 = float(np.median(endpoints[:, 1]))
        xy = local_xy(endpoints[:, 0], endpoints[:, 1], lon0, lat0)
        pairwise = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
        labels = DBSCAN(eps=float(eps_m), min_samples=int(min_samples)).fit_predict(xy)
        ordered_labels = sorted(
            np.unique(labels), key=lambda label: str(min(trip_ids[labels == label]))
        )
        for center_id, label in enumerate(ordered_labels):
            members = np.flatnonzero(labels == label)
            within = pairwise[np.ix_(members, members)]
            totals = within.sum(axis=1)
            tied = members[np.isclose(totals, totals.min())]
            medoid = int(sorted(tied, key=lambda idx: str(trip_ids[idx]))[0])
            support_distances = pairwise[medoid, members]
            row: dict[str, object] = {
                "catalog_eps_m": float(eps_m),
                "user_id": int(user_id),
                "center_id": int(center_id),
                "medoid_trip_id": str(trip_ids[medoid]),
                "medoid_lon": float(endpoints[medoid, 0]),
                "medoid_lat": float(endpoints[medoid, 1]),
                "lon0": lon0,
                "lat0": lat0,
                "center_x_m": float(xy[medoid, 0]),
                "center_y_m": float(xy[medoid, 1]),
                "support": int(len(members)),
            }
            for quantile in map(float, support_quantiles):
                row[f"radius_q{int(round(quantile * 100))}_m"] = float(
                    np.quantile(support_distances, quantile)
                )
            rows.append(row)
    return pd.DataFrame(rows)


def nearest_catalogue_center(
    user_id: int, lon: float, lat: float, catalogue: pd.DataFrame
) -> tuple[pd.Series, float]:
    centers = catalogue[catalogue["user_id"] == int(user_id)].reset_index(drop=True)
    point = local_xy(
        np.asarray([lon]),
        np.asarray([lat]),
        float(centers.loc[0, "lon0"]),
        float(centers.loc[0, "lat0"]),
    )[0]
    distances = np.hypot(
        centers["center_x_m"].to_numpy(dtype=float) - point[0],
        centers["center_y_m"].to_numpy(dtype=float) - point[1],
    )
    position = int(np.argmin(distances))
    return centers.iloc[position], float(distances[position])


def endpoint_cluster_map(
    train: pd.DataFrame, catalogue: pd.DataFrame
) -> dict[tuple[int, str], int]:
    mapping: dict[tuple[int, str], int] = {}
    for row in train.itertuples(index=False):
        lon, lat = row.wgs_seq[-1]
        center, _ = nearest_catalogue_center(
            int(row.user_id), float(lon), float(lat), catalogue
        )
        mapping[(int(row.user_id), str(row.trip_id))] = int(center.center_id)
    return mapping
