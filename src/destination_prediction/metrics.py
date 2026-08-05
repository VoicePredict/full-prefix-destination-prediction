"""Destination-region, fixed-distance, and user-macro evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from destination_prediction.geometry import haversine_m, local_xy


def evaluate_native_predictions(
    native: pd.DataFrame,
    truth: pd.DataFrame,
    catalogue: pd.DataFrame,
    support_quantiles: list[float],
    fixed_radii_m: list[float],
    point_thresholds_m: list[float],
) -> pd.DataFrame:
    """Map native predictions to the training catalogue and score them.

    A Hit@Rq is awarded only when the true endpoint lies inside the support of
    the *emitted* catalogue region.  If supports overlap, the closest eligible
    medoid defines the endpoint's reference region.  An endpoint outside every
    support is retained and scored as an error.
    """
    truth_by_key = {
        (int(row.user_id), str(row.trip_id)): row for row in truth.itertuples(index=False)
    }
    centers_by_user = {
        int(user_id): group.reset_index(drop=True)
        for user_id, group in catalogue.groupby("user_id", sort=True)
    }
    support_quantiles = list(map(float, support_quantiles))
    fixed_radii_m = list(map(float, fixed_radii_m))

    def cached_nearest(user_id: int, lon: float, lat: float) -> tuple[pd.Series, float]:
        centers = centers_by_user[int(user_id)]
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

    true_assignments = {}
    for key, actual in truth_by_key.items():
        true_lon, true_lat = map(float, actual.wgs_seq[-1])
        nearest, nearest_distance = cached_nearest(key[0], true_lon, true_lat)
        centers = centers_by_user[key[0]]
        point = local_xy(
            np.asarray([true_lon]),
            np.asarray([true_lat]),
            float(centers.loc[0, "lon0"]),
            float(centers.loc[0, "lat0"]),
        )[0]
        distances = np.hypot(
            centers["center_x_m"].to_numpy(dtype=float) - point[0],
            centers["center_y_m"].to_numpy(dtype=float) - point[1],
        )
        support_assignments: dict[int, tuple[bool, int]] = {}
        for quantile in support_quantiles:
            token = int(round(100 * quantile))
            eligible = np.flatnonzero(
                distances <= centers[f"radius_q{token}_m"].to_numpy(dtype=float)
            )
            if len(eligible):
                position = int(eligible[np.argmin(distances[eligible])])
                support_assignments[token] = (True, int(centers.iloc[position].center_id))
            else:
                support_assignments[token] = (False, -1)
        true_assignments[key] = (
            nearest,
            nearest_distance,
            true_lon,
            true_lat,
            support_assignments,
        )

    rows: list[dict[str, object]] = []
    for prediction in native.itertuples(index=False):
        key = (int(prediction.user_id), str(prediction.trip_id))
        (
            true_center,
            true_distance,
            true_lon,
            true_lat,
            support_assignments,
        ) = true_assignments[key]
        pred_lon = float(prediction.native_pred_lon)
        pred_lat = float(prediction.native_pred_lat)
        pred_center, pred_to_center = cached_nearest(key[0], pred_lon, pred_lat)
        projected_lon = float(pred_center.medoid_lon)
        projected_lat = float(pred_center.medoid_lat)
        row = prediction._asdict()
        row.update(
            {
                "true_dest_lon": true_lon,
                "true_dest_lat": true_lat,
                "true_center_id": int(true_center.center_id),
                "true_to_center_m": true_distance,
                "pred_center_id": int(pred_center.center_id),
                "pred_to_center_m": pred_to_center,
                "pred_dest_lon": projected_lon,
                "pred_dest_lat": projected_lat,
                "destination_error_m": float(
                    haversine_m(projected_lon, projected_lat, true_lon, true_lat)
                ),
                "native_destination_error_m": float(
                    haversine_m(pred_lon, pred_lat, true_lon, true_lat)
                ),
            }
        )
        for threshold in point_thresholds_m:
            token = int(round(float(threshold)))
            row[f"within_{token}m"] = bool(row["destination_error_m"] <= float(threshold))
            row[f"native_within_{token}m"] = bool(
                row["native_destination_error_m"] <= float(threshold)
            )
        for quantile in support_quantiles:
            token = int(round(100 * quantile))
            covered, support_center_id = support_assignments[token]
            row[f"covered_r{token}"] = bool(covered)
            row[f"true_center_id_r{token}"] = int(support_center_id)
            row[f"hit_r{token}_all"] = bool(
                covered and int(pred_center.center_id) == int(support_center_id)
            )
        for radius in fixed_radii_m:
            token = int(round(radius))
            covered = true_distance <= radius
            row[f"covered_fixed_{token}m"] = bool(covered)
            row[f"hit_fixed_{token}m_all"] = bool(
                covered and int(pred_center.center_id) == int(true_center.center_id)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def user_macro_mean(
    cases: pd.DataFrame, value_column: str, user_column: str = "user_id"
) -> float:
    """Give each user equal weight after within-user aggregation."""
    return float(cases.groupby(user_column, sort=True)[value_column].mean().mean())


def selection_summary(
    evaluated: pd.DataFrame, config_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_user_ratio = (
        evaluated.groupby([*config_columns, "ratio", "user_id"], as_index=False)
        .agg(
            hit_r90=("hit_r90_all", "mean"),
            within_1000m=("within_1000m", "mean"),
            median_error_m=("destination_error_m", "median"),
            n=("trip_id", "size"),
        )
    )
    leaderboard = (
        per_user_ratio.groupby(config_columns, as_index=False)
        .agg(
            macro_hit_r90=("hit_r90", "mean"),
            macro_within_1000m=("within_1000m", "mean"),
            macro_user_median_error_m=("median_error_m", "mean"),
        )
        .sort_values(
            [
                "macro_hit_r90",
                "macro_within_1000m",
                "macro_user_median_error_m",
                *config_columns,
            ],
            ascending=[False, False, True, *([True] * len(config_columns))],
        )
        .reset_index(drop=True)
    )
    return per_user_ratio, leaderboard
