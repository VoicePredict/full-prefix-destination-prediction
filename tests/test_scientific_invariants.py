from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

from destination_prediction.bootstrap import paired_hierarchical_bootstrap  # noqa: E402
from destination_prediction.catalogue import build_catalogue  # noqa: E402
from destination_prediction.geometry import local_xy, prefix_cutoff  # noqa: E402
from destination_prediction.metrics import evaluate_native_predictions, user_macro_mean  # noqa: E402
from destination_prediction.methods.grid_pattern_retrieval import (  # noqa: E402
    aligned_squared_distance,
    deterministic_descending_order,
)
from destination_prediction.preprocessing import even_indices  # noqa: E402


def trajectory(user: int, trip: str, endpoint: tuple[float, float]) -> dict:
    return {"user_id": user, "trip_id": trip, "wgs_seq": [[0.0, 0.0], list(endpoint)]}


class ScientificInvariantTests(unittest.TestCase):
    def test_prefix_cutoff_uses_ties_to_even_and_two_point_minimum(self) -> None:
        self.assertEqual(prefix_cutoff(11, 0.25), 3)
        self.assertEqual(prefix_cutoff(11, 0.75), 9)
        self.assertEqual(prefix_cutoff(3, 0.0), 2)

    def test_downsampling_preserves_both_endpoints(self) -> None:
        indices = even_indices(1001, 200)
        self.assertEqual(len(indices), 200)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 1000)

    def test_catalogue_is_constructed_only_from_passed_training_rows(self) -> None:
        train = pd.DataFrame(
            [trajectory(1, "train-a", (0.0, 0.0)), trajectory(1, "train-b", (0.01, 0.0))]
        )
        catalogue = build_catalogue(train, 10.0, [0.9], 1)
        self.assertEqual(set(catalogue["medoid_trip_id"]), {"train-a", "train-b"})
        self.assertNotIn("test", set(catalogue["medoid_trip_id"]))

    def test_singleton_region_has_zero_empirical_radius(self) -> None:
        train = pd.DataFrame([trajectory(1, "only", (0.0, 0.0))])
        catalogue = build_catalogue(train, 200.0, [0.9], 1)
        self.assertEqual(float(catalogue.iloc[0]["radius_q90_m"]), 0.0)

    def test_overlapping_supports_use_closest_eligible_medoid(self) -> None:
        medoids = np.asarray([[0.0, 0.0], [0.001, 0.0]])
        xy = local_xy(medoids[:, 0], medoids[:, 1], 0.0, 0.0)
        catalogue = pd.DataFrame(
            {
                "user_id": [1, 1],
                "center_id": [0, 1],
                "medoid_lon": medoids[:, 0],
                "medoid_lat": medoids[:, 1],
                "lon0": [0.0, 0.0],
                "lat0": [0.0, 0.0],
                "center_x_m": xy[:, 0],
                "center_y_m": xy[:, 1],
                "radius_q90_m": [100.0, 100.0],
            }
        )
        truth = pd.DataFrame([trajectory(1, "q", (0.0004, 0.0))])
        native = pd.DataFrame(
            [
                {"user_id": 1, "trip_id": "q", "native_pred_lon": 0.0, "native_pred_lat": 0.0},
                {"user_id": 1, "trip_id": "q", "native_pred_lon": 0.001, "native_pred_lat": 0.0},
            ]
        )
        scored = evaluate_native_predictions(native, truth, catalogue, [0.9], [200.0], [200.0])
        self.assertEqual(scored["hit_r90_all"].tolist(), [True, False])

    def test_unsupported_endpoint_is_retained_as_error(self) -> None:
        catalogue = pd.DataFrame(
            {
                "user_id": [1], "center_id": [0], "medoid_lon": [0.0], "medoid_lat": [0.0],
                "lon0": [0.0], "lat0": [0.0], "center_x_m": [0.0], "center_y_m": [0.0],
                "radius_q90_m": [0.0],
            }
        )
        truth = pd.DataFrame([trajectory(1, "q", (0.01, 0.0))])
        native = pd.DataFrame(
            [{"user_id": 1, "trip_id": "q", "native_pred_lon": 0.0, "native_pred_lat": 0.0}]
        )
        scored = evaluate_native_predictions(native, truth, catalogue, [0.9], [200.0], [200.0])
        self.assertFalse(bool(scored.iloc[0]["covered_r90"]))
        self.assertFalse(bool(scored.iloc[0]["hit_r90_all"]))

    def test_equal_scores_have_deterministic_training_order(self) -> None:
        self.assertEqual(deterministic_descending_order(np.ones(4)).tolist(), [0, 1, 2, 3])

    def test_user_macro_gives_users_equal_weight(self) -> None:
        cases = pd.DataFrame({"user_id": [1, 2, 2, 2], "hit": [1.0, 0.0, 0.0, 0.0]})
        self.assertEqual(user_macro_mean(cases, "hit"), 0.5)

    def test_paired_bootstrap_reuses_rows_across_methods(self) -> None:
        cases = pd.DataFrame(
            {"user_id": [1, 1, 2, 2], "full": [1.0, 2.0, 4.0, 5.0], "last": [0.0, 1.0, 3.0, 4.0]}
        )
        estimates = paired_hierarchical_bootstrap(cases, ["full", "last"], 50, 7)
        np.testing.assert_allclose(estimates[:, 0] - estimates[:, 1], 1.0)

    def test_matched_ablation_changes_only_included_aligned_states(self) -> None:
        patterns = np.asarray([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [1.0, 1.0]]])
        query = np.asarray([[0.0, 0.0], [0.0, 0.0]])
        full = aligned_squared_distance(patterns, query, False)
        last = aligned_squared_distance(patterns, query, True)
        np.testing.assert_allclose(full, [1.0, 5.0])
        np.testing.assert_allclose(last, [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
