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
from destination_prediction.catalogue import (  # noqa: E402
    build_catalogue,
    dbscan_component_assignment,
    endpoint_catalogue_assignment,
)
from destination_prediction.geometry import local_xy, prefix_cutoff  # noqa: E402
from destination_prediction.metrics import evaluate_native_predictions, user_macro_mean  # noqa: E402
from destination_prediction.methods.grid_pattern_retrieval import (  # noqa: E402
    aligned_squared_distance,
    candidate_order,
    deterministic_descending_order,
    emit_destination_region,
    gaussian_posterior,
    selected_candidate_indices,
)
from destination_prediction.methods.baselines import baseline_predictions  # noqa: E402
from destination_prediction.preprocessing import (  # noqa: E402
    even_indices,
    labelled_modes,
    overlapping_label_modes,
)


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

    def test_non_singleton_radius_uses_linear_empirical_quantile(self) -> None:
        endpoints = [(0.0, 0.0), (0.0001, 0.0), (0.0002, 0.0), (0.0004, 0.0)]
        train = pd.DataFrame(
            [trajectory(1, f"t{index}", endpoint) for index, endpoint in enumerate(endpoints)]
        )
        catalogue = build_catalogue(train, 30.0, [0.9], 1)
        self.assertEqual(len(catalogue), 1)
        center = catalogue.iloc[0]
        coordinates = np.asarray(endpoints)
        xy = local_xy(coordinates[:, 0], coordinates[:, 1], center.lon0, center.lat0)
        medoid = local_xy(
            np.asarray([center.medoid_lon]),
            np.asarray([center.medoid_lat]),
            center.lon0,
            center.lat0,
        )[0]
        expected = np.quantile(np.linalg.norm(xy - medoid, axis=1), 0.9, method="linear")
        self.assertAlmostEqual(float(center.radius_q90_m), float(expected), places=10)

    def test_training_endpoint_uses_frozen_nearest_medoid_assignment(self) -> None:
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
            }
        )
        train = pd.DataFrame([trajectory(1, "near-second", (0.0008, 0.0))])
        assignment = endpoint_catalogue_assignment(train, catalogue)
        self.assertEqual(assignment[(1, "near-second")], 1)

    def test_dbscan_membership_and_downstream_projection_are_distinct(self) -> None:
        endpoints = [0.0, 0.0008, 0.0016, 0.0024, 0.0032, 0.0042]
        train = pd.DataFrame(
            [trajectory(1, f"t{index}", (lon, 0.0)) for index, lon in enumerate(endpoints)]
        )
        catalogue = build_catalogue(train, 100.0, [0.9], 1)
        component = dbscan_component_assignment(train, 100.0, 1)
        projected = endpoint_catalogue_assignment(train, catalogue)
        self.assertEqual(component[(1, "t4")], 0)
        self.assertEqual(projected[(1, "t4")], 1)
        self.assertEqual(catalogue.set_index("center_id").loc[0, "support"], 5)

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

    def test_trajectory_id_tie_rule_is_invariant_to_training_row_order(self) -> None:
        distances = np.asarray([1.0, 0.0, 1.0, 1.0])
        posterior = np.asarray([0.2, 0.4, 0.2, 0.2])
        trip_ids = np.asarray(["trip-z", "trip-m", "trip-a", "trip-b"])
        expected = ["trip-m", "trip-a", "trip-b", "trip-z"]
        first = candidate_order(distances, posterior, trip_ids, "trajectory_id")
        self.assertEqual(trip_ids[first].tolist(), expected)

        shuffled = np.asarray([3, 0, 2, 1])
        second = candidate_order(
            distances[shuffled],
            posterior[shuffled],
            trip_ids[shuffled],
            "trajectory_id",
        )
        self.assertEqual(trip_ids[shuffled][second].tolist(), expected)

    def test_boundary_tie_rule_includes_every_candidate_at_kth_distance(self) -> None:
        squared = np.asarray([0.0, 1.0, 1.0, 1.0, 2.0])
        posterior = gaussian_posterior(squared, 2.5)
        selected = selected_candidate_indices(
            squared,
            posterior,
            np.asarray(["a", "d", "b", "c", "e"]),
            2,
            "include_boundary",
        )
        self.assertEqual(selected.tolist(), [0, 2, 3, 1])

    def test_equal_destination_mass_uses_catalogue_order(self) -> None:
        emitted, mass, pool_size, tied = emit_destination_region(
            np.asarray([0.0, 0.0]),
            np.asarray(["later", "earlier"]),
            np.asarray([1, 0]),
            2.5,
            2,
            "trajectory_id",
        )
        self.assertEqual(emitted, 0)
        self.assertEqual(pool_size, 2)
        self.assertTrue(tied)
        self.assertAlmostEqual(mass, 0.5)

    def test_most_frequent_baseline_tie_uses_catalogue_order(self) -> None:
        train = pd.DataFrame(
            [
                {
                    **trajectory(1, "a", (0.0, 0.0)),
                    "local_seq": [[0.0, 0.0], [0.0, 0.0]],
                }
            ]
        )
        evaluate = pd.DataFrame(
            [
                {
                    **trajectory(1, "q", (0.0, 0.0)),
                    "local_seq": [[0.0, 0.0], [0.0, 0.0]],
                }
            ]
        )
        catalogue = pd.DataFrame(
            {
                "user_id": [1, 1],
                "center_id": [1, 0],
                "support": [2, 2],
                "medoid_lon": [0.001, 0.0],
                "medoid_lat": [0.0, 0.0],
                "lon0": [0.0, 0.0],
                "lat0": [0.0, 0.0],
                "center_x_m": [111.195, 0.0],
                "center_y_m": [0.0, 0.0],
            }
        )
        predictions = baseline_predictions(train, evaluate, catalogue, [0.5])
        frequent = predictions[
            predictions["config_id"] == "most_frequent_personal_destination"
        ].iloc[0]
        self.assertEqual(float(frequent.native_pred_lon), 0.0)

    def test_non_ground_interval_between_sparse_points_is_detected(self) -> None:
        times = pd.Series(
            pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:10:00"])
        )
        labels = pd.DataFrame(
            {
                "start": pd.to_datetime(["2020-01-01 00:04:00"]),
                "end": pd.to_datetime(["2020-01-01 00:06:00"]),
                "mode": ["airplane"],
            }
        )
        point_modes, _ = labelled_modes(times, labels)
        self.assertEqual(point_modes, ["unknown", "unknown"])
        self.assertEqual(
            overlapping_label_modes(times.iloc[0], times.iloc[-1], labels),
            ["airplane"],
        )

        touching_boundaries = pd.DataFrame(
            {
                "start": pd.to_datetime(
                    ["2019-12-31 23:59:00", "2020-01-01 00:10:00"]
                ),
                "end": pd.to_datetime(
                    ["2020-01-01 00:00:00", "2020-01-01 00:11:00"]
                ),
                "mode": ["boat", "ferry"],
            }
        )
        self.assertEqual(
            overlapping_label_modes(
                times.iloc[0], times.iloc[-1], touching_boundaries
            ),
            ["boat", "ferry"],
        )

    def test_user_macro_gives_users_equal_weight(self) -> None:
        cases = pd.DataFrame({"user_id": [1, 2, 2, 2], "hit": [1.0, 0.0, 0.0, 0.0]})
        self.assertEqual(user_macro_mean(cases, "hit"), 0.5)

    def test_paired_bootstrap_reuses_rows_across_methods(self) -> None:
        cases = pd.DataFrame(
            {"user_id": [1, 1, 2, 2], "full": [1.0, 2.0, 4.0, 5.0], "last": [0.0, 1.0, 3.0, 4.0]}
        )
        estimates = paired_hierarchical_bootstrap(cases, ["full", "last"], 50, 7)
        np.testing.assert_allclose(estimates[:, 0] - estimates[:, 1], 1.0)

    def test_bootstrap_preserves_complete_four_ratio_case_vectors(self) -> None:
        cases = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "full_25": [1.0, 2.0, 4.0, 5.0],
                "last_25": [0.0, 1.0, 3.0, 4.0],
                "full_75": [6.0, 7.0, 9.0, 10.0],
                "last_75": [4.0, 5.0, 7.0, 8.0],
            }
        )
        columns = ["full_25", "last_25", "full_75", "last_75"]
        estimates = paired_hierarchical_bootstrap(cases, columns, 50, 11)
        np.testing.assert_allclose(estimates[:, 0] - estimates[:, 1], 1.0)
        np.testing.assert_allclose(estimates[:, 2] - estimates[:, 3], 2.0)
        np.testing.assert_allclose(
            (estimates[:, 2] - estimates[:, 3])
            - (estimates[:, 0] - estimates[:, 1]),
            1.0,
        )

    def test_matched_ablation_changes_only_included_aligned_states(self) -> None:
        patterns = np.asarray([[[0.0, 0.0], [1.0, 1.0]], [[2.0, 2.0], [1.0, 1.0]]])
        query = np.asarray([[0.0, 0.0], [0.0, 0.0]])
        full = aligned_squared_distance(patterns, query, False)
        last = aligned_squared_distance(patterns, query, True)
        np.testing.assert_allclose(full, [1.0, 5.0])
        np.testing.assert_allclose(last, [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
