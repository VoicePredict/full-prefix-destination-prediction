from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from destination_prediction import artifact  # noqa: E402


class ReproductionBundleTests(unittest.TestCase):
    def test_whitelist_exactly_covers_frozen_publication_files(self) -> None:
        frozen_results = {
            f"results/{path.relative_to(ROOT / 'reference' / 'results').as_posix()}"
            for path in (ROOT / "reference" / "results").rglob("*")
            if path.is_file() and path.name != "README.md"
        }
        whitelisted_results = {
            spec.destination for spec in artifact.RESULT_FILE_MAP
        } | {
            spec.destination for spec in artifact.SANITIZED_RESULT_SPECS
        } | {"results/cohort_characteristics.csv"}
        self.assertEqual(whitelisted_results, frozen_results)
        for spec in artifact.SANITIZED_RESULT_SPECS:
            frozen_columns = tuple(
                pd.read_csv(ROOT / "reference" / spec.destination, nrows=0).columns
            )
            self.assertEqual(spec.output_columns, frozen_columns)

        frozen_predictions = {
            f"predictions/{path.name}"
            for path in (ROOT / "reference" / "predictions").glob("*.csv.gz")
        }
        whitelisted_predictions = {
            spec.destination for spec in artifact.PREDICTION_SPECS
        }
        self.assertEqual(whitelisted_predictions, frozen_predictions)
        for spec in artifact.PREDICTION_SPECS:
            frozen_columns = tuple(
                pd.read_csv(ROOT / "reference" / spec.destination, nrows=0).columns
            )
            self.assertEqual(
                spec.output_columns,
                frozen_columns,
                f"public prediction schema differs for {spec.destination}",
            )

        result_sources = [spec.source for spec in artifact.RESULT_FILE_MAP]
        result_destinations = [spec.destination for spec in artifact.RESULT_FILE_MAP]
        sanitized_result_sources = [
            spec.source for spec in artifact.SANITIZED_RESULT_SPECS
        ]
        sanitized_result_destinations = [
            spec.destination for spec in artifact.SANITIZED_RESULT_SPECS
        ]
        prediction_sources = [spec.source for spec in artifact.PREDICTION_SPECS]
        prediction_destinations = [spec.destination for spec in artifact.PREDICTION_SPECS]
        self.assertEqual(len(result_sources), len(set(result_sources)))
        self.assertEqual(len(result_destinations), len(set(result_destinations)))
        self.assertFalse(set(result_sources) & set(sanitized_result_sources))
        self.assertFalse(
            set(result_destinations) & set(sanitized_result_destinations)
        )
        self.assertEqual(len(prediction_sources), len(set(prediction_sources)))
        self.assertEqual(len(prediction_destinations), len(set(prediction_destinations)))

    def test_prediction_sanitization_is_private_stable_and_deterministic(self) -> None:
        spec = artifact.PredictionSpec(
            "unused.csv",
            "unused.csv.gz",
            (
                "method",
                "ratio",
                "user_id",
                "case_id",
                "analogue_case_id",
                "score",
            ),
        )
        rows = [
            {
                "method": "grid",
                "ratio": 0.5,
                "user_id": 7,
                "trip_id": "original-trip-b",
                "analogue_trip_id": "training-trip-z",
                "score": 0.25,
                "native_pred_lon": 116.3,
                "native_pred_lat": 39.9,
                "query_runtime_ms": 2.0,
                "source_path": "/private/source/b.plt",
                "wgs_seq": "[[116.3, 39.9]]",
            },
            {
                "method": "grid",
                "ratio": 0.25,
                "user_id": 3,
                "trip_id": "original-trip-a",
                "analogue_trip_id": "training-trip-y",
                "score": 0.75,
                "native_pred_lon": 116.4,
                "native_pred_lat": 40.0,
                "query_runtime_ms": 3.0,
                "source_path": "/private/source/a.plt",
                "wgs_seq": "[[116.4, 40.0]]",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            first_source = temporary / "first.csv"
            second_source = temporary / "second.csv"
            first_output = temporary / "first.csv.gz"
            second_output = temporary / "second.csv.gz"
            pd.DataFrame(rows).to_csv(first_source, index=False)
            pd.DataFrame(reversed(rows)).to_csv(second_source, index=False)

            artifact.sanitize_prediction_csv(first_source, first_output, spec)
            artifact.sanitize_prediction_csv(second_source, second_output, spec)

            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())
            self.assertEqual(first_output.read_bytes()[4:8], b"\x00\x00\x00\x00")
            public = pd.read_csv(first_output, dtype="string")
            self.assertEqual(list(public.columns), list(spec.output_columns))
            self.assertFalse(artifact._forbidden_public_columns(public.columns))
            serialized = gzip.decompress(first_output.read_bytes()).decode("utf-8")
            for private_value in (
                "original-trip-a",
                "original-trip-b",
                "training-trip-y",
                "training-trip-z",
                "/private/source",
                "116.3",
                "39.9",
            ):
                self.assertNotIn(private_value, serialized)
            self.assertEqual(public["ratio"].tolist(), ["0.25", "0.5"])
            self.assertEqual(
                public["case_id"].tolist(),
                [
                    artifact.stable_trajectory_id(3, "original-trip-a", "c"),
                    artifact.stable_trajectory_id(7, "original-trip-b", "c"),
                ],
            )
            self.assertEqual(
                public["analogue_case_id"].tolist(),
                [
                    artifact.stable_trajectory_id(3, "training-trip-y", "a"),
                    artifact.stable_trajectory_id(7, "training-trip-z", "a"),
                ],
            )

    def test_evaluation_prediction_tables_share_the_frozen_case_namespace(self) -> None:
        index = pd.read_csv(
            ROOT / "reference/results/matched_ablation/bootstrap_case_index.csv.gz",
            dtype="string",
        )
        expected = set(zip(index["user_id"], index["case_id"]))
        evaluation_files = (
            "canonical_cross_method_cases.csv.gz",
            "geometric_distance_cases.csv.gz",
            "matched_grid_cases.csv.gz",
            "tie_sensitivity_cases.csv.gz",
            "tie_boundary_cases.csv.gz",
            "matched_familiarity_cases.csv.gz",
            "catalogue_assignment_sensitivity_cases.csv.gz",
            "matched_robustness_evaluation37_cases.csv.gz",
            "ambiguity_prediction_cases.csv.gz",
            "ambiguity_binned_cases.csv.gz",
        )
        for name in evaluation_files:
            frame = pd.read_csv(
                ROOT / "reference/predictions" / name,
                usecols=["user_id", "case_id"],
                dtype="string",
            )
            self.assertEqual(
                set(zip(frame["user_id"], frame["case_id"])),
                expected,
                f"evaluation case namespace differs in {name}",
            )

    def test_builder_copies_only_explicitly_whitelisted_files(self) -> None:
        result_spec = artifact.CopySpec(
            "one/output.csv", "results/kept.csv"
        )
        prediction_spec = artifact.PredictionSpec(
            "prediction/source.csv",
            "predictions/cases.csv.gz",
            ("method", "user_id", "case_id", "value"),
        )
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            runs = temporary / "runs"
            bundle = temporary / "candidate"
            result_source = runs / result_spec.source
            prediction_source = runs / prediction_spec.source
            result_source.parent.mkdir(parents=True)
            prediction_source.parent.mkdir(parents=True)
            result_source.write_text("metric,value\nhit,0.5\n", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "method": "grid",
                        "user_id": 1,
                        "trip_id": "private-trip",
                        "value": 0.5,
                        "native_pred_lon": 1.0,
                    }
                ]
            ).to_csv(prediction_source, index=False)
            secret = runs / "one" / "internal_training_history.csv"
            secret.write_text("must,not,ship\n", encoding="utf-8")

            with (
                patch.object(artifact, "RESULT_FILE_MAP", (result_spec,)),
                patch.object(artifact, "MANIFEST_FILE_MAP", ()),
                patch.object(artifact, "STATIC_MANIFEST_FILE_MAP", ()),
                patch.object(artifact, "COHORT_INPUTS", {}),
                patch.object(artifact, "COHORT_CHARACTERISTIC_INPUTS", {}),
                patch.object(artifact, "SANITIZED_RESULT_SPECS", ()),
                patch.object(artifact, "PREDICTION_SPECS", (prediction_spec,)),
            ):
                summary = artifact.build_candidate_bundle(runs, bundle)

            self.assertEqual(summary["result_files"], 1)
            self.assertEqual(summary["prediction_files"], 1)
            files = {
                path.relative_to(bundle).as_posix()
                for path in bundle.rglob("*")
                if path.is_file()
            }
            self.assertEqual(
                files,
                {
                    "manifests/python_packages_frozen.txt",
                    "manifests/reference_system.txt",
                    "predictions/cases.csv.gz",
                    "results/kept.csv",
                },
            )
            self.assertFalse((bundle / "one/internal_training_history.csv").exists())
            self.assertTrue(secret.exists())

    def test_bootstrap_case_index_keeps_plan_position_order(self) -> None:
        spec = artifact.SANITIZED_RESULT_SPECS[0]
        source_rows = [
            {
                "user_position": 1,
                "within_user_position": 0,
                "user_id": 2,
                "trip_id": "trip-c",
            },
            {
                "user_position": 0,
                "within_user_position": 1,
                "user_id": 1,
                "trip_id": "trip-b",
            },
            {
                "user_position": 0,
                "within_user_position": 0,
                "user_id": 1,
                "trip_id": "trip-a",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            source = temporary / "bootstrap_case_index.csv"
            destination = temporary / "bootstrap_case_index.csv.gz"
            pd.DataFrame(source_rows).to_csv(source, index=False)
            public = artifact.sanitize_prediction_csv(source, destination, spec)
        self.assertEqual(
            list(zip(public["user_position"], public["within_user_position"])),
            [(0, 0), (0, 1), (1, 0)],
        )
        self.assertEqual(
            public["case_id"].tolist(),
            [
                artifact.stable_trajectory_id(1, "trip-a"),
                artifact.stable_trajectory_id(1, "trip-b"),
                artifact.stable_trajectory_id(2, "trip-c"),
            ],
        )

    def test_reference_tree_is_rejected_before_any_write(self) -> None:
        destination = artifact.REFERENCE_ROOT / f"unit-test-{uuid.uuid4().hex}"
        self.assertFalse(destination.exists())
        with self.assertRaises(ValueError):
            artifact.build_candidate_bundle(Path("missing-run-root"), destination)
        self.assertFalse(destination.exists())

    def test_published_rounding_passes_but_material_perturbation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            frozen = temporary / "frozen"
            candidate = temporary / "candidate"
            for root in (frozen, candidate):
                (root / "paper_tables").mkdir(parents=True)
                (root / "figures").mkdir(parents=True)
                (root / "figures/figure.svg").write_text(
                    "<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8"
                )
            pd.DataFrame(
                [{"method": "grid", "ratio": 0.25, "estimate": 1.232}]
            ).to_csv(frozen / "paper_tables/table.csv", index=False)
            pd.DataFrame(
                [{"method": "grid", "ratio": 0.25, "estimate": 1.231}]
            ).to_csv(candidate / "paper_tables/table.csv", index=False)

            passed = artifact.compare_paper_outputs(candidate, frozen)
            self.assertEqual(passed["status"], "passed")

            pd.DataFrame(
                [{"method": "grid", "ratio": 0.25, "estimate": 1.25}]
            ).to_csv(candidate / "paper_tables/table.csv", index=False)
            report_path = temporary / "equivalence_diff.json"
            with self.assertRaises(artifact.ArtifactComparisonError) as captured:
                artifact.compare_paper_outputs(
                    candidate, frozen, report_path=report_path
                )
            self.assertEqual(captured.exception.report["status"], "failed")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")
            self.assertTrue(report["differences"])
            self.assertIn("estimate", report["differences"][0]["message"])

    def test_cross_platform_float_serialization_is_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            frozen = temporary / "frozen" / "paper_tables"
            candidate = temporary / "candidate" / "paper_tables"
            frozen.mkdir(parents=True)
            candidate.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "method": "probabilistic_grid_pattern_retrieval",
                        "ratio": 0.75,
                        "estimate": 46.800463235028474,
                    }
                ]
            ).to_csv(frozen / "main_hit_r90.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "method": "probabilistic_grid_pattern_retrieval",
                        "ratio": 0.75,
                        "estimate": 46.80046323502848,
                    }
                ]
            ).to_csv(candidate / "main_hit_r90.csv", index=False)

            report = artifact.compare_paper_outputs(candidate.parent, frozen.parent)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["files_compared"], 1)

    def test_entropy_comparison_uses_three_published_digits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            frozen = temporary / "frozen" / "paper_tables"
            candidate = temporary / "candidate" / "paper_tables"
            frozen.mkdir(parents=True)
            candidate.mkdir(parents=True)
            pd.DataFrame(
                [{"requested_k": 10, "ratio": 0.25, "mean_entropy_nats": 1.222}]
            ).to_csv(frozen / "training_only_ambiguity.csv", index=False)
            pd.DataFrame(
                [{"requested_k": 10, "ratio": 0.25, "mean_entropy_nats": 1.224}]
            ).to_csv(candidate / "training_only_ambiguity.csv", index=False)
            with self.assertRaises(artifact.ArtifactComparisonError):
                artifact.compare_paper_outputs(candidate.parent, frozen.parent)

    def test_committed_paper_outputs_have_complete_comparison_keys(self) -> None:
        report = artifact.compare_paper_outputs(
            ROOT / "results", ROOT / "results", raise_on_difference=False
        )
        self.assertEqual(report["status"], "passed", report["differences"])

    def test_cohort_characteristics_follow_paper_aggregation(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "user_id": 1,
                    "trip_id": "u1-a",
                    "start_datetime": "2020-01-01 00:00:00",
                    "end_datetime": "2020-01-01 00:10:00",
                    "path_length_m": 1000,
                },
                {
                    "user_id": 1,
                    "trip_id": "u1-b",
                    "start_datetime": "2020-01-02 00:00:00",
                    "end_datetime": "2020-01-02 00:30:00",
                    "path_length_m": 3000,
                },
                {
                    "user_id": 2,
                    "trip_id": "u2-a",
                    "start_datetime": "2020-01-03 00:00:00",
                    "end_datetime": "2020-01-03 01:00:00",
                    "path_length_m": 6000,
                },
            ]
        )
        catalogue = pd.DataFrame(
            [
                {"user_id": 1, "catalog_eps_m": 200, "support": 1},
                {"user_id": 1, "catalog_eps_m": 200, "support": 2},
                {"user_id": 2, "catalog_eps_m": 200, "support": 1},
                {"user_id": 2, "catalog_eps_m": 100, "support": 9},
            ]
        )
        metrics = pd.DataFrame(
            [
                {"user_id": 1, "catalog_eps_m": 200, "coverage_r90_mean": 0.5},
                {"user_id": 2, "catalog_eps_m": 200, "coverage_r90_mean": 0.7},
                {"user_id": 2, "catalog_eps_m": 100, "coverage_r90_mean": 0.9},
            ]
        )

        values = artifact._cohort_characteristics(manifest, catalogue, metrics)
        self.assertEqual(values["users"], 2)
        self.assertEqual(values["recorded_trajectories"], 3)
        self.assertEqual(values["median_trajectories_per_user"], 1.5)
        self.assertEqual(values["median_user_level_duration"], 40.0)
        self.assertEqual(values["median_user_level_distance"], 4.0)
        self.assertEqual(values["median_regions_per_user"], 1.5)
        self.assertAlmostEqual(values["user_macro_r90_coverage"], 60.0)
        self.assertAlmostEqual(values["singleton_regions"], 200.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
