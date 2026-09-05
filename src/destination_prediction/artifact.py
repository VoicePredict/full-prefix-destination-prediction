"""Build and compare reviewer-facing reproduction bundles.

The experiment runners deliberately retain rich internal outputs (coordinates,
model diagnostics, and training histories).  This module is the single,
explicit boundary between those run directories and the compact public
artifact.  Nothing is discovered by recursively copying a run directory:
every released file is named in one of the maps below.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = REPOSITORY_ROOT / "reference"


@dataclass(frozen=True)
class CopySpec:
    """One whitelisted source-to-bundle copy operation."""

    source: str
    destination: str
    gzip_csv: bool = False


@dataclass(frozen=True)
class PredictionSpec:
    """One public case table and its exact released schema."""

    source: str
    destination: str
    output_columns: tuple[str, ...]
    sort_columns: tuple[str, ...] = ()


class ArtifactComparisonError(AssertionError):
    """Raised when reproduced paper outputs differ from the frozen outputs."""

    def __init__(self, report: dict[str, object]):
        self.report = report
        differences = report.get("differences", [])
        preview = "; ".join(str(item.get("message", item)) for item in differences[:3])
        super().__init__(
            f"Reproduced paper outputs are not equivalent"
            + (f": {preview}" if preview else "")
        )


# Fresh-run files that are retained as publication-facing statistical output.
# The destination side mirrors reference/results exactly.  The recurrent
# classifier uses its current public name on both sides of this boundary.
RESULT_FILE_MAP: tuple[CopySpec, ...] = (
    CopySpec("discovery/outputs/preparation/preparation_summary.json", "results/discovery/preparation/preparation_summary.json"),
    CopySpec("discovery/outputs/preparation/partition_counts_by_user.csv", "results/discovery/preparation/partition_counts_by_user.csv"),
    CopySpec("discovery/outputs/non_neural/geometric_selection/leaderboard.csv", "results/discovery/non_neural/geometric_selection/leaderboard.csv"),
    CopySpec("discovery/outputs/non_neural/geometric_selection/selected_config.json", "results/discovery/non_neural/geometric_selection/selected_config.json"),
    CopySpec("discovery/outputs/non_neural/grid_pattern_selection/leaderboard.csv", "results/discovery/non_neural/grid_pattern_selection/leaderboard.csv"),
    CopySpec("discovery/outputs/non_neural/grid_pattern_selection/selected_config.json", "results/discovery/non_neural/grid_pattern_selection/selected_config.json"),
    CopySpec("discovery/outputs/recurrent/validation_leaderboard.csv", "results/discovery/recurrent/validation_leaderboard.csv"),
    CopySpec("discovery/outputs/recurrent/selected_config.json", "results/discovery/recurrent/selected_config.json"),
    CopySpec("discovery/outputs/tsmini/validation_leaderboard.csv", "results/discovery/tsmini/validation_leaderboard.csv"),
    CopySpec("discovery/outputs/tsmini/selected_config.json", "results/discovery/tsmini/selected_config.json"),
    CopySpec("discovery/outputs/final/metrics_user_macro_seed_summary.csv", "results/discovery/final/metrics_user_macro_seed_summary.csv"),
    CopySpec("discovery/outputs/final/paired_hierarchical_bootstrap.csv", "results/discovery/final/paired_hierarchical_bootstrap.csv"),
    CopySpec("discovery/outputs/validation_report.json", "results/discovery/validation_report.json"),
    CopySpec("evaluation37/outputs/preparation/preparation_summary.json", "results/evaluation37/preparation/preparation_summary.json"),
    CopySpec("evaluation37/outputs/preparation/partition_counts_by_user.csv", "results/evaluation37/preparation/partition_counts_by_user.csv"),
    CopySpec("evaluation37/outputs/final/catalogue_assignment_audit.json", "results/evaluation37/final/catalogue_assignment_audit.json"),
    CopySpec("evaluation37/outputs/final/evaluation_audit.json", "results/evaluation37/final/evaluation_audit.json"),
    CopySpec("evaluation37/outputs/final/metrics_by_user_seed.csv", "results/evaluation37/final/metrics_by_user_seed.csv"),
    CopySpec("evaluation37/outputs/final/metrics_trip_weighted_seed_summary.csv", "results/evaluation37/final/metrics_trip_weighted_seed_summary.csv"),
    CopySpec("evaluation37/outputs/final/metrics_user_macro_seed_summary.csv", "results/evaluation37/final/metrics_user_macro_seed_summary.csv"),
    CopySpec("evaluation37/outputs/final/paired_hierarchical_bootstrap.csv", "results/evaluation37/final/paired_hierarchical_bootstrap.csv"),
    CopySpec("evaluation37/outputs/final/quality_sensitivity_user_macro_seed_summary.csv", "results/evaluation37/final/quality_sensitivity_user_macro_seed_summary.csv"),
    CopySpec("evaluation37/outputs/analysis/destination_popularity_metrics.csv", "results/evaluation37/analysis/destination_popularity_metrics.csv"),
    CopySpec("evaluation37/outputs/analysis/familiarity_analogue_analysis_summary.json", "results/evaluation37/analysis/familiarity_analogue_analysis_summary.json"),
    CopySpec("evaluation37/outputs/analysis/prefix_analogue_metrics.csv", "results/evaluation37/analysis/prefix_analogue_metrics.csv"),
    CopySpec("evaluation37/outputs/analysis/stratified_bootstrap.csv", "results/evaluation37/analysis/stratified_bootstrap.csv"),
    CopySpec("evaluation37/outputs/validation_report.json", "results/evaluation37/validation_report.json"),
    CopySpec("expanded46/outputs/preparation/preparation_summary.json", "results/expanded46/preparation/preparation_summary.json"),
    CopySpec("expanded46/outputs/preparation/partition_counts_by_user.csv", "results/expanded46/preparation/partition_counts_by_user.csv"),
    CopySpec("expanded46/outputs/final/metrics_user_macro_seed_summary.csv", "results/expanded46/final/metrics_user_macro_seed_summary.csv"),
    CopySpec("expanded46/outputs/validation_report.json", "results/expanded46/validation_report.json"),
    CopySpec("ambiguity/outputs/analysis_summary.json", "results/ambiguity/analysis_summary.json"),
    CopySpec("ambiguity/outputs/cases/held_out_user_effects.csv", "results/ambiguity/cases/held_out_user_effects.csv"),
    CopySpec("ambiguity/outputs/cases/held_out_user_profiles.csv", "results/ambiguity/cases/held_out_user_profiles.csv"),
    CopySpec("ambiguity/outputs/checks/ambiguity_thresholds.csv", "results/ambiguity/checks/ambiguity_thresholds.csv"),
    CopySpec("ambiguity/outputs/checks/history_size_thresholds.json", "results/ambiguity/checks/history_size_thresholds.json"),
    CopySpec("ambiguity/outputs/tables/ambiguity_distribution.csv", "results/ambiguity/tables/ambiguity_distribution.csv"),
    CopySpec("ambiguity/outputs/tables/ambiguity_focal_bootstrap.csv", "results/ambiguity/tables/ambiguity_focal_bootstrap.csv"),
    CopySpec("ambiguity/outputs/tables/ambiguity_method_metrics.csv", "results/ambiguity/tables/ambiguity_method_metrics.csv"),
    CopySpec("ambiguity/outputs/tables/history_size_method_metrics.csv", "results/ambiguity/tables/history_size_method_metrics.csv"),
    CopySpec("ambiguity/outputs/tables/joint_familiarity_case_counts.csv", "results/ambiguity/tables/joint_familiarity_case_counts.csv"),
    CopySpec("ambiguity/outputs/tables/joint_familiarity_method_metrics.csv", "results/ambiguity/tables/joint_familiarity_method_metrics.csv"),
    CopySpec("ambiguity/outputs/tables/point_error_and_coverage_metrics.csv", "results/ambiguity/tables/point_error_and_coverage_metrics.csv"),
    CopySpec("ambiguity/outputs/tables/user_feature_effect_correlations.csv", "results/ambiguity/tables/user_feature_effect_correlations.csv"),
    CopySpec("ambiguity/outputs/tables/user_heterogeneity.csv", "results/ambiguity/tables/user_heterogeneity.csv"),
    CopySpec("ambiguity/outputs/validation_report.json", "results/ambiguity/validation_report.json"),
    CopySpec("geometric_distance/outputs/method_metrics.csv", "results/geometric_distance/method_metrics.csv"),
    CopySpec("geometric_distance/outputs/paired_bootstrap.csv", "results/geometric_distance/paired_bootstrap.csv"),
    CopySpec("geometric_distance/outputs/run_summary.json", "results/geometric_distance/run_summary.json"),
    CopySpec("geometric_distance/outputs/validation_report.json", "results/geometric_distance/validation_report.json"),
    CopySpec("matched_ablation/outputs/bootstrap_resample_ids.npz", "results/matched_ablation/bootstrap_resample_ids.npz"),
    CopySpec("matched_ablation/outputs/method_metrics.csv", "results/matched_ablation/method_metrics.csv"),
    CopySpec("matched_ablation/outputs/paired_bootstrap.csv", "results/matched_ablation/paired_bootstrap.csv"),
    CopySpec("matched_ablation/outputs/run_summary.json", "results/matched_ablation/run_summary.json"),
    CopySpec("matched_ablation/outputs/validation_report.json", "results/matched_ablation/validation_report.json"),
    CopySpec("tie_sensitivity/outputs/method_metrics.csv", "results/tie_sensitivity/method_metrics.csv"),
    CopySpec("tie_sensitivity/outputs/paired_tie_sensitivity_bootstrap.csv", "results/tie_sensitivity/paired_tie_sensitivity_bootstrap.csv"),
    CopySpec("tie_sensitivity/outputs/run_summary.json", "results/tie_sensitivity/run_summary.json"),
    CopySpec("tie_sensitivity/outputs/top10_boundary_tie_summary.csv", "results/tie_sensitivity/top10_boundary_tie_summary.csv"),
    CopySpec("tie_sensitivity/outputs/validation_report.json", "results/tie_sensitivity/validation_report.json"),
    CopySpec("matched_familiarity/outputs/matched_familiarity_bootstrap.csv", "results/matched_familiarity/matched_familiarity_bootstrap.csv"),
    CopySpec("matched_familiarity/outputs/matched_familiarity_metrics.csv", "results/matched_familiarity/matched_familiarity_metrics.csv"),
    CopySpec("matched_familiarity/outputs/run_summary.json", "results/matched_familiarity/run_summary.json"),
    CopySpec("matched_familiarity/outputs/validation_report.json", "results/matched_familiarity/validation_report.json"),
    CopySpec("matched_diagnostics/outputs/matched_ambiguity_bootstrap.csv", "results/matched_diagnostics/matched_ambiguity_bootstrap.csv"),
    CopySpec("matched_diagnostics/outputs/matched_ambiguity_metrics.csv", "results/matched_diagnostics/matched_ambiguity_metrics.csv"),
    CopySpec("matched_diagnostics/outputs/matched_history_size_metrics.csv", "results/matched_diagnostics/matched_history_size_metrics.csv"),
    CopySpec("matched_diagnostics/outputs/matched_user_effects.csv", "results/matched_diagnostics/matched_user_effects.csv"),
    CopySpec("matched_diagnostics/outputs/matched_user_feature_correlations.csv", "results/matched_diagnostics/matched_user_feature_correlations.csv"),
    CopySpec("matched_diagnostics/outputs/run_summary.json", "results/matched_diagnostics/run_summary.json"),
    CopySpec("matched_diagnostics/outputs/validation_report.json", "results/matched_diagnostics/validation_report.json"),
    CopySpec("matched_robustness/outputs/canonical_cross_method_bootstrap.csv", "results/matched_robustness/canonical_cross_method_bootstrap.csv"),
    CopySpec("matched_robustness/outputs/canonical_cross_method_metrics.csv", "results/matched_robustness/canonical_cross_method_metrics.csv"),
    CopySpec("matched_robustness/outputs/catalogue_assignment_bootstrap.csv", "results/matched_robustness/catalogue_assignment_bootstrap.csv"),
    CopySpec("matched_robustness/outputs/catalogue_assignment_change_audit.csv", "results/matched_robustness/catalogue_assignment_change_audit.csv"),
    CopySpec("matched_robustness/outputs/matched_robustness_bootstrap.csv", "results/matched_robustness/matched_robustness_bootstrap.csv"),
    CopySpec("matched_robustness/outputs/run_summary.json", "results/matched_robustness/run_summary.json"),
    CopySpec("matched_robustness/outputs/validation_report.json", "results/matched_robustness/validation_report.json"),
)


MANIFEST_FILE_MAP: tuple[CopySpec, ...] = (
    CopySpec("evaluation37/outputs/preparation/preparation_summary.json", "manifests/preparation_summary.json"),
    CopySpec("evaluation37/outputs/preparation/partition_counts_by_user.csv", "manifests/partition_counts_by_user.csv"),
    CopySpec("evaluation37/outputs/preparation/partition_manifest.csv", "manifests/partition_manifest.csv.gz", True),
    CopySpec("evaluation37/outputs/preparation/source_trajectory_audit.csv", "manifests/source_trajectory_audit.csv.gz", True),
)


STATIC_MANIFEST_FILE_MAP: tuple[CopySpec, ...] = (
    CopySpec("manifests/geolife_1.3_files.sha256", "manifests/geolife_1.3_files.sha256"),
    CopySpec("manifests/run_registry.json", "manifests/run_registry.json"),
)


COHORT_INPUTS = {
    "discovery": "discovery/outputs/preparation/cohort_sampling_frame.csv",
    "evaluation37": "evaluation37/outputs/preparation/cohort_sampling_frame.csv",
    "expanded46": "expanded46/outputs/preparation/cohort_sampling_frame.csv",
    "source_audit": "evaluation37/outputs/preparation/source_trajectory_audit.csv",
    "evaluation_summary": "evaluation37/outputs/preparation/preparation_summary.json",
}


COHORT_CHARACTERISTIC_INPUTS = {
    "discovery_manifest": "discovery/outputs/preparation/partition_manifest.csv",
    "evaluation_manifest": "evaluation37/outputs/preparation/partition_manifest.csv",
    "discovery_catalogue": "discovery/outputs/final/outer_training_destination_catalogs.csv",
    "evaluation_catalogue": "evaluation37/outputs/final/outer_training_destination_catalogs.csv",
    "discovery_metrics": "discovery/outputs/final/metrics_user_macro_seed_summary.csv",
    "evaluation_metrics": "evaluation37/outputs/final/metrics_user_macro_seed_summary.csv",
}


PREDICTION_SPECS: tuple[PredictionSpec, ...] = (
    PredictionSpec(
        "matched_robustness/outputs/canonical_cross_method_cases.csv",
        "predictions/canonical_cross_method_cases.csv.gz",
        ("method", "ratio", "user_id", "case_id", "hit_r90_all"),
    ),
    PredictionSpec(
        "geometric_distance/outputs/geometric_predictions.csv",
        "predictions/geometric_distance_cases.csv.gz",
        (
            "family", "method", "ratio", "user_id", "case_id",
            "analogue_case_id", "nearest_distance", "resample_points",
            "true_center_id", "true_to_center_m", "pred_center_id",
            "pred_to_center_m", "destination_error_m", "native_destination_error_m",
            "within_200m", "native_within_200m", "within_1000m",
            "native_within_1000m", "covered_r80", "true_center_id_r80",
            "hit_r80_all", "covered_r90", "true_center_id_r90", "hit_r90",
            "covered_r95", "true_center_id_r95", "hit_r95_all",
            "covered_fixed_200m", "hit_fixed_200m_all",
        ),
    ),
    PredictionSpec(
        "matched_ablation/outputs/comparison_cases.csv",
        "predictions/matched_grid_cases.csv.gz",
        ("method", "ratio", "user_id", "case_id", "hit_r90_all", "destination_error_m", "within_200m", "within_1000m"),
    ),
    PredictionSpec(
        "tie_sensitivity/outputs/tie_sensitivity_predictions.csv",
        "predictions/tie_sensitivity_cases.csv.gz",
        (
            "variant", "method", "ratio", "user_id", "case_id",
            "selected_pool_size", "destination_mass_tie", "true_center_id",
            "true_to_center_m", "pred_center_id", "pred_to_center_m",
            "destination_error_m", "native_destination_error_m", "within_200m",
            "native_within_200m", "within_1000m", "native_within_1000m",
            "covered_r80", "true_center_id_r80", "hit_r80_all", "covered_r90",
            "true_center_id_r90", "hit_r90_all", "covered_r95",
            "true_center_id_r95", "hit_r95_all", "covered_fixed_200m",
            "hit_fixed_200m_all",
        ),
    ),
    PredictionSpec(
        "tie_sensitivity/outputs/top10_boundary_tie_cases.csv",
        "predictions/tie_boundary_cases.csv.gz",
        ("method", "ratio", "user_id", "case_id", "boundary_distance", "strictly_closer", "boundary_equal_candidates", "boundary_tie_straddles_top_k"),
    ),
    PredictionSpec(
        "matched_familiarity/outputs/matched_familiarity_cases.csv",
        "predictions/matched_familiarity_cases.csv.gz",
        ("ratio", "user_id", "case_id", "grid_full_prefix", "grid_last_state_only", "paired_effect", "no_train_analogue_q90", "prefix_familiarity"),
    ),
    PredictionSpec(
        "matched_robustness/outputs/catalogue_assignment_cases.csv",
        "predictions/catalogue_assignment_sensitivity_cases.csv.gz",
        (
            "assignment_rule", "method", "ratio", "user_id", "case_id",
            "true_center_id", "true_to_center_m", "pred_center_id",
            "pred_to_center_m", "destination_error_m", "within_200m",
            "within_1000m", "covered_r90", "true_center_id_r90", "hit_r90_all",
        ),
    ),
    PredictionSpec(
        "matched_robustness/outputs/held_out_predictions.csv",
        "predictions/matched_robustness_evaluation37_cases.csv.gz",
        (
            "cohort", "observation_definition", "method", "ratio", "user_id",
            "case_id", "true_center_id", "true_to_center_m", "pred_center_id",
            "pred_to_center_m", "destination_error_m", "native_destination_error_m",
            "within_200m", "native_within_200m", "within_1000m",
            "native_within_1000m", "covered_r80", "true_center_id_r80",
            "hit_r80_all", "covered_r90", "true_center_id_r90", "hit_r90_all",
            "covered_r95", "true_center_id_r95", "hit_r95_all",
            "covered_fixed_200m", "hit_fixed_200m_all",
        ),
    ),
    PredictionSpec(
        "matched_robustness/outputs/expanded46_predictions.csv",
        "predictions/matched_robustness_expanded46_cases.csv.gz",
        (
            "cohort", "observation_definition", "method", "ratio", "user_id",
            "case_id", "true_center_id", "true_to_center_m", "pred_center_id",
            "pred_to_center_m", "destination_error_m", "native_destination_error_m",
            "within_200m", "native_within_200m", "within_1000m",
            "native_within_1000m", "covered_r80", "true_center_id_r80",
            "hit_r80_all", "covered_r90", "true_center_id_r90", "hit_r90_all",
            "covered_r95", "true_center_id_r95", "hit_r95_all",
            "covered_fixed_200m", "hit_fixed_200m_all",
        ),
    ),
    PredictionSpec(
        "ambiguity/outputs/cases/held_out_location_ambiguity_binned.csv",
        "predictions/ambiguity_binned_cases.csv.gz",
        (
            "cohort", "ratio", "user_id", "case_id", "requested_k",
            "effective_k", "distinct_destinations", "entropy_nats",
            "effective_destinations", "top1_share", "top2_share",
            "top1_top2_margin", "nearest_training_position_m",
            "entropy_low_upper", "entropy_medium_upper", "discovery_cases",
            "ambiguity_stratum",
        ),
    ),
    PredictionSpec(
        "ambiguity/outputs/cases/prediction_cases_with_ambiguity.csv",
        "predictions/ambiguity_prediction_cases.csv.gz",
        (
            "method", "ratio", "user_id", "case_id", "covered_r90",
            "hit_r90", "within_200m", "within_1000m", "destination_error_m",
            "native_destination_error_m", "cohort", "requested_k", "effective_k",
            "distinct_destinations", "entropy_nats", "effective_destinations",
            "top1_share", "top2_share", "top1_top2_margin",
            "nearest_training_position_m", "entropy_low_upper",
            "entropy_medium_upper", "discovery_cases", "ambiguity_stratum",
        ),
    ),
)


SANITIZED_RESULT_SPECS: tuple[PredictionSpec, ...] = (
    PredictionSpec(
        "matched_ablation/outputs/bootstrap_case_index.csv",
        "results/matched_ablation/bootstrap_case_index.csv.gz",
        ("user_position", "within_user_position", "user_id", "case_id"),
        ("user_position", "within_user_position"),
    ),
)


FORBIDDEN_PUBLIC_COLUMN_FRAGMENTS = (
    "longitude",
    "latitude",
    "_lon",
    "_lat",
    "runtime",
    "wgs_seq",
    "local_seq",
    "raw_sequence",
    "source_file",
    "source_path",
)

SORT_PRIORITY = (
    "cohort",
    "observation_definition",
    "assignment_rule",
    "variant",
    "family",
    "method",
    "config_id",
    "requested_k",
    "catalog_eps_m",
    "seed",
    "ratio",
    "user_id",
    "case_id",
    "analogue_case_id",
)


def _trajectory_key(user_id: object, trip_id: object) -> tuple[str, str]:
    """Normalize the private fields used to identify one personal trajectory."""

    try:
        user_token = str(int(user_id))
    except (TypeError, ValueError):
        user_token = str(user_id).strip()
    return user_token, str(trip_id).strip()


def stable_trajectory_id(user_id: object, trip_id: object, prefix: str = "c") -> str:
    """Return a portable pseudonymous identifier for one personal trajectory."""

    if prefix not in {"c", "a"}:
        raise ValueError("Trajectory identifier prefix must be 'c' or 'a'.")
    if pd.isna(trip_id) or str(trip_id).strip() == "":
        return ""
    user_token, trip_token = _trajectory_key(user_id, trip_id)
    payload = f"{user_token}\x1f{trip_token}".encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def _evaluation_case_id_map(run_root: Path) -> dict[tuple[str, str], str] | None:
    """Map private evaluation keys to the artifact's frozen public case IDs.

    Public identifiers are presentation-only.  Their namespace was frozen with
    the released bootstrap case index, while the accompanying fresh-run index
    supplies the private trajectory key at the same deterministic positions.
    """

    raw_path = run_root / "matched_ablation/outputs/bootstrap_case_index.csv"
    if not raw_path.is_file():
        return None
    public_path = (
        REFERENCE_ROOT / "results/matched_ablation/bootstrap_case_index.csv.gz"
    )
    if not public_path.is_file():
        raise FileNotFoundError(f"Frozen public case index is missing: {public_path}")

    raw = pd.read_csv(raw_path, dtype={"trip_id": "string"})
    public = pd.read_csv(public_path, dtype={"case_id": "string"})
    positions = ["user_position", "within_user_position", "user_id"]
    required_raw = positions + ["trip_id"]
    required_public = positions + ["case_id"]
    if any(column not in raw for column in required_raw):
        raise ValueError(f"{raw_path}: incomplete bootstrap case index")
    if any(column not in public for column in required_public):
        raise ValueError(f"{public_path}: incomplete public bootstrap case index")

    linked = raw.loc[:, required_raw].merge(
        public.loc[:, required_public],
        on=positions,
        how="left",
        validate="one_to_one",
    )
    if linked["case_id"].isna().any() or len(linked) != len(public):
        raise ValueError("Fresh and frozen bootstrap case indices are not aligned")

    mapping = {
        _trajectory_key(row.user_id, row.trip_id): str(row.case_id)
        for row in linked.itertuples(index=False)
    }
    if len(mapping) != len(linked) or linked["case_id"].nunique() != len(linked):
        raise ValueError("Evaluation trajectory keys and public case IDs must be unique")
    return mapping


def _write_deterministic_gzip(payload: bytes, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            compressed.write(payload)


def _write_gzip_csv(frame: pd.DataFrame, destination: Path) -> None:
    buffer = io.StringIO(newline="")
    frame.to_csv(
        buffer,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    _write_deterministic_gzip(buffer.getvalue().encode("utf-8"), destination)


def _forbidden_public_columns(columns: Iterable[str]) -> list[str]:
    forbidden: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if lowered in {"trip_id", "analogue_trip_id"} or any(
            token in lowered for token in FORBIDDEN_PUBLIC_COLUMN_FRAGMENTS
        ):
            forbidden.append(str(column))
    return forbidden


def sanitize_prediction_csv(
    source: Path,
    destination: Path,
    spec: PredictionSpec,
    case_id_map: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Create one coordinate-free, deterministically ordered public case file."""

    frame = pd.read_csv(
        source,
        dtype={"trip_id": "string", "analogue_trip_id": "string"},
    )
    if "user_id" not in frame or "trip_id" not in frame:
        raise ValueError(f"{source}: public case conversion requires user_id and trip_id")
    frame["case_id"] = [
        (
            case_id_map.get(
                _trajectory_key(user_id, trip_id),
                stable_trajectory_id(user_id, trip_id, "c"),
            )
            if case_id_map is not None
            else stable_trajectory_id(user_id, trip_id, "c")
        )
        for user_id, trip_id in zip(frame["user_id"], frame["trip_id"])
    ]
    if "analogue_case_id" in spec.output_columns:
        if "analogue_trip_id" not in frame:
            raise ValueError(f"{source}: analogue_trip_id is required by the public schema")
        frame["analogue_case_id"] = [
            stable_trajectory_id(user_id, trip_id, "a")
            for user_id, trip_id in zip(frame["user_id"], frame["analogue_trip_id"])
        ]

    missing = [column for column in spec.output_columns if column not in frame]
    if missing:
        raise ValueError(f"{source}: missing public columns {missing}")
    public = frame.loc[:, list(spec.output_columns)].copy()
    forbidden = _forbidden_public_columns(public.columns)
    if forbidden:
        raise ValueError(f"{source}: forbidden public columns survived: {forbidden}")

    if spec.sort_columns:
        sort_columns = list(spec.sort_columns)
        missing_sort = [column for column in sort_columns if column not in public]
        if missing_sort:
            raise ValueError(f"{source}: missing sort columns {missing_sort}")
    else:
        sort_columns = [column for column in SORT_PRIORITY if column in public]
        sort_columns.extend(
            column for column in public.columns if column not in sort_columns
        )
    public = public.sort_values(
        sort_columns, kind="mergesort", na_position="last"
    ).reset_index(drop=True)
    _write_gzip_csv(public, destination)
    return public


def _copy_spec(source_root: Path, output_root: Path, spec: CopySpec) -> None:
    source = source_root / spec.source
    destination = output_root / spec.destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if spec.gzip_csv:
        _write_deterministic_gzip(source.read_bytes(), destination)
    else:
        shutil.copyfile(source, destination)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").str.strip().str.lower().isin({"true", "1", "yes"})


def _write_cohort_manifest(run_root: Path, destination: Path) -> None:
    discovery = pd.read_csv(run_root / COHORT_INPUTS["discovery"])
    evaluation = pd.read_csv(run_root / COHORT_INPUTS["evaluation37"])
    expanded = pd.read_csv(run_root / COHORT_INPUTS["expanded46"])
    audit = pd.read_csv(run_root / COHORT_INPUTS["source_audit"])
    summary = json.loads(
        (run_root / COHORT_INPUTS["evaluation_summary"]).read_text(encoding="utf-8")
    )

    retained = audit.loc[audit["status"].astype(str) == "eligible"]
    counts = retained.groupby("user_id").size().to_dict()
    all_users = sorted(map(int, audit["user_id"].unique()))

    discovery = discovery.set_index("user_id")
    evaluation = evaluation.set_index("user_id")
    expanded = expanded.set_index("user_id")
    discovery_users = set(
        map(int, discovery.index[_as_bool(discovery["selected"])])
    )
    evaluation_users = set(
        map(int, evaluation.index[_as_bool(evaluation["selected"])])
    )
    expanded_users = set(map(int, expanded.index[_as_bool(expanded["selected"])]))
    prior_users = set(map(int, summary.get("excluded_prior_users", [])))
    eligible_users = set(map(int, discovery.index))

    rows: list[dict[str, object]] = []
    for user_id in all_users:
        eligible = user_id in eligible_users
        discovery_member = user_id in discovery_users
        evaluation_member = user_id in evaluation_users
        prior = user_id in prior_users
        if not eligible:
            reason = "minimum_history_not_met"
        elif discovery_member:
            reason = "discovery_configuration_selection"
        elif prior:
            reason = "prior_analysis"
        elif evaluation_member:
            reason = ""
        else:
            raise ValueError(f"Eligible user {user_id} has no normalized cohort role")
        rows.append(
            {
                "user_id": user_id,
                "eligibility": str(eligible).lower(),
                "eligible_trajectories": int(counts.get(user_id, 0)),
                "history_tertile": (
                    str(discovery.loc[user_id, "history_tertile"]) if eligible else ""
                ),
                "discovery_membership": str(discovery_member).lower(),
                "prior_use": str(prior).lower(),
                "evaluation_membership": str(evaluation_member).lower(),
                "expanded_membership": str(user_id in expanded_users).lower(),
                "exclusion_reason": reason,
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False, lineterminator="\n")


def _write_environment_manifests(manifest_root: Path) -> None:
    manifest_root.mkdir(parents=True, exist_ok=True)
    packages = sorted(
        {
            f"{name}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
            if (name := distribution.metadata.get("Name"))
        },
        key=str.casefold,
    )
    (manifest_root / "python_packages_frozen.txt").write_text(
        "\n".join(packages) + "\n", encoding="utf-8"
    )
    system = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    (manifest_root / "reference_system.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in system.items()) + "\n",
        encoding="utf-8",
    )


def _cohort_characteristics(
    manifest: pd.DataFrame,
    catalogue: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, float | int]:
    manifest = manifest.copy()
    manifest["start_datetime"] = pd.to_datetime(manifest["start_datetime"])
    manifest["end_datetime"] = pd.to_datetime(manifest["end_datetime"])
    manifest["duration_minutes"] = (
        manifest["end_datetime"] - manifest["start_datetime"]
    ).dt.total_seconds() / 60.0
    manifest["distance_km"] = pd.to_numeric(
        manifest["path_length_m"], errors="raise"
    ) / 1000.0
    per_user = manifest.groupby("user_id", sort=True).agg(
        trajectories=("trip_id", "size"),
        median_duration_minutes=("duration_minutes", "median"),
        median_distance_km=("distance_km", "median"),
    )

    primary_catalogue = catalogue.loc[
        pd.to_numeric(catalogue["catalog_eps_m"], errors="raise").eq(200.0)
    ].copy()
    regions_per_user = primary_catalogue.groupby("user_id", sort=True).size()
    primary_metrics = metrics.loc[
        pd.to_numeric(metrics["catalog_eps_m"], errors="raise").eq(200.0)
    ]
    if primary_catalogue.empty or primary_metrics.empty:
        raise ValueError("Cohort characteristics require the 200 m catalogue and metrics")

    return {
        "users": int(manifest["user_id"].nunique()),
        "recorded_trajectories": int(len(manifest)),
        "median_trajectories_per_user": float(per_user["trajectories"].median()),
        "median_user_level_duration": float(
            per_user["median_duration_minutes"].median()
        ),
        "median_user_level_distance": float(per_user["median_distance_km"].median()),
        "median_regions_per_user": float(regions_per_user.median()),
        "user_macro_r90_coverage": float(
            pd.to_numeric(primary_metrics["coverage_r90_mean"], errors="raise").mean()
            * 100.0
        ),
        "singleton_regions": float(
            pd.to_numeric(primary_catalogue["support"], errors="raise").eq(1).mean()
            * 100.0
        ),
    }


def _write_cohort_characteristics(run_root: Path, destination: Path) -> None:
    cohorts: dict[str, dict[str, float | int]] = {}
    for label in ("discovery", "evaluation"):
        manifest = pd.read_csv(
            run_root / COHORT_CHARACTERISTIC_INPUTS[f"{label}_manifest"],
            dtype={"trip_id": "string"},
        )
        catalogue = pd.read_csv(
            run_root / COHORT_CHARACTERISTIC_INPUTS[f"{label}_catalogue"]
        )
        metrics = pd.read_csv(
            run_root / COHORT_CHARACTERISTIC_INPUTS[f"{label}_metrics"]
        )
        cohorts[label] = _cohort_characteristics(manifest, catalogue, metrics)

    units = {
        "users": "count",
        "recorded_trajectories": "count",
        "median_trajectories_per_user": "trajectories",
        "median_user_level_duration": "minutes",
        "median_user_level_distance": "kilometres",
        "median_regions_per_user": "regions",
        "user_macro_r90_coverage": "percent",
        "singleton_regions": "percent",
    }
    reporting_precision = {
        "users": 0,
        "recorded_trajectories": 0,
        "median_trajectories_per_user": 1,
        "median_user_level_duration": 1,
        "median_user_level_distance": 2,
        "median_regions_per_user": 0,
        "user_macro_r90_coverage": 1,
        "singleton_regions": 1,
    }
    rows = [
        {
            "characteristic": characteristic,
            "discovery": f"{cohorts['discovery'][characteristic]:.{reporting_precision[characteristic]}f}",
            "evaluation": f"{cohorts['evaluation'][characteristic]:.{reporting_precision[characteristic]}f}",
            "unit": unit,
        }
        for characteristic, unit in units.items()
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        destination,
        index=False,
        lineterminator="\n",
    )


def _required_sources(run_root: Path) -> list[Path]:
    paths = [run_root / spec.source for spec in RESULT_FILE_MAP]
    paths.extend(run_root / spec.source for spec in MANIFEST_FILE_MAP)
    paths.extend(run_root / spec.source for spec in PREDICTION_SPECS)
    paths.extend(run_root / spec.source for spec in SANITIZED_RESULT_SPECS)
    paths.extend(run_root / value for value in COHORT_INPUTS.values())
    paths.extend(
        run_root / value for value in COHORT_CHARACTERISTIC_INPUTS.values()
    )
    paths.extend(REFERENCE_ROOT / spec.source for spec in STATIC_MANIFEST_FILE_MAP)
    return paths


def _is_within(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def build_candidate_bundle(run_root: Path, bundle_root: Path) -> dict[str, object]:
    """Normalize validated fresh runs into a new reviewer-facing bundle.

    ``reference/`` is immutable: a destination at or below that directory is
    rejected before any source is inspected or any directory is created.
    """

    runs = Path(run_root).expanduser().resolve()
    destination = Path(bundle_root).expanduser().resolve()
    reference = REFERENCE_ROOT.resolve()
    if _is_within(destination, reference):
        raise ValueError("Candidate bundles must not be written under reference/")
    if destination.exists():
        raise FileExistsError(f"Candidate bundle destination already exists: {destination}")
    if not runs.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {runs}")

    missing = sorted({path for path in _required_sources(runs) if not path.is_file()})
    if missing:
        rendered = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Candidate bundle inputs are incomplete:\n{rendered}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.building-", dir=destination.parent)
    )
    try:
        for spec in RESULT_FILE_MAP:
            _copy_spec(runs, temporary, spec)
        for spec in MANIFEST_FILE_MAP:
            _copy_spec(runs, temporary, spec)
        for spec in STATIC_MANIFEST_FILE_MAP:
            _copy_spec(REFERENCE_ROOT, temporary, spec)
        if COHORT_INPUTS:
            _write_cohort_manifest(
                runs, temporary / "manifests/cohort_manifest.csv"
            )
        _write_environment_manifests(temporary / "manifests")
        if COHORT_CHARACTERISTIC_INPUTS:
            _write_cohort_characteristics(
                runs, temporary / "results/cohort_characteristics.csv"
            )
        evaluation_case_ids = _evaluation_case_id_map(runs)
        for spec in SANITIZED_RESULT_SPECS:
            sanitize_prediction_csv(
                runs / spec.source,
                temporary / spec.destination,
                spec,
                evaluation_case_ids,
            )
        for spec in PREDICTION_SPECS:
            sanitize_prediction_csv(
                runs / spec.source,
                temporary / spec.destination,
                spec,
                evaluation_case_ids,
            )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        "status": "built",
        "bundle_root": str(destination),
        "manifest_files": (
            len(MANIFEST_FILE_MAP)
            + len(STATIC_MANIFEST_FILE_MAP)
            + 2
            + int(bool(COHORT_INPUTS))
        ),
        "prediction_files": len(PREDICTION_SPECS),
        "result_files": (
            len(RESULT_FILE_MAP)
            + len(SANITIZED_RESULT_SPECS)
            + int(bool(COHORT_CHARACTERISTIC_INPUTS))
        ),
    }


PAPER_KEY_COLUMNS = {
    "analysis",
    "assignment_rule",
    "analogue_availability",
    "analogue_threshold_quantile",
    "catalog_eps_m",
    "characteristic",
    "cohort",
    "destination_popularity_stratum",
    "display_name",
    "first_method",
    "history_size_stratum",
    "method",
    "metric",
    "observation_definition",
    "prefix_familiarity",
    "quantity",
    "ratio",
    "ratio_or_change",
    "reference_method",
    "requested_k",
    "second_method",
    "seed",
    "stratum",
    "user_id",
    "unit",
    "variant",
}

EXACT_NUMERIC_COLUMNS = {
    "cases",
    "discovery_cases",
    "effective_k",
    "n",
    "replicates",
    "seeds",
    "trips",
    "user_case_sum",
    "users",
}

PUBLISHED_PRECISION_OVERRIDES = {
    ("paper_tables/training_only_ambiguity.csv", "mean_entropy_nats"): 3,
}


def _paper_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.endswith((".csv", ".csv.gz", ".svg")):
            files[path.relative_to(root).as_posix()] = path
    return files


def _string_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("<NA>")


def _numeric_series(series: pd.Series) -> pd.Series | None:
    text = _string_values(series)
    nonempty = ~text.isin({"", "<NA>"})
    numeric = pd.to_numeric(series, errors="coerce")
    if bool(numeric[nonempty].notna().all()):
        return numeric.astype(float)
    return None


def _formatted_number(value: float, digits: int) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return f"{value:.{digits}f}"


def _compare_csv(
    relative: str,
    candidate_path: Path,
    frozen_path: Path,
    published_digits: int,
) -> list[dict[str, object]]:
    differences: list[dict[str, object]] = []
    candidate = pd.read_csv(candidate_path)
    frozen = pd.read_csv(frozen_path)
    if list(candidate.columns) != list(frozen.columns):
        return [
            {
                "file": relative,
                "kind": "schema",
                "message": "CSV columns differ",
                "candidate": list(candidate.columns),
                "frozen": list(frozen.columns),
            }
        ]
    if len(candidate) != len(frozen):
        return [
            {
                "file": relative,
                "kind": "row_count",
                "message": f"row count differs: {len(candidate)} != {len(frozen)}",
            }
        ]

    key_columns = [column for column in candidate.columns if column in PAPER_KEY_COLUMNS]
    if not key_columns:
        candidate = candidate.assign(__row_number__=range(len(candidate)))
        frozen = frozen.assign(__row_number__=range(len(frozen)))
        key_columns = ["__row_number__"]
    candidate_keys = candidate[key_columns].apply(_string_values)
    frozen_keys = frozen[key_columns].apply(_string_values)
    if candidate_keys.duplicated().any() or frozen_keys.duplicated().any():
        differences.append(
            {
                "file": relative,
                "kind": "keys",
                "message": f"comparison keys are not unique: {key_columns}",
            }
        )
        return differences

    candidate_index = pd.MultiIndex.from_frame(candidate_keys)
    frozen_index = pd.MultiIndex.from_frame(frozen_keys)
    candidate_key_set = set(candidate_index.tolist())
    frozen_key_set = set(frozen_index.tolist())
    if candidate_key_set != frozen_key_set:
        differences.append(
            {
                "file": relative,
                "kind": "keys",
                "message": "row keys differ",
                "missing_from_candidate": [str(value) for value in sorted(frozen_key_set - candidate_key_set)[:10]],
                "unexpected_in_candidate": [str(value) for value in sorted(candidate_key_set - frozen_key_set)[:10]],
            }
        )
        return differences

    candidate = candidate.set_index(candidate_index).sort_index()
    frozen = frozen.set_index(frozen_index).sort_index()
    compare_columns = [
        column for column in candidate.columns if column not in set(key_columns)
    ]
    for column in compare_columns:
        candidate_numeric = _numeric_series(candidate[column])
        frozen_numeric = _numeric_series(frozen[column])
        if candidate_numeric is not None and frozen_numeric is not None:
            if column in EXACT_NUMERIC_COLUMNS:
                equal = (
                    (candidate_numeric == frozen_numeric)
                    | (candidate_numeric.isna() & frozen_numeric.isna())
                )
            else:
                digits = PUBLISHED_PRECISION_OVERRIDES.get(
                    (relative, column), published_digits
                )
                candidate_rounded = candidate_numeric.map(
                    lambda value: _formatted_number(value, digits)
                )
                frozen_rounded = frozen_numeric.map(
                    lambda value: _formatted_number(value, digits)
                )
                equal = candidate_rounded == frozen_rounded
        else:
            equal = _string_values(candidate[column]) == _string_values(frozen[column])
        if not bool(equal.all()):
            positions = list(candidate.index[~equal][:5])
            differences.append(
                {
                    "file": relative,
                    "kind": "values",
                    "column": column,
                    "message": f"{int((~equal).sum())} values differ in {column}",
                    "examples": [
                        {
                            "key": str(position),
                            "candidate": str(candidate.loc[position, column]),
                            "frozen": str(frozen.loc[position, column]),
                        }
                        for position in positions
                    ],
                }
            )
    return differences


def compare_paper_outputs(
    candidate_output: Path,
    frozen_output: Path,
    *,
    published_digits: int = 2,
    report_path: Path | None = None,
    raise_on_difference: bool = True,
) -> dict[str, object]:
    """Compare generated paper tables/figures with the frozen source of truth."""

    candidate_root = Path(candidate_output).expanduser().resolve()
    frozen_root = Path(frozen_output).expanduser().resolve()
    candidate_files = _paper_files(candidate_root)
    frozen_files = _paper_files(frozen_root)
    differences: list[dict[str, object]] = []

    candidate_names = set(candidate_files)
    frozen_names = set(frozen_files)
    if candidate_names != frozen_names:
        differences.append(
            {
                "kind": "file_set",
                "message": "paper output file sets differ",
                "missing_from_candidate": sorted(frozen_names - candidate_names),
                "unexpected_in_candidate": sorted(candidate_names - frozen_names),
            }
        )

    compared = 0
    for relative in sorted(candidate_names & frozen_names):
        candidate_path = candidate_files[relative]
        frozen_path = frozen_files[relative]
        compared += 1
        if relative.endswith((".csv", ".csv.gz")):
            differences.extend(
                _compare_csv(relative, candidate_path, frozen_path, published_digits)
            )
        elif candidate_path.read_bytes() != frozen_path.read_bytes():
            differences.append(
                {
                    "file": relative,
                    "kind": "bytes",
                    "message": "generated figure bytes differ",
                }
            )

    report: dict[str, object] = {
        "status": "passed" if not differences else "failed",
        "published_digits": int(published_digits),
        "files_compared": compared,
        "differences": differences,
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if differences and raise_on_difference:
        raise ArtifactComparisonError(report)
    return report


__all__ = [
    "ArtifactComparisonError",
    "CopySpec",
    "MANIFEST_FILE_MAP",
    "PREDICTION_SPECS",
    "PredictionSpec",
    "RESULT_FILE_MAP",
    "SANITIZED_RESULT_SPECS",
    "build_candidate_bundle",
    "compare_paper_outputs",
    "sanitize_prediction_csv",
    "stable_trajectory_id",
]
