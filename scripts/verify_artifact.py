#!/usr/bin/env python3
"""Verify release integrity and the numerical claims used in the manuscript."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
RATIOS = [0.25, 0.50, 0.66, 0.75]
FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        FAILURES.append(message)


def close(actual: float, expected: float, digits: int, label: str) -> None:
    check(round(actual, digits) == round(expected, digits), f"{label}: {actual} != {expected}")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_checksums() -> None:
    manifest = ROOT / "SHA256SUMS"
    check(manifest.is_file(), "SHA256SUMS is missing")
    if not manifest.is_file():
        return
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        check(path.is_file(), f"Checksummed file is missing: {relative}")
        if path.is_file():
            check(digest(path) == expected, f"Checksum mismatch: {relative}")


def read_csv(relative: str) -> list[dict[str, str]]:
    path = REFERENCE / relative
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_public_cases() -> None:
    expected_rows = {
        "predictions/cross_method_cases.csv.gz": 283200,
        "predictions/geometric_distance_cases.csv.gz": 17700,
        "predictions/matched_grid_cases.csv.gz": 23600,
        "predictions/tie_sensitivity_cases.csv.gz": 59000,
        "predictions/tie_boundary_cases.csv.gz": 11800,
        "predictions/matched_familiarity_cases.csv.gz": 5900,
        "predictions/matched_robustness_evaluation37_cases.csv.gz": 35400,
        "predictions/matched_robustness_expanded46_cases.csv.gz": 23104,
        "predictions/ambiguity_prediction_cases.csv.gz": 106200,
        "predictions/ambiguity_binned_cases.csv.gz": 17700,
    }
    forbidden = ("_lon", "_lat", "longitude", "latitude", "trip_id", "runtime")
    for relative, count in expected_rows.items():
        path = REFERENCE / relative
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = sum(1 for _ in reader)
        check(rows == count, f"Unexpected row count in {relative}: {rows}")
        for column in header:
            lowered = column.lower()
            check(not any(token in lowered for token in forbidden), f"Non-public column {column} in {relative}")


def verify_dataset(expected: dict[str, object]) -> None:
    summary = json.loads(
        (REFERENCE / "manifests" / "preparation_summary.json").read_text(encoding="utf-8")
    )
    mapping = {
        "source_users": "source_users",
        "source_plt_files": "source_plt_files",
        "eligible_whole_trajectories": "eligible_whole_trajectories_before_user_selection",
        "eligible_users": "users_meeting_minimum_history",
        "evaluation_users": "selected_user_count",
        "evaluation_trajectories": "selected_trajectories",
    }
    for claim, source_key in mapping.items():
        check(summary[source_key] == expected[claim], f"Dataset claim mismatch: {claim}")
    partitions = summary["partition_counts"]
    check(partitions["fit"] == expected["fit_trajectories"], "Fit count mismatch")
    check(partitions["validation"] == expected["validation_trajectories"], "Validation count mismatch")
    check(partitions["test"] == expected["test_trajectories"], "Test count mismatch")

    manifest_lines = (
        REFERENCE / "manifests" / "geolife_1.3_files.sha256"
    ).read_text(encoding="utf-8").splitlines()
    plt_lines = [line for line in manifest_lines if line.lower().endswith(".plt")]
    check(len(manifest_lines) == 18740, "GeoLife manifest must contain 18,740 files")
    check(len(plt_lines) == expected["source_plt_files"], "GeoLife PLT manifest count mismatch")


def indexed(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def verify_main_rates(expected: dict[str, list[float]]) -> None:
    canonical = indexed(
        read_csv("results/matched_robustness/canonical_cross_method_metrics.csv"),
        ("method", "ratio"),
    )
    matched = read_csv("results/matched_robustness/matched_robustness_bootstrap.csv")
    last = {
        str(float(row["ratio_or_change"])): row
        for row in matched
        if row["analysis"] == "held_out_point_count"
        and row["metric"] == "hit_r90_all"
        and row["quantity"] == "rate"
        and row["first_method"] == "grid_last_state_only"
    }
    for method, values in expected.items():
        for ratio, expected_value in zip(RATIOS, values):
            ratio_key = str(float(ratio))
            if method == "grid_last_state_only":
                actual = float(last[ratio_key]["observed"])
            else:
                actual = 100 * float(canonical[(method, ratio_key)]["user_macro_hit_r90"])
            close(actual, expected_value, 2, f"Hit@R90 {method} {ratio}")


def verify_matched(expected: dict[str, list[float]]) -> None:
    rows = read_csv("results/matched_robustness/matched_robustness_bootstrap.csv")
    selected = {}
    for row in rows:
        if row["analysis"] != "held_out_point_count" or row["metric"] != "hit_r90_all":
            continue
        if row["first_method"] != "grid_full_prefix" or row["second_method"] != "grid_last_state_only":
            continue
        if row["quantity"] in {"paired_difference", "mean_difference", "difference_in_change"}:
            key = row["ratio_or_change"]
            if row["quantity"] == "mean_difference":
                key = "mean"
            selected[key] = row
    for key, values in expected.items():
        row = selected[key]
        for field, expected_value in zip(("observed", "ci_low", "ci_high"), values):
            close(float(row[field]), expected_value, 2, f"Matched {key} {field}")


def verify_ambiguity(expected: dict[str, list[float]]) -> None:
    rows = read_csv("predictions/ambiguity_binned_cases.csv.gz")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["requested_k"], row["ratio"])].append(float(row["entropy_nats"]))
    for k, values in expected.items():
        for ratio, expected_value in zip(RATIOS, values):
            samples = grouped[(k, str(ratio))]
            actual = sum(samples) / len(samples)
            close(actual, expected_value, 3, f"Ambiguity k={k} ratio={ratio}")


def verify_ties(expected: dict[str, list[float]]) -> None:
    rows = indexed(
        read_csv("results/tie_sensitivity/paired_tie_sensitivity_bootstrap.csv"),
        ("variant", "quantity", "ratio_or_change"),
    )
    for variant, values in expected.items():
        row = rows[(variant, "difference_in_change", "75_minus_25")]
        for field, expected_value in zip(("observed", "ci_low", "ci_high"), values):
            close(float(row[field]), expected_value, 2, f"Tie sensitivity {variant} {field}")


def verify_observation_definitions(expected: dict[str, list[float]]) -> None:
    aliases = {
        "point_count": "held_out_point_count",
        "elapsed_time": "held_out_elapsed_time",
        "cumulative_distance": "held_out_cumulative_distance",
    }
    rows = read_csv("results/matched_robustness/matched_robustness_bootstrap.csv")
    for definition, values in expected.items():
        matches = [
            row
            for row in rows
            if row["analysis"] == aliases[definition]
            and row["metric"] == "hit_r90_all"
            and row["quantity"] == "difference_in_change"
            and row["ratio_or_change"] == "75_minus_25"
        ]
        check(len(matches) == 1, f"Missing observation definition {definition}")
        if matches:
            for field, expected_value in zip(("observed", "ci_low", "ci_high"), values):
                close(float(matches[0][field]), expected_value, 2, f"Observation {definition} {field}")


def verify_familiarity_counts(expected: dict[str, list[int]]) -> None:
    rows = read_csv("results/matched_familiarity/matched_familiarity_bootstrap.csv")
    table = indexed(rows, ("prefix_familiarity", "ratio"))
    for stratum, counts in expected.items():
        for ratio, count in zip(RATIOS, counts):
            check(int(table[(stratum, str(ratio))]["cases"]) == count, f"Familiarity count {stratum} {ratio}")


def verify_miscellaneous() -> None:
    large = [path for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size >= 50 * 1024 * 1024]
    check(not large, "Files at or above 50 MiB require Git LFS: " + ", ".join(map(str, large)))
    check(
        (REFERENCE / "results" / "matched_ablation" / "bootstrap_resample_ids.npz").is_file(),
        "Saved bootstrap plan is missing",
    )
    check((REFERENCE / "manifests" / "partition_manifest.csv.gz").is_file(), "Partition manifest is missing")
    check((REFERENCE / "manifests" / "source_trajectory_audit.csv.gz").is_file(), "Source audit is missing")
    cohort_rows = read_csv("manifests/cohort_manifest.csv")
    required_cohort_columns = {
        "eligibility",
        "discovery_membership",
        "prior_use",
        "evaluation_membership",
        "exclusion_reason",
    }
    check(bool(cohort_rows), "Normalized cohort manifest is empty")
    if cohort_rows:
        check(
            required_cohort_columns.issubset(cohort_rows[0]),
            "Normalized cohort manifest is missing required provenance columns",
        )

    registry = json.loads(
        (REFERENCE / "manifests" / "run_registry.json").read_text(encoding="utf-8")
    )
    check(len(registry["runs"]) == 10, "Run registry must describe ten reported runs")
    for run in registry["runs"]:
        check((ROOT / run["config"]).is_file(), f"Missing registered config: {run['config']}")

    required_source = [
        "src/destination_prediction/context.py",
        "src/destination_prediction/protocol.py",
        "src/destination_prediction/preprocessing.py",
        "src/destination_prediction/catalogue.py",
        "src/destination_prediction/geometry.py",
        "src/destination_prediction/metrics.py",
        "src/destination_prediction/bootstrap.py",
        "src/destination_prediction/methods/grid_pattern_retrieval.py",
        "src/destination_prediction/evaluate.py",
        "scripts/reproduce.py",
        "scripts/verify_protocol.py",
    ]
    for relative in required_source:
        check((ROOT / relative).is_file(), f"Missing shared implementation: {relative}")
    check(not (ROOT / "experiments").exists(), "Duplicated historical experiment tree is present")

    def strings(value: object):
        if isinstance(value, dict):
            for child in value.values():
                yield from strings(child)
        elif isinstance(value, list):
            for child in value:
                yield from strings(child)
        elif isinstance(value, str):
            yield value

    for config_path in sorted((ROOT / "configs").glob("*.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for value in strings(config):
            check(
                not value.startswith("experiments/"),
                f"Historical experiment path in {config_path.name}: {value}",
            )

    forbidden_host_tokens = (
        "/" + "Users/",
        "/" + "home/" + "fila/",
        "100." + "79." + "183.108",
        "100." + "82." + "212.121",
    )
    text_suffixes = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yml", ".yaml", ".sh"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS" or "tests" in path.parts:
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden_host_tokens:
            check(token not in content, f"Host-specific path or address in {path.relative_to(ROOT)}")


def main() -> int:
    expected = json.loads((ROOT / "results/expected_claims.json").read_text(encoding="utf-8"))
    verify_checksums()
    verify_public_cases()
    verify_dataset(expected["dataset"])
    verify_main_rates(expected["main_user_macro_hit_r90_percent"])
    verify_matched(expected["matched_full_minus_last_percent_points"])
    verify_ambiguity(expected["ambiguity_mean_entropy_nats"])
    verify_ties(expected["tie_stage_contrast_percent_points"])
    verify_observation_definitions(expected["alternative_observation_stage_contrast_percent_points"])
    verify_familiarity_counts(expected["matched_familiarity_case_counts"])
    verify_miscellaneous()
    report = {"status": "passed" if not FAILURES else "failed", "checks": CHECKS, "failures": FAILURES}
    print(json.dumps(report, indent=2))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
