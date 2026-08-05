#!/usr/bin/env python3
"""Deterministic matched-grid robustness analyses for the EDBT paper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext
from destination_prediction.protocol import EvaluationProtocol


CONTEXT = RunContext.from_environment()
EXP_DIR = CONTEXT.run_directory
CONFIG = CONTEXT.config
HELD = CONTEXT.dependency_directory(CONFIG["held_out_source"])
EXPANDED = CONTEXT.dependency_directory(CONFIG["expanded_source"])
TIE = CONTEXT.dependency_directory(CONFIG["tie_source"])
BOOTSTRAP = CONTEXT.dependency_directory(CONFIG["bootstrap_source"])
OUTPUT = EXP_DIR / "outputs"
FULL = str(CONFIG["methods"]["full"])
LAST = str(CONFIG["methods"]["last"])
RATIOS = list(map(float, CONFIG["observation_ratios"]))


def load_protocol(source: str, name: str):
    del name
    return EvaluationProtocol(CONTEXT.dependency_context(source))


def progress_cutoff(
    local_sequence: list[list[float]],
    time_sequence: list[float],
    ratio: float,
    definition: str,
    protocol,
) -> int:
    length = len(local_sequence)
    if definition == "point_count":
        return int(protocol.prefix_cutoff(length, ratio))
    if definition == "elapsed_time":
        times = np.asarray(time_sequence, dtype=float)
        elapsed = np.maximum.accumulate(times - times[0])
        if elapsed[-1] <= 0:
            return int(protocol.prefix_cutoff(length, ratio))
        position = int(
            np.searchsorted(elapsed, ratio * float(elapsed[-1]), side="left")
        )
        return min(length, max(2, position + 1))
    if definition == "cumulative_distance":
        coordinates = np.asarray(local_sequence, dtype=float)
        segment = np.linalg.norm(coordinates[1:] - coordinates[:-1], axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment)])
        if cumulative[-1] <= 0:
            return int(protocol.prefix_cutoff(length, ratio))
        position = int(
            np.searchsorted(
                cumulative, ratio * float(cumulative[-1]), side="left"
            )
        )
        return min(length, max(2, position + 1))
    raise ValueError(definition)


def resampled_cells(
    row,
    ratio: float,
    definition: str,
    protocol,
    cell_size: float,
    states: int,
) -> np.ndarray:
    cutoff = progress_cutoff(
        row.local_seq, row.time_seq_s, ratio, definition, protocol
    )
    prefix = np.asarray(row.local_seq, dtype=float)[:cutoff]
    cells = np.floor(prefix / cell_size)
    keep = np.r_[True, np.any(cells[1:] != cells[:-1], axis=1)]
    return protocol.resample_path(cells[keep], states)


def emit_region(
    squared: np.ndarray,
    trip_ids: np.ndarray,
    regions: np.ndarray,
    sigma: float,
    top_k: int,
) -> int:
    scores = -0.5 * squared / max(sigma**2, 1e-12)
    scores -= float(np.max(scores))
    weights = np.exp(scores)
    weights /= float(weights.sum())
    order = np.lexsort((trip_ids.astype(str), squared))
    selected = order[: min(top_k, len(order))]
    mass: dict[int, float] = {}
    for position in selected:
        region = int(regions[position])
        mass[region] = mass.get(region, 0.0) + float(weights[position])
    return int(sorted(mass, key=lambda region: (-mass[region], region))[0])


def predict(
    source: Path,
    protocol,
    cohort: str,
    definitions: list[str],
) -> pd.DataFrame:
    marked = protocol.load_marked()
    train, test = protocol.partitions(marked, "outer")
    origins = protocol.user_origins(train)
    train = protocol.add_local_sequences(train, origins)
    test = protocol.add_local_sequences(test, origins)
    catalog = protocol.build_catalog(
        train, float(protocol.CONFIG["task"]["primary_dbscan_eps_m"])
    )
    cluster_map = protocol.endpoint_cluster_map(train, catalog)
    train_by_user = {
        int(user): frame.reset_index(drop=True)
        for user, frame in train.groupby("user_id", sort=True)
    }
    centers_by_user = {
        int(user): frame.set_index("center_id")
        for user, frame in catalog.groupby("user_id", sort=True)
    }
    cfg = CONFIG["grid_configuration"]
    cell_size = float(cfg["cell_size_m"])
    sigma = float(cfg["emission_sigma_cells"])
    top_k = int(cfg["posterior_top_k"])
    states = int(cfg["alignment_states"])
    rows: list[dict[str, object]] = []

    for definition in definitions:
        for ratio in RATIOS:
            patterns = {
                user: np.stack(
                    [
                        resampled_cells(
                            row,
                            ratio,
                            definition,
                            protocol,
                            cell_size,
                            states,
                        )
                        for row in frame.itertuples(index=False)
                    ]
                )
                for user, frame in train_by_user.items()
            }
            regions = {
                user: np.asarray(
                    [
                        cluster_map[(user, str(trip_id))]
                        for trip_id in frame["trip_id"]
                    ],
                    dtype=int,
                )
                for user, frame in train_by_user.items()
            }
            identifiers = {
                user: frame["trip_id"].astype(str).to_numpy()
                for user, frame in train_by_user.items()
            }
            for test_row in test.itertuples(index=False):
                user = int(test_row.user_id)
                query = resampled_cells(
                    test_row,
                    ratio,
                    definition,
                    protocol,
                    cell_size,
                    states,
                )
                difference = patterns[user] - query[None, :, :]
                distances = {
                    FULL: np.square(difference).sum(axis=2).mean(axis=1),
                    LAST: np.square(difference[:, -1, :]).sum(axis=1),
                }
                for method, squared in distances.items():
                    region = emit_region(
                        squared,
                        identifiers[user],
                        regions[user],
                        sigma,
                        top_k,
                    )
                    center = centers_by_user[user].loc[region]
                    rows.append(
                        {
                            "cohort": cohort,
                            "observation_definition": definition,
                            "method": method,
                            "ratio": ratio,
                            "user_id": user,
                            "trip_id": str(test_row.trip_id),
                            "native_pred_lon": float(center.medoid_lon),
                            "native_pred_lat": float(center.medoid_lat),
                        }
                    )

    native = pd.DataFrame(rows)
    evaluated_parts = []
    for definition, frame in native.groupby(
        "observation_definition", sort=False
    ):
        evaluated = protocol.evaluate_native_predictions(
            frame.drop(columns=["cohort", "observation_definition"]),
            test,
            catalog,
        )
        evaluated.insert(0, "observation_definition", definition)
        evaluated.insert(0, "cohort", cohort)
        evaluated_parts.append(evaluated)
    return pd.concat(evaluated_parts, ignore_index=True)


def reused_point_count_predictions() -> pd.DataFrame:
    """Reuse the deterministic point-count cases produced by tie sensitivity."""
    cases = pd.read_csv(
        TIE / "outputs/tie_sensitivity_predictions.csv",
        dtype={"trip_id": str},
    )
    cases = cases[
        (cases["variant"] == "id_top10")
        & cases["method"].isin([FULL, LAST])
    ].copy()
    cases = cases.drop(
        columns=["variant", "selected_pool_size", "destination_mass_tie"],
        errors="ignore",
    )
    cases.insert(0, "observation_definition", "point_count")
    cases.insert(0, "cohort", "held_out_37")
    expected = 2 * len(RATIOS) * 1475
    if len(cases) != expected:
        raise RuntimeError(
            f"Canonical point-count cache contains {len(cases)} rows; expected {expected}."
        )
    return cases


def arrays_from_cases(
    cases: pd.DataFrame,
    metric: str,
    methods: list[str],
    case_index: pd.DataFrame | None = None,
) -> tuple[list[np.ndarray], list[tuple[str, float]], list[int]]:
    selected = cases[cases["method"].isin(methods)].copy()
    selected["trip_id"] = selected["trip_id"].astype(str)
    pivot = selected.pivot(
        index=["user_id", "trip_id"],
        columns=["method", "ratio"],
        values=metric,
    ).sort_index()
    if pivot.isna().any().any():
        raise RuntimeError("Incomplete four-ratio method matrix.")
    arrays: list[np.ndarray] = []
    users: list[int] = []
    if case_index is None:
        for user in sorted(pivot.index.get_level_values("user_id").unique()):
            users.append(int(user))
            arrays.append(
                pivot.xs(int(user), level="user_id")
                .sort_index()
                .to_numpy(dtype=float)
            )
    else:
        case_index = case_index.copy()
        case_index["trip_id"] = case_index["trip_id"].astype(str)
        for user_position, user in enumerate(
            case_index.sort_values("user_position")["user_id"].drop_duplicates()
        ):
            expected = case_index.loc[
                case_index["user_position"] == user_position, "trip_id"
            ].tolist()
            frame = pivot.xs(int(user), level="user_id").sort_index()
            if frame.index.astype(str).tolist() != expected:
                raise RuntimeError(f"Saved case order mismatch for user {user}.")
            users.append(int(user))
            arrays.append(frame.to_numpy(dtype=float))
    return arrays, list(pivot.columns), users


def saved_estimates(arrays: list[np.ndarray]) -> np.ndarray:
    plan = np.load(BOOTSTRAP / "outputs/bootstrap_resample_ids.npz")
    user_draws = plan["user_draw_positions"]
    within_draws = plan["within_user_draw_positions"]
    offsets = plan["within_user_draw_offsets"]
    estimates = np.empty((len(user_draws), arrays[0].shape[1]), dtype=float)
    slots = user_draws.shape[1]
    for replicate in range(len(user_draws)):
        aggregate = np.zeros(arrays[0].shape[1], dtype=float)
        for slot in range(slots):
            draw_index = replicate * slots + slot
            start, end = int(offsets[draw_index]), int(offsets[draw_index + 1])
            user_position = int(user_draws[replicate, slot])
            aggregate += arrays[user_position][within_draws[start:end]].mean(axis=0)
        estimates[replicate] = aggregate / slots
    return estimates


def generated_estimates(
    arrays: list[np.ndarray], seed: int
) -> np.ndarray:
    replicates = int(CONFIG["bootstrap"]["replicates"])
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates, arrays[0].shape[1]), dtype=float)
    user_count = len(arrays)
    for replicate in range(replicates):
        sampled_users = rng.integers(0, user_count, size=user_count)
        aggregate = np.zeros(arrays[0].shape[1], dtype=float)
        for user_position in sampled_users:
            values = arrays[int(user_position)]
            positions = rng.integers(0, len(values), size=len(values))
            aggregate += values[positions].mean(axis=0)
        estimates[replicate] = aggregate / user_count
    return estimates


def summarize_bootstrap(
    label: str,
    metric: str,
    arrays: list[np.ndarray],
    columns: list[tuple[str, float]],
    estimates: np.ndarray,
    methods: list[str],
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    column_index = {key: position for position, key in enumerate(columns)}
    observed = np.mean(
        np.stack([array.mean(axis=0) for array in arrays]), axis=0
    )
    rows: list[dict[str, object]] = []
    method_ratio_values: dict[str, list[np.ndarray]] = {
        method: [] for method in methods
    }
    method_ratio_observed: dict[str, list[float]] = {
        method: [] for method in methods
    }
    for method in methods:
        for ratio in RATIOS:
            position = column_index[(method, ratio)]
            values = estimates[:, position] * 100.0
            value = float(observed[position] * 100.0)
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "analysis": label,
                    "metric": metric,
                    "quantity": "rate",
                    "first_method": method,
                    "second_method": "",
                    "ratio_or_change": f"{ratio:.2f}",
                    "observed": value,
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
            method_ratio_values[method].append(values)
            method_ratio_observed[method].append(value)
        mean_values = np.mean(
            np.column_stack(method_ratio_values[method]), axis=1
        )
        low, high = np.quantile(mean_values, [0.025, 0.975])
        rows.append(
            {
                "analysis": label,
                "metric": metric,
                "quantity": "mean_rate",
                "first_method": method,
                "second_method": "",
                "ratio_or_change": "mean",
                "observed": float(np.mean(method_ratio_observed[method])),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )

    for first, second in pairs:
        differences: list[np.ndarray] = []
        observed_differences: list[float] = []
        for ratio in RATIOS:
            first_position = column_index[(first, ratio)]
            second_position = column_index[(second, ratio)]
            values = (
                estimates[:, first_position] - estimates[:, second_position]
            ) * 100.0
            value = float(
                (observed[first_position] - observed[second_position]) * 100.0
            )
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "analysis": label,
                    "metric": metric,
                    "quantity": "paired_difference",
                    "first_method": first,
                    "second_method": second,
                    "ratio_or_change": f"{ratio:.2f}",
                    "observed": value,
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
            differences.append(values)
            observed_differences.append(value)
        mean_values = np.mean(np.column_stack(differences), axis=1)
        low, high = np.quantile(mean_values, [0.025, 0.975])
        rows.append(
            {
                "analysis": label,
                "metric": metric,
                "quantity": "mean_difference",
                "first_method": first,
                "second_method": second,
                "ratio_or_change": "mean",
                "observed": float(np.mean(observed_differences)),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
        stage = differences[-1] - differences[0]
        low, high = np.quantile(stage, [0.025, 0.975])
        rows.append(
            {
                "analysis": label,
                "metric": metric,
                "quantity": "difference_in_change",
                "first_method": first,
                "second_method": second,
                "ratio_or_change": "75_minus_25",
                "observed": float(
                    observed_differences[-1] - observed_differences[0]
                ),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def matched_analysis(
    held: pd.DataFrame, expanded: pd.DataFrame
) -> pd.DataFrame:
    case_index = pd.read_csv(BOOTSTRAP / "outputs/bootstrap_case_index.csv")
    rows = []
    for definition in CONFIG["observation_definitions"]:
        cases = held[held["observation_definition"] == definition]
        arrays, columns, _ = arrays_from_cases(
            cases, "hit_r90_all", [FULL, LAST], case_index
        )
        estimates = saved_estimates(arrays)
        rows.append(
            summarize_bootstrap(
                f"held_out_{definition}",
                "hit_r90_all",
                arrays,
                columns,
                estimates,
                [FULL, LAST],
                [(FULL, LAST)],
            )
        )
    point = held[held["observation_definition"] == "point_count"].copy()
    arrays, columns, _ = arrays_from_cases(
        point, "within_200m", [FULL, LAST], case_index
    )
    rows.append(
        summarize_bootstrap(
            "held_out_point_count",
            "within_200m",
            arrays,
            columns,
            saved_estimates(arrays),
            [FULL, LAST],
            [(FULL, LAST)],
        )
    )

    quality_flags = pd.read_csv(
        HELD / "outputs/final/common_predictions_all_sensitivities.csv",
        usecols=["user_id", "trip_id", "quality_sensitivity_eligible"],
        dtype={"trip_id": str},
    ).drop_duplicates()
    point["trip_id"] = point["trip_id"].astype(str)
    quality = point.merge(
        quality_flags,
        on=["user_id", "trip_id"],
        validate="many_to_one",
    )
    quality = quality[quality["quality_sensitivity_eligible"].astype(bool)]
    arrays, columns, _ = arrays_from_cases(
        quality, "hit_r90_all", [FULL, LAST]
    )
    rows.append(
        summarize_bootstrap(
            "held_out_quality_point_count",
            "hit_r90_all",
            arrays,
            columns,
            generated_estimates(
                arrays, int(CONFIG["bootstrap"]["seed"]) + 101
            ),
            [FULL, LAST],
            [(FULL, LAST)],
        )
    )

    arrays, columns, _ = arrays_from_cases(
        expanded, "hit_r90_all", [FULL, LAST]
    )
    rows.append(
        summarize_bootstrap(
            "expanded46_point_count",
            "hit_r90_all",
            arrays,
            columns,
            generated_estimates(
                arrays, int(CONFIG["bootstrap"]["seed"]) + 202
            ),
            [FULL, LAST],
            [(FULL, LAST)],
        )
    )
    return pd.concat(rows, ignore_index=True)


def canonical_cross_method(
    held_point: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(
        HELD / "outputs/final/common_predictions_all_sensitivities.csv",
        dtype={"trip_id": str},
    )
    methods = [
        "probabilistic_grid_pattern_retrieval",
        "bigru",
        "geometric_retrieval",
        "tsmini",
        "current_position_only_personal_retrieval",
        "most_frequent_personal_destination",
        "nearest_known_destination_to_current_point",
    ]
    source = source[
        np.isclose(source["catalog_eps_m"], 200.0)
        & source["method"].isin(methods)
    ].copy()
    source["hit_r90_all"] = source["hit_r90_all"].astype(float)
    aggregated = (
        source.groupby(
            ["method", "ratio", "user_id", "trip_id"], as_index=False
        )["hit_r90_all"]
        .mean()
    )
    canonical = held_point[held_point["method"] == FULL][
        ["ratio", "user_id", "trip_id", "hit_r90_all"]
    ].copy()
    canonical["method"] = "probabilistic_grid_pattern_retrieval"
    aggregated = aggregated[
        aggregated["method"] != "probabilistic_grid_pattern_retrieval"
    ]
    cases = pd.concat([aggregated, canonical], ignore_index=True)
    case_index = pd.read_csv(BOOTSTRAP / "outputs/bootstrap_case_index.csv")
    arrays, columns, _ = arrays_from_cases(
        cases, "hit_r90_all", methods, case_index
    )
    estimates = saved_estimates(arrays)
    pairs = [
        ("probabilistic_grid_pattern_retrieval", method)
        for method in methods
        if method != "probabilistic_grid_pattern_retrieval"
    ]
    summary = summarize_bootstrap(
        "canonical_cross_method",
        "hit_r90_all",
        arrays,
        columns,
        estimates,
        methods,
        pairs,
    )
    per_user = (
        cases.groupby(["method", "ratio", "user_id"], as_index=False)
        .agg(hit_r90=("hit_r90_all", "mean"), cases=("trip_id", "size"))
    )
    metrics = (
        per_user.groupby(["method", "ratio"], as_index=False)
        .agg(
            user_macro_hit_r90=("hit_r90", "mean"),
            users=("user_id", "size"),
            cases=("cases", "sum"),
        )
    )
    return summary, metrics


def point_count_audit(held: pd.DataFrame) -> pd.DataFrame:
    rerun = held[held["observation_definition"] == "point_count"].copy()
    saved = pd.read_csv(
        TIE / "outputs/tie_sensitivity_predictions.csv",
        dtype={"trip_id": str},
    )
    saved = saved[saved["variant"] == "id_top10"].copy()
    keys = ["method", "ratio", "user_id", "trip_id"]
    rerun["trip_id"] = rerun["trip_id"].astype(str)
    audit = rerun[keys + ["pred_center_id"]].merge(
        saved[keys + ["pred_center_id"]],
        on=keys,
        how="outer",
        suffixes=("_rerun", "_saved"),
        indicator=True,
        validate="one_to_one",
    )
    audit["exact_region_match"] = (
        (audit["_merge"] == "both")
        & (audit["pred_center_id_rerun"] == audit["pred_center_id_saved"])
    )
    return audit


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    held_protocol = load_protocol(CONFIG["held_out_source"], "matched_robustness_held_protocol")
    expanded_protocol = load_protocol(
        CONFIG["expanded_source"], "matched_robustness_expanded_protocol"
    )
    held_alternatives = predict(
        HELD,
        held_protocol,
        "held_out_37",
        [
            definition
            for definition in CONFIG["observation_definitions"]
            if definition != "point_count"
        ],
    )
    held = pd.concat(
        [reused_point_count_predictions(), held_alternatives],
        ignore_index=True,
        sort=False,
    )
    expanded = predict(
        EXPANDED,
        expanded_protocol,
        "expanded_46",
        ["point_count"],
    )
    held.to_csv(OUTPUT / "held_out_predictions.csv", index=False)
    expanded.to_csv(OUTPUT / "expanded46_predictions.csv", index=False)
    audit = point_count_audit(held)
    audit.to_csv(OUTPUT / "canonical_point_count_audit.csv", index=False)
    matched = matched_analysis(held, expanded)
    matched.to_csv(OUTPUT / "matched_robustness_bootstrap.csv", index=False)
    cross, metrics = canonical_cross_method(
        held[held["observation_definition"] == "point_count"]
    )
    cross.to_csv(OUTPUT / "canonical_cross_method_bootstrap.csv", index=False)
    metrics.to_csv(OUTPUT / "canonical_cross_method_metrics.csv", index=False)
    summary = {
        "status": "completed",
        "held_out_users": int(held["user_id"].nunique()),
        "held_out_trajectories": int(
            held[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ),
        "expanded_users": int(expanded["user_id"].nunique()),
        "expanded_trajectories": int(
            expanded[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ),
        "canonical_point_count_matches": int(
            audit["exact_region_match"].sum()
        ),
        "canonical_point_count_expected": int(len(audit)),
        "bootstrap_replicates": int(CONFIG["bootstrap"]["replicates"]),
        "deterministic_tie_rule": True,
        "point_count_predictions_reused": True,
        "point_count_cache_source": "tie_sensitivity/id_top10",
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
