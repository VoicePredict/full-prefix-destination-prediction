#!/usr/bin/env python3
"""Tie prevalence and pool-rule sensitivity for the matched grid ablation."""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from destination_prediction.context import RunContext, parse_run_context
from destination_prediction.methods.grid_pattern_retrieval import (
    emit_destination_region,
    resampled_cells,
)
from destination_prediction.protocol import EvaluationProtocol


def _configure(context: RunContext) -> None:
    global CONTEXT, CONFIG, SOURCE, BASE_CONFIG, OUTPUT
    global FULL, LAST, RATIOS, ATOL, PROTOCOL
    CONTEXT = context
    CONFIG = context.config
    SOURCE = context.dependency_directory(CONFIG["source_experiment"])
    BASE_CONFIG = context.dependency_context(CONFIG["source_experiment"]).config
    OUTPUT = context.run_directory / "outputs"
    FULL = str(CONFIG["full_method"])
    LAST = str(CONFIG["ablation_method"])
    RATIOS = list(map(float, CONFIG["observation_ratios"]))
    ATOL = float(CONFIG["distance_tolerance"])
    PROTOCOL = EvaluationProtocol(
        context.dependency_context(BASE_CONFIG["source_experiment"])
    )


def boundary_diagnostic(squared: np.ndarray, top_k: int) -> dict[str, object]:
    limit = min(int(top_k), len(squared))
    ordered = np.sort(squared)
    boundary = float(ordered[limit - 1])
    strictly_closer = int(np.sum(squared < boundary - ATOL))
    equal = int(np.sum(np.isclose(squared, boundary, rtol=0.0, atol=ATOL)))
    straddles = bool(strictly_closer < limit < strictly_closer + equal)
    return {
        "boundary_distance": boundary,
        "strictly_closer": strictly_closer,
        "boundary_equal_candidates": equal,
        "boundary_tie_straddles_top_k": straddles,
    }


def predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    marked = PROTOCOL.load_marked()
    train, test = PROTOCOL.partitions(marked, "outer")
    origins = PROTOCOL.user_origins(train)
    train = PROTOCOL.add_local_sequences(train, origins)
    test = PROTOCOL.add_local_sequences(test, origins)
    catalog = PROTOCOL.build_catalog(
        train, float(PROTOCOL.CONFIG["task"]["primary_dbscan_eps_m"])
    )
    cluster_map = PROTOCOL.endpoint_catalogue_assignment(train, catalog)
    train_by_user = {
        int(user): frame.reset_index(drop=True)
        for user, frame in train.groupby("user_id", sort=True)
    }
    centers_by_user = {
        int(user): frame.set_index("center_id")
        for user, frame in catalog.groupby("user_id", sort=True)
    }
    cfg = BASE_CONFIG["frozen_grid_configuration"]
    cell_size = float(cfg["cell_size_m"])
    sigma = float(cfg["emission_sigma_cells"])
    states = int(cfg["alignment_states"])
    rows: list[dict[str, object]] = []
    tie_rows: list[dict[str, object]] = []

    for ratio in RATIOS:
        patterns = {
            user: np.stack(
                [
                    resampled_cells(seq, ratio, cell_size, states)
                    for seq in frame["local_seq"]
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
                test_row.local_seq, ratio, cell_size, states
            )
            difference = patterns[user] - query[None, :, :]
            distances = {
                FULL: np.square(difference).sum(axis=2).mean(axis=1),
                LAST: np.square(difference[:, -1, :]).sum(axis=1),
            }
            for method, squared in distances.items():
                diagnostic = boundary_diagnostic(squared, 10)
                tie_rows.append(
                    {
                        "method": method,
                        "ratio": ratio,
                        "user_id": user,
                        "trip_id": str(test_row.trip_id),
                        **diagnostic,
                    }
                )
                for variant in CONFIG["variants"]:
                    tie_rule = str(variant["tie_rule"])
                    region, _mass, pool_size, mass_tie = emit_destination_region(
                        squared,
                        identifiers[user],
                        regions[user],
                        sigma,
                        int(variant["top_k"]),
                        tie_rule,
                        ATOL,
                    )
                    if tie_rule == "legacy":
                        # The historical audit did not diagnose mass ties.
                        mass_tie = False
                    center = centers_by_user[user].loc[region]
                    rows.append(
                        {
                            "variant": str(variant["name"]),
                            "method": method,
                            "ratio": ratio,
                            "user_id": user,
                            "trip_id": str(test_row.trip_id),
                            "native_pred_lon": float(center.medoid_lon),
                            "native_pred_lat": float(center.medoid_lat),
                            "selected_pool_size": pool_size,
                            "destination_mass_tie": mass_tie,
                        }
                    )

    native = pd.DataFrame(rows)
    evaluated_parts = []
    for variant, frame in native.groupby("variant", sort=False):
        evaluated = PROTOCOL.evaluate_native_predictions(
            frame.drop(columns="variant"), test, catalog
        )
        evaluated.insert(0, "variant", variant)
        evaluated_parts.append(evaluated)
    return pd.concat(evaluated_parts, ignore_index=True), pd.DataFrame(tie_rows)


def metric_summary(cases: pd.DataFrame) -> pd.DataFrame:
    per_user = (
        cases.groupby(["variant", "method", "ratio", "user_id"], as_index=False)
        .agg(hit_r90=("hit_r90_all", "mean"), cases=("trip_id", "size"))
    )
    return (
        per_user.groupby(["variant", "method", "ratio"], as_index=False)
        .agg(
            user_macro_hit_r90=("hit_r90", "mean"),
            users=("user_id", "size"),
            cases=("cases", "sum"),
        )
    )


def arrays_for_variant(
    cases: pd.DataFrame, variant: str
) -> tuple[list[np.ndarray], list[tuple[str, float]]]:
    selected = cases[cases["variant"] == variant].copy()
    selected["trip_id"] = selected["trip_id"].astype(str)
    pivot = selected.pivot(
        index=["user_id", "trip_id"],
        columns=["method", "ratio"],
        values="hit_r90_all",
    ).sort_index()
    case_index = pd.read_csv(SOURCE / "outputs/bootstrap_case_index.csv")
    case_index["trip_id"] = case_index["trip_id"].astype(str)
    arrays: list[np.ndarray] = []
    for user_position, user in enumerate(
        case_index.sort_values("user_position")["user_id"].drop_duplicates()
    ):
        expected = case_index.loc[
            case_index["user_position"] == user_position, "trip_id"
        ].tolist()
        frame = pivot.xs(int(user), level="user_id").sort_index()
        if frame.index.astype(str).tolist() != expected:
            raise RuntimeError(f"Case order mismatch for user {user}.")
        arrays.append(frame.to_numpy(dtype=float))
    return arrays, list(pivot.columns)


def bootstrap_estimates(arrays: list[np.ndarray]) -> np.ndarray:
    plan = np.load(SOURCE / "outputs/bootstrap_resample_ids.npz")
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


def paired_bootstrap(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in [str(item["name"]) for item in CONFIG["variants"]]:
        arrays, columns = arrays_for_variant(cases, variant)
        column_index = {key: position for position, key in enumerate(columns)}
        estimates = bootstrap_estimates(arrays)
        observed = np.mean(
            np.stack([array.mean(axis=0) for array in arrays]), axis=0
        )
        ratio_values: list[np.ndarray] = []
        ratio_observed: list[float] = []
        for ratio in RATIOS:
            full_position = column_index[(FULL, ratio)]
            last_position = column_index[(LAST, ratio)]
            values = (
                estimates[:, full_position] - estimates[:, last_position]
            ) * 100.0
            value = float(
                (observed[full_position] - observed[last_position]) * 100.0
            )
            low, high = np.quantile(values, [0.025, 0.975])
            rows.append(
                {
                    "variant": variant,
                    "quantity": "paired_difference",
                    "ratio_or_change": f"{ratio:.2f}",
                    "observed": value,
                    "ci_low": float(low),
                    "ci_high": float(high),
                }
            )
            ratio_values.append(values)
            ratio_observed.append(value)
        stage = ratio_values[-1] - ratio_values[0]
        low, high = np.quantile(stage, [0.025, 0.975])
        rows.append(
            {
                "variant": variant,
                "quantity": "difference_in_change",
                "ratio_or_change": "75_minus_25",
                "observed": float(ratio_observed[-1] - ratio_observed[0]),
                "ci_low": float(low),
                "ci_high": float(high),
            }
        )
    return pd.DataFrame(rows)


def reproduction_audit(cases: pd.DataFrame) -> pd.DataFrame:
    legacy = cases[cases["variant"] == "legacy_top10"].copy()
    saved = pd.read_csv(
        SOURCE / "outputs/matched_grid_predictions.csv",
        dtype={"trip_id": str},
    )
    keys = ["method", "ratio", "user_id", "trip_id"]
    legacy["trip_id"] = legacy["trip_id"].astype(str)
    audit = legacy[keys + ["pred_center_id"]].merge(
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


def run(context: RunContext) -> int:
    _configure(context)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases, ties = predictions()
    cases.to_csv(OUTPUT / "tie_sensitivity_predictions.csv", index=False)
    ties.to_csv(OUTPUT / "top10_boundary_tie_cases.csv", index=False)
    metric_summary(cases).to_csv(OUTPUT / "method_metrics.csv", index=False)
    paired_bootstrap(cases).to_csv(
        OUTPUT / "paired_tie_sensitivity_bootstrap.csv", index=False
    )
    tie_summary = (
        ties.groupby(["method", "ratio"], as_index=False)
        .agg(
            boundary_tie_rate=("boundary_tie_straddles_top_k", "mean"),
            mean_boundary_equal_candidates=("boundary_equal_candidates", "mean"),
            median_boundary_equal_candidates=("boundary_equal_candidates", "median"),
            maximum_boundary_equal_candidates=("boundary_equal_candidates", "max"),
            cases=("trip_id", "size"),
        )
    )
    tie_summary.to_csv(OUTPUT / "top10_boundary_tie_summary.csv", index=False)
    audit = reproduction_audit(cases)
    audit.to_csv(OUTPUT / "legacy_reproduction_audit.csv", index=False)
    summary = {
        "status": "completed",
        "users": int(cases["user_id"].nunique()),
        "test_trajectories": int(
            cases[["user_id", "trip_id"]].drop_duplicates().shape[0]
        ),
        "ratios": RATIOS,
        "variants": [item["name"] for item in CONFIG["variants"]],
        "legacy_predictions_reproduced": int(audit["exact_region_match"].sum()),
        "legacy_predictions_expected": int(len(audit)),
        "saved_resample_plan_reused": True,
    }
    (OUTPUT / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def main() -> int:
    return run(parse_run_context(description=__doc__))


if __name__ == "__main__":
    raise SystemExit(main())
