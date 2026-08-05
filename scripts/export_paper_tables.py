#!/usr/bin/env python3
"""Regenerate compact paper-facing tables from the frozen result bundle."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference"
FROZEN = REFERENCE / "results"
PREDICTIONS = REFERENCE / "predictions"
OUT = ROOT / "results/paper_tables"
RATIOS = [0.25, 0.50, 0.66, 0.75]


METHOD_NAMES = {
    "probabilistic_grid_pattern_retrieval": "Probabilistic grid-pattern retrieval",
    "grid_last_state_only": "Last-state grid ablation",
    "bigru": "Recurrent destination classifier",
    "geometric_retrieval": "Geometric prefix retrieval",
    "tsmini": "TSMini embedding retrieval",
    "current_position_only_personal_retrieval": "Current-position-only retrieval",
    "most_frequent_personal_destination": "Most-frequent destination",
    "nearest_known_destination_to_current_point": "Nearest known destination",
}


def write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / name, index=False)


def main_method_table() -> None:
    canonical = pd.read_csv(FROZEN / "matched_robustness/canonical_cross_method_bootstrap.csv")
    matched = pd.read_csv(FROZEN / "matched_robustness/matched_robustness_bootstrap.csv")
    last = matched[
        (matched["analysis"] == "held_out_point_count")
        & (matched["metric"] == "hit_r90_all")
        & (matched["first_method"] == "grid_last_state_only")
        & matched["quantity"].isin(["rate", "mean_rate"])
    ].copy()
    last["first_method"] = "grid_last_state_only"
    values = pd.concat(
        [canonical[canonical["quantity"].isin(["rate", "mean_rate"])], last],
        ignore_index=True,
    )
    rows = []
    order = list(METHOD_NAMES)
    for method in order:
        frame = values[values["first_method"] == method]
        row = {"method": METHOD_NAMES[method]}
        for ratio in RATIOS:
            selected = frame[
                (frame["quantity"] == "rate")
                & pd.to_numeric(frame["ratio_or_change"], errors="coerce").sub(ratio).abs().lt(1e-9)
            ].iloc[0]
            row[f"hit_r90_{int(round(ratio * 100))}_pct"] = selected["observed"]
        mean = frame[frame["quantity"] == "mean_rate"].iloc[0]
        row.update(
            mean_pct=mean["observed"],
            mean_ci_low_pct=mean["ci_low"],
            mean_ci_high_pct=mean["ci_high"],
        )
        rows.append(row)
    write(pd.DataFrame(rows), "main_hit_r90.csv")


def matched_tables() -> None:
    frame = pd.read_csv(FROZEN / "matched_robustness/matched_robustness_bootstrap.csv")
    focal = frame[
        (frame["analysis"] == "held_out_point_count")
        & (frame["metric"] == "hit_r90_all")
        & frame["quantity"].isin(["paired_difference", "mean_difference", "difference_in_change"])
    ]
    write(focal, "matched_grid_contrast.csv")

    selected = frame[
        frame["analysis"].isin(
            [
                "held_out_point_count",
                "held_out_elapsed_time",
                "held_out_cumulative_distance",
            ]
        )
        & (frame["metric"] == "hit_r90_all")
        & frame["quantity"].isin(["paired_difference", "difference_in_change"])
        & frame["ratio_or_change"].astype(str).isin(["0.25", "0.75", "75_minus_25"])
    ]
    write(selected, "observation_definition_contrasts.csv")

    d200 = frame[
        (frame["analysis"] == "held_out_point_count")
        & (frame["metric"] == "within_200m")
        & frame["quantity"].isin(["paired_difference", "difference_in_change"])
    ]
    write(d200, "matched_d200_contrast.csv")


def ambiguity_table() -> None:
    cases = pd.read_csv(PREDICTIONS / "ambiguity_binned_cases.csv.gz")
    table = (
        cases.groupby(["requested_k", "ratio"], as_index=False)
        .agg(
            cases=("case_id", "size"),
            mean_entropy_nats=("entropy_nats", "mean"),
            mean_effective_destinations=("effective_destinations", "mean"),
            mean_top1_share=("top1_share", "mean"),
        )
        .sort_values(["requested_k", "ratio"])
    )
    write(table, "training_only_ambiguity.csv")


def point_error_table() -> None:
    frame = pd.read_csv(FROZEN / "ambiguity/tables/point_error_and_coverage_metrics.csv")
    selected_methods = [key for key in METHOD_NAMES if key != "grid_last_state_only"]
    table = (
        frame[frame["method"].isin(selected_methods)]
        .groupby("method", as_index=False)
        .agg(
            d200_pct=("user_macro_d200", lambda values: 100 * values.mean()),
            d1km_pct=("user_macro_d1km", lambda values: 100 * values.mean()),
            mean_user_median_m=("mean_user_median_error_m", "mean"),
            mean_user_p90_km=("mean_user_p90_error_m", lambda values: values.mean() / 1000),
        )
    )
    # The cross-method table uses the deterministic trajectory-ID tie rule for
    # grid-pattern retrieval. Replace the legacy grid row with values computed
    # from the deterministic point-count case file.
    deterministic = pd.read_csv(
        PREDICTIONS / "matched_robustness_evaluation37_cases.csv.gz"
    )
    deterministic = deterministic[
        (deterministic["observation_definition"] == "point_count")
        & (deterministic["method"] == "grid_full_prefix")
    ]
    by_user_ratio = (
        deterministic.groupby(["user_id", "ratio"], as_index=False)
        .agg(
            user_d200=("within_200m", "mean"),
            user_d1km=("within_1000m", "mean"),
            user_median_m=("destination_error_m", "median"),
            user_p90_m=("destination_error_m", lambda values: values.quantile(0.90)),
        )
    )
    replacement = {
        "method": "probabilistic_grid_pattern_retrieval",
        "d200_pct": 100 * by_user_ratio["user_d200"].mean(),
        "d1km_pct": 100 * by_user_ratio["user_d1km"].mean(),
        "mean_user_median_m": by_user_ratio["user_median_m"].mean(),
        "mean_user_p90_km": by_user_ratio["user_p90_m"].mean() / 1000,
    }
    table = table[table["method"] != "probabilistic_grid_pattern_retrieval"]
    table = pd.concat([table, pd.DataFrame([replacement])], ignore_index=True)
    table.insert(0, "display_name", table["method"].map(METHOD_NAMES))
    table = table.sort_values("display_name").reset_index(drop=True)
    write(table, "point_error_summary.csv")


def copy_reporting_tables() -> None:
    copies = {
        "tie_stage_sensitivity.csv": FROZEN / "tie_sensitivity/paired_tie_sensitivity_bootstrap.csv",
        "matched_prefix_familiarity.csv": FROZEN / "matched_familiarity/matched_familiarity_bootstrap.csv",
        "matched_user_effects.csv": FROZEN / "matched_diagnostics/matched_user_effects.csv",
        "matched_ambiguity.csv": FROZEN / "matched_diagnostics/matched_ambiguity_bootstrap.csv",
        "geometric_distance_metrics.csv": FROZEN / "geometric_distance/method_metrics.csv",
        "geometric_distance_intervals.csv": FROZEN / "geometric_distance/paired_bootstrap.csv",
        "destination_familiarity.csv": FROZEN / "evaluation37/analysis/destination_popularity_metrics.csv",
        "prefix_analogue_metrics.csv": FROZEN / "evaluation37/analysis/prefix_analogue_metrics.csv",
    }
    for name, source in copies.items():
        write(pd.read_csv(source), name)


def main() -> int:
    main_method_table()
    matched_tables()
    ambiguity_table()
    point_error_table()
    copy_reporting_tables()
    print(f"Wrote paper-facing tables to {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
