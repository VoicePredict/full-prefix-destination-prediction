#!/usr/bin/env python3
"""Run baselines and the two non-neural retrieval method families."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from destination_prediction.catalogue import build_catalogue
from destination_prediction.context import RunContext, parse_run_context
from destination_prediction.data import (
    add_local_sequences,
    load_marked,
    partitions,
    user_origins,
)
from destination_prediction.metrics import evaluate_native_predictions, selection_summary
from destination_prediction.methods.baselines import baseline_predictions
from destination_prediction.methods.geometric_retrieval import geometric_predictions
from destination_prediction.methods.grid_pattern_retrieval import grid_pattern_predictions


GRID_FAMILY = "probabilistic_grid_pattern_retrieval"


def evaluate_predictions(
    native: pd.DataFrame,
    truth: pd.DataFrame,
    catalogue: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    evaluation = config["evaluation"]
    return evaluate_native_predictions(
        native,
        truth,
        catalogue,
        list(map(float, evaluation["support_quantiles"])),
        list(map(float, evaluation["fixed_radius_m"])),
        list(map(float, evaluation["point_thresholds_m"])),
    )


def build_training_catalogue(
    train: pd.DataFrame, eps_m: float, config: dict
) -> pd.DataFrame:
    return build_catalogue(
        train,
        eps_m,
        list(map(float, config["evaluation"]["support_quantiles"])),
        int(config["task"]["dbscan_min_samples"]),
    )


def write_selection(
    evaluated: pd.DataFrame,
    family: str,
    config_columns: list[str],
    output: Path,
    config: dict,
) -> dict[str, object]:
    cases = evaluated[evaluated["family"] == family].copy()
    per_user, leaderboard = selection_summary(cases, config_columns)
    output.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output / "validation_cases.csv", index=False)
    per_user.to_csv(output / "validation_by_user_ratio.csv", index=False)
    leaderboard.to_csv(output / "leaderboard.csv", index=False)
    selected = leaderboard.iloc[0].to_dict()
    selected.update(
        {
            "family": family,
            "selection_metric": config["selection"]["primary_metric"],
            "selection_ratios": config["task"]["observation_ratios"],
            "outer_test_read": False,
        }
    )
    (output / "selected_config.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    return selected


def run(context: RunContext) -> dict[str, object]:
    config = context.config
    context.require_fixed_configurations_match()
    marked = load_marked(context.run_directory)
    ratios = list(map(float, config["task"]["observation_ratios"]))
    primary_eps = float(config["task"]["primary_dbscan_eps_m"])
    output = context.run_directory / "outputs" / "non_neural"
    output.mkdir(parents=True, exist_ok=True)

    fixed = config["selection"].get("fixed_configs")
    if fixed is None:
        fit, validation = partitions(marked, "inner")
        origins = user_origins(fit)
        fit = add_local_sequences(fit, origins)
        validation = add_local_sequences(validation, origins)
        fit_catalogue = build_training_catalogue(fit, primary_eps, config)
        fit_catalogue.to_csv(output / "fit_only_destination_catalog.csv", index=False)
        geometric_native = geometric_predictions(
            fit,
            validation,
            ratios,
            list(map(int, config["sweeps"]["geometric_retrieval"]["resample_points"])),
        )
        grid_config = config["sweeps"][GRID_FAMILY]
        grid_native = grid_pattern_predictions(
            fit,
            validation,
            fit_catalogue,
            ratios,
            list(map(float, grid_config["cell_size_m"])),
            list(map(float, grid_config["emission_sigma_cells"])),
            list(map(int, grid_config["posterior_top_k"])),
            int(grid_config["alignment_states"]),
            tie_rule=str(grid_config["candidate_tie_rule"]),
        )
        validation_evaluated = evaluate_predictions(
            pd.concat([geometric_native, grid_native], ignore_index=True),
            validation,
            fit_catalogue,
            config,
        )
        validation_evaluated.to_csv(
            output / "all_validation_predictions.csv", index=False
        )
        geometric_selected = write_selection(
            validation_evaluated,
            "geometric_retrieval",
            ["config_id", "resample_points"],
            output / "geometric_selection",
            config,
        )
        grid_selected = write_selection(
            validation_evaluated,
            GRID_FAMILY,
            ["config_id", "cell_size_m", "emission_sigma_cells", "posterior_top_k"],
            output / "grid_pattern_selection",
            config,
        )
        grid_selected.update(
            {
                "alignment_states": int(grid_config["alignment_states"]),
                "candidate_tie_rule": str(grid_config["candidate_tie_rule"]),
            }
        )
        (output / "grid_pattern_selection" / "selected_config.json").write_text(
            json.dumps(grid_selected, indent=2) + "\n", encoding="utf-8"
        )
    else:
        geometric_selected = {
            **fixed["geometric_retrieval"],
            "selection_source": "frozen 30-user discovery validation",
            "outer_test_read": False,
        }
        grid_selected = {
            **fixed[GRID_FAMILY],
            "selection_source": "frozen 30-user discovery validation",
            "outer_test_read": False,
        }
        for directory, selected in (
            (output / "geometric_selection", geometric_selected),
            (output / "grid_pattern_selection", grid_selected),
        ):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "selected_config.json").write_text(
                json.dumps(selected, indent=2) + "\n", encoding="utf-8"
            )

    train, test = partitions(marked, "outer")
    origins = user_origins(train)
    train = add_local_sequences(train, origins)
    test = add_local_sequences(test, origins)
    catalogue = build_training_catalogue(train, primary_eps, config)
    catalogue.to_csv(output / "outer_training_destination_catalog.csv", index=False)
    geometric_final = geometric_predictions(
        train, test, ratios, [int(geometric_selected["resample_points"])]
    )
    grid_final = grid_pattern_predictions(
        train,
        test,
        catalogue,
        ratios,
        [float(grid_selected["cell_size_m"])],
        [float(grid_selected["emission_sigma_cells"])],
        [int(grid_selected["posterior_top_k"])],
        int(grid_selected["alignment_states"]),
        tie_rule=str(grid_selected["candidate_tie_rule"]),
    )
    final_evaluated = evaluate_predictions(
        pd.concat(
            [baseline_predictions(train, test, catalogue, ratios), geometric_final, grid_final],
            ignore_index=True,
        ),
        test,
        catalogue,
        config,
    )
    final_evaluated.to_csv(output / "test_predictions.csv", index=False)

    summary = {
        "outer_training_trips": int(len(train)),
        "test_trips": int(len(test)),
        "users": int(marked["user_id"].nunique()),
        "geometric_selected": geometric_selected,
        "grid_pattern_selected": grid_selected,
        "test_labels_used_for_selection": False,
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = run(parse_run_context(description=__doc__))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
