#!/usr/bin/env python3
"""Generate the article SVG figures from the frozen public result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


ARTIFACT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ARTIFACT / "reference"
MANIFESTS = DEFAULT_BUNDLE / "manifests"
REFERENCE = DEFAULT_BUNDLE / "results"
MATCHED = REFERENCE / "matched_ablation"
MATCHED_ROBUSTNESS = REFERENCE / "matched_robustness"
MATCHED_DIAGNOSTIC = REFERENCE / "matched_diagnostics"
OUT = ARTIFACT / "results/figures"

RATIOS = np.asarray([0.25, 0.50, 0.66, 0.75])
COLORS = {
    "probabilistic_grid_pattern_retrieval": "#0072B2",
    "grid_full_prefix": "#0072B2",
    "grid_last_state_only": "#D55E00",
    "current_position_only_personal_retrieval": "#4D4D4D",
    "geometric_retrieval": "#009E73",
    "tsmini": "#D55E00",
    "recurrent": "#CC79A7",
}
NAMES = {
    "probabilistic_grid_pattern_retrieval": "Grid pattern",
    "grid_full_prefix": "Full grid",
    "grid_last_state_only": "Last-state grid",
    "current_position_only_personal_retrieval": "Current position",
    "geometric_retrieval": "Geometric",
    "tsmini": "TSMini",
    "recurrent": "GRU",
}


def interval_for(bootstrap: pd.DataFrame, quantity: str) -> tuple[float, float]:
    row = bootstrap.loc[bootstrap["quantity"] == quantity]
    if len(row) != 1:
        raise ValueError(f"Expected one bootstrap row for {quantity}, found {len(row)}")
    return float(row.iloc[0]["ci_low"]), float(row.iloc[0]["ci_high"])


def line(x1: float, y1: float, x2: float, y2: float, **attrs: object) -> str:
    values = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attrs.items())
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {values}/>'


def circle(x: float, y: float, radius: float, color: str) -> str:
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
        f'fill="{color}" stroke="white" stroke-width="0.8"/>'
    )


def square(x: float, y: float, radius: float, color: str) -> str:
    return (
        f'<rect x="{x-radius:.2f}" y="{y-radius:.2f}" '
        f'width="{2*radius:.2f}" height="{2*radius:.2f}" '
        f'fill="{color}" stroke="white" stroke-width="0.8"/>'
    )


def text(x: float, y: float, value: str, **attrs: object) -> str:
    values = " ".join(f'{key.replace("_", "-")}="{val}"' for key, val in attrs.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {values}>{escape(value)}</text>'


def polyline(points: list[tuple[float, float]], color: str, width: float = 1.8) -> str:
    value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{value}" fill="none" stroke="{color}" '
        f'stroke-width="{width}" stroke-linejoin="round"/>'
    )


def box(
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    fill: str,
    stroke: str = "#555555",
) -> list[str]:
    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="7" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    ]
    line_height = 15
    start_y = y + height / 2 - (len(lines) - 1) * line_height / 2 + 4
    for index, value in enumerate(lines):
        parts.append(
            text(
                x + width / 2,
                start_y + index * line_height,
                value,
                text_anchor="middle",
                font_size="11",
                fill="#222222",
            )
        )
    return parts


def arrow(x1: float, y1: float, x2: float, y2: float) -> list[str]:
    angle = np.arctan2(y2 - y1, x2 - x1)
    size = 7
    left = (
        x2 - size * np.cos(angle) + size * 0.55 * np.sin(angle),
        y2 - size * np.sin(angle) - size * 0.55 * np.cos(angle),
    )
    right = (
        x2 - size * np.cos(angle) - size * 0.55 * np.sin(angle),
        y2 - size * np.sin(angle) + size * 0.55 * np.cos(angle),
    )
    return [
        line(x1, y1, x2, y2, stroke="#555555", stroke_width="1.4"),
        f'<polygon points="{x2:.2f},{y2:.2f} {left[0]:.2f},{left[1]:.2f} '
        f'{right[0]:.2f},{right[1]:.2f}" fill="#555555"/>',
    ]


def axis_elements(
    left: float,
    top: float,
    width: float,
    height: float,
    ymin: float,
    ymax: float,
    yticks: list[float],
    ylabel: str,
) -> tuple[list[str], callable, callable]:
    xmap = lambda ratio: left + (ratio - 0.25) / 0.50 * width
    ymap = lambda value: top + height - (value - ymin) / (ymax - ymin) * height
    parts = [
        line(left, top + height, left + width, top + height, stroke="#333333", stroke_width="1"),
        line(left, top, left, top + height, stroke="#333333", stroke_width="1"),
    ]
    for tick in yticks:
        y = ymap(tick)
        parts.append(line(left, y, left + width, y, stroke="#D9D9D9", stroke_width="0.8"))
        parts.append(
            text(
                left - 8,
                y + 4,
                f"{tick:g}",
                text_anchor="end",
                font_size="11",
                fill="#333333",
            )
        )
    for ratio, label in zip(RATIOS, ["25", "50", "66", "75"]):
        x = xmap(ratio)
        parts.append(line(x, top + height, x, top + height + 5, stroke="#333333"))
        parts.append(
            text(
                x,
                top + height + 19,
                label,
                text_anchor="middle",
                font_size="11",
                fill="#333333",
            )
        )
    parts.append(
        text(
            left + width / 2,
            top + height + 38,
            "Point-count observation (%)",
            text_anchor="middle",
            font_size="12",
            fill="#222222",
        )
    )
    parts.append(
        text(
            left - 45,
            top + height / 2,
            ylabel,
            text_anchor="middle",
            font_size="12",
            fill="#222222",
            transform=f"rotate(-90 {left-45:.2f} {top+height/2:.2f})",
        )
    )
    return parts, xmap, ymap


def observation_ratio_figure() -> None:
    secondary = pd.read_csv(MATCHED / "paired_bootstrap.csv")
    deterministic = pd.read_csv(
        MATCHED_ROBUSTNESS / "matched_robustness_bootstrap.csv"
    )
    deterministic = deterministic[
        (deterministic["analysis"] == "held_out_point_count")
        & (deterministic["metric"] == "hit_r90_all")
    ]
    ratio_numeric = pd.to_numeric(
        deterministic["ratio_or_change"], errors="coerce"
    )
    records: list[dict[str, object]] = []
    for method in ["grid_full_prefix", "grid_last_state_only"]:
        for ratio in RATIOS:
            row = deterministic[
                (deterministic["quantity"] == "rate")
                & (deterministic["first_method"] == method)
                & np.isclose(ratio_numeric, ratio, equal_nan=False)
            ].iloc[0]
            records.append(
                {
                    "quantity": f"rate|{method}|{ratio:.2f}",
                    "observed": row["observed"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                }
            )
        for ratio in RATIOS:
            row = deterministic[
                (deterministic["quantity"] == "paired_difference")
                & (deterministic["first_method"] == "grid_full_prefix")
                & (
                    deterministic["second_method"]
                    == "grid_last_state_only"
                )
                & np.isclose(ratio_numeric, ratio, equal_nan=False)
            ].iloc[0]
            records.append(
                {
                    "quantity": (
                        "difference|grid_full_prefix|"
                        f"grid_last_state_only|{ratio:.2f}"
                    ),
                    "observed": row["observed"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                }
            )
    geometric_rows = secondary[
        secondary["quantity"].str.startswith(
            "difference|geometric_retrieval|"
            "current_position_only_personal_retrieval|"
        )
    ][["quantity", "observed", "ci_low", "ci_high"]]
    bootstrap = pd.concat(
        [pd.DataFrame(records).drop_duplicates(subset=["quantity"]), geometric_rows],
        ignore_index=True,
    )

    width, height = 460, 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    axes = [
        (82, 45, 340, 180, 20, 57, [20, 30, 40, 50], "Hit@R90 (%)"),
        (82, 315, 340, 180, -6, 19, [-5, 0, 5, 10, 15], "Paired difference (points)"),
    ]
    left_parts, xmap_left, ymap_left = axis_elements(*axes[0])
    right_parts, xmap_right, ymap_right = axis_elements(*axes[1])
    parts.extend(left_parts)
    parts.extend(right_parts)
    parts.append(
        text(252, 24, "(a) User-macro Hit@R90", text_anchor="middle", font_size="13")
    )
    parts.append(
        text(
            252,
            294,
            "(b) Matched retrieval differences",
            text_anchor="middle",
            font_size="13",
        )
    )

    for method, marker in [
        ("grid_full_prefix", circle),
        ("grid_last_state_only", square),
    ]:
        points = []
        for ratio in RATIOS:
            row = bootstrap.loc[
                bootstrap["quantity"] == f"rate|{method}|{ratio:.2f}"
            ].iloc[0]
            value = float(row["observed"])
            low, high = interval_for(bootstrap, f"rate|{method}|{ratio:.2f}")
            x, y = xmap_left(ratio), ymap_left(value)
            parts.append(
                line(x, ymap_left(low), x, ymap_left(high), stroke=COLORS[method])
            )
            parts.append(line(x - 4, ymap_left(low), x + 4, ymap_left(low), stroke=COLORS[method]))
            parts.append(line(x - 4, ymap_left(high), x + 4, ymap_left(high), stroke=COLORS[method]))
            parts.append(marker(x, y, 4, COLORS[method]))
            points.append((x, y))
        parts.append(polyline(points, COLORS[method]))

    contrasts = [
        (
            "grid_full_prefix",
            "grid_last_state_only",
            COLORS["grid_full_prefix"],
            circle,
        ),
        (
            "geometric_retrieval",
            "current_position_only_personal_retrieval",
            COLORS["geometric_retrieval"],
            square,
        ),
    ]
    for (numerator, denominator, color, marker), offset in zip(
        contrasts, [-3.5, 3.5]
    ):
        points = []
        for ratio in RATIOS:
            quantity = f"difference|{numerator}|{denominator}|{ratio:.2f}"
            row = bootstrap.loc[bootstrap["quantity"] == quantity].iloc[0]
            value = float(row["observed"])
            low, high = interval_for(bootstrap, quantity)
            x, y = xmap_right(ratio) + offset, ymap_right(value)
            parts.append(
                line(x, ymap_right(low), x, ymap_right(high), stroke=color)
            )
            parts.append(line(x - 3, ymap_right(low), x + 3, ymap_right(low), stroke=color))
            parts.append(line(x - 3, ymap_right(high), x + 3, ymap_right(high), stroke=color))
            parts.append(marker(x, y, 3.6, color))
            points.append((x, y))
        parts.append(polyline(points, color, 1.4))

    parts.append(
        line(
            82,
            ymap_right(0),
            422,
            ymap_right(0),
            stroke="#777777",
            stroke_width="1",
            stroke_dasharray="4 4",
        )
    )
    legends = [
        ("grid_full_prefix", 104, 61, "Full grid"),
        ("grid_last_state_only", 260, 61, "Last-state grid"),
        ("grid_full_prefix", 96, 331, "Full grid − last-state grid"),
        ("geometric_retrieval", 270, 331, "Geometric − current"),
    ]
    for method, x, y, label in legends:
        parts.append(line(x, y, x + 16, y, stroke=COLORS[method], stroke_width="2"))
        parts.append(text(x + 20, y + 4, label, font_size="9", fill="#222222"))

    parts.append("</svg>")
    (OUT / "observation_ratio_contrast.svg").write_text("\n".join(parts), encoding="utf-8")


def user_effect_figure() -> None:
    data = pd.read_csv(MATCHED_DIAGNOSTIC / "matched_user_effects.csv")
    wide = data.pivot(index="user_id", columns="ratio", values="paired_effect")
    wide = wide.sort_values(0.25)

    width, height = 460, 300
    left, top, plot_width, plot_height = 68, 25, 365, 210
    ymin, ymax = -40, 45
    xmap = lambda rank: left + (rank - 1) / (len(wide) - 1) * plot_width
    ymap = lambda value: top + plot_height - (value - ymin) / (ymax - ymin) * plot_height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        line(left, top + plot_height, left + plot_width, top + plot_height, stroke="#333333"),
        line(left, top, left, top + plot_height, stroke="#333333"),
    ]
    for tick in [-40, -20, 0, 20, 40]:
        y = ymap(tick)
        parts.append(line(left, y, left + plot_width, y, stroke="#D9D9D9"))
        parts.append(text(left - 8, y + 4, str(tick), text_anchor="end", font_size="11"))
    parts.append(
        line(left, ymap(0), left + plot_width, ymap(0), stroke="#777777", stroke_dasharray="4 4")
    )
    for tick in [1, 10, 20, 30, 37]:
        x = xmap(tick)
        parts.append(text(x, top + plot_height + 18, str(tick), text_anchor="middle", font_size="11"))
    parts.append(
        text(
            left + plot_width / 2,
            height - 12,
            "User index (ordered by 25% effect)",
            text_anchor="middle",
            font_size="12",
        )
    )
    parts.append(
        text(
            18,
            top + plot_height / 2,
            "Full grid minus last-state grid (points)",
            text_anchor="middle",
            font_size="12",
            transform=f"rotate(-90 18 {top+plot_height/2})",
        )
    )
    early = wide[0.25].to_numpy(dtype=float) * 100
    late = wide[0.75].to_numpy(dtype=float) * 100
    for rank, (early_value, late_value) in enumerate(zip(early, late), start=1):
        x = xmap(rank)
        parts.append(
            line(
                x,
                ymap(early_value),
                x,
                ymap(late_value),
                stroke="#BDBDBD",
                stroke_width="0.9",
            )
        )
        parts.append(circle(x, ymap(early_value), 2.8, COLORS["grid_full_prefix"]))
        parts.append(square(x, ymap(late_value), 2.5, COLORS["grid_last_state_only"]))
    for x, color, label, marker in [
        (82, COLORS["grid_full_prefix"], "25%", circle),
        (144, COLORS["grid_last_state_only"], "75%", square),
    ]:
        parts.append(marker(x + 7, 42, 3, color))
        parts.append(text(x + 15, 46, label, font_size="11"))
    parts.append("</svg>")
    (OUT / "user_effects.svg").write_text("\n".join(parts), encoding="utf-8")


def protocol_flow_figure() -> None:
    cohorts = pd.read_csv(MANIFESTS / "cohort_manifest.csv", dtype="string")
    truth = cohorts.apply(lambda column: column.str.lower() == "true")
    eligible_users = int(truth["eligibility"].sum())
    discovery_users = int(truth["discovery_membership"].sum())
    evaluation_users = int(truth["evaluation_membership"].sum())
    preliminary_users = int(
        (
            truth["eligibility"]
            & truth["prior_use"]
            & ~truth["discovery_membership"]
        ).sum()
    )
    discovery_summary = json.loads(
        (REFERENCE / "discovery/preparation/preparation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    evaluation_summary = json.loads(
        (MANIFESTS / "preparation_summary.json").read_text(encoding="utf-8")
    )
    discovery_partitions = discovery_summary["partition_counts"]
    evaluation_partitions = evaluation_summary["partition_counts"]
    discovery_trajectories = int(discovery_summary["selected_trajectories"])
    evaluation_trajectories = int(evaluation_summary["selected_trajectories"])
    evaluation_training = int(evaluation_partitions["fit"]) + int(
        evaluation_partitions["validation"]
    )
    evaluation_test = int(evaluation_partitions["test"])

    width, height = 930, 245
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    parts.extend(
        box(20, 86, 120, 58, ["Eligible users", f"n = {eligible_users}"], "#F2F2F2")
    )
    parts.extend(arrow(140, 115, 185, 45))
    parts.extend(arrow(140, 115, 185, 115))
    parts.extend(arrow(140, 115, 185, 183))
    parts.extend(
        box(
            185,
            15,
            145,
            60,
            [
                "Discovery cohort",
                f"{discovery_users} users",
                f"{discovery_trajectories:,} trajectories",
            ],
            "#DDEBF7",
            "#0072B2",
        )
    )
    parts.extend(
        box(
            185,
            85,
            145,
            60,
            [
                "Earlier preliminary",
                f"{preliminary_users} users",
                "excluded from evaluation",
            ],
            "#F7E6D5",
            "#D55E00",
        )
    )
    parts.extend(
        box(
            185,
            153,
            145,
            60,
            [
                "Final evaluation",
                f"{evaluation_users} users",
                f"{evaluation_trajectories:,} trajectories",
            ],
            "#DFF0E8",
            "#009E73",
        )
    )
    parts.extend(arrow(330, 45, 380, 45))
    parts.extend(
        box(
            380,
            15,
            150,
            60,
            [
                "Selection partitions",
                f"{int(discovery_partitions['fit']):,} fit; "
                f"{int(discovery_partitions['validation']):,} validation",
            ],
            "#DDEBF7",
            "#0072B2",
        )
    )
    parts.extend(arrow(530, 45, 580, 45))
    parts.extend(
        box(
            580,
            15,
            150,
            60,
            ["Frozen configurations", "and prespecified settings"],
            "#DDEBF7",
            "#0072B2",
        )
    )
    # Frozen settings govern personal fitting on earlier evaluation history.
    # They do not bypass fitting or enter the held-out test labels.
    parts.append(line(655, 75, 655, 118, stroke="#555555", stroke_width="1.4"))
    parts.append(line(655, 118, 455, 118, stroke="#555555", stroke_width="1.4"))
    parts.extend(arrow(455, 118, 455, 153))
    parts.extend(arrow(330, 183, 380, 183))
    parts.extend(
        box(
            380,
            153,
            150,
            60,
            [
                "Earlier 75% history",
                f"{evaluation_training:,} trajectories",
                "personal fitting",
            ],
            "#DFF0E8",
            "#009E73",
        )
    )
    parts.extend(arrow(530, 183, 580, 183))
    parts.extend(
        box(
            580,
            153,
            150,
            60,
            [
                "Later 25% test",
                f"{evaluation_test:,} trajectories",
                "predict before labels",
            ],
            "#DFF0E8",
            "#009E73",
        )
    )
    parts.extend(arrow(730, 183, 775, 183))
    parts.extend(box(775, 153, 135, 60, ["Labels revealed", "paired analysis"], "#F2F2F2"))
    parts.append(
        text(
            455,
            91,
            f"{int(discovery_partitions['test']):,} discovery-test trajectories "
            "excluded from selection",
            text_anchor="middle",
            font_size="10",
            fill="#555555",
        )
    )
    parts.append(
        text(
            465,
            239,
            "No evaluation trajectory or label enters configuration selection",
            text_anchor="middle",
            font_size="11",
            fill="#555555",
        )
    )
    parts.append("</svg>")
    (OUT / "protocol_flow.svg").write_text("\n".join(parts), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Generate manuscript SVG figures from a normalized result bundle."
    )
    result.add_argument(
        "--bundle-root",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Bundle containing manifests/, predictions/, and results/.",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=OUT,
        help="Directory receiving the SVG figures.",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    global MANIFESTS, REFERENCE, MATCHED, MATCHED_ROBUSTNESS, MATCHED_DIAGNOSTIC, OUT
    arguments = parser().parse_args(argv)
    bundle_root = arguments.bundle_root.expanduser().resolve()
    MANIFESTS = bundle_root / "manifests"
    REFERENCE = bundle_root / "results"
    MATCHED = REFERENCE / "matched_ablation"
    MATCHED_ROBUSTNESS = REFERENCE / "matched_robustness"
    MATCHED_DIAGNOSTIC = REFERENCE / "matched_diagnostics"
    OUT = arguments.output_root.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    observation_ratio_figure()
    user_effect_figure()
    protocol_flow_figure()
    try:
        display = OUT.relative_to(ARTIFACT)
    except ValueError:
        display = OUT
    print(f"Generated SVG figures in {display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
