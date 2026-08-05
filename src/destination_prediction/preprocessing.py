#!/usr/bin/env python3
"""Prepare complete GeoLife PLT files without introducing trip segmentation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from destination_prediction.context import RunContext


EARTH_RADIUS_M = 6_371_008.8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def haversine_segments_m(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if len(lon) < 2:
        return np.empty(0, dtype=float)
    lon1 = np.radians(lon[:-1])
    lon2 = np.radians(lon[1:])
    lat1 = np.radians(lat[:-1])
    lat2 = np.radians(lat[1:])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def load_labels(user_dir: Path) -> pd.DataFrame:
    path = user_dir / "labels.txt"
    if not path.exists():
        return pd.DataFrame(columns=["start", "end", "mode"])
    labels = pd.read_csv(path, sep="\t")
    labels.columns = [str(col).strip().lower().replace(" ", "_") for col in labels.columns]
    labels = labels.rename(
        columns={"start_time": "start", "end_time": "end", "transportation_mode": "mode"}
    )
    labels["start"] = pd.to_datetime(
        labels["start"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
    )
    labels["end"] = pd.to_datetime(
        labels["end"], format="%Y/%m/%d %H:%M:%S", errors="coerce"
    )
    labels["mode"] = labels["mode"].fillna("unknown").astype(str).str.strip().str.lower()
    return labels.dropna(subset=["start", "end"])[["start", "end", "mode"]].reset_index(drop=True)


def read_plt(path: Path) -> tuple[pd.DataFrame, int]:
    rows: list[tuple[float, float, float, datetime]] = []
    raw_line_count = 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for _ in range(6):
            next(reader, None)
        for fields in reader:
            raw_line_count += 1
            if len(fields) < 7:
                continue
            try:
                latitude = float(fields[0])
                longitude = float(fields[1])
                altitude = float(fields[3])
                timestamp = datetime.fromisoformat(f"{fields[5]} {fields[6]}")
            except (TypeError, ValueError):
                continue
            if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
                continue
            rows.append((latitude, longitude, altitude, timestamp))
    return (
        pd.DataFrame(rows, columns=["latitude", "longitude", "altitude_ft", "datetime"]),
        raw_line_count,
    )


def labelled_modes(times: pd.Series, labels: pd.DataFrame) -> tuple[list[str], float]:
    assigned = np.full(len(times), "unknown", dtype=object)
    if labels.empty or len(times) == 0:
        return assigned.tolist(), 0.0
    values = times.to_numpy(dtype="datetime64[ns]")
    ordered = labels.sort_values(["start", "end"]).reset_index(drop=True)
    starts = ordered["start"].to_numpy(dtype="datetime64[ns]")
    ends = ordered["end"].to_numpy(dtype="datetime64[ns]")
    modes = ordered["mode"].astype(str).to_numpy()
    positions = np.searchsorted(starts, values, side="right") - 1
    valid = positions >= 0
    valid_positions = positions[valid]
    valid_indices = np.flatnonzero(valid)
    overlaps = values[valid] <= ends[valid_positions]
    assigned[valid_indices[overlaps]] = modes[valid_positions[overlaps]]
    known_share = float(np.mean(assigned != "unknown")) if len(assigned) else 0.0
    return assigned.tolist(), known_share


def even_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length, dtype=int)
    idx = np.unique(np.rint(np.linspace(0, length - 1, maximum)).astype(int))
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != length - 1:
        idx = np.append(idx, length - 1)
    return idx


def dominant_known_mode(modes: list[str]) -> str:
    counts = Counter(mode for mode in modes if mode != "unknown")
    if not counts:
        return "unknown"
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def inspect_trip(
    path: Path,
    user_id: int,
    labels: pd.DataFrame,
    cfg: dict,
    raw_root: Path,
) -> tuple[dict | None, dict]:
    frame, raw_line_count = read_plt(path)
    audit: dict[str, object] = {
        "user_id": user_id,
        "source_file": path.relative_to(raw_root).as_posix(),
        "source_stem": path.stem,
        "raw_points": raw_line_count,
        "valid_points": int(len(frame)),
        "invalid_rows": int(raw_line_count - len(frame)),
        "status": "eligible",
        "reason": "retained",
    }
    if len(frame) == 0:
        audit.update(status="excluded", reason="no_valid_points")
        return None, audit

    times_ns = frame["datetime"].astype("int64").to_numpy()
    delta_s = np.diff(times_ns) / 1e9
    duration_s = float((frame["datetime"].iloc[-1] - frame["datetime"].iloc[0]).total_seconds())
    lon = frame["longitude"].to_numpy(dtype=float)
    lat = frame["latitude"].to_numpy(dtype=float)
    segment_m = haversine_segments_m(lon, lat)
    positive = delta_s > 0
    speed_mps = np.full(len(delta_s), np.nan, dtype=float)
    speed_mps[positive] = segment_m[positive] / delta_s[positive]
    modes, labelled_share = labelled_modes(frame["datetime"], labels)
    known_modes = sorted(set(mode for mode in modes if mode != "unknown"))
    non_ground = sorted(set(known_modes).intersection(set(cfg["non_ground_modes"])))

    audit.update(
        start_datetime=frame["datetime"].iloc[0].isoformat(),
        end_datetime=frame["datetime"].iloc[-1].isoformat(),
        duration_s=duration_s,
        labelled_point_share=labelled_share,
        known_modes="|".join(known_modes),
        non_ground_modes="|".join(non_ground),
        duplicate_or_nonpositive_time_steps=int((delta_s <= 0).sum()),
        max_recording_gap_s=float(np.max(delta_s)) if len(delta_s) else 0.0,
        path_length_m=float(segment_m.sum()),
        max_implied_speed_mps=float(np.nanmax(speed_mps)) if np.isfinite(speed_mps).any() else np.nan,
    )
    if len(frame) < int(cfg["min_points_per_trajectory"]):
        audit.update(status="excluded", reason="too_few_valid_points")
        return None, audit
    if duration_s < float(cfg["min_duration_seconds"]):
        audit.update(status="excluded", reason="duration_below_minimum")
        return None, audit
    if bool(cfg["exclude_entire_trajectory_if_any_label_is_non_ground"]) and non_ground:
        audit.update(status="excluded", reason="whole_trajectory_non_ground_overlap")
        return None, audit

    idx = even_indices(len(frame), int(cfg["max_points_after_even_downsampling"]))
    selected = frame.iloc[idx]
    start = frame["datetime"].iloc[0]
    record = {
        "user_id": int(user_id),
        "trip_id": path.stem,
        "source_file": path.relative_to(raw_root).as_posix(),
        "start_datetime": start.isoformat(),
        "end_datetime": frame["datetime"].iloc[-1].isoformat(),
        "mode": dominant_known_mode(modes),
        "known_modes": known_modes,
        "labelled_point_share": labelled_share,
        "raw_points": int(raw_line_count),
        "valid_points": int(len(frame)),
        "trajlen": int(len(selected)),
        "wgs_seq": [
            [float(lon_value), float(lat_value)]
            for lon_value, lat_value in zip(selected["longitude"], selected["latitude"])
        ],
        "time_seq_s": (
            (selected["datetime"] - start).dt.total_seconds().astype(float).tolist()
        ),
        "path_length_m": float(segment_m.sum()),
        "max_recording_gap_s": audit["max_recording_gap_s"],
        "max_implied_speed_mps": audit["max_implied_speed_mps"],
        "duplicate_or_nonpositive_time_steps": audit["duplicate_or_nonpositive_time_steps"],
    }
    return record, audit


def split_labels(n: int, outer_fraction: float, validation_fraction: float) -> list[str]:
    n_outer = min(n - 1, max(1, int(math.floor(n * outer_fraction))))
    n_val = max(1, int(round(n_outer * validation_fraction)))
    n_fit = max(1, n_outer - n_val)
    return ["fit"] * n_fit + ["validation"] * (n_outer - n_fit) + ["test"] * (n - n_outer)


def choose_cohort(trips: pd.DataFrame, cfg: dict) -> tuple[list[int], pd.DataFrame]:
    counts = trips.groupby("user_id").size().rename("eligible_trajectories").reset_index()
    minimum = int(cfg["min_eligible_trajectories_per_user"])
    frame = counts[counts["eligible_trajectories"] >= minimum].sort_values(
        ["eligible_trajectories", "user_id"]
    ).reset_index(drop=True)
    if len(frame) < int(cfg["cohort_users"]):
        raise RuntimeError(f"Only {len(frame)} users meet the eligibility rule.")
    frame["history_tertile"] = pd.qcut(
        frame["eligible_trajectories"], 3, labels=["lower", "middle", "upper"], duplicates="drop"
    ).astype(str)
    target = int(cfg["cohort_users"])
    excluded = set(map(int, cfg.get("exclude_user_ids", [])))
    included = set(map(int, cfg.get("include_user_ids", [])))
    frame["excluded_by_historical_use"] = frame["user_id"].isin(excluded)
    frame["evaluation_membership"] = frame["user_id"].isin(included)
    if included:
        eligible = set(frame["user_id"].astype(int))
        if included.intersection(excluded):
            raise RuntimeError("Evaluation inclusion and prior-use exclusion lists overlap.")
        if not included.issubset(eligible):
            missing = sorted(included - eligible)
            raise RuntimeError(f"Frozen evaluation users are not eligible: {missing}")
        available = sorted(included)
        if len(available) != target:
            raise RuntimeError(
                f"Evaluation cohort has {len(available)} users, expected {target}."
            )
        frame["selected"] = frame["user_id"].isin(available)
        return sorted(available), frame.sort_values("user_id").reset_index(drop=True)
    if cfg["cohort_sampling"] == (
        "all eligible users not included in the frozen 30-user discovery cohort"
    ):
        frame["excluded_by_discovery_cohort"] = frame["user_id"].isin(excluded)
        available = frame.loc[
            ~frame["excluded_by_discovery_cohort"], "user_id"
        ].astype(int).tolist()
        if len(available) != target:
            raise RuntimeError(
                f"Expanded cohort has {len(available)} users, expected {target}."
            )
        frame["selected"] = frame["user_id"].isin(available)
        return sorted(available), frame.sort_values("user_id").reset_index(drop=True)
    levels = sorted(frame["history_tertile"].unique())
    base = target // len(levels)
    remainder = target % len(levels)
    rng = np.random.default_rng(int(cfg["cohort_seed"]))
    chosen: list[int] = []
    for position, level in enumerate(levels):
        candidates = frame.loc[frame["history_tertile"] == level, "user_id"].to_numpy(dtype=int)
        amount = base + (1 if position < remainder else 0)
        chosen.extend(map(int, rng.choice(candidates, size=amount, replace=False)))
    frame["selected"] = frame["user_id"].isin(chosen)
    return sorted(chosen), frame.sort_values("user_id").reset_index(drop=True)


def run(context: RunContext) -> dict[str, object]:
    config = context.config
    cfg = dict(config["dataset"])
    raw_root = context.data_root
    data_root = raw_root / "Data"
    if not data_root.is_dir():
        raise SystemExit(f"GeoLife Data directory not found: {data_root}")
    output = context.run_directory / "outputs" / "preparation"
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    audits: list[dict] = []
    user_dirs = sorted((path for path in data_root.iterdir() if path.is_dir()), key=lambda p: p.name)
    for user_dir in user_dirs:
        if not user_dir.name.isdigit():
            continue
        user_id = int(user_dir.name)
        labels = load_labels(user_dir)
        for path in sorted((user_dir / "Trajectory").glob("*.plt")):
            record, audit = inspect_trip(path, user_id, labels, cfg, raw_root)
            audits.append(audit)
            if record is not None:
                records.append(record)

    all_trips = pd.DataFrame(records)
    if all_trips.empty:
        raise RuntimeError("No eligible GeoLife trajectories were found.")
    chosen, cohort_frame = choose_cohort(all_trips, cfg)
    trips = all_trips[all_trips["user_id"].isin(chosen)].copy()
    trips = trips.sort_values(["user_id", "start_datetime", "trip_id"]).reset_index(drop=True)

    task = config["task"]
    manifests = []
    for _, group in trips.groupby("user_id", sort=True):
        group = group.sort_values(["start_datetime", "trip_id"]).copy()
        group["partition"] = split_labels(
            len(group),
            float(task["outer_train_fraction"]),
            float(task["inner_validation_fraction_of_outer_train"]),
        )
        group["chronological_index"] = np.arange(len(group), dtype=int)
        manifests.append(group)
    marked = pd.concat(manifests, ignore_index=True)
    trips = marked.drop(columns=["partition", "chronological_index"])
    manifest = marked[
        [
            "user_id",
            "trip_id",
            "source_file",
            "start_datetime",
            "end_datetime",
            "mode",
            "raw_points",
            "valid_points",
            "trajlen",
            "path_length_m",
            "max_recording_gap_s",
            "max_implied_speed_mps",
            "duplicate_or_nonpositive_time_steps",
            "partition",
            "chronological_index",
        ]
    ].copy()

    trip_file = output / "geolife_source_faithful_trips.pkl"
    trips.to_pickle(trip_file)
    manifest.to_csv(output / "partition_manifest.csv", index=False)
    pd.DataFrame(audits).to_csv(output / "source_trajectory_audit.csv", index=False)
    cohort_frame.to_csv(output / "cohort_sampling_frame.csv", index=False)

    per_user = (
        manifest.groupby(["user_id", "partition"]).size().unstack(fill_value=0).reset_index()
    )
    per_user["total"] = per_user[["fit", "validation", "test"]].sum(axis=1)
    per_user.to_csv(output / "partition_counts_by_user.csv", index=False)
    audit_frame = pd.DataFrame(audits)
    summary = {
        "raw_root": context.repository_relative(raw_root),
        "source_users": int(len(user_dirs)),
        "source_plt_files": int(len(audit_frame)),
        "eligible_whole_trajectories_before_user_selection": int(len(all_trips)),
        "users_meeting_minimum_history": int(cohort_frame.shape[0]),
        "selected_users": chosen,
        "excluded_prior_users": sorted(
            map(int, cfg.get("exclude_user_ids", []))
        ),
        "prespecified_evaluation_users": sorted(
            map(int, cfg.get("include_user_ids", []))
        ),
        "selected_user_count": len(chosen),
        "selected_trajectories": int(len(trips)),
        "partition_counts": {str(k): int(v) for k, v in manifest["partition"].value_counts().items()},
        "exclusion_reasons": {
            str(k): int(v) for k, v in audit_frame["reason"].value_counts().items()
        },
        "dominant_mode_counts_selected": {
            str(k): int(v) for k, v in trips["mode"].value_counts().items()
        },
        "selected_labelled_point_share": {
            "mean": float(trips["labelled_point_share"].mean()),
            "median": float(trips["labelled_point_share"].median()),
        },
        "quality_audit_selected": {
            "trajectories_with_nonpositive_time_steps": int(
                (trips["duplicate_or_nonpositive_time_steps"] > 0).sum()
            ),
            "trajectories_with_gap_over_1h": int((trips["max_recording_gap_s"] > 3600).sum()),
            "trajectories_with_implied_speed_over_100mps": int(
                (trips["max_implied_speed_mps"] > 100).sum()
            ),
        },
        "manual_segmentation": False,
        "row_level_mode_filtering": False,
        "source_file_retained_as_indivisible_unit": True,
        "trip_pickle_sha256": sha256(trip_file),
    }
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    summary = run(RunContext.from_environment())
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
