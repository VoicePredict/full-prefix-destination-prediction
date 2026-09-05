#!/usr/bin/env python3
"""Verify a separately downloaded GeoLife 1.3 tree against the frozen manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reference" / "manifests" / "geolife_1.3_files.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def parse_manifest() -> list[tuple[str, Path]]:
    records = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        records.append((expected, Path(relative)))
    return records


def unexpected_trajectory_paths(
    data_root: Path, records: list[tuple[str, Path]]
) -> list[str]:
    """Return PLT files that preprocessing could read but the manifest omits."""

    expected = {
        relative.as_posix()
        for _digest, relative in records
        if relative.suffix.lower() == ".plt"
    }
    actual = {
        path.relative_to(data_root).as_posix()
        for path in (data_root / "Data").rglob("*.plt")
        if path.is_file()
    }
    return sorted(actual - expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Directory containing Data/ and User Guide-1.3.pdf")
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Check file names and counts without hashing file contents.",
    )
    args = parser.parse_args()
    data_root = args.root.expanduser().resolve()
    if not (data_root / "Data").is_dir():
        raise SystemExit(f"Expected {data_root / 'Data'}; see docs/DATA.md")

    records = parse_manifest()
    missing = []
    changed = []
    for expected, relative in records:
        path = data_root / relative
        if not path.is_file():
            missing.append(relative.as_posix())
        elif not args.paths_only and digest(path) != expected:
            changed.append(relative.as_posix())
    unexpected_plt = unexpected_trajectory_paths(data_root, records)

    plt_count = sum(relative.suffix.lower() == ".plt" for _, relative in records)
    report = {
        "status": (
            "passed" if not missing and not changed and not unexpected_plt else "failed"
        ),
        "root": "data/Geolife" if data_root == (ROOT / "data" / "Geolife") else f"<external>/{data_root.name}",
        "manifest_files": len(records),
        "manifest_plt_files": plt_count,
        "mode": "paths_only" if args.paths_only else "full_sha256",
        "missing": missing[:20],
        "changed": changed[:20],
        "unexpected_plt": unexpected_plt[:20],
    }
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
