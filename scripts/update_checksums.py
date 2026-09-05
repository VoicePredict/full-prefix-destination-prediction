#!/usr/bin/env python3
"""Write the release-wide SHA-256 manifest in stable path order."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SHA256SUMS"


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    parts = relative.parts
    if relative == Path("SHA256SUMS"):
        return False
    if ".git" in parts or "__pycache__" in parts:
        return False
    if any(part.endswith(".egg-info") for part in parts):
        return False
    if parts[0] in {".venv", ".pytest_cache", ".mypy_cache", "work"}:
        return False
    if len(parts) >= 2 and parts[:2] == ("data", "Geolife"):
        return False
    if len(parts) >= 3 and parts[:2] == ("third_party", "TSMini"):
        return relative in {
            Path("third_party/TSMini/VENDOR_PROVENANCE.md"),
            Path("third_party/TSMini/VENDOR_FILES.sha256"),
        }
    if relative.suffix == ".pdf" and parts[:2] == ("results", "figures"):
        return False
    return path.is_file()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    paths = sorted(path for path in ROOT.rglob("*") if included(path))
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
