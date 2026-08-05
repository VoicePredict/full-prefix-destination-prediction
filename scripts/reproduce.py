#!/usr/bin/env python3
"""Single reviewer-facing entry point for the complete artifact.

Examples
--------
python3 scripts/reproduce.py verify
python3 scripts/reproduce.py results
python3 scripts/reproduce.py setup
python3 scripts/reproduce.py full --data-root data/Geolife
python3 scripts/reproduce.py full --target matched_robustness --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DEFAULT_OUTPUT_ROOT = ROOT / "work" / "runs"
REFERENCE = ROOT / "reference"
DOWNLOADS = ROOT / "work" / "downloads"
DEFAULT_DATA_ROOT = ROOT / "data" / "Geolife"
TSMINI_ROOT = ROOT / "third_party" / "TSMini"
GEOLIFE_URL = (
    "https://download.microsoft.com/download/f/4/8/"
    "f4894aa5-fdbc-481e-9285-d5f8c4c4f039/"
    "Geolife%20Trajectories%201.3.zip"
)
TSMINI_COMMIT = "1fac8436836fc5f03414c9d4fdcec8d2d5a3c4be"
TSMINI_URL = f"https://github.com/changyanchuan/TSMini/archive/{TSMINI_COMMIT}.zip"


@dataclass(frozen=True)
class Run:
    name: str
    config: str
    dependencies: tuple[str, ...]
    stages: tuple[tuple[str, str], ...]


RUNS: dict[str, Run] = {
    "discovery": Run(
        "discovery",
        "discovery.json",
        (),
        (
            ("prepare", "destination_prediction.preprocessing"),
            ("non_neural", "destination_prediction.methods.run_non_neural"),
            ("bigru", "destination_prediction.methods.bigru"),
            ("tsmini", "destination_prediction.methods.tsmini"),
            ("evaluate", "destination_prediction.evaluate"),
            ("validate", "destination_prediction.validation.discovery"),
        ),
    ),
    "evaluation37": Run(
        "evaluation37",
        "evaluation37.json",
        ("discovery",),
        (
            ("prepare", "destination_prediction.preprocessing"),
            ("non_neural", "destination_prediction.methods.run_non_neural"),
            ("bigru", "destination_prediction.methods.bigru"),
            ("tsmini", "destination_prediction.methods.tsmini"),
            ("evaluate", "destination_prediction.evaluate"),
            ("familiarity", "destination_prediction.analyses.familiarity"),
            ("validate", "destination_prediction.validation.evaluation"),
        ),
    ),
    "expanded46": Run(
        "expanded46",
        "expanded46.json",
        ("discovery",),
        (
            ("prepare", "destination_prediction.preprocessing"),
            ("non_neural", "destination_prediction.methods.run_non_neural"),
            ("bigru", "destination_prediction.methods.bigru"),
            ("tsmini", "destination_prediction.methods.tsmini"),
            ("evaluate", "destination_prediction.evaluate"),
            ("validate", "destination_prediction.validation.evaluation"),
        ),
    ),
    "ambiguity": Run(
        "ambiguity",
        "ambiguity.json",
        ("discovery", "evaluation37"),
        (
            ("analysis", "destination_prediction.analyses.ambiguity"),
            ("validate", "destination_prediction.validation.ambiguity"),
        ),
    ),
    "geometric_distance": Run(
        "geometric_distance",
        "geometric_distance.json",
        ("evaluation37",),
        (
            ("analysis", "destination_prediction.analyses.geometric_distance"),
            ("validate", "destination_prediction.validation.geometric_distance"),
        ),
    ),
    "matched_ablation": Run(
        "matched_ablation",
        "matched_ablation.json",
        ("evaluation37",),
        (
            ("analysis", "destination_prediction.analyses.matched_ablation"),
            ("validate", "destination_prediction.validation.matched_ablation"),
        ),
    ),
    "tie_sensitivity": Run(
        "tie_sensitivity",
        "tie_sensitivity.json",
        ("matched_ablation",),
        (
            ("analysis", "destination_prediction.analyses.tie_sensitivity"),
            ("validate", "destination_prediction.validation.tie_sensitivity"),
        ),
    ),
    "matched_familiarity": Run(
        "matched_familiarity",
        "matched_familiarity.json",
        ("evaluation37", "tie_sensitivity"),
        (
            ("analysis", "destination_prediction.analyses.matched_familiarity"),
            ("validate", "destination_prediction.validation.matched_familiarity"),
        ),
    ),
    "matched_diagnostics": Run(
        "matched_diagnostics",
        "matched_diagnostics.json",
        ("ambiguity", "tie_sensitivity"),
        (
            ("analysis", "destination_prediction.analyses.matched_diagnostics"),
            ("validate", "destination_prediction.validation.matched_diagnostics"),
        ),
    ),
    "matched_robustness": Run(
        "matched_robustness",
        "matched_robustness.json",
        ("evaluation37", "expanded46", "matched_ablation", "tie_sensitivity"),
        (
            ("analysis", "destination_prediction.analyses.matched_robustness"),
            ("validate", "destination_prediction.validation.matched_robustness"),
        ),
    ),
}

ORDER = tuple(RUNS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(paths: Iterable[Path]) -> str:
    value = hashlib.sha256()
    for path in sorted(paths):
        value.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        value.update(path.read_bytes())
    return value.hexdigest()


def source_fingerprint(run_name: str) -> str:
    sources = list(SRC.rglob("*.py"))
    sources.extend(
        ROOT / "configs" / RUNS[name].config
        for name in dependency_closure([run_name])
    )
    sources.append(Path(__file__).resolve())
    return digest(sources)


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def dependency_closure(targets: Iterable[str]) -> list[str]:
    needed: set[str] = set()

    def visit(name: str) -> None:
        if name in needed:
            return
        for dependency in RUNS[name].dependencies:
            visit(dependency)
        needed.add(name)

    for target in targets:
        visit(target)
    return [name for name in ORDER if name in needed]


def run_command(command: list[str], environment: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with status {completed.returncode}; inspect {portable(log)}"
        )


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "EDBT-artifact/1.0"})
    print(f"[download] {url}")
    with urllib.request.urlopen(request, timeout=60) as source, partial.open("wb") as sink:
        total = int(source.headers.get("Content-Length", 0))
        downloaded = 0
        next_report = 10
        while block := source.read(8 * 1024 * 1024):
            sink.write(block)
            downloaded += len(block)
            if total:
                percent = min(100, int(100 * downloaded / total))
                if percent >= next_report:
                    print(f"[download] {destination.name}: {percent}%")
                    next_report = percent + 10
    print(f"[download] saved {downloaded / (1024 * 1024):.1f} MiB")
    partial.replace(destination)


def extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            if not member_path.is_relative_to(resolved_destination):
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def install_geolife(keep_downloads: bool) -> None:
    target = DEFAULT_DATA_ROOT
    if (target / "Data").is_dir():
        print(f"[setup] GeoLife already present at {portable(target)}")
    else:
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"Refusing to overwrite non-empty {portable(target)}")
        archive_path = DOWNLOADS / "Geolife_Trajectories_1.3.zip"
        if not archive_path.is_file():
            download(GEOLIFE_URL, archive_path)
        work_root = ROOT / "work" / "setup"
        work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work_root) as temporary_name:
            temporary = Path(temporary_name)
            extract_zip(archive_path, temporary)
            candidates = sorted(
                path.parent
                for path in temporary.rglob("Data")
                if path.is_dir() and (path / "000" / "Trajectory").is_dir()
            )
            if len(candidates) != 1:
                raise RuntimeError(
                    "The GeoLife archive did not contain one recognizable Data directory"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                target.rmdir()
            shutil.move(str(candidates[0]), str(target))
        if not keep_downloads:
            archive_path.unlink(missing_ok=True)
        print(f"[setup] GeoLife installed at {portable(target)}")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_geolife.py"), str(target)],
        cwd=ROOT,
        check=True,
    )


def install_tsmini(keep_downloads: bool) -> None:
    target = TSMINI_ROOT
    if (target / "model" / "tsmini.py").is_file():
        print(f"[setup] TSMini already present at {portable(target)}")
        return

    archive_path = DOWNLOADS / f"TSMini-{TSMINI_COMMIT}.zip"
    if not archive_path.is_file():
        download(TSMINI_URL, archive_path)
    work_root = ROOT / "work" / "setup"
    work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as temporary_name:
        temporary = Path(temporary_name)
        extract_zip(archive_path, temporary)
        candidates = sorted(
            path for path in temporary.iterdir() if path.is_dir() and (path / "model").is_dir()
        )
        if len(candidates) != 1:
            raise RuntimeError("The TSMini archive did not contain one source directory")
        target.mkdir(parents=True, exist_ok=True)
        source = candidates[0]
        for directory_name in ("model", "task", "utils"):
            shutil.copytree(
                source / directory_name,
                target / directory_name,
                dirs_exist_ok=True,
            )
        for file_name in (
            ".gitignore",
            "config.py",
            "readme.md",
            "requirements.txt",
            "train_trajsimi.py",
        ):
            source_file = source / file_name
            if source_file.is_file():
                shutil.copy2(source_file, target / file_name)
    if not keep_downloads:
        archive_path.unlink(missing_ok=True)
    if not (target / "model" / "tsmini.py").is_file():
        raise RuntimeError("TSMini installation is incomplete")
    print(f"[setup] TSMini {TSMINI_COMMIT} installed at {portable(target)}")


def ensure_vendor(dry_run: bool) -> None:
    target = ROOT / "third_party" / "TSMini" / "model" / "tsmini.py"
    if target.is_file():
        return
    print("[setup] fetch pinned TSMini source")
    if not dry_run:
        install_tsmini(keep_downloads=False)


def verify_data(data_root: Path, mode: str, environment: dict[str, str], dry_run: bool) -> None:
    command = [sys.executable, str(ROOT / "scripts/verify_geolife.py"), str(data_root)]
    if mode == "paths":
        command.append("--paths-only")
    print(f"[data] {mode} GeoLife verification")
    if not dry_run:
        run_command(command, environment, ROOT / "work/logs/verify_geolife.log")


def execute_run(
    spec: Run,
    output_root: Path,
    data_root: Path,
    base_environment: dict[str, str],
    force: bool,
    dry_run: bool,
) -> None:
    run_dir = output_root / spec.name
    config_path = ROOT / "configs" / spec.config
    fingerprint = source_fingerprint(spec.name)
    complete_path = run_dir / "COMPLETE.json"
    if complete_path.is_file() and not force:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("source_fingerprint") == fingerprint:
            print(f"[skip] {spec.name}: validated output already exists")
            return

    print(f"[run]  {spec.name}")
    if dry_run:
        for stage, module in spec.stages:
            print(f"       {stage:<14} python -m {module}")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    state_dir = run_dir / "state"
    log_dir = run_dir / "logs"
    state_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    environment = dict(base_environment)
    environment.update(
        {
            "DP_REPOSITORY_ROOT": str(ROOT),
            "DP_OUTPUT_ROOT": str(output_root),
            "DP_RUN_ID": spec.name,
            "DP_DATA_ROOT": str(data_root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    python_path = str(SRC)
    if environment.get("PYTHONPATH"):
        python_path = os.pathsep.join([python_path, environment["PYTHONPATH"]])
    environment["PYTHONPATH"] = python_path

    manifest: dict[str, object] = {
        "run": spec.name,
        "config": f"configs/{spec.config}",
        "dependencies": list(spec.dependencies),
        "data_root": portable(data_root),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "geolife_manifest_sha256": hashlib.sha256(
            (REFERENCE / "manifests" / "geolife_1.3_files.sha256").read_bytes()
        ).hexdigest(),
        "source_fingerprint": fingerprint,
        "started_at_utc": utc_now(),
        "stages": [],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    for stage, module in spec.stages:
        marker = state_dir / f"{stage}.json"
        if marker.is_file() and not force:
            state = json.loads(marker.read_text(encoding="utf-8"))
            if state.get("source_fingerprint") == fingerprint:
                print(f"       {stage:<14} cached")
                manifest["stages"].append({"name": stage, "status": "cached"})
                continue
        print(f"       {stage:<14} running")
        stage_started = time.monotonic()
        command = [sys.executable, "-m", module]
        try:
            run_command(command, environment, log_dir / f"{stage}.log")
        except Exception:
            manifest["status"] = "failed"
            manifest["failed_stage"] = stage
            manifest["finished_at_utc"] = utc_now()
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            raise
        stage_state = {
            "name": stage,
            "module": module,
            "status": "passed",
            "elapsed_seconds": round(time.monotonic() - stage_started, 3),
            "source_fingerprint": fingerprint,
        }
        marker.write_text(json.dumps(stage_state, indent=2) + "\n", encoding="utf-8")
        manifest["stages"].append(stage_state)

    manifest.update(
        {
            "status": "passed",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    complete_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "run": spec.name,
                "source_fingerprint": fingerprint,
                "finished_at_utc": manifest["finished_at_utc"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Verify or reproduce all results reported in the paper."
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify", help="Verify frozen files and paper claims.")
    subparsers.add_parser("results", help="Regenerate paper tables and figures.")
    setup = subparsers.add_parser(
        "setup", help="Download and install GeoLife 1.3 and pinned TSMini."
    )
    setup.add_argument(
        "--component",
        choices=["all", "geolife", "tsmini"],
        default="all",
        help="Install both external inputs or one selected component.",
    )
    setup.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Retain downloaded ZIP archives under work/downloads.",
    )
    full = subparsers.add_parser("full", help="Run the complete experiment DAG.")
    full.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Extracted GeoLife 1.3 root (default: data/Geolife).",
    )
    full.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Run output directory (default: work/runs).",
    )
    full.add_argument(
        "--target",
        action="append",
        choices=list(RUNS),
        help="Run this target and its dependencies; repeat for multiple targets.",
    )
    full.add_argument(
        "--data-check",
        choices=["full", "paths"],
        default="full",
        help="Verify all GeoLife hashes or only its file layout.",
    )
    full.add_argument("--force", action="store_true", help="Rerun completed stages.")
    full.add_argument("--dry-run", action="store_true", help="Print the execution plan.")
    return result


def main() -> int:
    args = parser().parse_args()
    environment = dict(os.environ)
    if args.command == "verify":
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_protocol.py")],
            cwd=ROOT,
            check=True,
        )
        return subprocess.call([sys.executable, str(ROOT / "scripts/verify_artifact.py")], cwd=ROOT)
    if args.command == "results":
        subprocess.run([sys.executable, str(ROOT / "scripts/export_paper_tables.py")], cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/generate_figures.py")], cwd=ROOT, check=True)
        return 0
    if args.command == "setup":
        if args.component in {"all", "geolife"}:
            install_geolife(args.keep_downloads)
        if args.component in {"all", "tsmini"}:
            install_tsmini(args.keep_downloads)
        print("External inputs are installed and ready.")
        return 0

    targets = args.target or [ORDER[-1], "geometric_distance", "matched_familiarity", "matched_diagnostics"]
    plan = dependency_closure(targets)
    print("Execution plan: " + " -> ".join(plan))
    if not args.dry_run:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_protocol.py")],
            cwd=ROOT,
            check=True,
        )
    if not args.dry_run:
        args.output_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    verify_data(args.data_root.expanduser().resolve(), args.data_check, environment, args.dry_run)
    ensure_vendor(args.dry_run)
    for name in plan:
        execute_run(
            RUNS[name],
            args.output_root.expanduser().resolve(),
            args.data_root.expanduser().resolve(),
            environment,
            args.force,
            args.dry_run,
        )
    if not args.dry_run:
        print("Full reproduction completed; every requested validation stage passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
