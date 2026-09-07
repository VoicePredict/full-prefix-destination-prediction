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
GENERATED_MARKER = ".full-prefix-reproduction-output.json"
REFERENCE = ROOT / "reference"
DOWNLOADS = ROOT / "work" / "downloads"
DEFAULT_DATA_ROOT = ROOT / "data" / "Geolife"
TSMINI_ROOT = ROOT / "third_party" / "TSMini"
TSMINI_MANIFEST = TSMINI_ROOT / "VENDOR_FILES.sha256"
GEOLIFE_URL = (
    "https://download.microsoft.com/download/f/4/8/"
    "f4894aa5-fdbc-481e-9285-d5f8c4c4f039/"
    "Geolife%20Trajectories%201.3.zip"
)
TSMINI_COMMIT = "1fac8436836fc5f03414c9d4fdcec8d2d5a3c4be"
TSMINI_URL = f"https://github.com/changyanchuan/TSMini/archive/{TSMINI_COMMIT}.zip"
TSMINI_UPSTREAM_TOOL_FUNCS_SHA256 = (
    "21b2158f4c91a900bf8ad945eede5a2d989ec4c3f8e9defa08b004dc6b39d78f"
)
TSMINI_CPU_COMPATIBLE_TOOL_FUNCS_SHA256 = (
    "c6937267a854034783631ea872f892b84484b6622c121f13b2d0d559cc7ff099"
)
TSMINI_UPSTREAM_NVML_IMPORT = (
    "from pynvml import *\n"
    "import psutil\n"
    "from datetime import datetime, timezone, timedelta\n\n"
    "nvmlInit() # need initializztion here\n"
)
TSMINI_GUARDED_NVML_IMPORT = (
    "import psutil\n"
    "from datetime import datetime, timezone, timedelta\n\n"
    "try:\n"
    "    from pynvml import (\n"
    "        nvmlDeviceGetHandleByIndex,\n"
    "        nvmlDeviceGetMemoryInfo,\n"
    "        nvmlInit,\n"
    "    )\n"
    "    nvmlInit()\n"
    "    _NVML_AVAILABLE = True\n"
    "except Exception:\n"
    "    _NVML_AVAILABLE = False\n"
)
TSMINI_UPSTREAM_GPU_INFO = (
    "class GPUInfo:\n"
    "    _h = nvmlDeviceGetHandleByIndex(0)\n\n"
    "    @classmethod\n"
    "    def mem(cls):\n"
    "        info = nvmlDeviceGetMemoryInfo(cls._h)\n"
)
TSMINI_GUARDED_GPU_INFO = (
    "class GPUInfo:\n"
    "    _h = nvmlDeviceGetHandleByIndex(0) if _NVML_AVAILABLE else None\n\n"
    "    @classmethod\n"
    "    def mem(cls):\n"
    "        if cls._h is None:\n"
    "            return 0, 0\n"
    "        info = nvmlDeviceGetMemoryInfo(cls._h)\n"
)


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
            ("recurrent", "destination_prediction.methods.recurrent_classifier"),
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
            ("recurrent", "destination_prediction.methods.recurrent_classifier"),
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
            ("recurrent", "destination_prediction.methods.recurrent_classifier"),
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
    sources.extend(
        [ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "constraints.txt"]
    )
    if TSMINI_MANIFEST.is_file():
        sources.append(TSMINI_MANIFEST)
        for line in TSMINI_MANIFEST.read_text(encoding="utf-8").splitlines():
            _expected, relative = line.split("  ", 1)
            vendor_file = TSMINI_ROOT / relative
            if vendor_file.is_file():
                sources.append(vendor_file)
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


def run_command(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=ROOT,
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


def install_geolife(data_root: Path, keep_downloads: bool) -> None:
    target = data_root.expanduser().resolve()
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
        apply_tsmini_cpu_compatibility(target)
        verify_tsmini_source()
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
    apply_tsmini_cpu_compatibility(target)
    verify_tsmini_source()
    print(f"[setup] TSMini {TSMINI_COMMIT} installed at {portable(target)}")


def apply_tsmini_cpu_compatibility(target: Path) -> None:
    """Apply the operational CPU guard used by the frozen reference runs.

    The upstream utility initializes NVIDIA Management Library (NVML) during
    import even though the encoder does not use GPU telemetry.  The reference
    runs guarded that optional import so the unchanged model could run on a
    CPU-only host.  Exact before/after hashes make this transformation fail
    closed if the pinned upstream file is not the expected one.
    """

    path = target / "utils" / "tool_funcs.py"
    if not path.is_file():
        raise RuntimeError("Pinned TSMini utility file is missing")
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    if current == TSMINI_CPU_COMPATIBLE_TOOL_FUNCS_SHA256:
        return
    if current != TSMINI_UPSTREAM_TOOL_FUNCS_SHA256:
        raise RuntimeError(
            "Unexpected pinned TSMini utils/tool_funcs.py before CPU compatibility "
            f"patch: {current}"
        )

    source = path.read_text(encoding="utf-8")
    if (
        source.count(TSMINI_UPSTREAM_NVML_IMPORT) != 1
        or source.count(TSMINI_UPSTREAM_GPU_INFO) != 1
    ):
        raise RuntimeError("Pinned TSMini CPU compatibility patch did not match once")
    patched = source.replace(
        TSMINI_UPSTREAM_NVML_IMPORT, TSMINI_GUARDED_NVML_IMPORT
    ).replace(TSMINI_UPSTREAM_GPU_INFO, TSMINI_GUARDED_GPU_INFO)
    path.write_bytes(patched.encode("utf-8"))
    result = hashlib.sha256(path.read_bytes()).hexdigest()
    if result != TSMINI_CPU_COMPATIBLE_TOOL_FUNCS_SHA256:
        raise RuntimeError(f"Pinned TSMini CPU compatibility patch produced {result}")


def verify_tsmini_source() -> None:
    """Verify every installed source file used from the pinned TSMini revision."""
    if not TSMINI_MANIFEST.is_file():
        raise RuntimeError("Pinned TSMini source manifest is missing")
    failures: list[str] = []
    for line in TSMINI_MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = TSMINI_ROOT / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"checksum mismatch {relative}")
    if failures:
        raise RuntimeError("Invalid pinned TSMini source: " + "; ".join(failures))


def ensure_vendor(dry_run: bool) -> None:
    target = ROOT / "third_party" / "TSMini" / "model" / "tsmini.py"
    if target.is_file():
        verify_tsmini_source()
        return
    print("[setup] fetch pinned TSMini source")
    if not dry_run:
        install_tsmini(keep_downloads=False)


def verify_data(data_root: Path, mode: str, dry_run: bool) -> None:
    command = [sys.executable, str(ROOT / "scripts/verify_geolife.py"), str(data_root)]
    if mode == "paths":
        command.append("--paths-only")
    print(f"[data] {mode} GeoLife verification")
    if not dry_run:
        run_command(command, ROOT / "work/logs/verify_geolife.log")


def execute_run(
    spec: Run,
    output_root: Path,
    data_root: Path,
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
        stage_log = log_dir / f"{stage}.log"
        if marker.is_file() and not force:
            state = json.loads(marker.read_text(encoding="utf-8"))
            if state.get("source_fingerprint") == fingerprint:
                print(f"       {stage:<14} cached")
                manifest["stages"].append({"name": stage, "status": "cached"})
                continue
        print(f"       {stage:<14} running (log: {portable(stage_log)})")
        stage_started = time.monotonic()
        command = [
            sys.executable,
            "-u",
            "-m",
            module,
            "--repository-root",
            str(ROOT),
            "--output-root",
            str(output_root),
            "--data-root",
            str(data_root),
            "--run-id",
            spec.name,
        ]
        try:
            run_command(command, stage_log)
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


def _assert_replaceable_generated_directory(destination: Path, kind: str) -> None:
    """Refuse to replace a directory not created by this orchestrator."""
    if not destination.exists():
        return
    if not destination.is_dir():
        raise RuntimeError(f"Generated output path is not a directory: {destination}")
    marker = destination / GENERATED_MARKER
    if not marker.is_file():
        raise RuntimeError(
            f"Refusing to replace unmarked directory: {destination}. "
            "Choose a different output root or move it manually."
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("kind") != kind:
        raise RuntimeError(f"Generated output marker has the wrong kind: {destination}")


def _publish_generated_directory(source: Path, destination: Path, kind: str) -> None:
    """Atomically publish one verified generated directory, preserving safety."""
    _assert_replaceable_generated_directory(destination, kind)
    (source / GENERATED_MARKER).write_text(
        json.dumps(
            {
                "kind": kind,
                "generated_by": "scripts/reproduce.py full",
                "generated_at_utc": utc_now(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if not destination.exists():
        source.replace(destination)
        return

    backup = destination.with_name(f".{destination.name}.previous")
    if backup.exists():
        raise RuntimeError(f"Stale generated-output backup requires inspection: {backup}")
    destination.replace(backup)
    try:
        source.replace(destination)
    except Exception:
        backup.replace(destination)
        raise
    shutil.rmtree(backup)


def assemble_reproduction(output_root: Path) -> tuple[Path, Path]:
    """Build a fresh public bundle and prove its paper outputs are unchanged."""
    from destination_prediction.artifact import (  # deferred for stdlib-only verify
        ArtifactComparisonError,
        build_candidate_bundle,
        compare_paper_outputs,
    )

    workspace = output_root.parent
    workspace.mkdir(parents=True, exist_ok=True)
    published_bundle = workspace / "reproduced"
    published_paper = workspace / "reproduced-paper"
    _assert_replaceable_generated_directory(published_bundle, "candidate_bundle")
    _assert_replaceable_generated_directory(published_paper, "paper_outputs")

    with tempfile.TemporaryDirectory(
        prefix=".reproduction-building-", dir=workspace
    ) as temporary_name:
        temporary = Path(temporary_name)
        candidate_bundle = temporary / "bundle"
        candidate_paper = temporary / "paper"
        print("[bundle] normalize validated run outputs")
        build_candidate_bundle(output_root, candidate_bundle)
        print("[paper] regenerate tables and figures from the fresh bundle")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/reproduce.py"),
                "results",
                "--bundle-root",
                str(candidate_bundle),
                "--output-root",
                str(candidate_paper),
            ],
            cwd=ROOT,
            check=True,
        )
        report_path = candidate_bundle / "manifests/paper_output_comparison.json"
        try:
            report = compare_paper_outputs(
                candidate_paper,
                ROOT / "results",
                report_path=report_path,
            )
        except ArtifactComparisonError as error:
            failure_report = workspace / "paper_output_comparison.failed.json"
            failure_report.write_text(
                json.dumps(error.report, indent=2) + "\n", encoding="utf-8"
            )
            raise
        (workspace / "paper_output_comparison.failed.json").unlink(missing_ok=True)
        print(
            f"[compare] {report['files_compared']} paper outputs match the "
            "validated source of truth"
        )
        _publish_generated_directory(
            candidate_paper, published_paper, "paper_outputs"
        )
        _publish_generated_directory(
            candidate_bundle, published_bundle, "candidate_bundle"
        )
    return published_bundle, published_paper


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Verify or reproduce all results reported in the paper."
    )
    subparsers = result.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List stable run IDs and dependencies.")
    subparsers.add_parser("verify", help="Verify frozen files and paper claims.")
    reporting = subparsers.add_parser(
        "results", help="Regenerate paper tables and figures from a result bundle."
    )
    reporting.add_argument(
        "--bundle-root",
        type=Path,
        default=REFERENCE,
        help="Normalized bundle root (default: reference).",
    )
    reporting.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results",
        help="Root receiving paper_tables/ and figures/.",
    )
    reporting.add_argument(
        "--compare-to",
        type=Path,
        help=(
            "Optional frozen paper-output root. Generated CSV values are "
            "compared at their published precision and figures byte-for-byte."
        ),
    )
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
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Existing GeoLife 1.3 root or installation destination "
            "(default: data/Geolife)."
        ),
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
    if args.command == "list":
        registry = json.loads(
            (REFERENCE / "manifests" / "run_registry.json").read_text(encoding="utf-8")
        )
        roles = {run["id"]: run for run in registry["runs"]}
        for name in ORDER:
            dependencies = ", ".join(RUNS[name].dependencies) or "none"
            print(f"{name:20} dependencies={dependencies:34} {roles[name]['status']}")
        return 0
    if args.command == "verify":
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/verify_protocol.py")],
            cwd=ROOT,
            check=True,
        )
        return subprocess.call([sys.executable, str(ROOT / "scripts/verify_artifact.py")], cwd=ROOT)
    if args.command == "results":
        bundle_root = args.bundle_root.expanduser().resolve()
        output_root = args.output_root.expanduser().resolve()
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/export_paper_tables.py"),
                "--bundle-root",
                str(bundle_root),
                "--output-root",
                str(output_root / "paper_tables"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/generate_figures.py"),
                "--bundle-root",
                str(bundle_root),
                "--output-root",
                str(output_root / "figures"),
            ],
            cwd=ROOT,
            check=True,
        )
        if args.compare_to is not None:
            from destination_prediction.artifact import compare_paper_outputs

            frozen_root = args.compare_to.expanduser().resolve()
            report = compare_paper_outputs(output_root, frozen_root)
            print(
                f"Verified {report['files_compared']} paper outputs against "
                f"{frozen_root} at published precision."
            )
        return 0
    if args.command == "setup":
        if args.component in {"all", "geolife"}:
            install_geolife(args.data_root, args.keep_downloads)
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
    verify_data(args.data_root.expanduser().resolve(), args.data_check, args.dry_run)
    ensure_vendor(args.dry_run)
    for name in plan:
        execute_run(
            RUNS[name],
            args.output_root.expanduser().resolve(),
            args.data_root.expanduser().resolve(),
            args.force,
            args.dry_run,
        )
    complete_dag = set(plan) == set(RUNS)
    if args.dry_run and complete_dag:
        print("       assemble       fresh bundle -> paper outputs -> equivalence check")
    elif not args.dry_run and complete_dag:
        bundle, paper = assemble_reproduction(args.output_root.expanduser().resolve())
        print(
            "Full reproduction completed; every validation stage passed and "
            f"fresh paper outputs are equivalent.\nBundle: {portable(bundle)}\n"
            f"Paper outputs: {portable(paper)}"
        )
    elif not args.dry_run:
        print("Requested runs completed; every requested validation stage passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
