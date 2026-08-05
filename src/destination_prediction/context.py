"""Explicit paths and configuration for one reproducible run."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUN_CONFIGS = {
    "discovery": "discovery.json",
    "evaluation37": "evaluation37.json",
    "expanded46": "expanded46.json",
    "ambiguity": "ambiguity.json",
    "geometric_distance": "geometric_distance.json",
    "matched_ablation": "matched_ablation.json",
    "tie_sensitivity": "tie_sensitivity.json",
    "matched_familiarity": "matched_familiarity.json",
    "matched_diagnostics": "matched_diagnostics.json",
    "matched_robustness": "matched_robustness.json",
}


@dataclass(frozen=True)
class RunContext:
    """All external inputs needed by a scientific workflow.

    A context is constructed once by a command-line entry point and passed to
    scientific functions.  This keeps path resolution out of method code and
    makes dependencies between completed runs visible at call sites.
    """

    repository_root: Path
    output_root: Path
    data_root: Path
    run_id: str
    run_directory: Path
    config_path: Path
    config: dict[str, Any]

    @classmethod
    def create(
        cls,
        run_id: str,
        repository_root: Path,
        output_root: Path,
        data_root: Path,
    ) -> "RunContext":
        if run_id not in RUN_CONFIGS:
            raise KeyError(f"Unknown public run ID: {run_id!r}")
        root = repository_root.expanduser().resolve()
        runs = output_root.expanduser().resolve()
        data = data_root.expanduser().resolve()
        config_path = root / "configs" / RUN_CONFIGS[run_id]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(root, runs, data, run_id, runs / run_id, config_path, config)

    @classmethod
    def from_environment(cls) -> "RunContext":
        root = Path(
            os.environ.get(
                "DP_REPOSITORY_ROOT", Path(__file__).resolve().parents[2]
            )
        )
        output = Path(os.environ.get("DP_OUTPUT_ROOT", root / "work" / "runs"))
        data = Path(os.environ.get("DP_DATA_ROOT", root / "data" / "Geolife"))
        run_id = os.environ.get("DP_RUN_ID")
        if not run_id:
            raise RuntimeError("DP_RUN_ID is not set; use scripts/reproduce.py")
        return cls.create(run_id, root, output, data)

    def dependency_directory(self, run_id: str) -> Path:
        if run_id not in RUN_CONFIGS:
            raise KeyError(f"Unknown dependency run ID: {run_id!r}")
        return self.output_root / run_id

    def dependency_outputs(self, run_id: str) -> Path:
        return self.dependency_directory(run_id) / "outputs"

    def dependency_context(self, run_id: str) -> "RunContext":
        return RunContext.create(
            run_id,
            self.repository_root,
            self.output_root,
            self.data_root,
        )

    def repository_relative(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.repository_root).as_posix()
        except ValueError:
            return f"<external>/{resolved.name}"


def implementation_constants(config: dict[str, Any], family: str) -> dict[str, Any]:
    """Return fixed constants under the discovery or evaluation schema."""
    if family in config.get("implementation_constants", {}):
        return config["implementation_constants"][family]
    return config["sweeps"][family]
