"""Explicit paths and configuration for one reproducible run."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
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


DISCOVERY_SELECTION_FILES = {
    "geometric_retrieval": Path(
        "non_neural/geometric_selection/selected_config.json"
    ),
    "probabilistic_grid_pattern_retrieval": Path(
        "non_neural/grid_pattern_selection/selected_config.json"
    ),
    "recurrent": Path("recurrent/selected_config.json"),
    "tsmini": Path("tsmini/selected_config.json"),
}


def _normalized_scientific_value(value: object) -> tuple[str, object]:
    """Normalize scalar configuration values without coercing strings."""

    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, (int, float)):
        return "number", float(value)
    if isinstance(value, str):
        return "string", value
    raise TypeError(f"Unsupported scientific configuration value: {value!r}")


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

    def require_fixed_configurations_match(
        self, source_run_id: str = "discovery"
    ) -> None:
        """Require every frozen evaluation setting to match fresh selection output.

        Evaluation configurations intentionally retain a readable copy of the
        selected scientific settings.  This check makes the declared DAG edge
        operational: those copies must agree with the selection files produced
        by the current discovery run before any fixed-configuration method is
        fitted.
        """

        fixed = self.config.get("selection", {}).get("fixed_configs")
        if fixed is None:
            return
        required_families = set(DISCOVERY_SELECTION_FILES)
        if set(fixed) != required_families:
            raise RuntimeError(
                "Frozen method families do not match the discovery protocol: "
                f"{sorted(fixed)} != {sorted(required_families)}"
            )
        source = self.dependency_outputs(source_run_id)
        for family, relative_path in DISCOVERY_SELECTION_FILES.items():
            path = source / relative_path
            if not path.is_file():
                raise RuntimeError(
                    f"Fresh discovery selection is missing for {family}: {path}"
                )
            selected = json.loads(path.read_text(encoding="utf-8"))
            for key, expected in fixed[family].items():
                if key not in selected:
                    raise RuntimeError(
                        f"Fresh discovery selection for {family} lacks {key!r}"
                    )
                actual = selected[key]
                if _normalized_scientific_value(actual) != _normalized_scientific_value(
                    expected
                ):
                    raise RuntimeError(
                        "Frozen configuration differs from fresh discovery selection: "
                        f"{family}.{key}: evaluation={expected!r}, "
                        f"discovery={actual!r}"
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


def add_context_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the explicit filesystem and run identity required by one stage."""
    parser.add_argument(
        "--repository-root",
        type=Path,
        required=True,
        help="Root of the checked-out artifact repository.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory containing outputs for all registered runs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Extracted GeoLife 1.3 root.",
    )
    parser.add_argument(
        "--run-id",
        choices=tuple(RUN_CONFIGS),
        required=True,
        help="Stable public run identifier.",
    )


def parse_run_context(
    argv: Sequence[str] | None = None,
    *,
    description: str | None = None,
) -> RunContext:
    """Construct a run context solely from explicit command-line arguments."""
    parser = argparse.ArgumentParser(description=description)
    add_context_arguments(parser)
    arguments = parser.parse_args(argv)
    return RunContext.create(
        arguments.run_id,
        arguments.repository_root,
        arguments.output_root,
        arguments.data_root,
    )
