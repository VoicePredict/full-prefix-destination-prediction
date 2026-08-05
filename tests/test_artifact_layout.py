from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from destination_prediction.context import RUN_CONFIGS, RunContext  # noqa: E402
from scripts import reproduce  # noqa: E402


class ArtifactLayoutTests(unittest.TestCase):
    def test_reference_material_is_minimal(self) -> None:
        for name in ("manifests", "predictions", "results"):
            self.assertTrue((ROOT / "reference" / name).is_dir())
        for obsolete in (
            "experiments",
            "frozen_results",
            "provenance",
            "environment",
            "Dockerfile",
        ):
            self.assertFalse((ROOT / obsolete).exists())

    def test_setup_targets_are_relative_and_extraction_is_safe(self) -> None:
        self.assertEqual(reproduce.DEFAULT_DATA_ROOT, ROOT / "data" / "Geolife")
        self.assertEqual(reproduce.TSMINI_ROOT, ROOT / "third_party" / "TSMini")
        self.assertIn("download.microsoft.com", reproduce.GEOLIFE_URL)
        self.assertTrue(reproduce.TSMINI_URL.endswith(f"{reproduce.TSMINI_COMMIT}.zip"))
        with tempfile.TemporaryDirectory() as temporary_name:
            temporary = Path(temporary_name)
            archive_path = temporary / "unsafe.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("../outside.txt", "blocked")
            with self.assertRaises(RuntimeError):
                reproduce.extract_zip(archive_path, temporary / "extract")

    def test_run_registry_matches_orchestrator_and_configs(self) -> None:
        registry = json.loads(
            (ROOT / "reference" / "manifests" / "run_registry.json").read_text()
        )
        registered = {run["id"] for run in registry["runs"]}
        self.assertEqual(registered, set(reproduce.RUNS))
        self.assertEqual(registered, set(RUN_CONFIGS))

    def test_full_plan_is_topological_and_complete(self) -> None:
        plan = reproduce.dependency_closure(reproduce.RUNS)
        self.assertEqual(set(plan), set(reproduce.RUNS))
        positions = {name: index for index, name in enumerate(plan)}
        for name, run in reproduce.RUNS.items():
            for dependency in run.dependencies:
                self.assertLess(positions[dependency], positions[name])

    def test_context_resolves_stable_run_ids(self) -> None:
        context = RunContext.create(
            "evaluation37", ROOT, ROOT / "work" / "runs", ROOT / "data" / "Geolife"
        )
        self.assertEqual(context.run_directory, ROOT / "work" / "runs" / "evaluation37")
        self.assertEqual(
            context.dependency_directory("discovery"), ROOT / "work" / "runs" / "discovery"
        )

    def test_public_text_has_no_private_or_historical_paths(self) -> None:
        forbidden = (
            "/" + "Users/",
            "/" + "home/",
            "100." + "79.",
            "100." + "82.",
            "experiments/" + "2026-",
        )
        suffixes = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".yml", ".yaml", ".sh"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.name == "SHA256SUMS" or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, f"forbidden path in {path.relative_to(ROOT)}")


if __name__ == "__main__":
    unittest.main()
