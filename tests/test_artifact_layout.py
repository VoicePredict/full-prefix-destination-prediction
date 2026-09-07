from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from destination_prediction.context import (  # noqa: E402
    DISCOVERY_SELECTION_FILES,
    RUN_CONFIGS,
    RunContext,
)
from scripts import (  # noqa: E402
    reproduce,
    update_checksums,
    verify_artifact,
    verify_geolife,
)


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

    def test_geolife_verification_rejects_an_extra_trajectory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            expected = root / "Data" / "000" / "Trajectory" / "expected.plt"
            unexpected = root / "Data" / "000" / "Trajectory" / "extra.plt"
            expected.parent.mkdir(parents=True)
            expected.write_text("expected\n", encoding="utf-8")
            unexpected.write_text("extra\n", encoding="utf-8")
            records = [("unused", expected.relative_to(root))]
            self.assertEqual(
                verify_geolife.unexpected_trajectory_paths(root, records),
                [unexpected.relative_to(root).as_posix()],
            )

    def test_setup_reuses_an_external_geolife_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="GeoLife root with spaces ") as name:
            data_root = Path(name)
            (data_root / "Data").mkdir()
            with mock.patch.object(reproduce, "download") as download:
                with mock.patch.object(reproduce.subprocess, "run") as run:
                    reproduce.install_geolife(data_root, keep_downloads=False)
            download.assert_not_called()
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(Path(command[-1]), data_root.resolve())
            self.assertTrue(run.call_args.kwargs["check"])

    def test_setup_cli_accepts_a_data_root_with_spaces(self) -> None:
        data_root = Path("/reviewer inputs/Geolife Trajectories 1.3")
        arguments = reproduce.parser().parse_args(
            ["setup", "--data-root", str(data_root)]
        )
        self.assertEqual(arguments.data_root, data_root)

    def test_tsmini_cpu_patch_is_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            target = Path(temporary_name)
            utility = target / "utils" / "tool_funcs.py"
            utility.parent.mkdir(parents=True)
            upstream = (
                reproduce.TSMINI_UPSTREAM_NVML_IMPORT
                + "\n"
                + reproduce.TSMINI_UPSTREAM_GPU_INFO
            ).encode("utf-8")
            expected = (
                reproduce.TSMINI_GUARDED_NVML_IMPORT
                + "\n"
                + reproduce.TSMINI_GUARDED_GPU_INFO
            ).encode("utf-8")
            utility.write_bytes(upstream)
            with mock.patch.object(
                reproduce,
                "TSMINI_UPSTREAM_TOOL_FUNCS_SHA256",
                hashlib.sha256(upstream).hexdigest(),
            ):
                with mock.patch.object(
                    reproduce,
                    "TSMINI_CPU_COMPATIBLE_TOOL_FUNCS_SHA256",
                    hashlib.sha256(expected).hexdigest(),
                ):
                    reproduce.apply_tsmini_cpu_compatibility(target)
                    self.assertEqual(utility.read_bytes(), expected)
                    reproduce.apply_tsmini_cpu_compatibility(target)
                    self.assertEqual(utility.read_bytes(), expected)

    def test_generated_large_files_are_outside_the_public_release(self) -> None:
        generated = ROOT / "work" / "test-large-file.bin"
        generated.parent.mkdir(parents=True, exist_ok=True)
        try:
            with generated.open("wb") as handle:
                handle.truncate(50 * 1024 * 1024)
            self.assertNotIn(generated, verify_artifact.public_release_files())
        finally:
            generated.unlink(missing_ok=True)

        generated_metadata = (
            ROOT / "src" / "personal_trajectory_endpoint_prediction.egg-info" / "PKG-INFO"
        )
        self.assertNotIn(generated_metadata, verify_artifact.public_release_files())

    def test_checksum_manifest_exactly_covers_public_release(self) -> None:
        listed = {
            line.split("  ", 1)[1]
            for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
        }
        expected = {
            path.relative_to(ROOT).as_posix()
            for path in verify_artifact.public_release_files()
        }
        self.assertEqual(listed, expected)

    def test_ci_uses_published_precision_output_comparison(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "validate.yml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(workflow.count("--compare-to results"), 2)
        self.assertNotIn("diff --recursive", workflow)

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

    def test_changed_discovery_selection_rejects_frozen_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_name:
            output_root = Path(temporary_name) / "runs"
            context = RunContext.create(
                "evaluation37", ROOT, output_root, ROOT / "data" / "Geolife"
            )
            fixed = context.config["selection"]["fixed_configs"]
            discovery_output = output_root / "discovery" / "outputs"
            for family, relative in DISCOVERY_SELECTION_FILES.items():
                destination = discovery_output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(
                    json.dumps({**fixed[family], "outer_test_read": False}) + "\n",
                    encoding="utf-8",
                )

            context.require_fixed_configurations_match()
            changed = discovery_output / DISCOVERY_SELECTION_FILES[
                "geometric_retrieval"
            ]
            payload = json.loads(changed.read_text(encoding="utf-8"))
            payload["resample_points"] = 32
            changed.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "geometric_retrieval.resample_points"):
                context.require_fixed_configurations_match()

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
            if not update_checksums.included(path) or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, f"forbidden path in {path.relative_to(ROOT)}")


if __name__ == "__main__":
    unittest.main()
