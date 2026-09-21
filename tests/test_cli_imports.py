"""Cold-process regressions for the prepare-001 CLI module-shadowing failure."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from core.artifacts import ROOT, code_manifest
from core.cli_bootstrap import configure_cli_imports


# run_path executes the real entrypoint and its --help path. A new interpreter
# and explicit ops/ search entry recreate direct-script startup, with no cached
# packaging import from unittest/torch to hide the original failure. We never
# authenticate, connect to a worker, open images, or start a GPU job here.
PROBE = r'''
import json
from importlib.machinery import PathFinder
from pathlib import Path
import runpy
import sys

script = Path(sys.argv[1]).resolve()
deep_imports = "--deep-imports" in sys.argv[2:]
root = script.parent.parent
sys.path.insert(0, str(script.parent))
assert "packaging" not in sys.modules, "Probe must start without cached packaging"
sys.argv = [str(script), "--help"]
try:
    runpy.run_path(str(script), run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0, exc.code
else:
    raise AssertionError("CLI did not execute --help")

assert all(Path(p or ".").resolve() != script.parent for p in sys.path), sys.path
from packaging.version import Version
import packaging
assert Version("1.0") < Version("2.0")
assert not Path(packaging.__file__).resolve().is_relative_to(root)
# Kaggle is controller-only and need not be installed on the Linux worker.
# If installed, resolution MUST find a package, never our ops/kaggle.py script.
# Inspecting the spec avoids SDK import side effects or authentication.
sdk = PathFinder.find_spec("kaggle", sys.path)
if sdk is not None:
    assert sdk.submodule_search_locations is not None, sdk.origin
    assert not Path(sdk.origin).resolve().is_relative_to(root), sdk.origin
from ops.bundle_build import package_job, wheelhouse_job
assert callable(package_job) and callable(wheelhouse_job)
runtime = {}
if deep_imports:
    import tracksdata
    import geff
    import zarr
    runtime = {"tracksdata": tracksdata.__file__, "geff": geff.__file__, "zarr": zarr.__file__}
print(json.dumps({"packaging": packaging.__file__, "runtime": runtime,
    "kaggle": None if sdk is None else sdk.origin, "entrypoint": script.name}))
'''


class CliImportTests(unittest.TestCase):
    def test_fresh_entrypoints_resolve_external_dependencies(self):
        with tempfile.TemporaryDirectory() as unrelated:
            for name in ("biohub.py", "kaggle.py", "kaggle_runtime.py", "verify.py"):
                for cwd in (ROOT, ROOT / "ops", Path(unrelated)):
                    with self.subTest(entrypoint=name, cwd=str(cwd)):
                        result = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "ops" / name)],
                            cwd=cwd, text=True, capture_output=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        report = json.loads(result.stdout.splitlines()[-1])
                        self.assertEqual(report["entrypoint"], name)

    def test_direct_script_and_module_help_remain_supported(self):
        for name in ("biohub", "kaggle", "kaggle_runtime", "verify"):
            for command in ([str(ROOT / "ops" / (name + ".py"))], ["-m", "ops." + name]):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, *command, "--help"], cwd=ROOT,
                        text=True, capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_path_cleanup_is_idempotent_and_preserves_dependencies(self):
        original = [str(ROOT / "ops"), str(ROOT), str(ROOT / "ops" / ".." / "ops"), "/dependency/site-packages"]
        with patch.object(sys, "path", original):
            configure_cli_imports(ROOT)
            expected = [str(ROOT), "/dependency/site-packages"]
            self.assertEqual(sys.path, expected)
            configure_cli_imports(ROOT)
            self.assertEqual(sys.path, expected)

    def test_release_includes_bootstrap_and_renamed_builder(self):
        manifest = code_manifest()
        self.assertIn("core/cli_bootstrap.py", manifest)
        self.assertIn("ops/bundle_build.py", manifest)
        self.assertNotIn("ops/packaging.py", manifest)
        self.assertFalse((ROOT / "ops/packaging.py").exists())


class RuntimeImportTests(unittest.TestCase):
    """Mandatory in full remote discovery; no synthetic substitute for imports."""

    def test_cold_tracksdata_geff_zarr_chain(self):
        for name in ("biohub.py", "kaggle_runtime.py"):
            with self.subTest(entrypoint=name):
                result = subprocess.run([sys.executable, "-c", PROBE, str(ROOT / "ops" / name), "--deep-imports"],
                    cwd=ROOT, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(set(report["runtime"]), {"tracksdata", "geff", "zarr"})


if __name__ == "__main__":
    unittest.main()
