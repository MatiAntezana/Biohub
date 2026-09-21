"""Label-free Kaggle entrypoint, invoked by the generated offline notebook."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)
from core.artifacts import dump, digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--test-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    bundle, test, output = Path(args.bundle), Path(args.test_root), Path(args.output)
    if test.name != "test" or not str(test.resolve()).startswith("/kaggle/input/"):
        raise ValueError("Kaggle runtime accepts the competition test mount only")
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads((bundle / "bundle.json").read_text())
    for name, expected in spec["file_sha256"].items():
        if digest(bundle / name) != expected:
            raise ValueError("Immutable package hash mismatch: " + name)
    from core.nextgen.images import ImageSource
    from ops.workflow import cache_panel, infer_panel, infer_temporal_panel
    names = sorted(path.stem for path in test.glob("*.zarr") if path.is_dir())
    if not names:
        raise ValueError("No test movies found; there is no train fallback")
    for name in names:
        ImageSource(test / (name + ".zarr"), "0", [1.625, .40625, .40625])
    started = time.monotonic()
    cache_panel(names, test, bundle / "assets/champion", output / "detector")
    if "temporal_regularizer" in spec:
        variant_cfg = spec["temporal_regularizer"]["config"]
        result = infer_temporal_panel(output / "detector", names, variant_cfg,
            output / "prediction", scope="submission")
    else:
        from core.nextgen.pipeline import check_config
        check_config(spec["config"])
        model = {**spec["mitosis"], "checkpoint": str(bundle / spec["mitosis"]["checkpoint"])}
        hoct = dict(spec["hoct"])
        if hoct:
            hoct["checkpoint"] = str(bundle / hoct["checkpoint"])
        result = infer_panel(output / "detector", names, test, spec["config"], spec["arm"], {}, hoct,
            output / "prediction", final_model=model, scope="submission")
    destination = Path("/kaggle/working/submission.csv")
    if destination.exists():
        raise FileExistsError("A previous submission file exists; fresh kernel required")
    (output / "prediction/candidate.csv").rename(destination)
    dump(Path("/kaggle/working/runtime_receipt.json"), {"status": "completed", "run_id": spec["run_id"],
        "code_hash": spec["code_hash"], "bundle_sha256": digest(bundle / "bundle.json"),
        "arm": spec["arm"], "contract": result["contract"], "seconds": time.monotonic() - started,
        "ground_truth_accessed": False, "test_datasets": names, "kaggle_score": None})


if __name__ == "__main__":
    main()
