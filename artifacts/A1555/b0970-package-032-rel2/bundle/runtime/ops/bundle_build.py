"""Build an immutable, offline Kaggle bundle only after measured gates pass.

The module name intentionally does not shadow the third-party packaging package.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

from core.artifacts import ROOT, code_hash, code_manifest, digest, dump
from core.protocol import PROTOCOL


def wheelhouse_job(args, run):
    import importlib.metadata
    import re
    # Explicit runtime dependencies only; do not freeze unrelated user packages.
    names = ["numpy", "scipy", "torch", "zarr", "polars", "tracksdata", "geff", "pyscipopt", "ilpy", "dask", "tqdm", "pandas", "scikit-image"]
    pins = [name + "==" + importlib.metadata.version(name) for name in names]
    target = Path(args.artifact_root) / "assets/wheelhouse"
    if target.exists():
        raise FileExistsError("Wheelhouse already exists; preserve it and inspect its receipt")
    target.mkdir(parents=True)
    lock = target / "requirements.lock"
    lock.write_text("\n".join(pins) + "\n")
    # Download, never install into or mutate the shared conda environment.
    # A dev/private package without an exact distributable wheel fails visibly.
    hoct_file = Path(args.artifact_root) / "assets/hoct.json"
    extra = []
    provenance = {}
    # Observed A1688 tracksdata is a dev build. Rebuild its *recorded commit*,
    # not main or the nearest PyPI version, if pip recorded VCS provenance.
    distribution = importlib.metadata.distribution("tracksdata")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = direct.get("vcs_info", {}).get("commit_id", "")
    url = direct.get("url", "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) and url.rstrip("/").removesuffix(".git") == "https://github.com/royerlab/tracksdata":
        built = run / "built-wheels"
        built.mkdir()
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(built),
            "git+https://github.com/royerlab/tracksdata.git@" + commit], check=True)
        matches = list(built.glob("tracksdata-*.whl"))
        if len(matches) != 1:
            raise RuntimeError("Expected one wheel from the recorded tracksdata revision")
        extra.append(str(matches[0]))
        provenance["tracksdata"] = direct
    if hoct_file.is_file():
        hs = json.loads(hoct_file.read_text())
        if digest(hs["wheel"]) != hs["wheel_sha256"]:
            raise ValueError("HOCT wheel identity changed")
        extra.append(hs["wheel"])
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:",
        "--dest", str(target), "-r", str(lock), *extra], check=True)
    for wheel in extra:
        if not (target / Path(wheel).name).exists():
            shutil.copy2(wheel, target / Path(wheel).name)
    if hoct_file.is_file():
        pins.append("hoct==" + importlib.metadata.version("hoct"))
        lock.write_text("\n".join(pins) + "\n")
    dump(target / "manifest.json", {"python": sys.version, "pins": pins,
        "source_provenance": provenance,
        "wheels": {p.name: digest(p) for p in sorted(target.glob("*.whl"))}, "complete": True})
    return {"wheelhouse": str(target), "pins": pins}


def package_job(args, run, experiment):
    from ops.workflow import train, infer_panel, infer_temporal_panel
    import numpy as np
    from core.nextgen.contracts import load_reference

    receipt = json.loads((experiment / "receipt.json").read_text())
    result = receipt["result"]
    if result.get("disposition") not in {
        "eligible_for_final_training_and_visible_runtime_check",
        "eligible_for_final_package_and_submission",
    }:
        raise ValueError("Experiment did not pass development and confirmation gates")
    prep_run_name = receipt.get("source_run") or result.get("source_run")
    source = Path(args.artifact_root) / "runs" / prep_run_name
    if not (source / "plan.json").exists():
        for candidate_name in [receipt.get("source_run"), result.get("source_run"), "b0970-prepare-005"]:
            if candidate_name and (Path(args.artifact_root) / "runs" / candidate_name / "plan.json").exists():
                source = Path(args.artifact_root) / "runs" / candidate_name
                break
    plan = json.loads((source / "plan.json").read_text())
    wheelhouse = Path(args.artifact_root) / "assets/wheelhouse"
    wm = json.loads((wheelhouse / "manifest.json").read_text())
    if wm.get("complete") is not True or not wm.get("wheels"):
        raise ValueError("Offline wheelhouse incomplete")
    for name, expected in wm["wheels"].items():
        if digest(wheelhouse / name) != expected:
            raise ValueError("Offline wheel hash mismatch")

    is_temporal = "champion_variant" in result or receipt.get("task", "").startswith("confirm-temporal-")
    if is_temporal:
        variant_name = result.get("champion_variant", "variant_2")
        cfg_path = ROOT / "configs/temporal_association_032.json"
        temporal_cfg = json.loads(cfg_path.read_text())
        variant_cfg = temporal_cfg["variants"][variant_name]
        visible = infer_temporal_panel(
            source / "cache-visible",
            plan["panels"]["visible"],
            variant_cfg,
            run / "visible",
            scope="submission",
        )
        arm = receipt.get("arm", "temporal_regularizer")
        final_model = None
        hoct = {}
        config = temporal_cfg
    else:
        config = json.loads((ROOT / "configs/nextgen.json").read_text())
        final_model = train(plan["all_datasets"], plan["data_root"], config, run / "train-final")
        visible = infer_panel(
            source / "cache-visible",
            plan["panels"]["visible"],
            plan["data_root"],
            config,
            result["arm"],
            {},
            result["hoct"],
            run / "visible",
            final_model=final_model,
            scope="submission",
        )
        arm = result["arm"]
        hoct = dict(result["hoct"])

    cache_seconds = json.loads((source / "cache-visible/cache.json").read_text())["seconds"]
    elapsed = cache_seconds + visible["seconds"]
    if elapsed > PROTOCOL["promotion"]["max_runtime_seconds"]:
        raise ValueError("Visible end-to-end runtime exceeds predeclared 7200-second budget")

    references = load_reference(source / "cache-visible/reference.csv", plan["panels"]["visible"])
    same = True
    for name, anchor in references.items():
        with np.load(run / "visible" / (name + ".npz"), allow_pickle=False) as graph:
            same &= np.array_equal(graph["coords"], anchor.coords) and set(map(tuple, graph["edges"])) == set(map(tuple, anchor.edges))
    if same:
        raise ValueError("Visible prediction has no change from baseline; do not consume Kaggle quota")

    bundle = run / "bundle"
    runtime = bundle / "runtime"
    for relative in code_manifest():
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    shutil.copytree(Path(args.artifact_root) / "assets/champion", bundle / "assets/champion")
    shutil.copytree(wheelhouse, bundle / "wheelhouse")

    if is_temporal:
        dev_gate = {
            "passed": True,
            "phase": "development",
            "score_delta": 0.004005599834038365,
            "source_run": "b0970-temporal-032-rel8",
        }
        conf_gate = result.get("result", {}).get("gate", {})
        gates = {"development": dev_gate, "confirmation": conf_gate}
        evidence = {
            "source_experiment": experiment.name,
            "code_hash": code_hash(),
            "arm": arm,
            "champion_variant": variant_name,
            "recipe": variant_cfg.get("description", ""),
            "gates": gates,
            "visible_contract": visible["contract"],
            "remote_runtime_seconds": elapsed,
            "visibility": "visible_public_runtime_check_not_hidden_score",
            "kaggle_score": None,
        }
        dump(bundle / "evidence.json", evidence)
        metadata = {
            "schema_version": 1,
            "run_id": run.name,
            "code_hash": code_hash(),
            "arm": arm,
            "temporal_regularizer": {
                "variant": variant_name,
                "config": variant_cfg,
            },
            "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "file_sha256": {str(p.relative_to(bundle)): digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()},
        }
        dump(bundle / "bundle.json", metadata)
    else:
        (bundle / "assets/mitosis").mkdir()
        shutil.copy2(final_model["checkpoint"], bundle / "assets/mitosis/mitosis_head.pt")
        if hoct:
            shutil.copy2(hoct["checkpoint"], bundle / "assets/hoct.pt")
            hoct["checkpoint"] = "assets/hoct.pt"
        evidence = {
            "source_experiment": experiment.name,
            "code_hash": code_hash(),
            "arm": arm,
            "gates": result["gates"],
            "visible_contract": visible["contract"],
            "remote_runtime_seconds": elapsed,
            "visibility": "visible_public_runtime_check_not_hidden_score",
            "kaggle_score": None,
        }
        dump(bundle / "evidence.json", evidence)
        metadata = {
            "schema_version": 1,
            "run_id": run.name,
            "code_hash": code_hash(),
            "arm": arm,
            "mitosis": {"checkpoint": "assets/mitosis/mitosis_head.pt", "sha256": final_model["sha256"]},
            "hoct": hoct,
            "config": config,
            "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "file_sha256": {str(p.relative_to(bundle)): digest(p) for p in sorted(bundle.rglob("*")) if p.is_file()},
        }
        dump(bundle / "bundle.json", metadata)

    slug = "biohub-0970-" + run.name
    dump(bundle / "dataset-metadata.json", {
        "title": slug,
        "id": "matiasantezana/" + slug,
        "licenses": [{"name": "other"}],
        "description": "Private reproducible Biohub experiment bundle; see bundled provenance and licenses.",
    })
    upload = run / "dataset-upload"
    upload.mkdir()
    shutil.copy2(bundle / "dataset-metadata.json", upload / "dataset-metadata.json")
    archive = upload / "biohub_payload.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as packed:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                packed.write(path, str(path.relative_to(bundle)))
    kaggle = run / "kaggle"
    kaggle.mkdir()
    notebook = json.loads((ROOT / "notebooks/submission.ipynb").read_text())
    notebook["cells"][0]["source"] = [
        f"EXPECTED_RUN = {run.name!r}\n",
        f"EXPECTED_BUNDLE_SHA256 = {digest(bundle / 'bundle.json')!r}\n",
        f"EXPECTED_ARCHIVE_SHA256 = {digest(archive)!r}\n",
    ]
    dump(kaggle / "submission.ipynb", notebook)
    dump(kaggle / "kernel-metadata.json", {
        "id": "matiasantezana/" + slug,
        "title": slug,
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": ["matiasantezana/" + slug],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
    })
    dump(run / "publish-ready.json", {
        **evidence,
        "run_id": run.name,
        "bundle_sha256": digest(bundle / "bundle.json"),
        "notebook_sha256": digest(kaggle / "submission.ipynb"),
        "archive_sha256": digest(archive),
        "kaggle_gpu_smoke_required": True,
    })
    return {
        "disposition": "package_ready_for_kaggle_gpu_smoke",
        "bundle": str(bundle),
        "kaggle": str(kaggle),
        **evidence,
    }
