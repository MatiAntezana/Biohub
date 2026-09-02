#!/usr/bin/env python3
"""Finalize trained autonomous variants into guarded Kaggle deployments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from aienv import ensure_aienv
from experimentlib import (
    ExperimentError,
    atomic_write_json,
    atomic_write_text,
    experiment_directory,
    load_manifest,
    repo_root,
    utc_now,
    validate_manifest,
)
from kaggle_pipeline import static_preflight
from variant_planner import append_event, latest_proposals, render


def _checkpoint(experiment_id: str, root: Path) -> Path:
    directory = experiment_directory(experiment_id, root)
    candidates = sorted(
        (directory / "checkpoints").glob("*/training_checkpoint.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    
    # Intento de descarga bajo demanda desde Hugging Face
    try:
        from hf_sync import download_experiment_checkpoint
        downloaded = download_experiment_checkpoint(experiment_id, root=root)
        if downloaded and downloaded.exists():
            return downloaded
    except Exception as exc:
        raise ExperimentError(f"{experiment_id}: no se encontró checkpoint local ni se pudo descargar de Hugging Face ({exc})") from exc

    raise ExperimentError(f"{experiment_id}: checkpoint local no recuperado")


def _export_weights(checkpoint: Path, output: Path) -> dict[str, Any]:
    import torch

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ExperimentError(f"checkpoint no resumible o formato desconocido: {checkpoint}")
    state = bundle.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ExperimentError(f"checkpoint sin model_state_dict: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        torch.save(state, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"parameters": len(state), "bytes": output.stat().st_size}


def _run_logged(command: list[str], *, cwd: Path, log: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    atomic_write_text(
        log,
        f"[{utc_now()}] $ {' '.join(command)}\n[stdout]\n{result.stdout or ''}\n[stderr]\n{result.stderr or ''}\n",
    )
    return result


def _publish_dataset(experiment_id: str, directory: Path, root: Path) -> dict[str, Any]:
    weights_dir = directory / "artifacts" / "deployment" / "weights_dataset"
    metadata = json.loads((weights_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    marker = weights_dir / "UPLOAD_COMPLETE.json"
    log = root / ".state" / "kaggle-weights" / f"{experiment_id}.log"
    if marker.is_file():
        return json.loads(marker.read_text(encoding="utf-8"))
    create = _run_logged(["kaggle", "datasets", "create", "-p", str(weights_dir), "--dir-mode", "zip"], cwd=root, log=log)
    output = (create.stdout or "") + "\n" + (create.stderr or "")
    if create.returncode and "already" in output.lower() and "exist" in output.lower():
        version = _run_logged(
            ["kaggle", "datasets", "version", "-p", str(weights_dir), "-m", f"{experiment_id} autonomous checkpoint", "--dir-mode", "zip"],
            cwd=root,
            log=log,
        )
        output = (version.stdout or "") + "\n" + (version.stderr or "")
        if version.returncode:
            raise ExperimentError(f"{experiment_id}: falló versionado dataset de pesos; ver {log.relative_to(root)}")
        action = "version"
    elif create.returncode:
        raise ExperimentError(f"{experiment_id}: falló creación dataset de pesos; ver {log.relative_to(root)}")
    else:
        action = "create"
    receipt = {"dataset_slug": metadata["id"], "action": action, "uploaded_at": utc_now(), "output_tail": output.strip()[-500:]}
    atomic_write_json(marker, receipt)
    return receipt


def prepare_one(proposal: dict[str, Any], *, root: Path, dry_run: bool = False) -> dict[str, Any]:
    experiment_id = proposal["experiment_id"]
    directory = experiment_directory(experiment_id, root)
    checkpoint = _checkpoint(experiment_id, root)
    token = experiment_id.lower().replace("-", "")
    weights = directory / "artifacts" / "deployment" / "weights_dataset" / f"{token}_weights.pt"
    if dry_run:
        return {"experiment_id": experiment_id, "checkpoint": str(checkpoint.relative_to(root)), "would_export": str(weights.relative_to(root)), "would_upload_dataset": True, "would_enable_deployment": True}
    export = _export_weights(checkpoint, weights)
    dataset = _publish_dataset(experiment_id, directory, root)
    manifest_path, manifest = load_manifest(experiment_id, root)
    metadata = json.loads((directory / "kaggle" / "kernel-metadata.json").read_text(encoding="utf-8"))
    manifest["kaggle"].update(
        deployment_ready=True,
        kernel_slug=metadata["id"],
        weights_dataset=dataset["dataset_slug"],
        weights_file=weights.name,
    )
    atomic_write_json(manifest_path, manifest)
    builder = directory / "kaggle" / "build_nb.py"
    build = _run_logged([sys.executable, str(builder)], cwd=root, log=root / ".state" / "kaggle-weights" / f"{experiment_id}-build.log")
    errors = validate_manifest(experiment_id, root) + static_preflight(experiment_id, root)
    if build.returncode or errors:
        manifest["kaggle"]["deployment_ready"] = False
        atomic_write_json(manifest_path, manifest)
        raise ExperimentError(f"{experiment_id}: paquete Kaggle inválido: {errors or ['build_nb.py falló']}")
    append_event(
        proposal,
        "KAGGLE_PENDING",
        root=root,
        details={
            "checkpoint": str(checkpoint.relative_to(root)),
            "weights": str(weights.relative_to(root)),
            "weights_export": export,
            "weights_dataset": dataset,
            "kernel_slug": metadata["id"],
        },
    )
    render(root)
    return {"experiment_id": experiment_id, "deployment_ready": True, "kernel_slug": metadata["id"], "dataset": dataset["dataset_slug"], "weights": export}


def prepare_ready_variants(*, root: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    root = root or repo_root()
    prepared = []
    errors = []
    for proposal in latest_proposals(root).values():
        if proposal.get("status") != "TRAINED":
            continue
        try:
            _, manifest = load_manifest(proposal["experiment_id"], root)
            if manifest.get("kaggle", {}).get("deployment_ready") is True:
                continue
            prepared.append(prepare_one(proposal, root=root, dry_run=dry_run))
        except Exception as error:
            errors.append({"experiment_id": proposal["experiment_id"], "error": repr(error)})
    return {"prepared": prepared, "errors": errors, "dry_run": dry_run}


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = prepare_ready_variants(dry_run=args.dry_run)
    print(json.dumps({"ok": not result["errors"], "action": "variant-kaggle-prepare", **result}, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
