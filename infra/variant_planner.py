#!/usr/bin/env python3
"""Plan and materialize reproducible Biohub variants from Kaggle feedback.

The planner uses an explicit exploitation/exploration policy.  It never runs
training itself.  Under ``Modo automatico`` it may create isolated experiment
folders and queue them; the normal scheduler remains the only launcher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any

from aienv import ensure_aienv
from experimentlib import (
    ExperimentError,
    atomic_write_json,
    atomic_write_text,
    canonical_json,
    exclusive_lock,
    experiment_directories,
    latest_status_for_experiment,
    load_queue,
    queue_experiment,
    read_json,
    repo_root,
    sha256_bytes,
    utc_now,
)
from kaggle_scores import best_by_experiment


LANES = ("exploitation", "adjacent_exploration", "novel_exploration")
ACTIVE_STATES = {
    "PROPOSED",
    "MATERIALIZED",
    "QUEUED",
    "READY",
    "RUNNING",
    "DOC_PENDING",
    "TRAINED",
    "KAGGLE_PENDING",
}


def ledger_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "experiments" / "variant_proposals.jsonl"


def read_events(root: Path | None = None) -> list[dict[str, Any]]:
    path = ledger_path(root)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError as error:
            raise ExperimentError(f"variant_proposals.jsonl corrupto en línea {line_number}: {error}") from error
    return events


def verify_ledger(root: Path | None = None) -> list[str]:
    errors: list[str] = []
    previous: str | None = None
    seen: set[str] = set()
    for index, event in enumerate(read_events(root), 1):
        event_id = event.get("event_id")
        if not event_id or event_id in seen:
            errors.append(f"evento {index}: event_id ausente o duplicado")
        seen.add(str(event_id))
        if event.get("previous_event_hash") != previous:
            errors.append(f"evento {index}: previous_event_hash inválido")
        stored = event.get("event_hash")
        computed = sha256_bytes(canonical_json({k: v for k, v in event.items() if k != "event_hash"}))
        if stored != computed:
            errors.append(f"evento {index}: event_hash inválido")
        previous = stored
    return errors


def latest_proposals(root: Path | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in read_events(root):
        latest[event["proposal_id"]] = event
    return latest


def append_event(proposal: dict[str, Any], status: str, *, root: Path, details: dict[str, Any] | None = None) -> dict[str, Any]:
    path = ledger_path(root)
    identity = f"{proposal['proposal_id']}|{status}|{canonical_json(details or {}).decode()}"
    event_id = sha256_bytes(identity.encode())[:24]
    with exclusive_lock(path.with_suffix(".lock")):
        events = read_events(root)
        existing = next((event for event in events if event.get("event_id") == event_id), None)
        if existing:
            return existing
        clean_proposal = {
            k: v for k, v in proposal.items()
            if k not in {"event_id", "event_hash", "previous_event_hash", "recorded_at"}
        }
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "recorded_at": utc_now(),
            **clean_proposal,
            "status": status,
            **(details or {}),
            "previous_event_hash": events[-1].get("event_hash") if events else None,
        }
        event["event_hash"] = sha256_bytes(canonical_json({k: v for k, v in event.items() if k != "event_hash"}))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return event


def _config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    policy = read_json(root / ".agents" / "autonomous-exploration.json", {})
    recipes = read_json(root / ".agents" / "variant-recipes.json", {})
    if policy.get("schema_version") != 1 or recipes.get("schema_version") != 1:
        raise ExperimentError("política o catálogo de variantes inválido")
    allocation = policy.get("allocation", {})
    total = sum(float(allocation.get(lane, 0)) for lane in LANES)
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ExperimentError(f"la asignación explotación/exploración debe sumar 1.0, suma {total}")
    return policy, recipes


def _parents(root: Path, policy: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    selection = policy["parent_selection"]
    excluded = set(selection.get("excluded_experiments", []))
    families = dict(catalog.get("families", {}))
    for proposal in latest_proposals(root).values():
        architecture = proposal.get("training", {}).get("architecture")
        if architecture:
            families[proposal["experiment_id"]] = architecture
    ranked = []
    for rank, row in enumerate(best_by_experiment(root), 1):
        experiment_id = row["experiment_id"]
        family = families.get(experiment_id)
        if experiment_id in excluded or not family or family == "stardist3d":
            continue
        ranked.append({**row, "rank": rank, "family": family})
    ranked = ranked[: int(selection.get("top_k", 3))]
    if not ranked:
        raise ExperimentError("no hay padres elegibles con Public Score de Kaggle y familia reproducible")
    return ranked


def _lane_counts(count: int, policy: dict[str, Any]) -> dict[str, int]:
    allocation = policy["allocation"]
    raw = {lane: count * float(allocation[lane]) for lane in LANES}
    counts = {lane: int(math.floor(raw[lane])) for lane in LANES}
    for lane in sorted(LANES, key=lambda name: (-(raw[name] - counts[name]), LANES.index(name))):
        if sum(counts.values()) >= count:
            break
        counts[lane] += 1
    if count >= int(allocation.get("planning_batch_size", 5)):
        minimum_novel = int(allocation.get("minimum_novel_per_batch", 1))
        if counts["novel_exploration"] < minimum_novel:
            donor = max((lane for lane in LANES if lane != "novel_exploration"), key=counts.get)
            counts[donor] -= minimum_novel - counts["novel_exploration"]
            counts["novel_exploration"] = minimum_novel
        minimum_explore = int(allocation.get("minimum_exploration_per_batch", 2))
        explore = counts["adjacent_exploration"] + counts["novel_exploration"]
        if explore < minimum_explore:
            delta = minimum_explore - explore
            counts["exploitation"] -= delta
            counts["adjacent_exploration"] += delta
    return counts


def _weighted_parent(parents: list[dict[str, Any]], rng: random.Random, temperature: float) -> dict[str, Any]:
    weights = [math.exp(-(index) / max(temperature, 1e-6)) for index in range(len(parents))]
    return rng.choices(parents, weights=weights, k=1)[0]


def _plateau(policy: dict[str, Any], root: Path) -> dict[str, Any] | None:
    limit = int(policy["stopping"].get("plateau_scored_variants", 6))
    minimum = float(policy["stopping"].get("plateau_minimum_improvement", 0.005))
    scored = [
        row for row in latest_proposals(root).values()
        if row.get("status") == "SCORED" and row.get("public_score") is not None
    ]
    scored.sort(key=lambda row: row.get("recorded_at", ""))
    if len(scored) < limit:
        return None
    window = scored[-limit:]
    prior_scores = [float(row["public_score"]) for row in scored[:-limit]]
    reference = max(prior_scores + [float(row["parent_public_score"]) for row in window])
    observed = max(float(row["public_score"]) for row in window)
    improvement = observed - reference
    if improvement < minimum:
        return {
            "reason": "kaggle_plateau",
            "window": limit,
            "reference_score": reference,
            "best_window_score": observed,
            "improvement": improvement,
            "required_improvement": minimum,
        }
    return None


def _next_ids(root: Path, count: int) -> list[str]:
    numbers = []
    for experiment_id in experiment_directories(root):
        match = re.fullmatch(r"EXP-(\d{3})", experiment_id)
        if match:
            numbers.append(int(match.group(1)))
    for proposal in latest_proposals(root).values():
        match = re.fullmatch(r"EXP-(\d{3})", str(proposal.get("experiment_id", "")))
        if match:
            numbers.append(int(match.group(1)))
    start = max(numbers or [0]) + 1
    return [f"EXP-{number:03d}" for number in range(start, start + count)]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _proposal(
    *,
    experiment_id: str,
    lane: str,
    parent: dict[str, Any],
    recipe: dict[str, Any],
    defaults: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    training = {**defaults, **recipe["training"]}
    architecture = parent["family"] if recipe["architecture"] == "$parent" else recipe["architecture"]
    training.update(architecture=architecture, seed=rng.randrange(1, 2**31 - 1))
    config = {
        "parent_experiment_id": parent["experiment_id"],
        "parent_public_score": float(parent["public_score"]),
        "lane": lane,
        "recipe_id": recipe["recipe_id"],
        "training": training,
    }
    config_hash = sha256_bytes(canonical_json(config))
    proposal_id = f"VAR-{config_hash[:16]}"
    return {
        "proposal_id": proposal_id,
        "experiment_id": experiment_id,
        "parent_experiment_id": parent["experiment_id"],
        "parent_public_score": float(parent["public_score"]),
        "parent_family": parent["family"],
        "lane": lane,
        "recipe_id": recipe["recipe_id"],
        "config_sha256": config_hash,
        "training": training,
        "scientific_rationale": (
            "Mejora acotada del mejor linaje Kaggle" if lane == "exploitation" else
            "Cambio arquitectónico cercano para diversificar el linaje" if lane == "adjacent_exploration" else
            "Exploración deliberada de una familia distante con presupuesto limitado"
        ),
    }


def plan(count: int, *, seed: int, root: Path | None = None, persist: bool = True) -> list[dict[str, Any]]:
    root = root or repo_root()
    policy, catalog = _config(root)
    parents = _parents(root, policy, catalog)
    counts = _lane_counts(count, policy)
    lanes = [lane for lane in LANES for _ in range(counts[lane])]
    rng = random.Random(seed)
    recipes_by_lane = {
        lane: [recipe for recipe in catalog["recipes"] if recipe["lane"] == lane]
        for lane in LANES
    }
    existing_hashes = {event.get("config_sha256") for event in latest_proposals(root).values()}
    failed_recipes: dict[str, int] = {}
    for event in latest_proposals(root).values():
        if event.get("status") in {"FAILED", "ORPHANED"}:
            recipe_id = event.get("recipe_id")
            failed_recipes[recipe_id] = failed_recipes.get(recipe_id, 0) + 1
    failure_limit = int(policy["budgets"].get("max_failed_recipe_attempts", 2))
    proposals: list[dict[str, Any]] = []
    ids = _next_ids(root, count)
    used_recipes: set[str] = set()
    top_parent_id = parents[0]["experiment_id"]
    exploitation_top_target = math.ceil(counts["exploitation"] * 0.5)
    exploitation_top_used = 0
    prior = sorted(latest_proposals(root).values(), key=lambda row: row.get("recorded_at", ""))
    parent_history = [row.get("parent_experiment_id") for row in prior]
    family_history = [row.get("parent_family") for row in prior]
    for experiment_id, lane in zip(ids, lanes, strict=True):
        for _attempt in range(100):
            eligible_parents = list(parents)
            max_parent = int(policy["diversity"].get("max_consecutive_same_parent", 2))
            max_family = int(policy["diversity"].get("max_consecutive_same_family", 2))
            if len(parent_history) >= max_parent and len(set(parent_history[-max_parent:])) == 1:
                eligible_parents = [row for row in eligible_parents if row["experiment_id"] != parent_history[-1]] or eligible_parents
            if len(family_history) >= max_family and len(set(family_history[-max_family:])) == 1:
                eligible_parents = [row for row in eligible_parents if row["family"] != family_history[-1]] or eligible_parents
            if lane == "exploitation" and exploitation_top_used < exploitation_top_target:
                top_only = [row for row in eligible_parents if row["experiment_id"] == top_parent_id]
                eligible_parents = top_only or eligible_parents
            parent = _weighted_parent(eligible_parents, rng, float(policy["parent_selection"].get("rank_temperature", 1.0)))
            compatible = [
                recipe for recipe in recipes_by_lane[lane]
                if parent["family"] in recipe["compatible_parent_families"]
                and failed_recipes.get(recipe["recipe_id"], 0) < failure_limit
            ]
            if not compatible:
                raise ExperimentError(f"no quedan recetas habilitadas para carril={lane}, familia={parent['family']}")
            if policy["diversity"].get("prefer_distinct_recipes_within_batch"):
                unused = [recipe for recipe in compatible if recipe["recipe_id"] not in used_recipes]
                compatible = unused or compatible
            proposal = _proposal(
                experiment_id=experiment_id,
                lane=lane,
                parent=parent,
                recipe=rng.choice(compatible),
                defaults=catalog["training_defaults"],
                rng=rng,
            )
            if proposal["config_sha256"] not in existing_hashes:
                break
        else:
            raise ExperimentError("no se pudo producir una configuración inédita en 100 intentos")
        existing_hashes.add(proposal["config_sha256"])
        used_recipes.add(proposal["recipe_id"])
        if lane == "exploitation" and proposal["parent_experiment_id"] == top_parent_id:
            exploitation_top_used += 1
        parent_history.append(proposal["parent_experiment_id"])
        family_history.append(proposal["parent_family"])
        proposals.append(proposal)
    if persist:
        for proposal in proposals:
            append_event(proposal, "PROPOSED", root=root, details={"planner_seed": seed})
        render(root)
    return proposals


def _experiment_name(proposal: dict[str, Any]) -> str:
    return f"{proposal['experiment_id']}_Auto_{proposal['lane']}_{proposal['recipe_id']}"


def _write_kaggle_scaffold(proposal: dict[str, Any], directory: Path, root: Path) -> None:
    """Create a test-only notebook package without pretending weights exist."""

    template_dir = root / "experiments" / "EXP-017_Wide_UNet3D_90_10" / "kaggle"
    template = (template_dir / "inference.py").read_text(encoding="utf-8")
    architecture = proposal["training"]["architecture"]
    model_files = {
        "anisotropic_unet3d": ("src/models/unet3d_anisotropic.py", "AnisotropicUNet3D"),
        "resunet3d": ("src/models/resnet_unet3d.py", "ResUNet3D"),
        "groupnorm_unet3d": ("src/models/unet3d_groupnorm.py", "GroupNormUNet3D"),
        "attention_unet3d": ("src/models/attention_unet3d.py", "AttentionUNet3D"),
        "convnext_unet3d": ("src/models/convnext_unet3d.py", "ConvNeXtUNet3D"),
    }
    model_relative, class_name = model_files[architecture]
    model_source = (root / model_relative).read_text(encoding="utf-8")
    start = template.index("# ----------------- 1.")
    end = template.index("# ----------------- 2.")
    generated = template[:start] + "# ----------------- 1. Generated Architecture -----------------\n\n" + model_source + "\n\n" + template[end:]
    experiment_token = proposal["experiment_id"].lower().replace("-", "")
    generated = generated.replace("EXP-017", proposal["experiment_id"])
    generated = generated.replace("exp017", experiment_token)
    generated = generated.replace("AnisotropicUNet3D", class_name)
    generated = generated.replace(
        "if \"%s\" in p.lower():" % experiment_token,
        "if \"%s\" in p.lower():" % experiment_token,
    )
    kaggle_dir = directory / "kaggle"
    atomic_write_text(kaggle_dir / "inference.py", generated)
    shutil.copy2(template_dir / "build_nb.py", kaggle_dir / "build_nb.py")
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [generated]}],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    atomic_write_json(kaggle_dir / "submission_notebook.ipynb", notebook)
    owner = "matiasantezana"
    short = proposal["experiment_id"].lower()
    kernel_slug = f"{owner}/biohub-{short}-auto"
    weights_slug = f"{owner}/biohub-{short}-auto-weights"
    metadata = {
        "id": kernel_slug,
        "title": f"Biohub {proposal['experiment_id']} Auto",
        "code_file": "submission_notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "false",
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [weights_slug],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
    }
    atomic_write_json(kaggle_dir / "kernel-metadata.json", metadata)
    weights_dir = directory / "artifacts" / "deployment" / "weights_dataset"
    weights_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        weights_dir / "dataset-metadata.json",
        {"title": f"Biohub {proposal['experiment_id']} Auto Weights", "id": weights_slug, "licenses": [{"name": "CC0-1.0"}]},
    )


def materialize(proposal: dict[str, Any], *, root: Path, auto_queue: bool) -> dict[str, Any]:
    directory = root / "experiments" / _experiment_name(proposal)
    if directory.exists():
        raise ExperimentError(f"la carpeta de variante ya existe: {directory}")
    for name in ("logs", "artifacts", "checkpoints", "code_snapshot", "validation", "kaggle"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "infra" / "templates" / "autonomous_variant_train.py", directory / "train.py")
    variant = {"schema_version": 1, **proposal, "created_at": utc_now()}
    atomic_write_json(directory / "variant.json", variant)
    model_files = {
        "anisotropic_unet3d": "src/models/unet3d_anisotropic.py",
        "resunet3d": "src/models/resnet_unet3d.py",
        "groupnorm_unet3d": "src/models/unet3d_groupnorm.py",
        "attention_unet3d": "src/models/attention_unet3d.py",
        "convnext_unet3d": "src/models/convnext_unet3d.py",
    }
    narrative = root / "Librería-Registros" / "Experimentos" / f"{proposal['experiment_id']}_Auto_{proposal['recipe_id']}.md"
    atomic_write_text(
        narrative,
        "\n".join([
            f"# {proposal['experiment_id']} — variante autónoma {proposal['recipe_id']}",
            "",
            f"- Carril: `{proposal['lane']}`",
            f"- Padre: `{proposal['parent_experiment_id']}` (Public Score Kaggle `{proposal['parent_public_score']:.4f}`)",
            f"- Arquitectura: `{proposal['training']['architecture']}`",
            f"- Config SHA-256: `{proposal['config_sha256']}`",
            f"- Hipótesis: {proposal['scientific_rationale']}",
            "",
            "## Resultado",
            "",
            "Pendiente. Registrar aquí el resultado local, Public Score de Kaggle, conclusión y decisión.",
            "",
        ]),
    )
    policy = read_json(root / ".agents" / "autonomous-exploration.json", {})
    materialization = policy["materialization"]
    experiment_id = proposal["experiment_id"]
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "name": _experiment_name(proposal).replace("_", " "),
        "entrypoint": "train.py",
        "arguments": [],
        "code_dependencies": [
            "src/strategies/autonomous_detection_training.py",
            "src/strategies/experiment_runtime.py",
            "src/dataset.py",
            model_files[proposal["training"]["architecture"]],
        ],
        "allowed_nodes": materialization["allowed_nodes"],
        "default_node": materialization["default_node"],
        "execution_policy": {
            "node_selection": "max_free_vram",
            "min_free_vram_mib": int(materialization["min_free_vram_mib"]),
            "auto_launch": bool(materialization["auto_launch"]),
            "max_parallel_runs": 1,
        },
        "priority": 100 if proposal["lane"] == "exploitation" else 110 if proposal["lane"] == "adjacent_exploration" else 120,
        "dependencies": [],
        "data_policy": {"mode": "full-fit", "validation": materialization["validation"], "full_fit": True, "train_fraction": 1.0},
        "variant_policy": {
            "proposal_id": proposal["proposal_id"],
            "lane": proposal["lane"],
            "parent_experiment_id": proposal["parent_experiment_id"],
            "parent_public_score": proposal["parent_public_score"],
            "recipe_id": proposal["recipe_id"],
            "config_sha256": proposal["config_sha256"],
        },
        "artifact_contract": {
            "terminal_receipt": "artifacts/{run_id}/TRAINING_COMPLETE.json",
            "checkpoints": ["checkpoints/{run_id}/training_checkpoint.pth"],
            "minimum_checkpoints": 1,
            "maximum_checkpoints": 1,
        },
        "documentation": {"narrative": str(narrative.relative_to(root)), "strategy": ".agents/autonomous-exploration.json"},
        "kaggle": {
            "enabled": True,
            "deployment_ready": False,
            "training_allowed": False,
            "test_only": True,
            "accelerator": "NvidiaTeslaT4",
            "competition": "biohub-cell-tracking-during-development",
            "output_file": "submission.csv",
            "auto_publish_after_train": True,
            "auto_push": True,
            "auto_submit": True,
            "max_retries": 2,
            "poll_seconds": 60,
            "submission_columns": ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"],
        },
    }
    atomic_write_json(directory / "experiment.json", manifest)
    _write_kaggle_scaffold(proposal, directory, root)
    atomic_write_text(
        directory / "kaggle" / "README.md",
        "# Gate Kaggle\n\nEl notebook test-only ya está generado. `deployment_ready=false` hasta recuperar el checkpoint, exportar y publicar su dataset de pesos y pasar preflight.\n",
    )
    append_event(proposal, "MATERIALIZED", root=root, details={"experiment_directory": str(directory.relative_to(root))})
    queued = None
    if auto_queue:
        queued = queue_experiment(experiment_id, root)
        append_event(proposal, queued["status"], root=root, details={"queued_at": queued.get("queued_at")})
    render(root)
    return {"proposal_id": proposal["proposal_id"], "experiment_id": experiment_id, "directory": str(directory.relative_to(root)), "queue_status": queued and queued["status"]}


def reconcile(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or repo_root()
    queue = {item["experiment_id"]: item for item in load_queue(root).get("items", [])}
    scores = {row["experiment_id"]: row for row in best_by_experiment(root)}
    updates = []
    for proposal in list(latest_proposals(root).values()):
        experiment_id = proposal["experiment_id"]
        desired = proposal["status"]
        details: dict[str, Any] = {}
        if experiment_id in scores:
            desired = "SCORED"
            details = {"public_score": float(scores[experiment_id]["public_score"])}
        elif latest_status_for_experiment(experiment_id, root) in {"SUCCEEDED", "DOC_PENDING", "CLOSED"}:
            desired = "TRAINED"
        elif experiment_id in queue:
            desired = queue[experiment_id]["status"]
        if desired != proposal["status"]:
            updated = append_event(proposal, desired, root=root, details=details)
            updates.append(updated)
    if updates:
        render(root)
    return updates


def maintain_autonomous_queue(*, root: Path | None = None, dry_run: bool = False) -> dict[str, Any]:
    root = root or repo_root()
    policy, _ = _config(root)
    if policy.get("enabled_in_automatic_mode") is not True:
        return {"enabled": False, "planned": [], "materialized": []}
    bootstrap = list(policy.get("bootstrap_sequence", []))
    pending_bootstrap = [
        experiment_id
        for experiment_id in bootstrap
        if latest_status_for_experiment(experiment_id, root) not in {"SUCCEEDED", "DOC_PENDING", "CLOSED"}
    ]
    if pending_bootstrap:
        return {
            "enabled": True,
            "blocked_by_bootstrap": pending_bootstrap,
            "bootstrap_sequence": bootstrap,
            "planned": [],
            "materialized": [],
            "dry_run": dry_run,
        }
    plateau = _plateau(policy, root)
    if plateau:
        return {"enabled": True, "paused": True, **plateau, "planned": [], "materialized": [], "dry_run": dry_run}
    if not dry_run:
        reconcile(root)
    latest = latest_proposals(root)
    active = [event for event in latest.values() if event.get("status") in ACTIVE_STATES]
    target = int(policy["budgets"]["target_active_variants"])
    missing = max(0, target - len(active))
    if missing == 0:
        return {"enabled": True, "active": len(active), "planned": [], "materialized": []}
    count = min(missing, int(policy["allocation"]["planning_batch_size"]))
    seed = int(sha256_bytes(f"{utc_now()[:10]}|{len(read_events(root))}".encode())[:8], 16)
    proposals = plan(count, seed=seed, root=root, persist=not dry_run)
    materialized = []
    if not dry_run and policy["materialization"].get("auto_materialize"):
        for proposal in proposals:
            materialized.append(materialize(proposal, root=root, auto_queue=bool(policy["materialization"].get("auto_queue"))))
    return {"enabled": True, "active": len(active), "planner_seed": seed, "planned": proposals, "materialized": materialized, "dry_run": dry_run}


def render(root: Path | None = None) -> Path:
    root = root or repo_root()
    latest = list(latest_proposals(root).values())
    lines = [
        "# Variantes autónomas",
        "",
        "> Vista generada desde `experiments/variant_proposals.jsonl`. Kaggle Public Score es la señal primaria; el test local permanece separado.",
        "",
        "| Experimento | Carril | Padre | Score padre | Receta | Estado | Public Score |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for row in sorted(latest, key=lambda item: item["experiment_id"]):
        score = row.get("public_score")
        lines.append(
            f"| {row['experiment_id']} | {row['lane']} | {row['parent_experiment_id']} | {float(row['parent_public_score']):.4f} | "
            f"{row['recipe_id']} | **{row['status']}** | {float(score):.4f} |" if score is not None else
            f"| {row['experiment_id']} | {row['lane']} | {row['parent_experiment_id']} | {float(row['parent_public_score']):.4f} | "
            f"{row['recipe_id']} | **{row['status']}** | — |"
        )
    if not latest:
        lines.append("| — | — | — | — | — | Sin propuestas | — |")
    lines += [
        "",
        "## Política",
        "",
        "- 60% explotación de los mejores linajes Kaggle.",
        "- 25% exploración adyacente con cambios arquitectónicos cercanos.",
        "- 15% exploración novedosa; en cada lote completo hay al menos una variante novedosa.",
        "- Ninguna variante puede promoverse por `train_loss`; solo sirve como diagnóstico de ejecución.",
        "- Una carpeta materializada sigue con `kaggle.deployment_ready=false` hasta tener pesos exportados y notebook test-only validado.",
        "",
    ]
    path = root / "experiments" / "AUTONOMOUS_VARIANTS.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--count", type=int, default=5)
    plan_parser.add_argument("--seed", type=int, default=20260828)
    plan_parser.add_argument("--persist", action="store_true")
    materialize_parser = sub.add_parser("materialize")
    materialize_parser.add_argument("proposal_id")
    materialize_parser.add_argument("--queue", action="store_true")
    sub.add_parser("reconcile")
    sub.add_parser("render")
    sub.add_parser("verify")
    sub.add_parser("status")
    args = parser.parse_args()
    root = repo_root()
    if args.command == "plan":
        result = {"proposals": plan(max(1, args.count), seed=args.seed, root=root, persist=args.persist), "persisted": args.persist}
    elif args.command == "materialize":
        proposal = latest_proposals(root).get(args.proposal_id)
        if proposal is None:
            raise ExperimentError(f"proposal_id desconocido: {args.proposal_id}")
        result = materialize(proposal, root=root, auto_queue=args.queue)
    elif args.command == "reconcile":
        result = {"updates": reconcile(root)}
    elif args.command == "render":
        result = {"view": str(render(root).relative_to(root))}
    elif args.command == "verify":
        errors = verify_ledger(root)
        result = {"valid": not errors, "errors": errors}
    else:
        result = {"proposals": list(latest_proposals(root).values())}
    print(json.dumps({"ok": True, "action": f"variants-{args.command}", **result}, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
