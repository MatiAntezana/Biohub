"""008: fixed, low-capacity representation probes, never production models.

Use only the historical source-group encoder (NOT the rejected 007 encoder).
Fit all six convex probes before cross-group development scoring. No validation
selection, threshold sweep, graph construction, or confirmation evaluation.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch

from core.artifacts import ROOT, code_manifest, digest, dump
from core.mitosis_learning import (config as learning_config, verify_learning_source,
    open_cache, batch_tensors, development_indices, classification_metrics)
from core.nextgen.contracts import verify_file
from core.nextgen.mitosis import load_head, FEATURE_DIM
from core.protocol import PROTOCOL

SCOPE = "representation_assay_only_no_submission"
VARIANTS = ("geometry", "frozen_image", "geometry_frozen_image")
ADDITIONS = {"core/mitosis_probe.py", "configs/mitosis_probe.json", "tests/test_mitosis_probe.py"}
DRIVERS = {"ops/biohub.py", "ops/verify.py"}
GEOMETRY_NAMES = ("min_distance", "max_distance", "daughter_distance", "cosine",
    "distance_asymmetry", "midpoint_distance", "axial_separation",
    "frame_t_minus_1", "frame_t", "frame_t_plus_1", "frame_t_plus_2")
IMAGE_NAMES = tuple(f"frozen_image_{i:02d}" for i in range(64))


def config():
    return json.loads((ROOT / "configs/mitosis_probe.json").read_text())


def verify_probe_source(source, manifest):
    cfg = config()
    previous = {p: sha for p, sha in manifest.items() if p not in ADDITIONS}
    science = {p: sha for p, sha in previous.items() if p not in DRIVERS}
    identity = hashlib.sha256(json.dumps(science, sort_keys=True).encode()).hexdigest()
    if identity != cfg["parent_science_sha256"]:
        raise ValueError("Probe requires unchanged 007 science; no unreviewed additions")
    if source.name != cfg["source_run"]:
        raise ValueError("Probe requires the pinned 005 prepared source")
    verify_learning_source(source, previous)


def validate_recipe(cfg):
    expected = config()
    if cfg != expected or tuple(cfg["variants"]) != VARIANTS or cfg["scope"] != SCOPE:
        raise ValueError("Only the exact predeclared probe recipe is supported")


def feature_matrix(geometry, image, variant):
    geometry, image = np.asarray(geometry, float), np.asarray(image, float)
    if geometry.ndim != 2 or geometry.shape[1] != FEATURE_DIM or image.shape != (len(geometry), 64):
        raise ValueError("Probe feature dimensions differ from the frozen encoder")
    if not np.isfinite(geometry).all() or not np.isfinite(image).all():
        raise ValueError("Non-finite probe features")
    if variant == "geometry":
        return geometry, GEOMETRY_NAMES
    if variant == "frozen_image":
        return image, IMAGE_NAMES
    if variant == "geometry_frozen_image":
        return np.concatenate([geometry, image], axis=1), GEOMETRY_NAMES + IMAGE_NAMES
    raise ValueError("Unregistered representation variant")


def standardize_training(x, floor):
    x = np.asarray(x, float)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all() or floor <= 0:
        raise ValueError("Invalid training features for standardization")
    mean, std = x.mean(axis=0), x.std(axis=0)
    active = std >= floor
    scale = np.where(active, std, 1.)
    stats = {"mean": mean.tolist(), "scale": scale.tolist(), "active": active.tolist()}
    return transform(x, stats), stats


def transform(x, stats):
    x = np.asarray(x, float)
    mean, scale, active = (np.asarray(stats[k]) for k in ("mean", "scale", "active"))
    if (x.ndim != 2 or mean.shape != (x.shape[1],) or scale.shape != mean.shape
            or active.shape != mean.shape or not np.isfinite(x).all()
            or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("Invalid fixed training standardizer")
    # Constant train columns cannot acquire an effect from the evaluated group.
    return ((x - mean) / scale) * active


def class_weights(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Probe fitting requires both confirmed binary classes")
    return np.where(labels == 1, .5 / (labels == 1).sum(), .5 / (labels == 0).sum())


def objective(theta, x, labels, sample_weight, l2):
    """Half of each class-mean BCE + lambda/2 ||w||²; intercept unpenalized."""
    w, intercept = theta[:-1], theta[-1]
    logits = x @ w + intercept
    loss = float(sample_weight @ (np.logaddexp(0, logits) - labels * logits) + .5 * l2 * (w @ w))
    residual = sample_weight * (expit(logits) - labels)
    grad = np.r_[x.T @ residual + l2 * w, residual.sum()]
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise ValueError("Non-finite regularized probe objective")
    return loss, grad


def fit_probe(x, labels, cfg):
    validate_recipe(cfg)
    x, stats = standardize_training(x, cfg["std_floor"])
    labels = np.asarray(labels, float)
    weight = class_weights(labels)
    if len(labels) != len(x):
        raise ValueError("Feature/label length mismatch")
    fitted = minimize(objective, np.zeros(x.shape[1] + 1), args=(x, labels, weight, cfg["l2"]),
        method=cfg["optimizer"], jac=True,
        options={k: cfg[k] for k in ("maxiter", "ftol", "gtol")})
    loss, gradient = objective(fitted.x, x, labels, weight, cfg["l2"])
    norm = float(np.max(np.abs(gradient)))
    if not fitted.success or norm > cfg["max_gradient_inf"]:
        raise ValueError(f"Probe optimizer did not converge: {fitted.message}; gradient_inf={norm}")
    return {"weights": fitted.x[:-1].tolist(), "intercept": float(fitted.x[-1]),
        "standardizer": stats, "fit": {"success": True, "iterations": int(fitted.nit),
        "objective": loss, "gradient_inf": norm, "message": str(fitted.message)}}


def predict_probe(model, x):
    x = transform(x, model["standardizer"])
    weights, intercept = np.asarray(model["weights"], float), float(model["intercept"])
    if weights.shape != (x.shape[1],) or not np.isfinite(weights).all() or not np.isfinite(intercept):
        raise ValueError("Invalid linear probe coefficients")
    return x @ weights + intercept


def probe_metrics(labels, logits, threshold):
    result = classification_metrics(labels, logits, threshold)
    labels, logits = np.asarray(labels, int), np.asarray(logits, float)
    npos = int(labels.sum())
    # Non-interpolated average precision, processing complete tied-score groups.
    # AP summarizes ranking; no operating threshold is selected or exported.
    if npos:
        order = np.argsort(-logits, kind="stable")
        cumulative = np.cumsum(labels[order])
        ends = np.r_[np.flatnonzero(np.diff(logits[order]) != 0), len(labels) - 1]
        tp = cumulative[ends]
        ap = float((np.diff(np.r_[0, tp]) / npos) @ (tp / (ends + 1)))
    else:
        ap = None
    result.update(average_precision=ap, positive_prevalence=npos / len(labels) if len(labels) else None)
    return result


def historical_spec(source, training_group):
    relative = f"train-{training_group}/mitosis_head.pt"
    return {"checkpoint": str(source / relative), "sha256": learning_config()["source_sha256"][relative]}


def extract_features(source, data_group, encoder_group, indices, device="cuda"):
    """No optimizer or gradients; one frozen historical encoder per training fold."""
    records, images, geometry = open_cache(source, data_group)
    spec = historical_spec(source, encoder_group)
    # Guard own-training extraction too: pin and verify exact encoder provenance.
    verify_file(spec["checkpoint"], spec["sha256"])
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    if payload["training_groups"] != [encoder_group] or payload["selection_groups"]:
        raise ValueError("Encoder provenance differs from the fixed source group")
    row = {"group": data_group, "split": "train" if data_group == encoder_group else "confirmation"}
    head = load_head(spec, row, device).requires_grad_(False).eval()
    features, historical_logits = [], []
    batch_size = config()["batch_size"]
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = indices[start:start + batch_size]
            im, geom = batch_tensors(images, geometry, selected, device)
            encoded = head.image(im)
            features.append(encoded.cpu().numpy())
            historical_logits.append(head.classifier(torch.cat([encoded, geom], dim=1)).squeeze(-1).cpu().numpy())
    encoded = np.concatenate(features) if features else np.empty((0, 64))
    logits = np.concatenate(historical_logits) if historical_logits else np.empty(0)
    geom = np.asarray(geometry[indices], float)
    feature_matrix(geom, encoded, "geometry_frozen_image")
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite frozen historical logits")
    return [records[i] for i in indices], geom, encoded, logits


def train_group(source, group, out, cfg):
    records, _, _ = open_cache(source, group)
    records, geometry, encoded, _ = extract_features(source, group, group, np.arange(len(records)))
    labels = np.asarray([r["label"] for r in records])
    models = {}
    for variant in VARIANTS:
        x, names = feature_matrix(geometry, encoded, variant)
        model = fit_probe(x, labels, cfg)
        model.update(schema="mitosis_linear_probe_v1", scope=SCOPE, variant=variant,
            feature_names=list(names), training_groups=[group], selection_groups=[], calibrated=False,
            encoder=historical_spec(source, group) if variant != "geometry" else None, recipe=cfg)
        model["training_only_metrics"] = probe_metrics(labels, predict_probe(model, x), cfg["evaluation_threshold"])
        path = out / (variant + ".json")
        with path.open("x") as stream:
            json.dump(model, stream, indent=2, allow_nan=False)
        models[variant] = {"path": str(path), "sha256": digest(path)}
        print(json.dumps({"training_group": group, "variant": variant, "fit": model["fit"]}), flush=True)
    return models


def load_probe(spec, data_group, expected_group, variant):
    verify_file(spec["path"], spec["sha256"])
    model = json.loads(Path(spec["path"]).read_text())
    if (model.get("schema") != "mitosis_linear_probe_v1" or model.get("scope") != SCOPE
            or model.get("variant") != variant or model.get("training_groups") != [expected_group]
            or model.get("selection_groups") != [] or data_group == expected_group):
        raise ValueError("Probe must exclude evaluated biological group and remain assay-only")
    validate_recipe(model["recipe"])
    if variant != "geometry" and model["encoder"]["sha256"] != historical_spec(ROOT, expected_group)["sha256"]:
        raise ValueError("Probe fitted with a different frozen image encoder")
    return model


def compare_development(source, models, cfg, output):
    result = {}
    names = set(PROTOCOL["development"])
    for group in PROTOCOL["group_prefixes"]:
        records, _, _ = open_cache(source, group)
        indices = development_indices(records, names)
        other = next(g for g in PROTOCOL["group_prefixes"] if g != group)
        rows, geometry, encoded, historical = extract_features(source, group, other, indices)
        labels = np.asarray([r["label"] for r in rows])
        logits = {"historical": historical}
        for variant in VARIANTS:
            model = load_probe(models[other][variant], group, other, variant)
            x, _ = feature_matrix(geometry, encoded, variant)
            logits[variant] = predict_probe(model, x)
        comparisons = {}
        for variant, values in logits.items():
            metrics = probe_metrics(labels, values, cfg["evaluation_threshold"])
            per_sample = {name: probe_metrics(labels[mask], values[mask], cfg["evaluation_threshold"])
                for name in sorted(names) if (mask := np.array([r["dataset"] == name for r in rows])).any()}
            comparisons[variant] = {"metrics": metrics, "per_sample": per_sample}
        for variant in VARIANTS:
            comparisons[variant]["delta_vs_historical"] = {key:
                comparisons[variant]["metrics"][key] - comparisons["historical"]["metrics"][key]
                if comparisons[variant]["metrics"][key] is not None and comparisons["historical"]["metrics"][key] is not None else None
                for key in ("auroc", "average_precision", "balanced_bce")}
        per_record = [{"dataset": r["dataset"], "source_record_index": int(i), "label": int(r["label"]),
            **{variant + "_logit": float(values[j]) for variant, values in logits.items()}}
            for j, (i, r) in enumerate(zip(indices, rows))]
        dump(output / (group + ".scores.json"), per_record)
        result[group] = {"evaluated_data_group": group, "encoder_and_probe_trained_on_group": other,
            "scope": "teacher_coordinates_development_not_proposals_or_tracking", "by_variant": comparisons}
    return result


def probe_job(args, run, source):
    verify_probe_source(source, code_manifest())
    cfg = config()
    validate_recipe(cfg)
    models = {}
    for group in PROTOCOL["group_prefixes"]:
        out = run / ("train-" + group)
        out.mkdir()
        models[group] = train_group(source, group, out, cfg)
        torch.cuda.empty_cache()
    dump(run / "models.json", models)
    # All fixed fits complete before reading cross-group development for scoring.
    development = compare_development(source, models, cfg, run)
    verify_probe_source(source, code_manifest())
    result = {"disposition": SCOPE, "recipe": cfg, "source_run": source.name,
        "source_assets_sha256": learning_config()["source_sha256"], "source_unchanged": True,
        "models": models, "development": development, "confirmation_evaluated": False,
        "tracking_evaluated": False, "threshold_selected": False, "model_selected": False,
        "kaggle_score": None}
    dump(run / "probe.json", result)
    return {"disposition": SCOPE, "probe_sha256": digest(run / "probe.json"),
        "source_unchanged": True, "confirmation_evaluated": False, "tracking_evaluated": False,
        "threshold_selected": False, "model_selected": False}
