"""Fixed balanced-learning assay, isolated from tracking and publication.

Only the sampling/loss recipe changes. Source tensors and historical heads are
read-only and pinned. Development is scored after training, never for selection.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn

from core.artifacts import ROOT, code_manifest, dump, digest
from core.mitosis_diagnostic import verify_diagnostic_source, score_summary
from core.nextgen.contracts import verify_file
from core.nextgen.mitosis import MitosisHead, FEATURE_DIM, PATCH_SHAPE, load_head
from core.protocol import PROTOCOL

ASSAY_ADDITIONS = {"core/mitosis_learning.py", "configs/mitosis_learning.json", "tests/test_mitosis_learning.py"}


def config():
    return json.loads((ROOT / "configs/mitosis_learning.json").read_text())


def verify_learning_source(source, current_manifest):
    cfg = config()
    if source.name != cfg["source_run"]:
        raise ValueError("Learning assay requires the exact pinned prepared run")
    for relative, sha in cfg["source_sha256"].items():
        verify_file(source / relative, sha)
    receipt = json.loads((source / "receipt.json").read_text())
    if receipt["code_hash"] != cfg["source_code_hash"]:
        raise ValueError("Learning source code hash differs")
    # Apply the previous exact-science check after excluding only the new assay.
    # Changes to the production trainer/head, features or protocol still reject.
    verify_diagnostic_source(receipt, {p: sha for p, sha in current_manifest.items() if p not in ASSAY_ADDITIONS})


def balanced_batches(labels, batch_size, steps, rng):
    labels = np.asarray(labels)
    if labels.ndim != 1 or set(labels.tolist()) != {0, 1}:
        raise ValueError("Balanced batches require both confirmed binary classes")
    if batch_size < 2 or batch_size % 2 or steps < 1:
        raise ValueError("Even positive batch size and positive step count required")
    pools = [np.flatnonzero(labels == k) for k in (0, 1)]
    buffers = [np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)]
    half = batch_size // 2
    for _ in range(steps):
        parts = []
        for k in (0, 1):
            while len(buffers[k]) < half:
                buffers[k] = np.concatenate([buffers[k], rng.permutation(pools[k])])
            parts.append(buffers[k][:half])
            buffers[k] = buffers[k][half:]
        yield rng.permutation(np.concatenate(parts))


def classification_metrics(labels, logits, threshold):
    labels, logits = np.asarray(labels), np.asarray(logits, float)
    if labels.ndim != 1 or logits.shape != labels.shape or not np.isfinite(logits).all() or not set(labels) <= {0, 1}:
        raise ValueError("Invalid learning evaluation arrays")
    # Stable sigmoid/loss; these are balanced-training scores, not calibrated posteriors.
    scores = np.exp(-np.logaddexp(0, -logits))
    positive, negative = labels == 1, labels == 0
    npos, nneg = int(positive.sum()), int(negative.sum())
    predictions = scores >= threshold
    tp, fp = int((predictions & positive).sum()), int((predictions & negative).sum())
    losses = np.logaddexp(0, logits) - labels * logits
    auroc = ((rankdata(logits)[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg)) if npos and nneg else None
    return {"examples": len(labels), "positive": npos, "negative": nneg,
        "auroc": float(auroc) if auroc is not None else None,
        "balanced_bce": float(.5 * (losses[positive].mean() + losses[negative].mean())) if npos and nneg else None,
        "unweighted_bce": float(losses.mean()) if len(labels) else None,
        "frozen_threshold": threshold, "tp": tp, "fp": fp, "fn": npos - tp, "tn": nneg - fp,
        "positive_recall": tp / npos if npos else None,
        "false_positive_rate": fp / nneg if nneg else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "positive_scores": score_summary(scores[positive], threshold),
        "negative_scores": score_summary(scores[negative], threshold)}


def open_cache(source, group):
    folder = source / ("train-" + group)
    records = [json.loads(s) for s in (folder / "examples.jsonl").read_text().splitlines()]
    images = np.load(folder / "mitosis_images.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(folder / "mitosis_geometry.npy", mmap_mode="r", allow_pickle=False)
    if images.shape != (len(records), 4, *PATCH_SHAPE) or geometry.shape != (len(records), FEATURE_DIM):
        raise ValueError("Pinned tensor cache shape mismatch")
    if images.dtype != np.float32 or geometry.dtype != np.float32:
        raise ValueError("Unexpected training tensor dtype")
    for row in records:
        if row["group"] != group or not row["dataset"].startswith(group + "_") or row["split"] != "train":
            raise ValueError("Training cache crosses biological groups")
        if row["label"] not in (0, 1) or row["evidence"] != {0: "confirmed_different_parent", 1: "gt_division"}[row["label"]]:
            raise ValueError("Unconfirmed training label")
    return records, images, geometry


def batch_tensors(images, geometry, indices, device):
    im, geom = np.asarray(images[indices]), np.asarray(geometry[indices])
    if not np.isfinite(im).all() or not np.isfinite(geom).all():
        raise ValueError("Non-finite source tensors")
    return torch.as_tensor(im, device=device), torch.as_tensor(geom, device=device)


def predict_cached(head, images, geometry, indices, batch_size, device):
    values = []
    head.eval()
    with torch.inference_mode():
        for i in range(0, len(indices), batch_size):
            im, geom = batch_tensors(images, geometry, indices[i:i + batch_size], device)
            values.extend(head(im, geom).cpu().numpy().tolist())
    logits = np.asarray(values, float)
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite head logits")
    return logits


def train_balanced(source, group, output, cfg, device="cuda"):
    if cfg["loss"] != "unweighted_bce_with_logits" or cfg["positive_loss_weight"] != 1 or cfg["positive_per_batch"] * 2 != cfg["batch_size"]:
        raise ValueError("Balanced sampling must not double-count positive weighting")
    if cfg["selection"] != "fixed_final_epoch":
        raise ValueError("Assay must not select epochs on development")
    records, images, geometry = open_cache(source, group)
    labels = np.asarray([r["label"] for r in records], np.float32)
    torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(cfg["seed"])
    head = MitosisHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    steps = math.ceil(len(records) / cfg["batch_size"])
    history = []
    for epoch in range(cfg["epochs"]):
        head.train()
        running, clipped = 0., 0
        norms, exposure = [], np.zeros(len(records), dtype=np.int64)
        for indices in balanced_batches(labels, cfg["batch_size"], steps, rng):
            im, geom = batch_tensors(images, geometry, indices, device)
            target = torch.as_tensor(labels[indices], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(im, geom), target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite balanced training loss")
            loss.backward()
            norm = float(nn.utils.clip_grad_norm_(head.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True))
            optimizer.step()
            running += float(loss.detach())
            norms.append(norm); clipped += int(norm > cfg["gradient_clip_norm"])
            np.add.at(exposure, indices, 1)
        row = {"epoch": epoch + 1, "steps": steps, "train_balanced_sampling_bce": running / steps,
            "all_negative_batches": 0, "gradient_norm_median": float(np.median(norms)),
            "gradient_clipped_fraction": clipped / steps,
            "draws_positive": int(exposure[labels == 1].sum()), "draws_negative": int(exposure[labels == 0].sum()),
            "unique_positive_seen": int((exposure[labels == 1] > 0).sum()),
            "unique_negative_seen": int((exposure[labels == 0] > 0).sum())}
        history.append(row)
        dump(output / "progress.json", {"status": "training", "history": history})
        print(json.dumps({"training_group": group, **row}), flush=True)
    payload = {"schema_version": 1, "feature_dim": FEATURE_DIM,
        "state_dict": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        "training_groups": [group], "selection_groups": [], "calibrated": False,
        "scope": "learning_assay_only_no_submission", "config": cfg,
        "class_balance": {"positive": int((labels == 1).sum()), "negative": int((labels == 0).sum()),
            "positive_loss_weight": 1., "sampled_positive_fraction": .5}}
    checkpoint = output / "mitosis_head.pt"
    with checkpoint.open("xb") as handle:
        torch.save(payload, handle)
    train_logits = predict_cached(head, images, geometry, np.arange(len(records)), cfg["batch_size"], device)
    metrics = classification_metrics(labels, train_logits, cfg["evaluation_threshold"])
    spec = {"checkpoint": str(checkpoint), "sha256": digest(checkpoint)}
    dump(output / "training.json", {"history": history, "model": spec,
        "final_checkpoint_training_only": metrics, "selection_groups": [], "training_groups": [group],
        "source_examples_sha256": digest(source / ("train-" + group) / "examples.jsonl")})
    return spec


def development_indices(records, names):
    return np.asarray([i for i, row in enumerate(records) if row["dataset"] in names], dtype=np.int64)


def compare_development(source, models, cfg, output, device="cuda"):
    result = {}
    names = set(PROTOCOL["development"])
    for group in PROTOCOL["group_prefixes"]:
        records, images, geometry = open_cache(source, group)
        indices = development_indices(records, names)
        other = next(g for g in PROTOCOL["group_prefixes"] if g != group)
        old_path = source / ("train-" + other) / "mitosis_head.pt"
        specs = {"historical": {"checkpoint": str(old_path), "sha256": cfg["source_sha256"][str(old_path.relative_to(source))]},
                 "balanced": models[other]}
        labels = np.asarray([records[i]["label"] for i in indices], dtype=int)
        comparisons = {}
        per_record = [{"dataset": records[i]["dataset"], "source_record_index": int(i), "label": int(records[i]["label"])} for i in indices]
        for variant, spec in specs.items():
            # Reuse the strict opposite-group loader; split name only invokes its guard.
            head = load_head(spec, {"group": group, "split": "confirmation"}, device)
            logits = predict_cached(head, images, geometry, indices, cfg["batch_size"], device)
            metrics = classification_metrics(labels, logits, cfg["evaluation_threshold"])
            for row, value in zip(per_record, logits):
                row[variant + "_logit"] = float(value)
            per_sample = {name: classification_metrics(labels[mask], logits[mask], cfg["evaluation_threshold"])
                for name in sorted(names) if (mask := np.array([r["dataset"] == name for r in per_record])).any()}
            comparisons[variant] = {"checkpoint_sha256": spec["sha256"], "metrics": metrics, "per_sample": per_sample}
            del head
        dump(output / (group + ".scores.json"), per_record)
        result[group] = {"evaluated_data_group": group, "head_trained_on_group": other,
            "examples": len(indices), "by_variant": comparisons,
            "scope": "teacher_coordinates_development_not_detector_proposals_or_tracking_score"}
    return result


def learning_job(args, run, source):
    verify_learning_source(source, code_manifest())
    cfg = config()
    models = {}
    # Complete both fixed training jobs BEFORE any development evaluation.
    for group in PROTOCOL["group_prefixes"]:
        out = run / ("train-" + group)
        out.mkdir()
        models[group] = train_balanced(source, group, out, cfg)
        torch.cuda.empty_cache()
    dump(run / "models.json", models)
    comparisons = compare_development(source, models, cfg, run)
    result = {"disposition": "learning_assay_only_no_submission", "models": models,
        "source_run": source.name, "source_code_hash": cfg["source_code_hash"],
        "source_assets_sha256": cfg["source_sha256"], "recipe": cfg,
        "development": comparisons, "confirmation_evaluated": False,
        "threshold_selected": False, "tracking_evaluated": False, "kaggle_score": None}
    dump(run / "learning.json", result)
    # Detect accidental mutation of every source file, including read-only tensors.
    verify_learning_source(source, code_manifest())
    return {"disposition": result["disposition"], "learning_sha256": digest(run / "learning.json"),
        "confirmation_evaluated": False, "tracking_evaluated": False, "source_unchanged": True}
