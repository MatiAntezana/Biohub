"""Small, permutation-invariant image/geometry head for division hypotheses.

Scores are experimental, not claimed calibrated probabilities. Only confirmed
annotated parents generate negative labels; missing annotations are unknown.
"""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from src.tracking.contracts import verify_file

FEATURE_DIM = 11
PATCH_SHAPE = (9, 25, 25)
PATCH_HALF_UM = 16.0


class MitosisHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv3d(4, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(64 + FEATURE_DIM, 32), nn.SiLU(), nn.Dropout(.2), nn.Linear(32, 1))

    def forward(self, image, geometry):
        return self.classifier(torch.cat([self.image(image), geometry], dim=1)).squeeze(-1)


def example_tensors(triple, source):
    mother, a, b = np.asarray(triple, float)
    if a[0] != mother[0] + 1 or b[0] != a[0]:
        raise ValueError("Mitosis examples require a mother and two daughters in the next frame")
    va, vb = (a[1:] - mother[1:]) * source.scale, (b[1:] - mother[1:]) * source.scale
    da, db = np.linalg.norm(va), np.linalg.norm(vb)
    cosine = float(va @ vb / max(da * db, 1e-6))
    geometry = [min(da, db) / 12, max(da, db) / 12, np.linalg.norm(va - vb) / 12,
                cosine, abs(da - db) / 12, np.linalg.norm((va + vb) * .5) / 12, abs(va[0] - vb[0]) / 12]
    images, flags = [], []
    for t in range(int(mother[0]) - 1, int(mother[0]) + 3):
        valid = 0 <= t < source.shape[0]
        flags.append(float(valid))
        images.append(source.patch(t, mother[1:], half_um=PATCH_HALF_UM, shape=PATCH_SHAPE)
                      if valid else np.zeros(PATCH_SHAPE, np.float32))
    return np.stack(images), np.asarray(geometry + flags, np.float32)


def confirmed_label(mother, a, b, gt_parents):
    if a not in gt_parents or b not in gt_parents:
        return None
    return int(gt_parents[a] == mother and gt_parents[b] == mother)


def annotation_examples(coords, edges, scale, config):
    """Generate positives and confirmed different-parent negatives from a GT graph.

    No random train/validation split, no unknown-to-negative conversion. A mother
    with an annotated outgoing edge supplies the anchor; both proposed daughters
    must have an explicitly annotated parent. Negative sampling is deterministic.
    """
    children, parents = defaultdict(list), {}
    for u, v in edges:
        children[int(u)].append(int(v))
        if int(v) in parents:
            raise ValueError("GT child has multiple parents")
        parents[int(v)] = int(u)
    examples = []
    for t in np.unique(coords[:, 0]):
        next_ids = np.flatnonzero(coords[:, 0] == t + 1)
        if not len(next_ids):
            continue
        tree = cKDTree(coords[next_ids, 1:] * scale)
        for mother in np.flatnonzero(coords[:, 0] == t):
            actual = sorted(children[int(mother)])
            if not actual:
                continue
            if len(actual) == 2:
                examples.append((int(mother), *actual, 1))
            candidates = tree.query_ball_point(coords[mother, 1:] * scale, config["radius_um"])
            candidates = sorted((int(next_ids[i]) for i in candidates),
                key=lambda i: (np.linalg.norm((coords[i, 1:] - coords[mother, 1:]) * scale), i))[:config["max_children"]]
            negatives = 0
            for a, b in combinations(candidates, 2):
                if confirmed_label(int(mother), a, b, parents) == 0:
                    examples.append((int(mother), a, b, 0))
                    negatives += 1
                    if negatives >= config["max_negatives_per_mother"]:
                        break
    return examples


def load_head(spec, dataset_row, device):
    verify_file(spec["checkpoint"], spec["sha256"])
    payload = torch.load(spec["checkpoint"], map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("feature_dim") != FEATURE_DIM:
        raise ValueError("Unsupported mitosis head checkpoint")
    if dataset_row["split"] == "confirmation":
        used = set(payload["training_groups"]) | set(payload["selection_groups"])
        if dataset_row["group"] in used:
            raise ValueError("Confirmation biological group was used to train or select the head")
    head = MitosisHead().to(device)
    head.load_state_dict(payload["state_dict"], strict=True)
    return head.eval()


def score_divisions(coords, pairs, probabilities, source, spec, row, config, device="cuda", fixed_count=None):
    head = load_head(spec, row, device)
    fixed_count = len(coords) if fixed_count is None else fixed_count
    outgoing = defaultdict(list)
    for (u, v), p in zip(pairs, probabilities):
        if u < fixed_count and v < fixed_count and p >= config["min_edge_probability"]:
            distance = np.linalg.norm((coords[v, 1:] - coords[u, 1:]) * source.scale)
            if distance <= config["radius_um"]:
                outgoing[int(u)].append((int(v), float(p)))
    approved, batch = [], []

    def flush():
        if not batch:
            return
        tensors = [example_tensors(coords[list(ids)], source) for ids in batch]
        images = torch.as_tensor(np.stack([x[0] for x in tensors]), device=device)
        geometry = torch.as_tensor(np.stack([x[1] for x in tensors]), device=device)
        with torch.inference_mode():
            scores = head(images, geometry).sigmoid().cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite mitosis scores")
        for ids, score in zip(batch, scores):
            if score >= config["threshold"]:
                approved.append((*ids, float(score)))
        batch.clear()

    for mother, candidates in sorted(outgoing.items()):
        children = [v for v, _ in sorted(candidates, key=lambda item: (-item[1], item[0]))[:config["max_children"]]]
        for a, b in combinations(sorted(children), 2):
            batch.append((mother, a, b))
            if len(batch) >= config["batch_size"]:
                flush()
    flush()
    del head
    return approved


def add_confirmed_pairs(anchor, divisions):
    """Conservative ablation: keep every anchor edge, add only a free second child.

    A pair must include the mother's existing single child and a child without
    a parent. Competing hypotheses are resolved in descending learned score.
    No nodes are added or moved and no existing divisions are deleted.
    """
    from src.tracking.contracts import Lineage
    children, parents = defaultdict(set), {}
    edges = set(map(tuple, anchor.edges))
    for u, v in edges:
        children[int(u)].add(int(v))
        parents[int(v)] = int(u)
    added = []
    for mother, a, b, score in sorted(divisions, key=lambda item: (-item[3], *item[:3])):
        existing = children[mother]
        pair = {a, b}
        if len(existing) != 1 or not existing.issubset(pair):
            continue
        other = next(iter(pair - existing))
        if other in parents:
            continue
        if anchor.coords[other, 0] != anchor.coords[mother, 0] + 1:
            raise ValueError("Non-consecutive mitosis hypothesis")
        edges.add((mother, other))
        existing.add(other)
        parents[other] = mother
        added.append({"mother": mother, "daughter": other, "head_score": score})
    result = Lineage(anchor.coords.copy(), np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 2)).validate()
    return result, {"approved_pairs": len(divisions), "added_divisions": added,
        "preserved_reference_nodes": len(anchor.coords), "preserved_reference_edges": len(anchor.edges)}


def train_head(examples_path, manifest, output, config, device="cuda"):
    from src.tracking.images import ImageSource
    rows = {r["name"]: r for r in manifest["datasets"]}
    records = [json.loads(line) for line in Path(examples_path).read_text().splitlines() if line.strip()]
    train = [r for r in records if rows[r["dataset"]]["split"] == "train"]
    validation = [r for r in records if rows[r["dataset"]]["split"] == "validation"]
    fixed_epochs = config.get("selection") == "fixed_epochs"
    if fixed_epochs and validation:
        raise ValueError("Fixed-epoch training must not read an evaluation group")
    for split in ((train,) if fixed_epochs else (train, validation)):
        if {r["label"] for r in split} != {0, 1}:
            raise ValueError("Both train and validation need confirmed positive and negative examples")
    for r in records:
        if r["group"] != rows[r["dataset"]]["group"] or r["split"] != rows[r["dataset"]]["split"]:
            raise ValueError("Example provenance differs from the frozen manifest")
        if r["evidence"] != {0: "confirmed_different_parent", 1: "gt_division"}.get(r["label"]):
            raise ValueError("Unknown annotations cannot be used as negative examples")
        coords = np.asarray(r["coords"], float)
        if coords.shape != (3, 4) or not np.isfinite(coords).all() or (coords < 0).any():
            raise ValueError("Invalid annotated mother/daughter coordinates")
    training_groups = sorted({rows[r["dataset"]]["group"] for r in train})
    selection_groups = sorted({rows[r["dataset"]]["group"] for r in validation})
    if set(training_groups) & set(selection_groups):
        raise ValueError("Mitosis train/validation biological groups overlap")
    torch.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    train_positive = sum(int(r["label"]) for r in train)
    train_negative = len(train) - train_positive
    if not train_positive or not train_negative:
        raise ValueError("Both classes are required for mitosis training")
    # Division examples are sparse.  Without this correction, the BCE optimum
    # is a near-all-negative classifier (the public panel is ~1:48).  This
    # only changes the training loss; the held-out loss remains unweighted and
    # the decision threshold still requires separate remote validation.
    positive_weight = float(train_negative / train_positive)
    head = MitosisHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["learning_rate"], weight_decay=.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    selection_loss_fn = nn.BCEWithLogitsLoss()
    # Materialise each public image patch exactly once. Re-reading four 3-D
    # frames per example *per epoch* turns a small CNN into an hours-long I/O
    # job and risks an interrupted run before any useful checkpoint is saved.
    # The cache lives under this unique run directory and is never shared with
    # inference or Kaggle.
    cache_dir = Path(output).parent
    image_cache = cache_dir / "mitosis_images.npy"
    geometry_cache = cache_dir / "mitosis_geometry.npy"
    images = np.lib.format.open_memmap(image_cache, mode="w+", dtype=np.float32,
        shape=(len(records), 4, *PATCH_SHAPE))
    geometry = np.lib.format.open_memmap(geometry_cache, mode="w+", dtype=np.float32,
        shape=(len(records), FEATURE_DIM))
    by_dataset = defaultdict(list)
    for index, record in enumerate(records):
        by_dataset[record["dataset"]].append((index, record))
    for name in sorted(by_dataset):
        row = rows[name]
        source = ImageSource(row["image"], row.get("image_key", "0"), row["scale_um"])
        # Sorting by mother frame lets the source's bounded four-frame cache
        # cover each temporal context without holding an entire movie in RAM.
        for index, record in sorted(by_dataset[name], key=lambda item: (item[1]["coords"][0][0], item[0])):
            im, geom = example_tensors(np.asarray(record["coords"], float), source)
            images[index], geometry[index] = im, geom
    images.flush()
    geometry.flush()
    labels = np.asarray([record["label"] for record in records], dtype=np.float32)
    record_index = {id(record): index for index, record in enumerate(records)}

    def batch_tensors(batch):
        indices = np.asarray([record_index[id(record)] for record in batch], dtype=np.int64)
        return (torch.as_tensor(np.asarray(images[indices]), device=device),
                torch.as_tensor(np.asarray(geometry[indices]), device=device),
                torch.as_tensor(labels[indices], dtype=torch.float32, device=device))

    best, best_state, history = float("inf"), None, []
    progress_path = Path(output).parent / "training_progress.json"
    for epoch in range(config["epochs"]):
        head.train()
        # Shuffle sequences and examples within them, not frames between splits.
        shuffled = []
        sequence_names = sorted({r["dataset"] for r in train})
        rng.shuffle(sequence_names)
        for name in sequence_names:
            group = [r for r in train if r["dataset"] == name]
            rng.shuffle(group)
            shuffled.extend(group)
        train_loss = 0.0
        for begin in range(0, len(shuffled), config["batch_size"]):
            im, geom, target = batch_tensors(shuffled[begin:begin + config["batch_size"]])
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(head(im, geom), target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite mitosis training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            train_loss += float(loss.detach()) * len(target)
        head.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for begin in range(0, len(validation), config["batch_size"]):
                im, geom, target = batch_tensors(validation[begin:begin + config["batch_size"]])
                validation_loss += float(selection_loss_fn(head(im, geom), target)) * len(target)
        validation_loss = validation_loss / len(validation) if validation else None
        history.append({"epoch": epoch + 1, "train_bce": train_loss / len(train), "selection_bce": validation_loss})
        print(json.dumps(history[-1]), flush=True)
        if fixed_epochs or validation_loss < best:
            best = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        progress_path.write_text(json.dumps({"status": "training", "history": history,
            "best_selection_bce": best, "class_balance": {"positive": train_positive,
            "negative": train_negative, "positive_loss_weight": positive_weight}}, indent=2) + "\n")
    if best_state is None:
        raise RuntimeError("No finite mitosis checkpoint was selected")
    payload = {"schema_version": 1, "feature_dim": FEATURE_DIM, "state_dict": best_state,
        "training_groups": training_groups, "selection_groups": selection_groups,
        "calibrated": False, "class_balance": {"positive": train_positive,
        "negative": train_negative, "positive_loss_weight": positive_weight},
        "scope": "fixed_epochs_excluded_evaluation_group" if fixed_epochs else "head_selection_only_detectors_are_full_fit", "config": config}
    with Path(output).open("xb") as handle:
        torch.save(payload, handle)
    progress_path.write_text(json.dumps({"status": "completed", "history": history,
        "best_selection_bce": best, "checkpoint": str(output), "class_balance": payload["class_balance"]}, indent=2) + "\n")
    return {"history": history, "training_groups": training_groups, "selection_groups": selection_groups,
            "training_examples": len(train), "selection_examples": len(validation),
            "class_balance": {"positive": train_positive, "negative": train_negative,
                "positive_loss_weight": positive_weight}, "tensor_cache": {
                "images": str(image_cache), "geometry": str(geometry_cache),
                "examples": len(records), "augmentation": "none_cached_exact_image_patches"},
            "calibrated": False, "threshold_requires_remote_validation": True}
