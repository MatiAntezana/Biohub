"""Predeclared folds, paired gates and provenance checks; no leaderboard tuning."""
import hashlib
import json
from pathlib import Path
import numpy as np
from core.artifacts import ROOT, digest

PROTOCOL = json.loads((ROOT / "configs/protocol.json").read_text())


def panel_names(names, public_info):
    names = sorted(names)
    development = PROTOCOL["development"]
    visible = PROTOCOL["visible"]
    if not set(development + visible).issubset(names):
        raise ValueError("Missing predeclared public samples")
    excluded = set(development + PROTOCOL["previously_exposed"])
    confirmation = []
    for group in PROTOCOL["group_prefixes"]:
        eligible = [n for n in names if n.startswith(group + "_") and n not in excluded]
        rank = lambda n: hashlib.sha256((PROTOCOL["split_seed"] + n).encode()).hexdigest()
        # Stratify by annotation counts and estimated density, never prediction scores.
        rich = sorted(eligible, key=lambda n: (-public_info[n]["divisions"], rank(n)))[:2]
        remaining = sorted(set(eligible) - set(rich), key=lambda n: (public_info[n]["estimated_nodes"], rank(n)))
        if len(remaining) < 4:
            raise ValueError("Insufficient unexposed confirmation samples")
        confirmation += rich + [remaining[len(remaining)//4], remaining[3*len(remaining)//4]]
    return {"development": development, "confirmation": confirmation, "visible": visible}


def assess(evaluation, phase):
    rules = PROTOCOL["promotion"]
    reasons = []
    summary = evaluation["summary"]
    for metric, minimum in (("score", rules[phase + "_score_delta_min"]),
        ("edge_jaccard", rules["minimum_edge_jaccard_delta"]),
        ("division_jaccard", rules["minimum_division_jaccard_delta"])):
        a, b = summary["reference"].get(metric), summary["candidate"].get(metric)
        if a is None or b is None or not np.isfinite([a, b]).all() or b - a < minimum:
            reasons.append(f"{phase}: {metric} improvement below {minimum} or undefined")
    for group, rows in evaluation["by_group"].items():
        a, b = rows["reference"].get("score"), rows["candidate"].get("score")
        if a is None or b is None or not np.isfinite([a, b]).all() or b - a < rules["minimum_group_score_delta"]:
            reasons.append(f"{phase}: score regression/undefined in group {group}")
    ref = {r["dataset"]: r for r in evaluation["rows"]["reference"]}
    cand = {r["dataset"]: r for r in evaluation["rows"]["candidate"]}
    if not ref or ref.keys() != cand.keys():
        reasons.append("Incomplete paired sample denominator")
    for name in ref.keys() & cand.keys():
        delta = cand[name]["adj_edge_jaccard"] - ref[name]["adj_edge_jaccard"]
        if not np.isfinite(delta) or delta < rules["minimum_sample_adjusted_jaccard_delta"]:
            reasons.append(f"{phase}: excessive sample regression: {name}")
    return {"passed": not reasons, "reasons": reasons, "phase": phase,
        "score_delta": summary["candidate"]["score"] - summary["reference"]["score"],
        "scope": "public_paired_diagnostic_full_fit_detector_not_kaggle_score"}


def resolve_dataset_row(name, data_root, *, split, gt=False):
    row = {"name": name, "group": name.split("_", 1)[0], "split": split,
        "image": str(Path(data_root) / (name + ".zarr")), "image_key": "0",
        "scale_um": [1.625, .40625, .40625]}
    if gt:
        row["gt"] = str(Path(data_root) / (name + ".geff"))
    return row
