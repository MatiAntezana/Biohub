"""035: Drift-Compensated Linear Track Extension (DC-LTE) & Bio-Conserved Mitosis Gating (BC-MG).

Builds on Campaign 032's DDKS-SPI (smoothing + fork pruning) and introduces
DC-LTE: linear extension of truncated cell tracks in the rest frame of tissue
drift vec{v}_global(t).

By matching terminal track endpoints (out_degree == 0 at frame t) exclusively
with orphan track initiations (in_degree == 0 at frame t+1):
  - Every extended node maintains out_degree <= 1 and in_degree <= 1.
  - Zero branches or fork divisions are introduced.
  - Mitosis false-positive risk is strictly ZERO.
  - Biological tissue continuity is recovered across transient detection dropouts.

Optionally, Variant 2 includes BC-MG (Bio-Conserved Mitosis Gate) requiring a
strict 6-layer biophysical conservation audit (mother track history >= 3,
daughter forward persistence >= 2, momentum conservation cos >= 0.45,
sister morphology similarity >= 0.75, mass conservation delta <= 0.35,
and detector probability margin >= 1.30x) before any cytokinesis second daughter
is reassigned.

Label-free candidate generation (GT never accessed during inference).
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from core.artifacts import ROOT, code_hash, code_manifest, digest, dump
from core import mitosis_counterfactual as official_eval
from core.nextgen.association import Lineage
from core.nextgen.contracts import load_reference, verify_file
from core.protocol import assess

# Re-use 032's smoothing and fork-pruning functions identically
from core.temporal_association_032 import (
    apply_harmonized_axial_regularizer,
    prune_spurious_cytokinesis_forks,
)

ADDITIONS = {
    "core/temporal_association_035.py",
    "configs/temporal_association_035.json",
    "tests/test_temporal_association_035.py",
    "ops/audit_temporal_inputs_035.py",
}


def config():
    return json.loads((ROOT / "configs/temporal_association_035.json").read_text())


def verify_sources(folder, manifest):
    folder = Path(folder)
    cfg = config()
    if folder.name != cfg["source_run"] and folder.name != "b0970-prepare-005":
        exp = cfg["source_run"]
        raise ValueError(f"Expected source run {exp} or b0970-prepare-005, got {folder.name}")
    receipt_path = folder / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing receipt in source run: {folder}")
    receipt = json.loads(receipt_path.read_text())
    task = receipt.get("task")
    status = receipt.get("status")
    if status != "completed":
        raise ValueError(f"Source run {folder.name} status is {status}, expected completed")

    if task == "prepare":
        plan_path = folder / "plan.json"
        if not plan_path.is_file():
            raise FileNotFoundError(f"Missing plan in prepare run: {folder}")
        plan = json.loads(plan_path.read_text())
        return plan

    corpus_path = folder / "corpus.json"
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Missing corpus in source run: {folder}")
    corpus = json.loads(corpus_path.read_text())
    for rel, expected in cfg["source_artifacts"].items():
        actual = digest(folder / rel)
        if actual != expected:
            raise ValueError(f"Corrupted source artifact {rel}: expected {expected}, got {actual}")
    return corpus


def audit_inputs(manifest):
    cfg = config()
    source_folder = ROOT / "artifacts" / cfg["node"] / cfg["source_run"]
    if not source_folder.is_dir():
        source_folder = ROOT / "artifacts/A1555" / cfg["source_run"]
    if not source_folder.is_dir():
        source_folder = ROOT / "artifacts/A1688" / cfg["source_run"]
    corpus = verify_sources(source_folder, manifest)
    names = corpus["development"]
    npz_stats = {}
    for name in names:
        p = source_folder / (name + ".evidence.npz")
        if not p.is_file():
            raise FileNotFoundError(f"Missing evidence NPZ: {p}")
        f = np.load(p)
        coords = f["coords"]
        ref_edges = f["reference_edges"]
        scale = f["scale"]
        npz_stats[name] = {
            "nodes": len(coords),
            "reference_edges": len(ref_edges),
            "scale": scale.tolist(),
            "group": name.split("_", 1)[0],
        }
    candidates_available = {}
    for name in names:
        cand_path = ROOT / "artifacts/A1688/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz")
        candidates_available[name] = cand_path.is_file()
    return {
        "status": "audit_passed",
        "source_run": cfg["source_run"],
        "development_datasets": len(names),
        "stats": npz_stats,
        "candidates_available": candidates_available,
    }, corpus


# ---------------------------------------------------------------------------
# Global drift computation
# ---------------------------------------------------------------------------

def compute_global_drift(coords, edges, scale):
    """Per-frame global median drift vector v_global(t) in physical units (um).

    For each frame t, computes the median displacement of all forward edges
    (t -> t+1).  Returns dict mapping int(t) -> np.array([dz, dy, dx]) in um.
    """
    if len(edges) == 0:
        return {}
    t_sources = coords[edges[:, 0], 0].astype(int)
    drifts = {}
    for t in np.unique(t_sources):
        mask = t_sources == t
        sub = edges[mask]
        t_targets = coords[sub[:, 1], 0].astype(int)
        fwd = t_targets == t + 1
        if np.any(fwd):
            disps = (coords[sub[fwd, 1], 1:] - coords[sub[fwd, 0], 1:]) * scale
            drifts[int(t)] = np.median(disps, axis=0)
        else:
            drifts[int(t)] = np.zeros(3)
    return drifts


def _build_anchor_to_cand_map(coords, scale, cand_npz):
    """Maps every anchor node to the nearest candidate-graph node using KDTree.

    Returns anchor_to_cand (int array) and cand_edges probability lookup dict.
    """
    from scipy.spatial import cKDTree

    cand_coords = cand_npz["coords"]
    cand_by_t = {}
    for t_val in np.unique(cand_coords[:, 0].astype(int)):
        mask = cand_coords[:, 0] == t_val
        idxs = np.flatnonzero(mask)
        tree = cKDTree(cand_coords[idxs, 1:] * scale)
        cand_by_t[int(t_val)] = (idxs, tree)

    anchor_to_cand = np.full(len(coords), -1, dtype=int)
    for t_val in np.unique(coords[:, 0].astype(int)):
        t_val = int(t_val)
        if t_val not in cand_by_t:
            continue
        mask_a = coords[:, 0].astype(int) == t_val
        idxs_a = np.flatnonzero(mask_a)
        idxs_c, tree = cand_by_t[t_val]
        dists, local_idxs = tree.query(coords[idxs_a, 1:] * scale)
        anchor_to_cand[idxs_a] = idxs_c[local_idxs]

    cand_edges = cand_npz["edges"]
    prob_map = {}
    for row in cand_edges:
        prob_map[(int(row[0]), int(row[1]))] = float(row[2])

    return anchor_to_cand, prob_map


# ---------------------------------------------------------------------------
# DC-LTE: Drift-Compensated Linear Track Extension
# ---------------------------------------------------------------------------

def apply_dc_lte(coords, edges, scale, lte_cfg, cand_npz=None, descriptors=None):
    """Drift-Compensated Linear Track Extension.

    Identifies terminal nodes u (out_degree == 0 at frame t) and orphan nodes v
    (in_degree == 0 at frame t+1). Evaluates candidate links (u -> v) in the
    tissue rest frame:
      vec{d}* = (r_v - r_u) - vec{v}_drift(t)
      dist* = ||vec{d}*|| <= d_max_um

    Filters by appearance similarity or detector probability:
      cos_sim(q_u, q_v) >= sim_min OR p(u -> v) >= p_min

    Performs greedy 1-to-1 disjoint matching ordered by distance dist*.
    Strictly preserves out_degree <= 1 and in_degree <= 1 (zero division risk).

    Returns (new_edges, audit_dict).
    """
    if not lte_cfg.get("enabled", False):
        return edges, {"extensions": 0, "terminals_scanned": 0, "orphans_scanned": 0}

    d_max_um = float(lte_cfg.get("d_max_um", 2.0))
    p_min = float(lte_cfg.get("p_candidate_min", 0.65))
    sim_min = float(lte_cfg.get("sim_min", 0.70))
    use_drift = bool(lte_cfg.get("global_drift_compensation", True))

    in_deg = Counter()
    out_deg = Counter()
    for u, v in edges:
        out_deg[int(u)] += 1
        in_deg[int(v)] += 1

    n_nodes = len(coords)
    t_coords = coords[:, 0].astype(int)

    # Compute global drift
    drifts = compute_global_drift(coords, edges, scale) if use_drift else {}

    # Probability map from candidate detector NPZ
    anchor_to_cand = None
    prob_map = {}
    if cand_npz is not None:
        anchor_to_cand, prob_map = _build_anchor_to_cand_map(coords, scale, cand_npz)

    # Group terminal and orphan nodes by frame
    terminals_by_t = defaultdict(list)
    orphans_by_t = defaultdict(list)
    for i in range(n_nodes):
        t_val = int(t_coords[i])
        if out_deg[i] == 0:
            terminals_by_t[t_val].append(i)
        if in_deg[i] == 0:
            orphans_by_t[t_val].append(i)

    # Normalize descriptors if available for cosine similarity
    norm_desc = None
    if descriptors is not None and len(descriptors) == n_nodes:
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        norm_desc = descriptors / norms

    candidate_extensions = []
    total_terminals = 0
    total_orphans = 0

    all_t = sorted(set(terminals_by_t.keys()))
    for t_val in all_t:
        terms = terminals_by_t.get(t_val, [])
        orphs = orphans_by_t.get(t_val + 1, [])
        if not terms or not orphs:
            continue

        total_terminals += len(terms)
        total_orphans += len(orphs)
        v_drift = drifts.get(t_val, np.zeros(3))

        for u in terms:
            r_u = coords[u, 1:] * scale
            for v in orphs:
                r_v = coords[v, 1:] * scale
                disp_star = (r_v - r_u) - v_drift
                dist_star = float(np.linalg.norm(disp_star))

                if dist_star > d_max_um:
                    continue

                # Verification check: appearance or detector probability
                p_u_v = 0.0
                if anchor_to_cand is not None:
                    raw_u = int(anchor_to_cand[u])
                    raw_v = int(anchor_to_cand[v])
                    p_u_v = prob_map.get((raw_u, raw_v), 0.0)

                sim_u_v = 0.0
                if norm_desc is not None:
                    sim_u_v = float(np.dot(norm_desc[u], norm_desc[v]))

                # Pass criteria: meets detector probability OR appearance similarity OR close geometric proximity
                pass_check = False
                if p_u_v >= p_min:
                    pass_check = True
                elif sim_u_v >= sim_min:
                    pass_check = True
                elif dist_star <= 0.75 * d_max_um and (p_u_v > 0.0 or norm_desc is None):
                    # Pure close proximity in rest frame when no negative evidence exists
                    pass_check = True

                if not pass_check:
                    continue

                candidate_extensions.append({
                    "u": u, "v": v, "dist_star": dist_star,
                    "p_u_v": p_u_v, "sim_u_v": sim_u_v, "t": t_val
                })

    # Sort candidate extensions by distance (greedy nearest-neighbor in rest frame)
    candidate_extensions.sort(key=lambda x: x["dist_star"])

    matched_u = set()
    matched_v = set()
    edges_to_add = []
    audit_items = []

    for item in candidate_extensions:
        u, v = item["u"], item["v"]
        if u in matched_u or v in matched_v:
            continue
        matched_u.add(u)
        matched_v.add(v)
        edges_to_add.append((u, v))
        audit_items.append(item)

    # Construct new edge array
    if edges_to_add:
        add_arr = np.array(edges_to_add, dtype=edges.dtype)
        new_edges = np.vstack([edges, add_arr])
    else:
        new_edges = edges.copy()

    audit = {
        "extensions": len(edges_to_add),
        "total_terminals": total_terminals,
        "total_orphans": total_orphans,
        "candidates_evaluated": len(candidate_extensions),
        "items": audit_items[:50],  # sample for inspection
    }
    return new_edges, audit


# ---------------------------------------------------------------------------
# BC-MG: Bio-Conserved Mitosis Gating
# ---------------------------------------------------------------------------

def apply_bc_mg(coords, edges, scale, reassign_cfg, cand_npz=None, descriptors=None, stats=None):
    """6-Layer Bio-Conserved Mitosis Gate for cytokinesis second daughters.

    Applies the 6 biophysical conservation criteria:
      Layer 1: Mother track history >= mother_history_min_frames (default 3)
      Layer 2: Daughter forward persistence >= daughter_persistence_min_frames (default 2)
      Layer 3: Center-of-mass momentum alignment cos(v_mid, v_in) >= momentum_cos_min (default 0.45)
      Layer 4: Sister morphological similarity cos(q_d1, q_d2) >= sister_morphology_sim_min (default 0.75)
      Layer 5: Mass/photometric conservation |(s_d1 + s_d2) - s_m| <= mass_conservation_delta_max (default 0.35)
      Layer 6: Probability margin over competitor p(m -> d2) / p(c -> d2) >= prob_margin_ratio_min (default 1.30)

    Returns (new_edges, audit_dict).
    """
    if not reassign_cfg.get("enabled", False):
        return edges, {"reassignments": 0, "candidates_scanned": 0}

    cos_max = float(reassign_cfg.get("cos_max", -0.60))
    sister_min_um = float(reassign_cfg.get("sister_distance_min_um", 3.0))
    sister_max_um = float(reassign_cfg.get("sister_distance_max_um", 12.0))
    ratio_min = float(reassign_cfg.get("distance_ratio_min", 0.35))
    p_cand_min = float(reassign_cfg.get("p_candidate_min", 0.85))
    comp_max_out = int(reassign_cfg.get("competitor_max_out_deg", 1))
    hist_min = int(reassign_cfg.get("mother_history_min_frames", 3))
    pers_min = int(reassign_cfg.get("daughter_persistence_min_frames", 2))
    momentum_min = float(reassign_cfg.get("momentum_cos_min", 0.45))
    morph_min = float(reassign_cfg.get("sister_morphology_sim_min", 0.75))
    mass_delta_max = float(reassign_cfg.get("mass_conservation_delta_max", 0.35))
    margin_min = float(reassign_cfg.get("prob_margin_ratio_min", 1.30))

    # Graph topology
    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    drifts = compute_global_drift(coords, edges, scale)

    anchor_to_cand = None
    prob_map = {}
    if cand_npz is not None:
        anchor_to_cand, prob_map = _build_anchor_to_cand_map(coords, scale, cand_npz)

    # Normalize descriptors if available
    norm_desc = None
    if descriptors is not None and len(descriptors) == len(coords):
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        norm_desc = descriptors / norms

    # Helper: compute backward track history length
    def track_history_len(node):
        length = 1
        curr = node
        while curr in parents and length <= hist_min + 1:
            curr = parents[curr]
            length += 1
        return length

    # Helper: compute forward track persistence length
    def track_forward_len(node):
        length = 1
        curr = node
        while curr in children and len(children[curr]) == 1 and length <= pers_min + 1:
            curr = children[curr][0]
            length += 1
        return length

    parents_by_t = defaultdict(list)
    for d2, c in parents.items():
        parents_by_t[int(coords[d2, 0])].append((d2, c))

    mothers = [m for m, ch in children.items() if len(ch) == 1]
    candidate_reassignments = []
    candidates_scanned = 0

    for m in mothers:
        d1 = children[m][0]
        t_m = int(coords[m, 0])
        if int(coords[d1, 0]) != t_m + 1:
            continue

        # Layer 1: Mother history check
        if track_history_len(m) < hist_min:
            continue

        r_m = coords[m, 1:] * scale
        r_d1 = coords[d1, 1:] * scale
        v_drift = drifts.get(t_m, np.zeros(3))

        v1_star = (r_d1 - r_m) - v_drift
        norm_v1 = float(np.linalg.norm(v1_star))
        if norm_v1 < 1e-6:
            continue

        # Ingoing mother velocity (for momentum conservation)
        v_in = None
        if m in parents:
            p_m = parents[m]
            r_pm = coords[p_m, 1:] * scale
            v_in = r_m - r_pm - drifts.get(int(coords[p_m, 0]), np.zeros(3))
            norm_vin = float(np.linalg.norm(v_in))
            if norm_vin > 1e-6:
                v_in = v_in / norm_vin
            else:
                v_in = None

        for d2, c in parents_by_t.get(t_m + 1, ()):
            if d2 == d1 or c == m:
                continue
            if int(coords[c, 0]) != t_m:
                continue
            if len(children[c]) > comp_max_out:
                continue

            candidates_scanned += 1
            r_d2 = coords[d2, 1:] * scale

            # Sister distance check
            sister_dist = float(np.linalg.norm(r_d1 - r_d2))
            if sister_dist < sister_min_um or sister_dist > sister_max_um:
                continue

            v2_star = (r_d2 - r_m) - v_drift
            norm_v2 = float(np.linalg.norm(v2_star))
            if norm_v2 < 1e-6:
                continue

            ratio = min(norm_v1, norm_v2) / max(norm_v1, norm_v2)
            if ratio < ratio_min:
                continue

            cos_theta = float(np.dot(v1_star, v2_star) / (norm_v1 * norm_v2))
            if cos_theta > cos_max:
                continue

            # Layer 2: Daughter forward persistence
            if track_forward_len(d1) < pers_min or track_forward_len(d2) < pers_min:
                continue

            # Layer 3: Momentum conservation
            if v_in is not None:
                v_mid = 0.5 * (r_d1 + r_d2) - r_m - v_drift
                norm_vmid = float(np.linalg.norm(v_mid))
                if norm_vmid > 1e-6:
                    cos_mom = float(np.dot(v_mid / norm_vmid, v_in))
                    if cos_mom < momentum_min:
                        continue

            # Layer 4: Sister morphological similarity
            if norm_desc is not None:
                sim_sisters = float(np.dot(norm_desc[d1], norm_desc[d2]))
                if sim_sisters < morph_min:
                    continue

            # Layer 5: Mass conservation & Tri-Photometric Invariant
            if stats is not None and len(stats) == len(coords):
                s_m = float(stats[m, 0]) if stats.ndim > 1 else float(stats[m])
                s_d1 = float(stats[d1, 0]) if stats.ndim > 1 else float(stats[d1])
                s_d2 = float(stats[d2, 0]) if stats.ndim > 1 else float(stats[d2])
                if s_m > 1e-4:
                    # 5a. Total mass conservation error
                    mass_rel_err = abs((s_d1 + s_d2) - s_m) / s_m
                    if mass_rel_err > mass_delta_max:
                        continue
                    # 5b. Bilateral sister contrast symmetry
                    min_s, max_s = min(s_d1, s_d2), max(s_d1, s_d2)
                    if (min_s / max(max_s, 1e-6)) < 0.60:
                        continue
                    # 5c. Individual daughter partition halving bounds [0.35, 0.70]
                    if not (0.35 <= (s_d1 / s_m) <= 0.70 and 0.35 <= (s_d2 / s_m) <= 0.70):
                        continue

            # Layer 6: Probability margin over competitor
            p_m_d2 = 0.0
            p_c_d2 = 0.0
            if anchor_to_cand is not None:
                raw_m = int(anchor_to_cand[m])
                raw_d2 = int(anchor_to_cand[d2])
                raw_c = int(anchor_to_cand[c])
                p_m_d2 = prob_map.get((raw_m, raw_d2), 0.0)
                p_c_d2 = prob_map.get((raw_c, raw_d2), 0.0)

            if p_m_d2 < p_cand_min:
                continue

            if p_c_d2 > 0 and (p_m_d2 / p_c_d2) < margin_min:
                continue

            candidate_reassignments.append({
                "m": m, "d1": d1, "d2": d2, "c": c,
                "cos_theta": cos_theta,
                "sister_dist": sister_dist,
                "ratio": ratio,
                "p_m_d2": p_m_d2,
            })

    candidate_reassignments.sort(key=lambda x: x["cos_theta"])
    assigned_mothers = set()
    assigned_daughters = set()
    reassign_edges_add = []
    reassign_edges_remove = []
    audit_items = []

    for cand in candidate_reassignments:
        m, d1, d2, c = cand["m"], cand["d1"], cand["d2"], cand["c"]
        if m in assigned_mothers or d2 in assigned_daughters:
            continue
        assigned_mothers.add(m)
        assigned_daughters.add(d2)
        reassign_edges_remove.append((c, d2))
        reassign_edges_add.append((m, d2))
        audit_items.append(cand)

    remove_set = set(reassign_edges_remove)
    new_edges_list = [e for e in edges if (int(e[0]), int(e[1])) not in remove_set]
    for u, v in reassign_edges_add:
        new_edges_list.append(np.array([u, v], dtype=edges.dtype))

    new_edges = np.array(new_edges_list, dtype=edges.dtype) if new_edges_list else edges[:0]
    audit = {
        "reassignments": len(audit_items),
        "candidates_scanned": candidates_scanned,
        "items": audit_items,
    }
    return new_edges, audit


# ---------------------------------------------------------------------------
# build_candidate_graph: 032 smoothing + fork pruning + DC-LTE + optional BC-MG
# ---------------------------------------------------------------------------

def build_candidate_graph(anchor, scale, descriptors, stats, variant_cfg, group,
                          cand_npz=None):
    """Constructs verified Lineage candidate under Campaign 035 rules.

    Pipeline:
      1. Harmonized axial smoothing (same as 032)
      2. Fork pruning DDKS-SPI     (same as 032)
      3. DC-LTE track extension    (NEW in 035 - zero division risk)
      4. BC-MG mitosis gate        (Optional in Variant 2)
    """
    smoothed_coords, audit_smoothing = apply_harmonized_axial_regularizer(
        anchor.coords, anchor.edges, scale, variant_cfg.get("smoothing", {})
    )

    pruned_edges, audit_pruning = prune_spurious_cytokinesis_forks(
        smoothed_coords, anchor.edges, scale, descriptors,
        variant_cfg.get("fork_pruning", {}), group=group
    )

    # 3. DC-LTE: Drift-Compensated Linear Track Extension
    lte_cfg = variant_cfg.get("dc_lte", {})
    extended_edges, audit_lte = apply_dc_lte(
        smoothed_coords, pruned_edges, scale, lte_cfg,
        cand_npz=cand_npz, descriptors=descriptors
    )

    # 4. Optional BC-MG: Bio-Conserved Mitosis Gate
    reassign_cfg = variant_cfg.get("mitosis_reassignment", {})
    final_edges, audit_reassign = apply_bc_mg(
        smoothed_coords, extended_edges, scale, reassign_cfg,
        cand_npz=cand_npz, descriptors=descriptors, stats=stats
    )

    cand = Lineage(smoothed_coords, final_edges)
    cand.validate()
    return cand, {
        "smoothing": audit_smoothing,
        "fork_pruning": audit_pruning,
        "dc_lte": audit_lte,
        "mitosis_reassignment": audit_reassign,
    }


def evaluate_association_panel(reference, candidate):
    """Evaluates paired panel summary without arbitrary node cardinality constraints."""
    if not reference or sorted(reference) != sorted(candidate):
        raise ValueError("Incomplete paired association panel")
    metric = official_eval.official_metrics()
    groups = sorted({r["group"] for r in reference.values()})
    rows = {"reference": list(reference.values()), "candidate": list(candidate.values())}
    summary = {k: metric.summarise(v) for k, v in rows.items()}
    grouped = {g: {k: metric.summarise([r for r in v if r["group"] == g]) for k, v in rows.items()} for g in groups}
    for summaries in [summary, *grouped.values()]:
        for s in summaries.values():
            if not np.isfinite([s[k] for k in ("score", "edge_jaccard", "adj_edge_jaccard", "division_jaccard")]).all():
                raise ValueError("Undefined association metric; no sample dropping")
    return dict(rows=rows, summary=summary, by_group=grouped, uses_gt_for_edits=False,
        scope="public_paired_diagnostic_full_fit_detector_not_kaggle_score", promoted=False)


def _load_cand_npz(name, source_path):
    search_paths = [
        source_path / "cache-development/raw" / (name + "_candidates.npz"),
        source_path.parent / "b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
        source_path.parent / "b0970-prepare-005/cache-development/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1688/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1555/b0970-prepare-004/cache-development/raw" / (name + "_candidates.npz"),
    ]
    for p in search_paths:
        if p.is_file():
            return np.load(p)
    return None


def _load_cand_npz_confirm(name, source_path):
    search_paths = [
        source_path / "cache-confirmation/raw" / (name + "_candidates.npz"),
        source_path.parent / "b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1688/b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
        ROOT / "artifacts/A1555/b0970-prepare-005/cache-confirmation/raw" / (name + "_candidates.npz"),
    ]
    for p in search_paths:
        if p.is_file():
            return np.load(p)
    return None


def run_paired_evaluation(source_run, corpus, names, variant_cfg):
    """Runs strict label-free candidate generation, then official GT evaluation on remote node."""
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph

    source_path = Path(source_run)
    refs = {}
    cands = {}
    audits = {}
    metric = official_eval.official_metrics()

    for name in sorted(names):
        group = name.split("_", 1)[0]
        evidence_path = source_path / (name + ".evidence.npz")
        f = np.load(evidence_path)
        raw_coords = f["coords"]
        raw_edges = f["reference_edges"]
        desc = f["descriptors"] if "descriptors" in f else None
        stats = f["stats"] if "stats" in f else None
        scale = f["scale"]

        anchor = Lineage(raw_coords, raw_edges)
        anchor.validate()

        cand_npz = _load_cand_npz(name, source_path)

        # 1. Label-free candidate generation (smoothing + pruning + DC-LTE + optional BC-MG)
        cand_graph, audit = build_candidate_graph(
            anchor, scale, desc, stats, variant_cfg, group=group,
            cand_npz=cand_npz
        )
        audits[name] = audit

        # 2. Open GT for official metric evaluation ONLY
        gt_path = Path(corpus["data_root"]) / (name + ".geff")
        if not gt_path.exists():
            local_alt = ROOT / "cache-gt" / (name + ".geff")
            if local_alt.exists():
                gt_path = local_alt
            else:
                raise FileNotFoundError(f"GT geff missing: {gt_path}")

        gt = load_graph(gt_path)
        estimated = (GeffMetadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
        if estimated is None or not np.isfinite(estimated) or estimated <= 0:
            estimated = float(len(anchor.coords))

        row = {"name": name, "group": group, "split": "development",
               "scale_um": scale.tolist(), "gt": str(gt_path)}

        def eval_graph(g):
            nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(g.coords)}
            edges_list = [{"source_id": int(u), "target_id": int(v)} for u, v in g.edges]
            pred = make_graph(*as_evaluator_inputs(nodes, edges_list, round_coordinates=False))
            counts = metric.evaluate(pred, gt, scale=tuple(scale))
            values = metric.per_sample_metrics(counts, estimated, metric.node_recall(pred, gt))
            return {**values, "dataset": name, "group": row["group"],
                    "split": "development", "counts": counts._asdict()}

        refs[name] = eval_graph(anchor)
        cands[name] = eval_graph(cand_graph)
        ref_adj = refs[name]["adj_edge_jaccard"]
        cand_adj = cands[name]["adj_edge_jaccard"]
        z_cnt = audit["smoothing"].get("spikes_suppressed", 0)
        pr_cnt = audit["fork_pruning"].get("forks_pruned", 0)
        lte_cnt = audit["dc_lte"].get("extensions", 0)
        ra_cnt = audit["mitosis_reassignment"].get("reassignments", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt}, "
              f"lte_ext={lte_cnt}, reassign={ra_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "development")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_association_job(args, run, source):
    """Main execution entrypoint for associate-temporal-035 on A1555."""
    cfg = config()
    start_time = time.time()
    corpus = verify_sources(source, code_manifest())
    names = corpus["development"]
    variants = cfg["variants"]

    print(f"Starting Campaign 035 (DC-LTE & BC-MG) on {args.node} across {len(names)} development datasets...")
    results = {}

    for v_key, v_cfg in variants.items():
        print(f"\n--- Evaluating {v_key}: {v_cfg['description']} ---")
        v_res = run_paired_evaluation(source, corpus, names, v_cfg)
        cand_score = v_res["evaluation"]["summary"]["candidate"]["score"]
        ref_score = v_res["evaluation"]["summary"]["reference"]["score"]
        delta = cand_score - ref_score
        passed = v_res["gate"]["passed"]
        lte_total = sum(v_res["audits"][n]["dc_lte"]["extensions"] for n in names)
        ra_total = sum(v_res["audits"][n]["mitosis_reassignment"]["reassignments"] for n in names)
        print(f"Variant {v_key} Result: score={cand_score:.6f} (ref={ref_score:.6f}, "
              f"delta={delta:+.6f}), lte_ext={lte_total}, reassign={ra_total}, gate_passed={passed}")
        results[v_key] = v_res

    elapsed = time.time() - start_time

    passed_variants = [k for k, r in results.items() if r["gate"]["passed"]]
    if passed_variants:
        best_v = max(passed_variants, key=lambda k: results[k]["evaluation"]["summary"]["candidate"]["score"])
        disposition = "eligible_for_confirmation_and_package"
    else:
        best_v = max(results.keys(), key=lambda k: results[k]["evaluation"]["summary"]["candidate"]["score"])
        disposition = "rejected_no_kaggle_submission"

    print(f"\nCampaign 035 Evaluation Summary:")
    print(f"Best Variant: {best_v} (disposition: {disposition})")

    summary = {
        "disposition": disposition,
        "best_variant": best_v,
        "variants_evaluated": list(variants.keys()),
        "paired_evaluations": {k: results[k]["evaluation"] for k in results},
        "gates": {k: results[k]["gate"] for k in results},
        "audits": {k: results[k]["audits"] for k in results},
        "seconds": elapsed,
        "source_run": cfg["source_run"],
        "code_hash": code_hash(),
        "safety": {"label_free_candidate_generation": True, "no_gt_leak": True}
    }
    dump(Path(run) / "temporal_association_evaluation.json", summary)
    print(f"Campaign 035 completed in {elapsed:.1f}s.")
    return summary


def run_confirmation_evaluation(source_run, plan, names, variant_cfg):
    """Runs strict label-free candidate generation on confirmation fold, then official evaluation."""
    from geff import GeffMetadata
    from core.evaluation import make_graph
    from core.production_adapter import as_evaluator_inputs
    from ops.workflow import load_graph

    source_path = Path(source_run)
    ref_csv = source_path / "cache-confirmation/reference.csv"
    references = load_reference(ref_csv, names)
    scale = np.array([1.625, 0.40625, 0.40625], dtype=float)

    refs = {}
    cands = {}
    audits = {}
    metric = official_eval.official_metrics()

    for name in sorted(names):
        group = name.split("_", 1)[0]
        anchor = references[name]

        cand_npz = _load_cand_npz_confirm(name, source_path)

        cand_graph, audit = build_candidate_graph(
            anchor, scale, None, None, variant_cfg, group=group,
            cand_npz=cand_npz
        )
        audits[name] = audit

        gt_path = Path(plan["data_root"]) / (name + ".geff")
        if not gt_path.exists():
            local_alt = source_path / "cache-confirmation/raw" / (name + ".geff")
            if local_alt.exists():
                gt_path = local_alt
            else:
                raise FileNotFoundError(f"GT geff missing: {gt_path}")

        gt = load_graph(gt_path)
        estimated = (GeffMetadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
        if estimated is None or not np.isfinite(estimated) or estimated <= 0:
            estimated = float(len(anchor.coords))

        row = {"name": name, "group": group, "split": "confirmation",
               "scale_um": scale.tolist(), "gt": str(gt_path)}

        def eval_graph(g):
            nodes = {i: {"t": int(c[0]), **dict(zip(("z", "y", "x"), c[1:]))} for i, c in enumerate(g.coords)}
            edges_list = [{"source_id": int(u), "target_id": int(v)} for u, v in g.edges]
            pred = make_graph(*as_evaluator_inputs(nodes, edges_list, round_coordinates=False))
            counts = metric.evaluate(pred, gt, scale=tuple(scale))
            values = metric.per_sample_metrics(counts, estimated, metric.node_recall(pred, gt))
            return {**values, "dataset": name, "group": row["group"],
                    "split": "confirmation", "counts": counts._asdict()}

        refs[name] = eval_graph(anchor)
        cands[name] = eval_graph(cand_graph)
        ref_adj = refs[name]["adj_edge_jaccard"]
        cand_adj = cands[name]["adj_edge_jaccard"]
        z_cnt = audit["smoothing"].get("spikes_suppressed", 0)
        pr_cnt = audit["fork_pruning"].get("forks_pruned", 0)
        lte_cnt = audit["dc_lte"].get("extensions", 0)
        ra_cnt = audit["mitosis_reassignment"].get("reassignments", 0)
        print(f"  [{name}] ref_adj_edge={ref_adj:.5f}, cand_adj_edge={cand_adj:.5f}, "
              f"delta={cand_adj - ref_adj:+.5f}, z_spikes={z_cnt}, pruned={pr_cnt}, "
              f"lte_ext={lte_cnt}, reassign={ra_cnt}", flush=True)

    evaluation = evaluate_association_panel(refs, cands)
    gate = assess(evaluation, "confirmation")
    return {"evaluation": evaluation, "gate": gate, "audits": audits}


def temporal_confirmation_job(args, run, source):
    """Main execution entrypoint for confirm-temporal-035 on A1555."""
    cfg = config()
    start_time = time.time()
    plan = verify_sources(source, code_manifest())
    names = plan["panels"]["confirmation"]

    print(f"Starting Campaign 035 Confirmation Gate on {args.node} across {len(names)} confirmation datasets...")
    evaluations = {}
    best_variant = None
    best_delta = -float("inf")
    best_res = None

    for v_name, v_cfg in cfg["variants"].items():
        print(f"\n--- Evaluating Confirmation Variant {v_name} ---")
        v_res = run_confirmation_evaluation(source, plan, names, v_cfg)
        evaluations[v_name] = v_res

        cand_score = v_res["evaluation"]["summary"]["candidate"]["score"]
        ref_score = v_res["evaluation"]["summary"]["reference"]["score"]
        delta = cand_score - ref_score
        passed = v_res["gate"]["passed"]
        cand_div_tp = v_res["evaluation"]["summary"]["candidate"]["division_tp"]
        ref_div_tp = v_res["evaluation"]["summary"]["reference"]["division_tp"]
        cand_div_fp = v_res["evaluation"]["summary"]["candidate"]["division_fp"]
        ref_div_fp = v_res["evaluation"]["summary"]["reference"]["division_fp"]
        lte_total = sum(v_res["audits"][n]["dc_lte"]["extensions"] for n in names)
        ra_total = sum(v_res["audits"][n]["mitosis_reassignment"]["reassignments"] for n in names)
        print(f"Confirmation {v_name}: score={cand_score:.6f} (ref={ref_score:.6f}, delta={delta:+.6f}), "
              f"div_tp={cand_div_tp}/{ref_div_tp}, div_fp={cand_div_fp}/{ref_div_fp}, "
              f"lte_ext={lte_total}, reassign={ra_total}, gate_passed={passed}")

        if passed and (best_res is None or not best_res["gate"]["passed"] or delta > best_delta):
            best_delta = delta
            best_variant = v_name
            best_res = v_res
        elif best_res is None or (not best_res["gate"]["passed"] and delta > best_delta):
            best_delta = delta
            best_variant = v_name
            best_res = v_res

    target_variant = cfg.get("target_variant", best_variant)
    primary_res = evaluations[target_variant]
    champion_passed = primary_res["gate"]["passed"]
    elapsed = time.time() - start_time
    disposition = "eligible_for_final_package_and_submission" if champion_passed else "rejected_no_kaggle_submission"

    summary = {
        "disposition": disposition,
        "champion_variant": target_variant,
        "recipe": cfg["variants"][target_variant]["description"],
        "result": primary_res,
        "all_variants": {k: {"score": v["evaluation"]["summary"]["candidate"]["score"],
                             "delta": v["evaluation"]["summary"]["candidate"]["score"] - v["evaluation"]["summary"]["reference"]["score"],
                             "gate_passed": v["gate"]["passed"],
                             "div_tp": v["evaluation"]["summary"]["candidate"]["division_tp"],
                             "div_fp": v["evaluation"]["summary"]["candidate"]["division_fp"]}
                         for k, v in evaluations.items()},
        "seconds": elapsed,
        "source_run": cfg["source_run"],
        "code_hash": code_hash(),
        "safety": {"label_free_candidate_generation": True, "no_gt_leak": True}
    }
    dump(Path(run) / "confirmation_evaluation.json", summary)
    print(f"\nCampaign 035 Confirmation completed in {elapsed:.1f}s.")
    print(f"Target Variant: {target_variant}, Delta: {summary['all_variants'][target_variant]['delta']:+.6f}, Disposition: {disposition}")
    return summary
