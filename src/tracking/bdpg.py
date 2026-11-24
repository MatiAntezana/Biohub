"""Bounded Divergence Persistence Gating (BDPG) and axial regularization.

Refines cytokinesis fork discrimination to eliminate ephemeral spur divisions
while preserving true biological cytokinesis events under convective tissue drift:
1. Obtuse divergent cytokinesis (cos_3d <= 0.0):
   Naturally diverging daughters; protected if both tracks persist (len >= 3)
   or have positive forward heading (head > 0.3).
2. Drift-tilted cytokinesis (0.0 < cos_3d <= heading_guard_cos_max):
   True divisions under convective tissue drift angle slightly forward.
   Protected only if both daughters exhibit robust persistence (len >= 5)
   and bounded birth distance (max_d <= 6.0 um).
3. Acute crossovers (cos_3d > heading_guard_cos_max):
   Unphysical parallel jumps; never immunized by track persistence and pruned
   via Distance-Dominant Keeper Selection & Spatial Proximity Invariant (DDKS-SPI).
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from src.tracking.contracts import Lineage


DEFAULT_BDPG_CONFIG = {
    "smoothing": {
        "enabled": True,
        "reversal_only": True,
        "symmetric_only": True,
        "max_asymmetry_um": 0.0,
        "jitter_threshold_um": 0.8,
        "max_correction_um": 0.6,
        "weight_z": 0.8,
    },
    "fork_pruning": {
        "enabled": True,
        "cos_thresh": 0.0,
        "min_ratio": 0.18,
        "min_max_dist_um": 3.0,
        "spi_enabled": True,
        "spi_max_close_um": 1.5,
        "spi_min_far_um": 2.5,
        "spi_min_ratio": 2.0,
        "heading_guard": True,
        "heading_guard_cos_max": 0.20,
        "guard_requires_cos": True,
        "divergent_pers_min": 3,
        "tilted_pers_min": 5,
        "tilted_max_birth_dist_um": 6.0,
        "distance_cost_mode": "linear",
        "distance_scale_d0_um": 2.0,
        "distance_cost_alpha": 1.0,
        "photometric_branch_selection": False,
    },
}


def apply_harmonized_axial_regularizer(coords, edges, scale, sm_cfg):
    """Harmonized quasi-symmetric axial regularizer suppressing unphysical z-reversal jitter."""
    if not sm_cfg.get("enabled", False):
        return coords.copy(), {"spikes_suppressed": 0, "max_displacement_um": 0.0}

    reversal_only = sm_cfg.get("reversal_only", True)
    symmetric_only = sm_cfg.get("symmetric_only", True)
    max_asym_um = sm_cfg.get("max_asymmetry_um", 0.0)
    jitter_thresh_um = sm_cfg.get("jitter_threshold_um", 0.8)
    max_correction_um = sm_cfg.get("max_correction_um", 0.6)
    w_z = sm_cfg.get("weight_z", 0.8)

    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    new_coords = coords.copy().astype(np.float64)
    count = 0
    max_disp = 0.0

    for i in range(len(coords)):
        if i in parents and len(children[i]) == 1:
            p = parents[i]
            c = children[i][0]

            if coords[p, 0] + 1 == coords[i, 0] and coords[i, 0] + 1 == coords[c, 0]:
                zp = coords[p, 1] * scale[0]
                zi = coords[i, 1] * scale[0]
                zc = coords[c, 1] * scale[0]

                d1 = zi - zp
                d2 = zc - zi

                is_reversal = d1 * d2 < 0
                if reversal_only and not is_reversal:
                    continue

                if symmetric_only and is_reversal:
                    asym = abs(abs(d1) - abs(d2))
                    if asym > max_asym_um:
                        continue

                mag = min(abs(d1), abs(d2))
                if mag >= jitter_thresh_um:
                    z_target = 0.5 * (zp + zc)
                    disp_um = abs(zi - z_target) * w_z
                    if disp_um > max_correction_um:
                        disp_um = max_correction_um
                    corr_um = math.copysign(disp_um, z_target - zi)
                    new_coords[i, 1] += corr_um / scale[0]
                    count += 1
                    max_disp = max(max_disp, disp_um)

    return new_coords, {
        "spikes_suppressed": count,
        "max_displacement_um": float(max_disp),
    }


def prune_spurious_cytokinesis_forks(coords, edges, scale, descriptors, pr_cfg):
    """Bounded Divergence Persistence Gating (BDPG) & DDKS-SPI."""
    if not pr_cfg.get("enabled", False):
        return edges.copy(), {"forks_inspected": 0, "forks_pruned": 0, "spi_triggered": 0}

    cos_thresh = pr_cfg.get("cos_thresh", 0.0)
    min_ratio = pr_cfg.get("min_ratio", 0.18)
    min_max_dist_um = pr_cfg.get("min_max_dist_um", 3.0)
    photometric_branch_selection = pr_cfg.get("photometric_branch_selection", False)

    spi_enabled = pr_cfg.get("spi_enabled", True)
    spi_max_close_um = pr_cfg.get("spi_max_close_um", 1.5)
    spi_min_far_um = pr_cfg.get("spi_min_far_um", 2.5)
    spi_min_ratio = pr_cfg.get("spi_min_ratio", 2.0)

    heading_guard = pr_cfg.get("heading_guard", True)
    heading_guard_cos_max = float(pr_cfg.get("heading_guard_cos_max", 0.20))
    guard_requires_cos = bool(pr_cfg.get("guard_requires_cos", True))
    div_pers = int(pr_cfg.get("divergent_pers_min", 3))
    tilt_pers = int(pr_cfg.get("tilted_pers_min", 5))
    max_birth_d = float(pr_cfg.get("tilted_max_birth_dist_um", 6.0))

    dist_cost_mode = pr_cfg.get("distance_cost_mode", "linear")
    d0_um = pr_cfg.get("distance_scale_d0_um", 2.0)
    alpha_dist = pr_cfg.get("distance_cost_alpha", 1.0)

    children = defaultdict(list)
    parents = {}
    for u, v in edges:
        children[int(u)].append(int(v))
        parents[int(v)] = int(u)

    def track_length(node):
        curr = node
        length = 0
        while curr in children and len(children[curr]) > 0:
            curr = children[curr][0]
            length += 1
            if length > 50:
                break
        return length

    divs = [(m, ch) for m, ch in children.items() if len(ch) == 2]
    edges_to_remove = set()
    spi_triggered = 0

    for m, (d1, d2) in divs:
        vm = coords[m, 1:] * scale
        v1 = coords[d1, 1:] * scale - vm
        v2 = coords[d2, 1:] * scale - vm
        d_v1 = float(np.linalg.norm(v1))
        d_v2 = float(np.linalg.norm(v2))
        cos_3d = float(np.dot(v1, v2) / max(d_v1 * d_v2, 1e-6))
        ratio = float(min(d_v1, d_v2) / max(max(d_v1, d_v2), 1e-6))
        max_d = max(d_v1, d_v2)
        min_d = min(d_v1, d_v2)

        sim1, sim2 = 0.5, 0.5
        has_sim = False
        if descriptors is not None and len(descriptors) > max(m, d1, d2):
            d_m = descriptors[m]
            norm_m = np.linalg.norm(d_m)
            if norm_m > 0:
                norm1 = np.linalg.norm(descriptors[d1])
                norm2 = np.linalg.norm(descriptors[d2])
                if norm1 > 0 and norm2 > 0:
                    sim1 = float(np.dot(d_m, descriptors[d1]) / (norm_m * norm1))
                    sim2 = float(np.dot(d_m, descriptors[d2]) / (norm_m * norm2))
                    has_sim = True

        is_spurious = False
        if cos_3d > cos_thresh:
            is_spurious = True
        elif ratio < min_ratio and max_d > min_max_dist_um:
            is_spurious = True

        if is_spurious:
            len1 = track_length(d1)
            len2 = track_length(d2)

            ch1 = children.get(d1, [])
            ch2 = children.get(d2, [])
            head1, head2 = 0.0, 0.0
            if len(ch1) >= 1:
                w1 = coords[ch1[0], 1:] * scale - coords[d1, 1:] * scale
                d_w1 = float(np.linalg.norm(w1))
                if d_w1 > 1e-3:
                    head1 = float(np.dot(v1, w1) / max(d_v1 * d_w1, 1e-6))
            if len(ch2) >= 1:
                w2 = coords[ch2[0], 1:] * scale - coords[d2, 1:] * scale
                d_w2 = float(np.linalg.norm(w2))
                if d_w2 > 1e-3:
                    head2 = float(np.dot(v2, w2) / max(d_v2 * d_w2, 1e-6))

            if heading_guard:
                if guard_requires_cos:
                    if cos_3d <= 0.0 and (
                        (len1 >= div_pers and len2 >= div_pers)
                        or (len1 >= 2 and len2 >= 2 and head1 > 0.3 and head2 > 0.3)
                    ):
                        continue

                    if 0.0 < cos_3d <= heading_guard_cos_max and (
                        len1 >= tilt_pers and len2 >= tilt_pers and max_d <= max_birth_d
                    ):
                        continue
                else:
                    if (len1 >= 5 and len2 >= 5) or (
                        cos_3d <= heading_guard_cos_max
                        and ((len1 >= 3 and len2 >= 3) or (len1 >= 2 and len2 >= 2 and head1 > 0.3 and head2 > 0.3))
                    ):
                        continue

            is_extreme_asymmetry = (
                (min_d <= spi_max_close_um and max_d >= spi_min_far_um)
                or (max_d >= spi_min_far_um and (max_d / max(min_d, 1e-6)) >= spi_min_ratio)
            )

            if spi_enabled and is_extreme_asymmetry:
                if d_v1 <= d_v2:
                    edges_to_remove.add((m, d2))
                else:
                    edges_to_remove.add((m, d1))
                spi_triggered += 1
                continue

            if dist_cost_mode == "quadratic":
                score1 = -alpha_dist * ((d_v1 / d0_um) ** 2)
                score2 = -alpha_dist * ((d_v2 / d0_um) ** 2)
            else:
                score1 = -d_v1 / 10.0
                score2 = -d_v2 / 10.0

            if m in parents:
                p = parents[m]
                u_in = vm - coords[p, 1:] * scale
                d_u = float(np.linalg.norm(u_in))
                align1 = float(np.dot(u_in, v1) / max(d_u * d_v1, 1e-6))
                align2 = float(np.dot(u_in, v2) / max(d_u * d_v2, 1e-6))
                score1 += 1.0 * align1
                score2 += 1.0 * align2

            if heading_guard:
                score1 += 0.15 * min(len1, 10) + 0.6 * head1
                score2 += 0.15 * min(len2, 10) + 0.6 * head2

            if photometric_branch_selection and has_sim:
                score1 += 1.5 * sim1
                score2 += 1.5 * sim2

            if score1 >= score2:
                edges_to_remove.add((m, d2))
            else:
                edges_to_remove.add((m, d1))

    surviving = [e for e in edges if (int(e[0]), int(e[1])) not in edges_to_remove]
    return np.asarray(surviving, dtype=edges.dtype).reshape(-1, 2), {
        "forks_inspected": len(divs),
        "forks_pruned": len(edges_to_remove),
        "spi_triggered": spi_triggered,
    }


def build_candidate_graph(anchor: Lineage, scale, descriptors=None, variant_cfg=None):
    """Construct a verified Lineage graph after axial smoothing and BDPG fork pruning."""
    cfg = variant_cfg or DEFAULT_BDPG_CONFIG
    smoothed_coords, audit_smoothing = apply_harmonized_axial_regularizer(
        anchor.coords, anchor.edges, scale, cfg.get("smoothing", {})
    )
    pruned_edges, audit_pruning = prune_spurious_cytokinesis_forks(
        smoothed_coords, anchor.edges, scale, descriptors, cfg.get("fork_pruning", {})
    )
    cand = Lineage(smoothed_coords, pruned_edges)
    cand.validate()
    return cand, {
        "smoothing": audit_smoothing,
        "fork_pruning": audit_pruning,
    }
