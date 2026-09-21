"""Reject interpolated gap nodes that collide with an observed rounded voxel.

This is a structural guard, not NMS or cardinality tuning. It never moves nodes,
merges identities, reconnects edges, or chooses between observed detections.
"""
from collections import defaultdict
import math
from core.node_provenance import SYNTHETIC_STAGES


class RoundingCollisionError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"Unresolved rounded detections in {report['dataset']}: {report['unresolved']}")


def rounded_tzyx(node):
    """The exact legacy integer conversion, with explicit finite/frame checks."""
    values = [float(node[k]) for k in ("t", "z", "y", "x")]
    if not all(math.isfinite(v) for v in values) or values[0] < 0 or not values[0].is_integer():
        raise ValueError("Invalid coordinates before legacy rounding")
    return (int(values[0]), *(max(0, int(round(v))) for v in values[1:]))


def resolve_rounding_collisions(nodes, edges, *, dataset, observed_node_ids, synthetic_origins):
    # An unmarked node is NOT proof of an observation: strict Gap-2 creates
    # unmarked nodes. Only the input snapshot and measured creation stages count.
    observed_node_ids = frozenset(observed_node_ids)
    if observed_node_ids & set(synthetic_origins):
        raise ValueError("Conflicting observed/synthetic node provenance")
    if any(stage not in SYNTHETIC_STAGES for stage in synthetic_origins.values()):
        raise ValueError("Unrecognised synthetic creation stage")
    groups = defaultdict(list)
    for node_id, node in nodes.items():
        if node_id != node["node_id"]:
            raise ValueError("Inconsistent node identity before quantization")
        groups[rounded_tzyx(node)].append(node_id)
    for edge in edges:
        if edge["source_id"] not in nodes or edge["target_id"] not in nodes:
            raise ValueError("Dangling edge before quantization")
    report = {"policy": "observed_over_gap_synthetic_exact_voxel_v2", "dataset": dataset,
        "provenance_policy": "input_snapshot_and_captured_creation_stage_v1",
        "input_nodes": len(nodes), "input_edges": len(edges), "collisions": [], "unresolved": [],
        "input_observed_nodes": len(set(nodes) & observed_node_ids),
        "input_synthetic_nodes_by_stage": {stage: sum(synthetic_origins.get(i) == stage for i in nodes)
                                            for stage in SYNTHETIC_STAGES},
        "unknown_provenance_ids": sorted(set(nodes) - observed_node_ids - set(synthetic_origins))}
    removed = set()
    for voxel, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        synthetic = [i for i in ids if i in synthetic_origins]
        observed = [i for i in ids if i in observed_node_ids]
        evidence = {"voxel_tzyx": list(voxel), "node_ids": ids,
            "coordinates_tzyx": [[float(nodes[i][k]) for k in ("t", "z", "y", "x")] for i in ids],
            "observed_ids": observed, "gap_synthetic_ids": synthetic,
            "gap2_synthetic_ids": [i for i in synthetic if synthetic_origins[i] == "recover_strict_gap2"],
            "unknown_ids": [i for i in ids if i not in observed_node_ids and i not in synthetic_origins],
            "node_origins": {str(i): "observed_input" if i in observed_node_ids else synthetic_origins.get(i, "unknown") for i in ids}}
        # Multiple observed identities, only synthetic identities, or unknown
        # provenance have no safe automatic winner. Preserve inputs and fail.
        if len(observed) != 1 or len(synthetic) != len(ids) - 1:
            report["unresolved"].append(evidence)
            continue
        removed.update(synthetic)
        report["collisions"].append({**evidence, "kept_node_id": observed[0], "removed_node_ids": synthetic})
    if report["unresolved"]:
        # Transactional: do not partially repair some groups before failing.
        report.update(status="rejected", removed_nodes=0, removed_edges=0)
        raise RoundingCollisionError(report)
    dropped_edges = [e for e in edges if e["source_id"] in removed or e["target_id"] in removed]
    clean_nodes = {i: n for i, n in nodes.items() if i not in removed} if removed else nodes
    clean_edges = [e for e in edges if e["source_id"] not in removed and e["target_id"] not in removed] if removed else edges
    report.update(status="repaired" if removed else "unchanged", removed_nodes=len(removed),
        removed_edges=len(dropped_edges), removed_node_ids=sorted(removed),
        removed_edge_endpoints=[[int(e["source_id"]), int(e["target_id"])] for e in dropped_edges],
        output_nodes=len(clean_nodes), output_edges=len(clean_edges))
    return clean_nodes, clean_edges, report
