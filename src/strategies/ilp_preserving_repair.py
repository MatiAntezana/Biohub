"""Pure, auditable graph repair that never replaces valid ILP topology."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RepairConfig:
    """Physical and confidence gates for the corrected EXP-031 branch."""

    scale_um: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
    min_ilp_confidence: float = 0.48
    tight_motion_gate_um: float = 6.0
    relaxed_motion_gate_um: float = 10.0
    velocity_weight: float = 0.5
    learned_probability_bonus: float = 1.0
    raw_distance_weight: float = 0.05
    division_parent_candidate_um: float = 8.0
    division_existing_child_um: float = 10.0
    division_sibling_um: float = 11.0
    division_min_angle_deg: float = 35.0
    division_min_separation_growth_um: float = 2.25


@dataclass
class TrackingGraph:
    """Small dependency-free representation used between solver and export."""

    nodes: dict[int, tuple[int, float, float, float]]
    edges: dict[tuple[int, int], float] = field(default_factory=dict)

    def copy(self) -> "TrackingGraph":
        return TrackingGraph(dict(self.nodes), dict(self.edges))


def _position_um(graph: TrackingGraph, node_id: int, config: RepairConfig) -> np.ndarray:
    return np.asarray(graph.nodes[node_id][1:], dtype=np.float64) * np.asarray(config.scale_um)


def _degrees(graph: TrackingGraph) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = {}
    for source, target in graph.edges:
        successors.setdefault(source, []).append(target)
        predecessors.setdefault(target, []).append(source)
    return successors, predecessors


def audit_graph(graph: TrackingGraph) -> dict[str, int]:
    """Raise on invalid lineage topology and return deterministic counts."""

    successors, predecessors = _degrees(graph)
    dangling = 0
    temporal = 0
    for source, target in graph.edges:
        if source not in graph.nodes or target not in graph.nodes:
            dangling += 1
            continue
        if graph.nodes[target][0] != graph.nodes[source][0] + 1:
            temporal += 1
    indegree_violations = sum(len(value) > 1 for value in predecessors.values())
    outdegree_violations = sum(len(value) > 2 for value in successors.values())
    report = {
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "dangling_edges": dangling,
        "nonconsecutive_edges": temporal,
        "indegree_violations": indegree_violations,
        "outdegree_violations": outdegree_violations,
    }
    if any(report[key] for key in report if key.endswith("violations") or key in {"dangling_edges", "nonconsecutive_edges"}):
        raise ValueError(f"invalid tracking graph: {report}")
    return report


def preserve_confident_ilp_edges(graph: TrackingGraph, config: RepairConfig) -> TrackingGraph:
    """Keep valid solver edges above threshold; never create replacements here."""

    output = TrackingGraph(dict(graph.nodes))
    ordered = sorted(graph.edges.items(), key=lambda item: (-item[1], item[0]))
    successors: dict[int, int] = {}
    predecessors: set[int] = set()
    for (source, target), probability in ordered:
        if probability < config.min_ilp_confidence:
            continue
        if source not in graph.nodes or target not in graph.nodes:
            continue
        if graph.nodes[target][0] != graph.nodes[source][0] + 1:
            continue
        if target in predecessors or successors.get(source, 0) >= 2:
            continue
        output.edges[(source, target)] = float(probability)
        predecessors.add(target)
        successors[source] = successors.get(source, 0) + 1
    return output


def _predicted_position(
    graph: TrackingGraph,
    source: int,
    predecessors: dict[int, list[int]],
    config: RepairConfig,
) -> np.ndarray:
    current = _position_um(graph, source, config)
    parents = predecessors.get(source, [])
    if len(parents) != 1:
        return current
    previous = _position_um(graph, parents[0], config)
    return current + config.velocity_weight * (current - previous)


def repair_unmatched_motion(
    graph: TrackingGraph,
    learned_probabilities: dict[tuple[int, int], float],
    config: RepairConfig = RepairConfig(),
) -> tuple[TrackingGraph, dict[str, int]]:
    """Run Hungarian only over sources/destinations left unmatched by ILP."""

    output = graph.copy()
    report = {"candidates": 0, "accepted_tight": 0, "accepted_relaxed": 0, "rejected_gate": 0}

    times = sorted({value[0] for value in output.nodes.values()})
    for time in times[:-1]:
        successors, predecessors = _degrees(output)
        sources = sorted(
            node_id for node_id, value in output.nodes.items()
            if value[0] == time and not successors.get(node_id)
        )
        targets = sorted(
            node_id for node_id, value in output.nodes.items()
            if value[0] == time + 1 and not predecessors.get(node_id)
        )
        if not sources or not targets:
            continue

        for label, gate in (("tight", config.tight_motion_gate_um), ("relaxed", config.relaxed_motion_gate_um)):
            successors, predecessors = _degrees(output)
            active_sources = [node for node in sources if not successors.get(node)]
            active_targets = [node for node in targets if not predecessors.get(node)]
            if not active_sources or not active_targets:
                break

            sentinel = 1e6
            cost = np.full((len(active_sources), len(active_targets)), sentinel, dtype=np.float64)
            for row, source in enumerate(active_sources):
                predicted = _predicted_position(output, source, predecessors, config)
                source_position = _position_um(output, source, config)
                for column, target in enumerate(active_targets):
                    report["candidates"] += 1
                    target_position = _position_um(output, target, config)
                    motion_distance = float(np.linalg.norm(predicted - target_position))
                    if motion_distance > gate:
                        report["rejected_gate"] += 1
                        continue
                    raw_distance = float(np.linalg.norm(source_position - target_position))
                    learned = float(np.clip(learned_probabilities.get((source, target), 0.0), 0.0, 1.0))
                    cost[row, column] = (
                        motion_distance
                        + config.raw_distance_weight * raw_distance
                        - config.learned_probability_bonus * learned
                    )

            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns, strict=True):
                if cost[row, column] >= sentinel:
                    continue
                source, target = active_sources[row], active_targets[column]
                output.edges[(source, target)] = float(learned_probabilities.get((source, target), 0.0))
                report[f"accepted_{label}"] += 1

    audit_graph(output)
    return output, report


def _single_successor(successors: dict[int, list[int]], node: int) -> int | None:
    values = successors.get(node, [])
    return values[0] if len(values) == 1 else None


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def add_conservative_divisions(
    graph: TrackingGraph,
    candidate_probabilities: dict[tuple[int, int], float],
    config: RepairConfig = RepairConfig(),
) -> tuple[TrackingGraph, dict[str, int]]:
    """Add a second daughter using true mutual-NN and angular divergence."""

    output = graph.copy()
    report = {
        "parents_considered": 0,
        "geometric_candidates": 0,
        "rejected_mutual_nn": 0,
        "rejected_continuity": 0,
        "rejected_divergence": 0,
        "accepted": 0,
    }
    successors, predecessors = _degrees(output)

    for parent in sorted(output.nodes):
        existing_children = successors.get(parent, [])
        if len(existing_children) != 1 or len(predecessors.get(parent, [])) != 1:
            continue
        report["parents_considered"] += 1
        child = existing_children[0]
        parent_t = output.nodes[parent][0]
        if output.nodes[child][0] != parent_t + 1:
            continue
        parent_position = _position_um(output, parent, config)
        child_position = _position_um(output, child, config)
        if np.linalg.norm(parent_position - child_position) > config.division_existing_child_um:
            continue

        orphan_ids = sorted(
            node_id for node_id, value in output.nodes.items()
            if value[0] == parent_t + 1 and not predecessors.get(node_id) and node_id != child
        )
        geometric = []
        for orphan in orphan_ids:
            orphan_position = _position_um(output, orphan, config)
            if np.linalg.norm(parent_position - orphan_position) > config.division_parent_candidate_um:
                continue
            if np.linalg.norm(child_position - orphan_position) > config.division_sibling_um:
                continue
            geometric.append(orphan)
        report["geometric_candidates"] += len(geometric)
        if not geometric:
            continue

        closest_to_child = min(geometric, key=lambda node: np.linalg.norm(_position_um(output, node, config) - child_position))
        orphan_position = _position_um(output, closest_to_child, config)
        sibling_candidates = [
            node_id for node_id, value in output.nodes.items()
            if value[0] == parent_t + 1 and node_id != closest_to_child
        ]
        closest_to_orphan = min(
            sibling_candidates,
            key=lambda node: np.linalg.norm(_position_um(output, node, config) - orphan_position),
            default=None,
        )
        if closest_to_orphan != child:
            report["rejected_mutual_nn"] += 1
            continue

        next_child = _single_successor(successors, child)
        next_orphan = _single_successor(successors, closest_to_child)
        if next_child is None or next_orphan is None:
            report["rejected_continuity"] += 1
            continue
        next_child_position = _position_um(output, next_child, config)
        next_orphan_position = _position_um(output, next_orphan, config)
        growth = float(
            np.linalg.norm(next_child_position - next_orphan_position)
            - np.linalg.norm(child_position - orphan_position)
        )
        angle = _angle_degrees(
            next_child_position - child_position,
            next_orphan_position - orphan_position,
        )
        if growth < config.division_min_separation_growth_um or angle < config.division_min_angle_deg:
            report["rejected_divergence"] += 1
            continue

        output.edges[(parent, closest_to_child)] = float(
            candidate_probabilities.get((parent, closest_to_child), 0.0)
        )
        predecessors.setdefault(closest_to_child, []).append(parent)
        successors.setdefault(parent, []).append(closest_to_child)
        report["accepted"] += 1

    audit_graph(output)
    classified = (
        report["accepted"]
        + report["rejected_mutual_nn"]
        + report["rejected_continuity"]
        + report["rejected_divergence"]
    )
    if classified > report["geometric_candidates"]:
        raise AssertionError(f"division report is inconsistent: {report}")
    return output, report
