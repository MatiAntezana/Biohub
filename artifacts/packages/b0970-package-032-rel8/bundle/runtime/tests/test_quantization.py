"""Regression for the reported real/synthetic voxel collision in prepare-003."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from core.nextgen.contracts import Lineage
from core.production import Production
from core.quantization import RoundingCollisionError, resolve_rounding_collisions, rounded_tzyx


def node(i, t, z, y, x, **attrs):
    return {"node_id": i, "t": t, "z": z, "y": y, "x": x, **attrs}


def incident_fixture():
    # Coordinates/provenance from the operator's 6bba_48816121 diagnostic.
    # Neighbouring endpoints/edges below are synthetic test fixtures, not GT.
    nodes = {14434: node(14434, 48, 24.64, 181.92, 64.0),
        26981: node(26981, 48, 24.9036, 182.397, 64.126, gap_synthetic=1),
        1: node(1, 47, 24, 180, 63), 2: node(2, 49, 26, 183, 65),
        3: node(3, 47, 24, 184, 63), 4: node(4, 49, 26, 185, 65)}
    edges = [{"source_id": u, "target_id": v, "edge_prob": .9}
             for u, v in ((1, 14434), (14434, 2), (3, 26981), (26981, 4))]
    return nodes, edges


def resolve_fixture(nodes, edges):
    synthetic = {i: "close_single_frame_gaps" for i in (26981, 30000) if i in nodes}
    return resolve_rounding_collisions(nodes, edges, dataset="fixture",
        observed_node_ids=set(nodes) - set(synthetic), synthetic_origins=synthetic)


class QuantizationTests(unittest.TestCase):
    def test_reported_pair_collides_and_only_gap_node_and_its_edges_are_removed(self):
        nodes, edges = incident_fixture()
        original = copy.deepcopy((nodes, edges))
        self.assertEqual(rounded_tzyx(nodes[14434]), (48, 25, 182, 64))
        self.assertEqual(rounded_tzyx(nodes[26981]), (48, 25, 182, 64))
        before = np.asarray([rounded_tzyx(n) for n in nodes.values()], float)
        with self.assertRaisesRegex(ValueError, "Duplicate detections"):
            Lineage(before, np.empty((0, 2), dtype=np.int64)).validate()
        clean, links, report = resolve_fixture(nodes, edges)
        self.assertEqual(set(clean), set(nodes) - {26981})
        self.assertIs(clean[14434], nodes[14434])
        self.assertEqual(links, edges[:2])  # No transfer to the observed identity.
        self.assertEqual(report["removed_node_ids"], [26981])
        self.assertEqual(report["removed_edges"], 2)
        self.assertEqual((nodes, edges), original)

    def test_no_collision_preserves_input_objects_and_legacy_rounding(self):
        nodes, edges = incident_fixture()
        nodes[26981]["y"] = 182.6
        clean, links, report = resolve_fixture(nodes, edges)
        self.assertIs(clean, nodes)
        self.assertIs(links, edges)
        self.assertEqual(report["status"], "unchanged")
        self.assertEqual(rounded_tzyx(node(9, 0, 2.5, 3.5, -.1)), (0, 2, 4, 0))

    def test_observed_only_synthetic_only_and_unknown_collisions_fail(self):
        for provenance in ("both_observed", "both_synthetic", "unknown"):
            with self.subTest(provenance=provenance):
                nodes, edges = incident_fixture()
                observed, synthetic = set(nodes), {}
                if provenance == "both_synthetic":
                    synthetic = {i: "close_single_frame_gaps" for i in (14434, 26981)}
                    observed -= set(synthetic)
                elif provenance == "unknown":
                    observed.remove(26981)
                original = copy.deepcopy((nodes, edges))
                with self.assertRaises(RoundingCollisionError) as caught:
                    resolve_rounding_collisions(nodes, edges, dataset="fixture",
                        observed_node_ids=observed, synthetic_origins=synthetic)
                self.assertEqual(caught.exception.report["status"], "rejected")
                self.assertEqual(caught.exception.report["removed_nodes"], 0)
                self.assertEqual((nodes, edges), original)

    def test_multiple_synthetics_keep_one_observed_without_order_dependence(self):
        nodes, edges = incident_fixture()
        nodes[30000] = {**nodes[26981], "node_id": 30000}
        a = resolve_fixture(nodes, edges)
        b = resolve_fixture(dict(reversed(list(nodes.items()))), edges)
        self.assertEqual(a, b)
        self.assertEqual(a[2]["removed_node_ids"], [26981, 30000])

    def test_same_space_at_different_frames_is_not_a_collision(self):
        nodes, edges = incident_fixture()
        nodes[26981]["t"] = 50
        clean, _, report = resolve_fixture(nodes, [])
        self.assertIs(clean, nodes)
        self.assertEqual(report["removed_nodes"], 0)

    def test_invalid_coordinates_or_dangling_edges_are_not_hidden_by_repair(self):
        for values in ((-1, 0, 0, 0), (.5, 0, 0, 0), (0, float("nan"), 0, 0)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                rounded_tzyx(node(0, *values))
        nodes, edges = incident_fixture()
        edges.append({"source_id": 99999, "target_id": 26981})
        with self.assertRaisesRegex(ValueError, "Dangling edge"):
            resolve_fixture(nodes, edges)

    def test_production_uses_guard_before_lineage_validation_and_writes_audit(self):
        nodes, edges = incident_fixture()
        with tempfile.TemporaryDirectory() as folder:
            runtime = Production.__new__(Production)  # No weights, torch or GPU initialization.
            runtime.output = Path(folder)
            runtime.deepcenter = None
            def add_gap(current, links, *args, **kwargs):
                current[26981] = copy.deepcopy(nodes[26981])
                return current, copy.deepcopy(edges)
            def filter_graph(current, links, **kwargs):
                current, links = runtime.ns["close_single_frame_gaps"](current, links)
                return current, links, {"fixture": True}
            runtime.ns = {"close_single_frame_gaps": add_gap,
                "recover_strict_gap2": lambda current, links, *a, **kw: (current, links),
                "filter_output_graph": filter_graph}
            raw = {i: n for i, n in nodes.items() if i != 26981}
            with patch("core.production_adapter.graph_to_plain", side_effect=lambda _: (copy.deepcopy(raw), copy.deepcopy(edges[:2]))):
                graph, stats = runtime.postprocess(object(), "fixture")
                graph.validate((100, 100, 300, 300))
                self.assertEqual(len(graph.coords), len(nodes) - 1)
                self.assertEqual(len(graph.edges), 2)
                report = json.loads((runtime.output / "fixture.rounding-collisions.json").read_text())
                self.assertEqual(report, stats["rounding_collision_guard"])
                # The same production path still rejects collisions of observed nodes.
                raw = nodes  # Both IDs now genuinely belong to the input snapshot.
                with self.assertRaises(RoundingCollisionError):
                    runtime.postprocess(object(), "rejected")
                self.assertEqual(json.loads((runtime.output / "rejected.rounding-collisions.json").read_text())["status"], "rejected")


if __name__ == "__main__":
    unittest.main()
