"""Creation-stage provenance, including the actual unmarked frozen Gap-2 code."""
import ast
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.artifacts import ROOT
from core.node_provenance import capture_node_creation, SYNTHETIC_STAGES
from core.production import Production
from core.quantization import RoundingCollisionError, resolve_rounding_collisions


def node(i, t, z, y=10., x=10.):
    return dict(node_id=i, t=t, z=z, y=y, x=x)


def noop(nodes_by_id, edges, *args, **kwargs):
    return nodes_by_id, edges


def actual_gap2_namespace():
    """Execute the real frozen Gap-2 function; image refinement alone is stubbed."""
    nb = json.loads((ROOT / "legacy/v19_restored.ipynb").read_text())
    source = "".join(nb["cells"][5]["source"])
    tree = ast.parse(source)
    names = {"recover_strict_gap2", "_next_node_id", "_single_successor_map", "_single_predecessor_map"}
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == len(names)
    # Synthetic fixture settings, not edits to the campaign's configuration.
    ns = dict(np=np, OUTPUT_GAP2_RECOVERY=True, GAP2_MAX_LINKS_ABS=10, GAP2_MAX_LINKS_FRAC=1.,
        GAP2_MAX_TOTAL_UM=8., GAP2_MAX_STEP_UM=3., GAP2_REQUIRE_CONTEXT=True, GAP2_FRAME_FRAC_CAP=1.,
        VOXEL_SCALE_UM=(1.625, .40625, .40625), close_single_frame_gaps=noop,
        refine_synthetic_midpoint=lambda dataset, t, midpoint, frames, stats: midpoint,
        edge_distance_um=lambda a, b: float(np.linalg.norm(
            (np.array([a[k] for k in ("z", "y", "x")]) - np.array([b[k] for k in ("z", "y", "x")]))
            * np.array([1.625, .40625, .40625]))))
    exec(compile(ast.Module(body=functions, type_ignores=[]), "actual-frozen-gap2", "exec"), ns)
    return ns


def gap2_fixture():
    nodes = {i: node(i, t, z) for i, t, z in ((1, 0, 10), (2, 1, 11), (3, 4, 14), (4, 5, 15), (7, 2, 12))}
    return nodes, [dict(source_id=u, target_id=v, edge_prob=.9) for u, v in ((1, 2), (3, 4))]


class NodeProvenanceTests(unittest.TestCase):
    def test_in_place_creation_is_tracked_by_stage_without_mutating_outputs(self):
        def single(nodes_by_id, edges, *args, **kwargs):
            nodes_by_id[2] = node(2, 1, 11)
            return nodes_by_id, edges
        def gap2(nodes_by_id, edges, *args, **kwargs):
            nodes_by_id[3] = node(3, 2, 12)
            return nodes_by_id, edges
        ns = dict(close_single_frame_gaps=single, recover_strict_gap2=gap2)
        nodes, edges = {100: node(100, 0, 10)}, []
        with capture_node_creation(ns, set(nodes)) as origins:
            out = ns["close_single_frame_gaps"](nodes, edges)
            self.assertIs(out[0], nodes)
            out = ns["recover_strict_gap2"](nodes_by_id=nodes, edges=edges)
            self.assertIs(out[1], edges)
        self.assertEqual(origins, {2: "close_single_frame_gaps", 3: "recover_strict_gap2"})
        self.assertEqual(nodes[2], node(2, 1, 11))  # No new tag/coordinate side effect.
        self.assertIs(ns["close_single_frame_gaps"], single)
        self.assertIs(ns["recover_strict_gap2"], gap2)

    def test_actual_frozen_gap2_unmarked_nodes_have_measured_provenance(self):
        ns = actual_gap2_namespace()
        nodes, edges = gap2_fixture()
        expected_stats, measured_stats = Counter(), Counter()
        expected = ns["recover_strict_gap2"](copy.deepcopy(nodes), copy.deepcopy(edges), expected_stats)
        with capture_node_creation(ns, set(nodes)) as origins:
            actual = ns["recover_strict_gap2"](copy.deepcopy(nodes), copy.deepcopy(edges), measured_stats)
        self.assertEqual(actual, expected)
        self.assertEqual(measured_stats, expected_stats)
        self.assertEqual(origins, {8: "recover_strict_gap2", 9: "recover_strict_gap2"})
        self.assertNotIn("gap_synthetic", actual[0][8])  # Reproduces the real omission.
        self.assertEqual(expected_stats["gap2_added_nodes"], 2)
        clean, links, report = resolve_rounding_collisions(*actual, dataset="fixture",
            observed_node_ids=set(nodes), synthetic_origins=origins)
        self.assertEqual(report["removed_node_ids"], [8])
        self.assertEqual(report["collisions"][0]["gap2_synthetic_ids"], [8])
        self.assertTrue({2, 3, 7, 9} <= set(clean))  # Gap endpoints remain observed.
        self.assertEqual(len(links), 3)

    def test_production_captures_gap2_before_applying_the_shared_guard(self):
        ns = actual_gap2_namespace()
        def frozen_filter(nodes, edges, **kwargs):
            stats = Counter()
            nodes, edges = ns["recover_strict_gap2"](nodes, edges, stats)
            return nodes, edges, dict(stats)
        ns["filter_output_graph"] = frozen_filter
        with tempfile.TemporaryDirectory() as folder:
            runtime = Production.__new__(Production)
            runtime.ns, runtime.deepcenter, runtime.output = ns, None, Path(folder)
            with patch("core.production_adapter.graph_to_plain", side_effect=lambda _: gap2_fixture()):
                graph, stats = runtime.postprocess(object(), "fixture")
            graph.validate((10, 100, 100, 100))
            self.assertEqual(len(graph.coords), 6)
            self.assertEqual(stats["rounding_collision_guard"]["removed_node_ids"], [8])
            report = json.loads((Path(folder)/"fixture.rounding-collisions.json").read_text())
            self.assertEqual(report["collisions"][0]["node_origins"], {"7": "observed_input", "8": "recover_strict_gap2"})

    def test_reported_confirmation_collision_is_gap2_not_two_observed_nodes(self):
        nodes = {3146: node(3146, 12, 44.84, 112.8, 31.36),
            24135: node(24135, 12, 44.64138552343303, 113.029204389648, 30.868682423763254),
            2901: node(2901, 11, 44), 24136: node(24136, 13, 46)}
        edges = [dict(source_id=u, target_id=v, gap2_recovered=1) for u,v in ((2901,24135),(24135,24136))]
        clean, links, report = resolve_rounding_collisions(nodes, edges, dataset="fixture",
            observed_node_ids={3146,2901}, synthetic_origins={24135:"recover_strict_gap2",24136:"recover_strict_gap2"})
        self.assertEqual(report["removed_node_ids"], [24135])
        self.assertEqual(report["collisions"][0]["voxel_tzyx"], [12,45,113,31])
        self.assertEqual(set(clean), {3146,2901,24136})
        self.assertEqual(links, [])

    def test_edge_flags_and_missing_node_flags_never_prove_observed_or_synthetic(self):
        nodes = {5: node(5, 1, 10.1), 999: node(999, 1, 10.2), 1: node(1, 0, 9)}
        edges = [dict(source_id=1, target_id=999, gap2_recovered=1)]
        for observed in ({1,5,999}, {1,5}):
            with self.subTest(observed=observed), self.assertRaises(RoundingCollisionError) as caught:
                resolve_rounding_collisions(nodes, edges, dataset="fixture",
                    observed_node_ids=observed, synthetic_origins={})
            self.assertEqual(caught.exception.report["removed_nodes"], 0)
            self.assertEqual(caught.exception.report["unresolved"][0]["gap_synthetic_ids"], [])

    def test_capture_restores_functions_after_failure_and_starts_fresh_per_dataset(self):
        def fail(nodes, edges):
            raise RuntimeError("fixture failure")
        ns = dict(close_single_frame_gaps=noop, recover_strict_gap2=fail)
        with self.assertRaisesRegex(RuntimeError, "fixture failure"):
            with capture_node_creation(ns, {1}):
                ns["recover_strict_gap2"]({1:node(1,0,0)}, [])
        self.assertIs(ns["recover_strict_gap2"], fail)
        self.assertIs(ns["close_single_frame_gaps"], noop)
        with capture_node_creation(ns, {2}) as origins:
            self.assertEqual(origins, {})

    def test_observed_id_reuse_and_conflicting_provenance_are_rejected(self):
        def reuse(nodes, edges):
            nodes[1] = node(1, 1, 1)
            return nodes, edges
        ns = dict(close_single_frame_gaps=noop, recover_strict_gap2=reuse)
        with self.assertRaisesRegex(ValueError, "reused"):
            with capture_node_creation(ns, {1}):
                ns["recover_strict_gap2"]({}, [])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            resolve_rounding_collisions({1:node(1,0,0)}, [], dataset="fixture",
                observed_node_ids={1}, synthetic_origins={1:"recover_strict_gap2"})

    def test_missing_creation_stage_fails_before_any_namespace_mutation(self):
        ns = {SYNTHETIC_STAGES[0]: noop}
        with self.assertRaises(KeyError):
            with capture_node_creation(ns, {}):
                self.fail("Missing stage cannot enter the filter")
        self.assertEqual(ns, {SYNTHETIC_STAGES[0]: noop})


if __name__ == "__main__":
    unittest.main()
