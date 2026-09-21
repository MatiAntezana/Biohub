"""Sparse-ID replay regression; local fixtures require no GPU or GEFF install."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.artifacts import ROOT, dump
from core.production import Production
from ops import workflow


EXPECTED = (
    "id,dataset,row_type,node_id,t,z,y,x,source_id,target_id\r\n"
    "0,fixture,node,0,0,2,3,4,-1,-1\r\n"
    "1,fixture,node,3,1,5,6,7,-1,-1\r\n"
    "2,fixture,node,10,1,8,9,10,-1,-1\r\n"
    "3,fixture,edge,-1,-1,-1,-1,-1,0,10\r\n"
    "4,fixture,edge,-1,-1,-1,-1,-1,0,3\r\n"
).encode()


def frozen_loader(td):
    """Compile only the actual notebook loader, with an explicit test backend."""
    notebook = json.loads((ROOT / "legacy/v19_restored.ipynb").read_text())
    tree = ast.parse("".join(notebook["cells"][5]["source"]))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "graph_from_geff")
    namespace = {"td": td, "Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "frozen-graph-loader", "exec"), namespace)
    return namespace["graph_from_geff"]


def fixture_graph(*, renumber=False):
    ids = {0: 0, 3: 1, 10: 2} if renumber else {0: 0, 3: 3, 10: 10}
    # Non-sorted source rows and edges catch accidental ID/order normalization.
    nodes = [dict(node_id=ids[i], t=t, z=z, y=y, x=x)
             for i, t, z, y, x in ((10, 1, 8, 9, 10), (0, 0, 2, 3, 4), (3, 1, 5, 6, 7))]
    edges = [dict(source_id=ids[0], target_id=ids[i], edge_prob=.9) for i in (10, 3)]
    return SimpleNamespace(node_attrs=lambda: SimpleNamespace(iter_rows=lambda **_: iter(nodes)),
        edge_attrs=lambda: SimpleNamespace(iter_rows=lambda **_: iter(edges)))


class ReplayTests(unittest.TestCase):
    def test_frozen_loader_uses_indexed_backend_and_handles_tuple(self):
        graph = fixture_graph()
        for returned in (graph, (graph, {"metadata": "fixture"})):
            with self.subTest(tuple_result=isinstance(returned, tuple)):
                indexed = Mock(return_value=returned)
                dense = Mock(side_effect=AssertionError("Must not renumber archived IDs"))
                td = SimpleNamespace(graph=SimpleNamespace(IndexedRXGraph=SimpleNamespace(from_geff=indexed),
                    InMemoryGraph=SimpleNamespace(from_geff=dense)))
                self.assertIs(frozen_loader(td)(Path("fixture.geff")), graph)
                indexed.assert_called_once_with(Path("fixture.geff"))
                dense.assert_not_called()

    def check_replay(self, renumber):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            raw = base / "archive/inputs/v19/raw_predictions/fixture.geff"
            raw.mkdir(parents=True)
            dump(base / "legacy/provenance.json", {"expected_v19_csv_sha256": hashlib.sha256(EXPECTED).hexdigest()})
            loader = Mock(return_value=fixture_graph(renumber=renumber))
            runtime = SimpleNamespace(ns={"graph_from_geff": loader,
                "close_single_frame_gaps": lambda nodes, edges, *a, **kw: (nodes, edges),
                "recover_strict_gap2": lambda nodes, edges, *a, **kw: (nodes, edges),
                "filter_output_graph": lambda nodes, edges, **_: (nodes, edges, {})},
                deepcenter=None, close=Mock(), output=base / "replay")
            runtime.filtered_graph = MethodType(Production.filtered_graph, runtime)
            with patch("core.production.Production", return_value=runtime), patch.object(workflow, "ROOT", base), \
                 patch.object(workflow, "PROTOCOL", {"visible": ["fixture"]}), \
                 patch.object(workflow, "load_graph", side_effect=AssertionError("Dense loader entered replay")):
                if renumber:
                    with self.assertRaisesRegex(RuntimeError, "did not reproduce archived CSV"):
                        workflow.replay(base / "assets", base / "data", base / "archive", base / "replay")
                else:
                    result = workflow.replay(base / "assets", base / "data", base / "archive", base / "replay")
                    self.assertTrue(result["passed"])
                    self.assertEqual((base / "replay/replayed.csv").read_bytes(), EXPECTED)
            loader.assert_called_once_with(raw)
            runtime.close.assert_called_once()
            receipt = json.loads((base / "replay/parity.json").read_text())
            self.assertEqual(receipt["passed"], not renumber)
            self.assertEqual(receipt["expected_sha256"], hashlib.sha256(EXPECTED).hexdigest())

    def test_replay_preserves_sparse_ids_edge_order_and_exact_bytes(self):
        self.check_replay(renumber=False)

    def test_same_geometry_with_renumbered_ids_still_fails_byte_gate(self):
        self.check_replay(renumber=True)

    def test_parity_failure_stops_before_plan_or_training(self):
        args = SimpleNamespace(artifact_root="fixture", legacy_root="fixture", data_root="fixture")
        with patch.object(workflow, "provision", return_value="fixture"), \
             patch.object(workflow, "replay", side_effect=RuntimeError("fixture parity failure")), \
             patch.object(workflow, "make_plan") as plan, patch.object(workflow, "train") as train:
            with self.assertRaisesRegex(RuntimeError, "fixture parity failure"):
                workflow.prepare_job(args, Path("never-created"))
            plan.assert_not_called()
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
