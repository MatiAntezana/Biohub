"""Frozen dual-seed detector and reconstructed V19 postprocessing.

All callers, including the Kaggle notebook, use these exact functions. Training
annotations never enter this module. Restoration is not a claim of score parity.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np

from core.artifacts import ROOT, dump, digest
from core.nextgen.contracts import Lineage, verify_file
from core.node_provenance import capture_node_creation
from core.quantization import RoundingCollisionError, resolve_rounding_collisions, rounded_tzyx


def verify_assets(assets):
    spec = json.loads((ROOT / "configs/models.json").read_text())
    paths = {}
    for role, row in spec.items():
        filename = "checkpoint_last.pt" if role == "deepcenter" else "edge_predictor_best.pth"
        paths[role] = Path(assets) / role / filename
        verify_file(paths[role], row["sha256"])
        config = json.loads((paths[role].parent / "config.json").read_text())
        if config != row["config"]:
            raise ValueError(f"Architecture config drift: {role}")
    return paths


class Production:
    def __init__(self, assets, image_root, output, *, load_detector=True):
        import torch
        from core.production_adapter import load_production_namespace
        self.torch = torch
        self.paths = verify_assets(assets)
        # Freeze every legacy option, including ambient controller variables.
        for key in list(os.environ):
            if key.startswith("BIOHUB_"):
                del os.environ[key]
        self.ns = load_production_namespace(ROOT / "legacy/v19_restored.ipynb")
        self.ns["TEST_DIR"] = Path(image_root)
        os.environ["BIOHUB_DEEPCENTER_CHECKPOINT"] = str(self.paths["deepcenter"])
        self.deepcenter = self.ns["load_deepcenter_veto_detector"]()
        if self.deepcenter is None or Path(self.deepcenter["path"]).resolve() != self.paths["deepcenter"].resolve():
            raise ValueError("DeepCenter loader did not use the verified pinned checkpoint")
        self.image_root = Path(image_root)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = self.secondary = None
        if load_detector:
            production_root = ROOT / "vendor/production"
            sys.path.insert(0, str(production_root / "src"))
            sys.path.insert(0, str(production_root / "scripts"))
            path = production_root / "scripts/predict_unet_transformer.py"
            source = path.read_text()
            if source.count('Path("/kaggle/working")') != 2:
                raise ValueError("Frozen production log relocation boundary changed")
            os.environ["BIOHUB_RUNTIME_LOG_DIR"] = str(self.output)
            source = source.replace('Path("/kaggle/working")', 'Path(os.environ["BIOHUB_RUNTIME_LOG_DIR"])')
            module = types.ModuleType("frozen_dual_seed")
            module.__file__ = str(path)
            sys.modules[module.__name__] = module
            exec(compile(source, str(path), "exec"), module.__dict__)
            self.module = module
            self.device = torch.device("cuda")
            self.model, self.window, self.downsample = module.load_model(self.paths["primary"], self.device)
            self.secondary, window2, downsample2 = module.load_model(self.paths["secondary"], self.device)
            if (self.window, self.downsample) != (window2, downsample2):
                raise ValueError("Dual-seed architecture mismatch")
            self.cfg = module.PredictConfig(det_threshold=.965, det_tta=True, pool_kernel_um=3.0,
                threshold=.05, use_ilp=True, ilp_edge_weight=-1.0, ilp_appearance_weight=0.0,
                ilp_disappearance_weight=2.0, ilp_division_weight=1.2)

    def filtered_graph(self, graph, name):
        from core.production_adapter import graph_to_plain
        nodes, edges = graph_to_plain(graph)
        observed_ids = frozenset(nodes)
        with capture_node_creation(self.ns, observed_ids) as synthetic_origins:
            nodes, edges, stats = self.ns["filter_output_graph"](nodes, edges, dataset=name, deepcenter_bundle=self.deepcenter)
        audit = self.output / (name + ".rounding-collisions.json")
        try:
            nodes, edges, collisions = resolve_rounding_collisions(nodes, edges, dataset=name,
                observed_node_ids=observed_ids, synthetic_origins=synthetic_origins)
        except RoundingCollisionError as exc:
            dump(audit, exc.report)
            raise
        dump(audit, collisions)
        if collisions["removed_nodes"]:
            print(f"[{name}] rounding guard: removed {collisions['removed_nodes']} gap-synthetic node(s), "
                  f"{collisions['removed_edges']} incident edge(s); "
                  f"output {len(nodes)} nodes, {len(edges)} edges", flush=True)
        return nodes, edges, {**stats, "rounding_collision_guard": collisions}

    def postprocess(self, graph, name):
        nodes, edges, stats = self.filtered_graph(graph, name)
        ids = sorted(nodes)
        remap = {old: new for new, old in enumerate(ids)}
        # Exact historical rounding is part of the champion, not a new ablation.
        coords = np.asarray([rounded_tzyx(nodes[i]) for i in ids], float)
        links = np.asarray([(remap[e["source_id"]], remap[e["target_id"]]) for e in edges], dtype=np.int64).reshape(-1, 2)
        return Lineage(coords, links).validate(), stats

    def predict(self, name):
        import tracksdata as td
        from core.nextgen.images import ImageSource
        shape = ImageSource(self.image_root / f"{name}.zarr", "0", [1.625, .40625, .40625]).shape
        with self.torch.inference_mode():
            coords, edges = self.module.predict_video(self.model, self.image_root / name,
                self.device, cfg=self.cfg, window_size=self.window, downsample=self.downsample,
                secondary_model=self.secondary, secondary_edge_weight=.15,
                secondary_detection_weight=.80, secondary_link_mode="low_margin_consensus",
                secondary_mix_temperature=1.0, secondary_low_margin_max=.35)
        candidate = self.output / (name + "_candidates.npz")
        with candidate.open("xb") as handle:
            np.savez_compressed(handle, coords=coords, edges=np.asarray(edges, float).reshape(-1, 4))
        graph = self.module.build_graph(coords, [edge for edge in edges if edge[2] > .48])
        if graph.num_edges():
            graph = td.solvers.ILPSolver(edge_weight=-td.EdgeAttr("edge_prob"), appearance_weight=0.0,
                disappearance_weight=2.0, division_weight=1.2, timeout=60.0).solve(graph)
        graph.to_geff(self.output / (name + ".geff"))
        anchor, stats = self.postprocess(graph, name)
        anchor.validate(shape)
        receipt = {"dataset": name, "candidate_threshold": .05, "raw_graph_threshold": .48,
            "primary_sha256": digest(self.paths["primary"]), "secondary_sha256": digest(self.paths["secondary"]),
            "deepcenter_sha256": digest(self.paths["deepcenter"]), "candidates_sha256": digest(candidate),
            "reference_nodes": len(anchor.coords), "reference_edges": len(anchor.edges),
            "baseline": "V19_with_synthetic_collision_guard_v2_parity_required", "stats": stats}
        dump(self.output / (name + ".json"), receipt)
        return anchor

    def close(self):
        self.model = self.secondary = self.deepcenter = None
        self.ns.clear()
        self.torch.cuda.empty_cache()
