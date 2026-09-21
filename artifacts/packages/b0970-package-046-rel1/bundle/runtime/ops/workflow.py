"""Executable experiment stages. Heavy work runs only inside a locked UCBV job."""
from collections import Counter
import csv
import json
from pathlib import Path
import shutil
import time

import numpy as np

from core.artifacts import ROOT, dump, digest, code_hash
from core.nextgen.contracts import Lineage, sha256, write_submission, load_reference, check_manifest
from core.protocol import PROTOCOL, assess, panel_names, resolve_dataset_row


def load_graph(path):
    import tracksdata as td
    loaded = td.graph.InMemoryGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def annotation_graph(path):
    graph = load_graph(path)
    attrs = list(graph.node_attrs().iter_rows(named=True))
    ids = {int(r["node_id"]): i for i, r in enumerate(attrs)}
    coords = np.asarray([[r[k] for k in ("t", "z", "y", "x")] for r in attrs], float)
    edges = np.asarray([(ids[int(u)], ids[int(v)]) for u, v in graph.edge_list()], dtype=np.int64).reshape(-1, 2)
    return Lineage(coords, edges).validate()


def provision(artifact_root, legacy_root):
    from core.production import verify_assets
    spec = json.loads((ROOT / "configs/models.json").read_text())
    assets = Path(artifact_root) / "assets/champion"
    for role, row in spec.items():
        filename = "checkpoint_last.pt" if role == "deepcenter" else "edge_predictor_best.pth"
        source = Path(legacy_root) / "models" / role / filename
        target = assets / role / filename
        if not target.exists():
            if digest(source) != row["sha256"]:
                raise ValueError("Source model identity mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as inp, target.open("xb") as out:
                shutil.copyfileobj(inp, out)
            dump(target.parent / "config.json", row["config"])
    verify_assets(assets)
    return assets


def make_plan(data_root, output):
    from geff import GeffMetadata
    info = {}
    for path in sorted(Path(data_root).glob("*.geff")):
        if path.stem.split("_", 1)[0] not in PROTOCOL["group_prefixes"]:
            raise ValueError("Unrecognised biological prefix; design review required")
        if not path.with_suffix(".zarr").is_dir():
            raise FileNotFoundError(path.with_suffix(".zarr"))
        graph = annotation_graph(path)
        info[path.stem] = {"divisions": sum(n == 2 for n in Counter(graph.edges[:, 0]).values()),
            "estimated_nodes": (GeffMetadata.read(path).extra or {})["estimated_number_of_nodes"]}
    result = {"schema_version": 1, "all_datasets": sorted(info), "panels": panel_names(info, info),
        "group_policy": "embryo_prefix_proxy_not_independently_verified_biological_identity",
        "protocol_sha256": digest(ROOT / "configs/protocol.json"), "data_root": str(data_root), "public_info": info}
    dump(output / "plan.json", result)
    return result


def replay(assets, data_root, legacy_root, output):
    from core.production import Production
    from core.submission_contract import COLUMNS
    from core.quantization import rounded_tzyx
    output.mkdir()
    runtime = Production(assets, data_root, output, load_detector=False)
    target = output / "replayed.csv"
    raw_root = Path(legacy_root) / "inputs/v19/raw_predictions"
    paths = sorted(raw_root.rglob("*.geff"))
    if sorted(p.stem for p in paths) != sorted(PROTOCOL["visible"]):
        raise ValueError("V19 archived raw graphs unavailable or wrong sample coverage")
    try:
        with target.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            row_id = 0
            for path in paths:
                # Byte replay must use the notebook's IndexedRXGraph loader.
                # InMemoryGraph silently renumbers sparse GEFF IDs, changing
                # node_id/source_id/target_id even when the lineage is identical.
                nodes, edges, _ = runtime.filtered_graph(runtime.ns["graph_from_geff"](path), path.stem)
                for n in sorted(nodes.values(), key=lambda r: r["node_id"]):
                    writer.writerow(dict(id=row_id, dataset=path.stem, row_type="node", node_id=n["node_id"],
                        **dict(zip(("t", "z", "y", "x"), rounded_tzyx(n))), source_id=-1, target_id=-1))
                    row_id += 1
                for e in edges:
                    writer.writerow(dict(id=row_id, dataset=path.stem, row_type="edge", node_id=-1, t=-1,z=-1,y=-1,x=-1,
                        source_id=e["source_id"], target_id=e["target_id"]))
                    row_id += 1
    finally:
        runtime.close()
    expected = json.loads((ROOT / "legacy/provenance.json").read_text())["expected_v19_csv_sha256"]
    receipt = {"expected_sha256": expected, "actual_sha256": digest(target),
        "passed": digest(target) == expected, "scope": "postprocess_byte_parity_on_archived_raw_graphs"}
    dump(output / "parity.json", receipt)
    if not receipt["passed"]:
        raise RuntimeError("V19 restoration did not reproduce archived CSV; stop before new training")
    return receipt


def prepare_examples(rows, config, target):
    from core.nextgen.mitosis import annotation_examples
    counts = Counter()
    with target.open("x") as handle:
        for row in rows:
            graph = annotation_graph(row["gt"])
            for m, a, b, label in annotation_examples(graph.coords, graph.edges, np.asarray(row["scale_um"]), config["mitosis"]):
                handle.write(json.dumps({"dataset": row["name"], "group": row["group"], "split": row["split"],
                    "coords": graph.coords[[m, a, b]].tolist(), "label": label,
                    "evidence": "gt_division" if label else "confirmed_different_parent"}) + "\n")
                counts[str(label)] += 1
    return dict(counts)


def train(names, data_root, config, output):
    from core.nextgen.mitosis import train_head
    output.mkdir()
    rows = [resolve_dataset_row(n, data_root, split="train", gt=True) for n in names]
    manifest = {"schema_version": 1, "scope": "diagnostic", "datasets": rows}
    check_manifest(manifest)
    examples = output / "examples.jsonl"
    counts = prepare_examples(rows, config, examples)
    checkpoint = output / "mitosis_head.pt"
    metrics = train_head(examples, manifest, checkpoint, config["training"])
    spec = {"checkpoint": str(checkpoint), "sha256": digest(checkpoint)}
    dump(output / "training.json", {**metrics, "model": spec, "counts": counts,
        "examples_sha256": digest(examples), "datasets": names})
    return spec


def cache_panel(names, data_root, assets, output):
    from core.production import Production
    output.mkdir()
    runtime = Production(assets, data_root, output / "raw")
    predictions = {}
    started = time.monotonic()
    try:
        for name in sorted(names):
            predictions[name] = runtime.predict(name)
            dump(output / "progress.json", {"completed": sorted(predictions), "total": len(names)})
    finally:
        runtime.close()
    write_submission(output / "reference.csv", predictions, enforce_budget=False)
    dump(output / "cache.json", {"datasets": sorted(names), "reference_sha256": digest(output / "reference.csv"),
        "seconds": time.monotonic() - started, "code_hash": code_hash()})


def infer_panel(cache, names, data_root, config, arm, models, hoct, output, *, final_model=None, scope="diagnostic"):
    from core.nextgen.pipeline import infer_dataset, preflight, evaluate_pairs
    from core.evaluation import metric_root
    output.mkdir()
    reference = load_reference(cache / "reference.csv", names)
    predictions, reports, rows = {}, {}, []
    started = time.monotonic()
    for name in sorted(names):
        row = resolve_dataset_row(name, data_root, split="confirmation" if scope == "diagnostic" and not final_model else "test",
            gt=scope == "diagnostic")
        row.update(candidates=str(cache / "raw" / (name + "_candidates.npz")),
            candidates_receipt=str(cache / "raw" / (name + ".json")))
        row["candidates_sha256"] = digest(row["candidates"])
        # The checkpoint was trained exclusively on the OTHER prefix. Selection
        # is fixed-epoch and never reads this group's images or annotations.
        other = next(g for g in PROTOCOL["group_prefixes"] if g != row["group"])
        manifest = {"schema_version": 1, "scope": scope, "datasets": [row],
            "reference_csv": str(cache / "reference.csv"), "reference_sha256": digest(cache / "reference.csv"),
            "mitosis": final_model or models[other], "hoct": hoct}
        # Full panel submission cardinality is checked after all samples, not per sample.
        if scope == "diagnostic":
            preflight(manifest, config, arm)
        graph, report = infer_dataset(reference[name], row, manifest, config, arm)
        predictions[name], reports[name] = graph, report
        rows.append(row)
        with (output / (name + ".npz")).open("xb") as handle:
            np.savez_compressed(handle, coords=graph.coords, edges=graph.edges)
        dump(output / "progress.json", {"completed": sorted(predictions), "total": len(names)})
    contract = write_submission(output / "candidate.csv", predictions, enforce_budget=scope == "submission")
    result = {"arm": arm, "contract": contract, "datasets": reports, "seconds": time.monotonic() - started,
        "reference_sha256": digest(cache / "reference.csv"), "config_sha256": digest(ROOT / "configs/nextgen.json"),
        "code_hash": code_hash(), "models": models, "hoct": hoct, "kaggle_score": None}
    if scope == "diagnostic" and not final_model:
        result["evaluation"] = evaluate_pairs(predictions, reference, rows, metric_root())
    dump(output / "result.json", result)
    return result


def infer_temporal_panel(cache, names, variant_cfg, output, *, scope="diagnostic", module_name=None):
    import importlib
    from core.nextgen.contracts import load_reference, write_submission
    import numpy as np

    target_mod = module_name or variant_cfg.get("module")
    if target_mod:
        mod = importlib.import_module(target_mod)
        build_candidate_graph = mod.build_candidate_graph
    elif variant_cfg.get("dndp", {}).get("enabled"):
        from core.temporal_association_043 import build_candidate_graph
    elif variant_cfg.get("fork_pruning", {}).get("aacg_enabled"):
        from core.temporal_association_042 import build_candidate_graph
    elif variant_cfg.get("fork_pruning", {}).get("mother_maturation_enabled"):
        from core.temporal_association_040 import build_candidate_graph
    else:
        from core.temporal_association_032 import build_candidate_graph

    output.mkdir(parents=True, exist_ok=False)
    reference = load_reference(cache / "reference.csv", names)
    scale = np.array([1.625, 0.40625, 0.40625], dtype=float)
    predictions, reports = {}, {}
    started = time.monotonic()
    for name in sorted(names):
        group = name.split("_", 1)[0]
        anchor = reference[name]
        cand_graph, audit = build_candidate_graph(anchor, scale, None, {}, variant_cfg, group=group)
        predictions[name] = cand_graph
        reports[name] = audit
        with (output / (name + ".npz")).open("xb") as handle:
            np.savez_compressed(handle, coords=cand_graph.coords, edges=cand_graph.edges)
        dump(output / "progress.json", {"completed": sorted(predictions), "total": len(names)})
    contract = write_submission(output / "candidate.csv", predictions, enforce_budget=scope == "submission")
    result = {
        "variant": variant_cfg.get("description", "temporal_regularizer"),
        "contract": contract,
        "datasets": reports,
        "seconds": time.monotonic() - started,
        "reference_sha256": digest(cache / "reference.csv"),
        "code_hash": code_hash(),
        "kaggle_score": None,
    }
    dump(output / "result.json", result)
    return result


def prepare_job(args, run):
    config = json.loads((ROOT / "configs/nextgen.json").read_text())
    assets = provision(args.artifact_root, args.legacy_root)
    replay(assets, args.data_root, args.legacy_root, run / "replay")
    plan = make_plan(args.data_root, run)
    models = {}
    for group in PROTOCOL["group_prefixes"]:
        models[group] = train([n for n in plan["all_datasets"] if n.startswith(group + "_")],
            args.data_root, config, run / ("train-" + group))
    dump(run / "models.json", models)
    for panel, names in plan["panels"].items():
        cache_panel(names, args.data_root, assets, run / ("cache-" + panel))
    return {"ready_for_experiments": True, "plan_sha256": digest(run / "plan.json"), "models": models}


def experiment_job(args, run, source):
    plan = json.loads((source / "plan.json").read_text())
    models = json.loads((source / "models.json").read_text())
    config = json.loads((ROOT / "configs/nextgen.json").read_text())
    hoct = json.loads((Path(args.artifact_root) / "assets/hoct.json").read_text()) if args.arm not in {"control", "mitosis_add"} else {}
    gates = {}
    for phase in ("development", "confirmation"):
        result = infer_panel(source / ("cache-" + phase), plan["panels"][phase], plan["data_root"], config, args.arm,
            models, hoct, run / phase)
        gates[phase] = assess(result["evaluation"], phase)
        dump(run / "gates.json", gates)
        if not gates[phase]["passed"]:
            return {"disposition": "rejected_no_kaggle_submission", "gates": gates}
    return {"disposition": "eligible_for_final_training_and_visible_runtime_check", "gates": gates,
        "source_run": source.name, "arm": args.arm, "hoct": hoct}
