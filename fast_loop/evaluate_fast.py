#!/usr/bin/env python3
"""
Fast-Loop Evaluation Runner for Autoresearch V2.
Evaluates `core/graph_builder.py` over cached predictions in seconds.
Outputs rich diagnostic telemetry with mathematical error decomposition.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.graph_builder import build_lineage_graph
from core.diagnostic_evaluator import evaluate_tracking
from fast_loop.mock_cache import (
    generate_synthetic_validation_dataset,
    load_dataset_from_cache,
    save_dataset_to_cache,
)


def run_fast_evaluation(
    cache_path: Path | None = None,
    quiet: bool = False,
) -> dict:
    """Runs evaluation of graph_builder over validation cache and returns telemetry dict."""
    if cache_path is None or not cache_path.exists():
        default_cache = PROJECT_ROOT / "fast_loop" / "val_cache.json"
        if not default_cache.exists():
            if not quiet:
                print(f"[FAST-EVAL] Validation cache not found. Generating new benchmark at: {default_cache}")
            ds = generate_synthetic_validation_dataset(seed=42)
            save_dataset_to_cache(ds, default_cache)
        cache_path = default_cache

    t_start = time.time()
    dataset = load_dataset_from_cache(cache_path)
    t_load = time.time() - t_start

    t_graph_start = time.time()
    final_nodes, final_edges, mitoses = build_lineage_graph(
        dataset["raw_detections"],
        dataset["estimated_nodes_per_frame"],
    )
    t_graph = time.time() - t_graph_start

    t_eval_start = time.time()
    total_est_nodes = sum(dataset["estimated_nodes_per_frame"].values())
    telemetry = evaluate_tracking(
        pred_nodes_by_t=final_nodes,
        pred_edges=final_edges,
        gt_nodes_by_t=dataset["gt_nodes"],
        gt_edges=dataset["gt_edges"],
        estimated_total_nodes=total_est_nodes,
    )
    t_eval = time.time() - t_eval_start
    total_wall_time = time.time() - t_start

    result_dict = telemetry.to_dict()
    result_dict["timing"] = {
        "load_seconds": round(t_load, 3),
        "graph_builder_seconds": round(t_graph, 3),
        "evaluator_seconds": round(t_eval, 3),
        "total_wall_seconds": round(total_wall_time, 3),
    }
    result_dict["mitoses_predicted_count"] = len(mitoses)

    if not quiet:
        print_diagnostic_banner(result_dict)

    return result_dict


def print_diagnostic_banner(res: dict) -> None:
    """Prints a structured scientific diagnostic dashboard."""
    score = res["total_score"]
    adj_edge = res["adjusted_edge_jaccard"]
    div_jaccard = res["division_jaccard"]
    penalty = res["node_penalty_multiplier"]
    recall = res["node_recall_7um"]
    t_pred = res["total_predicted_nodes"]
    t_true = res["estimated_true_nodes"]
    wall_time = res["timing"]["total_wall_seconds"]

    print("=" * 80)
    print(f"🔬 AUTORESEARCH V2 - FAST-LOOP DIAGNOSTIC REPORT ({wall_time:.2f}s)")
    print("=" * 80)
    print(f" 🏆 TOTAL SCORE:              {score:.4f}   (Adj Edge Jaccard + 0.1 * Division Jaccard)")
    print(f" ├─ Adjusted Edge Jaccard:    {adj_edge:.4f}  (Unadjusted: {res['unadjusted_edge_jaccard']:.4f})")
    print(f" ├─ Node Penalty Multiplier:  {penalty:.4f}  (T_pred: {t_pred} vs T_true: {t_true})")
    print(f" └─ Division Jaccard:         {div_jaccard:.4f}  (TP: {res['div_tp']} | FP: {res['div_fp']} | FN: {res['div_fn']})")
    print("-" * 80)
    print(" 📊 DETAILED METRIC DECOMPOSITION:")
    matched_gt = t_pred - res['spurious_nodes']
    total_gt = matched_gt + res['missed_gt_nodes']
    print(f"   • Node Recall (<=7.0um):   {recall * 100:.1f}% ({matched_gt}/{total_gt} GT recovered)")
    print(f"   • Spurious (FP) Nodes:     {res['spurious_nodes']}")
    print(f"   • Edge Confusion:          TP={res['edge_tp']} | FP={res['edge_fp']} | FN={res['edge_fn']}")
    print("   • Root Cause Errors:")
    print(f"     - Edges Recovered:       {res['edges_recovered']}")
    print(f"     - Lost to Detection:     {res['edges_lost_to_detection']} (Cell not detected in frame)")
    print(f"     - Fragmented Tracking:   {res['edges_fragmented']} (Cells detected but link broke)")
    print(f"     - Wrong Association:     {res['wrong_association_edges']} (Connected to wrong cell)")
    print("   • Anisotropic Spatial Error:")
    print(f"     - Z-axis error:          {res['z_axis_displacement_error']:.3f} um")
    print(f"     - XY error:              {res['xy_displacement_error']:.3f} um")
    print("=" * 80)

    # Diagnostic actionable suggestions
    if penalty < 0.98:
        print(" ⚠️  ALERT: Node penalty is hurting your score! Reduce NODE_BUDGET_FACTOR or increase CONF_THRESHOLD.")
    if div_jaccard < 0.40:
        print(" 💡 OPPORTUNITY: Division Jaccard is low! Adjust MITOSIS_CONF_THRESHOLD, SAFE_DIV_MAX_UM or FRAME_CAP.")
    if res["edges_fragmented"] > res["edges_recovered"] * 0.15:
        print(" 💡 OPPORTUNITY: High edge fragmentation! Increase MAX_GAP2_DISTANCE_UM or MAX_LINK_DISTANCE_UM.")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Autoresearch V2 Fast-Loop Evaluator")
    parser.add_argument("--cache", type=str, default=None, help="Path to validation cache JSON")
    parser.add_argument("--output-json", type=str, default=None, help="Export diagnostic metrics to JSON")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal banner")
    parser.add_argument("--json-line", action="store_true", help="Print compact AUTORESEARCH_METRICS_JSON line to stdout")
    parser.add_argument("--local", action="store_true", help="Force local execution on Mac (bypasses remote cluster)")
    parser.add_argument("--node", type=str, default=None, help="Target remote cluster node (e.g. A1556-UBU)")
    args = parser.parse_args()

    # Remote Execution Enforcement:
    # If not explicitly --local, dispatch execution to remote GPU cluster node!
    if not args.local:
        from core.remote_executor import run_remote_python, find_best_online_node, DEFAULT_NODES
        target_node = DEFAULT_NODES.get(args.node) if args.node else find_best_online_node()
        remote_args = ["--local"]
        if args.cache:
            remote_args.extend(["--cache", args.cache])
        if args.output_json:
            remote_args.extend(["--output-json", args.output_json])
        if args.quiet:
            remote_args.append("--quiet")
        if args.json_line:
            remote_args.append("--json-line")

        code, stdout, stderr = run_remote_python(
            "fast_loop/evaluate_fast.py",
            remote_args,
            node=target_node,
            quiet=args.quiet,
        )
        if stdout:
            print(stdout)
        if code != 0:
            if stderr:
                print(f"[FAST-EVAL-REMOTE ERROR]: {stderr}", file=sys.stderr)
            sys.exit(code)
        sys.exit(0)

    cache_path = Path(args.cache) if args.cache else None
    res = run_fast_evaluation(cache_path=cache_path, quiet=args.quiet)

    if args.json_line:
        print("AUTORESEARCH_METRICS_JSON=" + json.dumps(res, sort_keys=True))

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        if not args.quiet:
            print(f"[FAST-EVAL] Metrics exported to {out_p}")

    # Return exit code based on success
    sys.exit(0)


if __name__ == "__main__":
    main()
