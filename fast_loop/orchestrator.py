#!/usr/bin/env python3
"""
Autoresearch V2 Fast-Loop Autonomous Orchestrator.
Manages the closed-loop scientific experiment cycle:
  1. Reads baseline from results.tsv.
  2. Runs fast evaluation on core/graph_builder.py.
  3. Compares total_score against current best.
  4. If improved: records new best in results.tsv and creates git commit.
  5. If regressed: reverts core/graph_builder.py and logs failure analysis into MEMORY_FAILURES.md.
"""

from __future__ import annotations
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fast_loop.evaluate_fast import run_fast_evaluation

RESULTS_FILE = PROJECT_ROOT / "results.tsv"
FAILURES_FILE = PROJECT_ROOT / "MEMORY_FAILURES.md"
GRAPH_BUILDER_FILE = PROJECT_ROOT / "core" / "graph_builder.py"


def get_current_best_score() -> Tuple[float, Optional[str]]:
    """Reads results.tsv and returns (best_score, best_run_id)."""
    if not RESULTS_FILE.exists():
        return 0.0, None

    best_score = -1.0
    best_id = None
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                sc = float(row.get("score", 0.0))
                if sc > best_score:
                    best_score = sc
                    best_id = row.get("run_id")
            except (ValueError, TypeError):
                continue

    return (best_score if best_score >= 0.0 else 0.0), best_id


def append_to_results(
    run_id: str,
    res: Dict[str, Any],
    status: str,
    hypothesis: str,
    target_metric: str,
) -> None:
    """Appends experiment outcome to results.tsv."""
    file_exists = RESULTS_FILE.exists()
    fieldnames = [
        "run_id",
        "timestamp",
        "score",
        "delta_vs_best",
        "adj_edge_jaccard",
        "division_jaccard",
        "node_penalty",
        "node_recall",
        "fragmented_edges",
        "status",
        "target_submetric",
        "hypothesis",
    ]

    best_score, _ = get_current_best_score()
    score = float(res.get("total_score", 0.0))
    delta = score - best_score if best_score > 0 else 0.0

    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": f"{score:.4f}",
        "delta_vs_best": f"{delta:+.4f}",
        "adj_edge_jaccard": f"{res.get('adjusted_edge_jaccard', 0.0):.4f}",
        "division_jaccard": f"{res.get('division_jaccard', 0.0):.4f}",
        "node_penalty": f"{res.get('node_penalty_multiplier', 1.0):.4f}",
        "node_recall": f"{res.get('node_recall_7um', 0.0):.4f}",
        "fragmented_edges": str(res.get("edges_fragmented", 0)),
        "status": status,
        "target_submetric": target_metric,
        "hypothesis": hypothesis.replace("\t", " ").strip(),
    }

    with open(RESULTS_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_to_memory_failures(
    run_id: str,
    hypothesis: str,
    target_metric: str,
    score: float,
    best_score: float,
    diagnostics: Dict[str, Any],
    post_mortem: str,
) -> None:
    """Logs a failed hypothesis into MEMORY_FAILURES.md so the agent never repeats it."""
    entry = f"""
### ❌ Run {run_id} ({datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")})
- **Hipótesis**: {hypothesis}
- **Sub-métrica objetivo**: `{target_metric}`
- **Resultado**: Score `{score:.4f}` vs Mejor `{best_score:.4f}` (Delta: `{score - best_score:+.4f}`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: {diagnostics.get('adjusted_edge_jaccard')} (Unadjusted: {diagnostics.get('unadjusted_edge_jaccard')})
  - `Node Penalty Multiplier`: {diagnostics.get('node_penalty_multiplier')} (T_pred: {diagnostics.get('total_predicted_nodes')} / T_true: {diagnostics.get('estimated_true_nodes')})
  - `Division Jaccard`: {diagnostics.get('division_jaccard')} (TP: {diagnostics.get('div_tp')} | FP: {diagnostics.get('div_fp')} | FN: {diagnostics.get('div_fn')})
  - `Fragmented Edges`: {diagnostics.get('edges_fragmented')} | `Wrong Associations`: {diagnostics.get('wrong_association_edges')}
- **Post-Mortem / Aprendizaje**:
  {post_mortem.strip()}

---
"""
    with open(FAILURES_FILE, "a", encoding="utf-8") as f:
        f.write(entry)


def git_commit_change(run_id: str, hypothesis: str, score: float, best_score: float) -> bool:
    """Commits graph_builder.py when score improves."""
    try:
        subprocess.run(["git", "add", "core/graph_builder.py"], cwd=PROJECT_ROOT, check=True)
        msg = f"feat(fast-loop): {run_id} - {hypothesis[:60]} (Score: {best_score:.4f} -> {score:.4f})"
        subprocess.run(["git", "commit", "-m", msg], cwd=PROJECT_ROOT, check=True)
        return True
    except Exception as e:
        print(f"[ORCHESTRATOR] Git commit notice: {e}")
        return False


def git_revert_change() -> bool:
    """Reverts graph_builder.py to last committed state."""
    try:
        subprocess.run(["git", "checkout", "core/graph_builder.py"], cwd=PROJECT_ROOT, check=True)
        return True
    except Exception as e:
        print(f"[ORCHESTRATOR] Git revert notice: {e}")
        return False


def run_experiment_cycle(
    hypothesis: str,
    target_metric: str = "total_score",
    post_mortem: str = "",
    auto_git: bool = True,
) -> Dict[str, Any]:
    """
    Executes a single fast-loop research iteration.
    """
    best_score, best_run_id = get_current_best_score()
    run_idx = 1
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            run_idx = max(1, sum(1 for _ in f))
    run_id = f"FAST-{run_idx:03d}"

    print(f"\n🚀 [ORCHESTRATOR] Launching {run_id}")
    print(f"   Hypothesis:     {hypothesis}")
    print(f"   Target Metric:  {target_metric}")
    print(f"   Current Best:   {best_score:.4f} ({best_run_id or 'None'})")

    try:
        telemetry = run_fast_evaluation(quiet=False)
        score = telemetry["total_score"]
    except Exception as e:
        print(f"❌ [ORCHESTRATOR] Experiment crashed: {e}")
        telemetry = {"total_score": 0.0, "error": str(e)}
        score = 0.0

    improved = score > best_score

    if improved:
        print(f"\n🎉 [SUCCESS] New record achieved! {best_score:.4f} ──► {score:.4f} (+{score - best_score:.4f})")
        append_to_results(run_id, telemetry, status="KEEP", hypothesis=hypothesis, target_metric=target_metric)
        if auto_git:
            git_commit_change(run_id, hypothesis, score, best_score)
    else:
        print(f"\n❌ [DISCARD] Score did not improve ({score:.4f} <= {best_score:.4f}). Reverting changes.")
        append_to_results(run_id, telemetry, status="REVERT", hypothesis=hypothesis, target_metric=target_metric)
        
        pm = post_mortem or f"Hypothesis '{hypothesis}' resulted in score {score:.4f} (delta: {score - best_score:.4f}). Reverting."
        append_to_memory_failures(run_id, hypothesis, target_metric, score, best_score, telemetry, pm)
        
        if auto_git:
            git_revert_change()

    return {
        "run_id": run_id,
        "score": score,
        "best_score": max(score, best_score),
        "improved": improved,
        "telemetry": telemetry,
    }


def main():
    parser = argparse.ArgumentParser(description="Autoresearch V2 Fast-Loop Orchestrator")
    parser.add_argument("--hypothesis", type=str, default="Baseline evaluation", help="Falsifiable hypothesis description")
    parser.add_argument("--target-metric", type=str, default="total_score", help="Sub-metric targeted by hypothesis")
    parser.add_argument("--post-mortem", type=str, default="", help="Explanation of why the hypothesis failed if degraded")
    parser.add_argument("--no-git", action="store_true", help="Disable git commit/checkout on outcome")
    args = parser.parse_args()

    res = run_experiment_cycle(
        hypothesis=args.hypothesis,
        target_metric=args.target_metric,
        post_mortem=args.post_mortem,
        auto_git=not args.no_git,
    )
    sys.exit(0 if res["improved"] else 1)


if __name__ == "__main__":
    main()
