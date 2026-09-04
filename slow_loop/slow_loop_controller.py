"""
Slow-Loop Controller for Autoresearch V2.
Supervises heavy GPU representation training on the remote cluster (50 epochs, 24GB VRAM).
When Fast-Loop algorithm space plateaus on current cached representations,
triggers a Slow-Loop experiment to learn richer features (e.g., mitosis heads, T=5 windows).
"""

from __future__ import annotations
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIOHUB_ROOT = Path("/Users/matias/Proyects/Biohub")


@dataclass
class SlowLoopProposal:
    experiment_id: str
    name: str
    architecture: str
    epochs: int
    batch_size: int
    learning_rate: float
    temporal_window: int
    target_node: str
    rationale: str


def check_fast_loop_plateau(results_tsv: Path, window: int = 5, min_gain: float = 0.003) -> bool:
    """Checks if the last `window` experiments in the Fast-Loop improved less than `min_gain`."""
    if not results_tsv.exists():
        return False

    scores: List[float] = []
    with open(results_tsv, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]  # skip header
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) > 2:
                try:
                    scores.append(float(parts[2]))
                except ValueError:
                    continue

    if len(scores) < window:
        return False

    recent = scores[-window:]
    gain = max(recent) - min(recent)
    return gain < min_gain


def generate_slow_loop_proposal(plateau_reason: str = "Fast-loop representation saturation") -> SlowLoopProposal:
    """Generates a structured Slow-Loop proposal targeted at representation learning."""
    return SlowLoopProposal(
        experiment_id="EXP-061",
        name="DeepGraph Temporal-6 Mitosis-Aware Anisotropic",
        architecture="deepgraph_multiscale_t5",
        epochs=50,
        batch_size=1,
        learning_rate=0.0003,
        temporal_window=5,
        target_node="A1560-UBU",
        rationale=(
            f"{plateau_reason}. Increasing temporal context to T=5 and joint training "
            "with mitosis focal loss to improve representation separability."
        ),
    )


def plan_and_queue_slow_loop(proposal: SlowLoopProposal, auto_queue: bool = False) -> Dict[str, Any]:
    """Prepares and validates Slow-Loop proposal for Biohub execution."""
    manifest = {
        "experiment_id": proposal.experiment_id,
        "name": proposal.name,
        "architecture": proposal.architecture,
        "epochs": proposal.epochs,
        "batch_size": proposal.batch_size,
        "learning_rate": proposal.learning_rate,
        "temporal_window": proposal.temporal_window,
        "default_node": proposal.target_node,
        "rationale": proposal.rationale,
    }

    print("=" * 80)
    print(f"🐢 SLOW-LOOP GPU PROPOSAL GENERATED: {proposal.experiment_id}")
    print("=" * 80)
    print(f" Name:            {proposal.name}")
    print(f" Architecture:    {proposal.architecture} (Window: T={proposal.temporal_window})")
    print(f" Epochs:          {proposal.epochs} on {proposal.target_node}")
    print(f" Rationale:       {proposal.rationale}")
    print("=" * 80)

    return manifest


if __name__ == "__main__":
    prop = generate_slow_loop_proposal()
    plan_and_queue_slow_loop(prop)
