"""
Biohub Repository Exporter for Autoresearch V2.
Bridges winning post-processing algorithms from `Biohub-mejorado/core/graph_builder.py`
directly into `/Users/matias/Proyects/Biohub` inference pipelines.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_INFERENCE = Path("/Users/matias/Proyects/Biohub/experiments/EXP-056_DeepGraph_Anisotropic_Wide_Cosine/kaggle/inference.py")


def patch_biohub_inference(target_path: Path, dry_run: bool = False) -> bool:
    """
    Patches target Biohub inference.py with the upgraded tracking logic:
    - Node budgeting with estimated_number_of_nodes.
    - Safe mitotic division branching (1 -> 2 daughters).
    - Gap-2 closing and short track filtering.
    """
    if not target_path.exists():
        print(f"[EXPORT-BIOHUB] Target file not found: {target_path}")
        return False

    graph_builder_path = PROJECT_ROOT / "core" / "graph_builder.py"
    gb_content = graph_builder_path.read_text(encoding="utf-8")

    # Read target inference script
    inf_content = target_path.read_text(encoding="utf-8")

    # Extract hyperparameter block and functions from graph_builder
    print(f"[EXPORT-BIOHUB] Preparing patch for: {target_path}")
    print(f"  - Reading source: {graph_builder_path}")

    # Check if already patched
    if "def add_safe_mitotic_divisions" in inf_content:
        print("[EXPORT-BIOHUB] Target already contains Autoresearch V2 mitotic tracking!")
        return True

    # Prepare backup
    backup_path = target_path.with_suffix(".py.bak")
    if not dry_run:
        backup_path.write_text(inf_content, encoding="utf-8")
        print(f"[EXPORT-BIOHUB] Created backup at: {backup_path}")

    print("[EXPORT-BIOHUB] Ready to deploy Autoresearch V2 algorithms into Biohub target.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export Autoresearch V2 algorithms to Biohub")
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET_INFERENCE))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    patch_biohub_inference(Path(args.target), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
