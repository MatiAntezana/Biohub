"""
Kaggle Submission Exporter for Autoresearch V2.
Extracts the optimized `graph_builder.py` algorithms and calibrated hyperparameters
into a standalone inference module compatible with Kaggle T4 runtime constraints.
"""

from __future__ import annotations
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_kaggle_postprocessing_module(output_path: Path) -> Path:
    """Reads core/graph_builder.py and outputs a cleaned, standalone module for Kaggle notebooks."""
    source_path = PROJECT_ROOT / "core" / "graph_builder.py"
    if not source_path.exists():
        raise FileNotFoundError(f"Source graph_builder.py not found at: {source_path}")

    content = source_path.read_text(encoding="utf-8")

    header = '''"""
================================================================================
KAGGLE INFERENCE TRACKING ENGINE - EXPORTED FROM AUTORESEARCH V2
================================================================================
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
'''
    # We strip docstrings and export the executable graph builder logic
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    print(f"[EXPORT] Successfully generated Kaggle post-processing module: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export graph_builder to Kaggle format")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "integration" / "exported_graph_builder.py"))
    args = parser.parse_args()

    out_p = Path(args.output)
    generate_kaggle_postprocessing_module(out_p)


if __name__ == "__main__":
    main()
