"""Verify the completed011 retrieval locally without training, GT or network."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT/"artifacts/A1555")
    args = parser.parse_args()
    from core.repair_ranking import audit_inputs
    report, _ = audit_inputs(args.runs_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
