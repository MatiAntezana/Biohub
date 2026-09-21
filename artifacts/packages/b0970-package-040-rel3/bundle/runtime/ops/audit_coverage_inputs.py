"""Read-only014 input/novelty audit. Does not open images or GT, or fit models."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)
from core.coverage_audit import audit_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT/"artifacts/A1555")
    parser.add_argument("--plan", type=Path, default=ROOT/"artifacts/A1688/b0970-prepare-005/plan.json")
    args = parser.parse_args()
    result, _, _, _ = audit_inputs(args.runs, args.plan)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
