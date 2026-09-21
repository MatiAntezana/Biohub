"""Read-only 029 input audit. Does not open images or GT, or mutate graphs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports

configure_cli_imports(ROOT)
from core.temporal_association_029 import audit_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "artifacts/A1555")
    args = parser.parse_args()
    result, _ = audit_inputs(args.runs)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
