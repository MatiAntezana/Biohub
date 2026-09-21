"""Read-only local input audit for013; no model fitting or annotation access."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)
from core.event_ranking import audit_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT/"artifacts/A1555")
    args = parser.parse_args()
    report, _, _ = audit_inputs(args.runs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
