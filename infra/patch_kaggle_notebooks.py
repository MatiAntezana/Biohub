#!/usr/bin/env python3
"""Patch all modern submission notebooks to permanently eliminate Kaggle timeouts.

Fixes applied:
1. Cell 0: Injects BIOHUB_VALIDATOR_ENABLE = "0" (never run 2.5h train validation in submission rerun).
2. Cell 0: Injects BIOHUB_ILP_TIMEOUT = "60.0" (caps SCIP solver to 60s per video).
3. Cell 0: Enforces bounded safe division parameters (matching EXP-056 0.933 record):
   - BIOHUB_SAFE_DIV_MAX_UM = "7.0"
   - BIOHUB_SAFE_DIV_SISTER_MAX_UM = "12.0"
   - BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP = "0.00375"
   - BIOHUB_SAFE_DIV_FRAME_FRAC_CAP = "0.0076"
   - BIOHUB_ILP_DIVISION_WEIGHT = "1.2"
4. Cell 4: Injects ILPSolver timeout patch into predict_unet_transformer.py runtime.
5. Cell 7: Changes default to VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "0") == "1".
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ILP_INJECTION = """
# Patch ILPSolver timeout
_s = _ps.read_text()
_ilp_old = \"\"\"        if cfg.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
            )\"\"\"
_ilp_new = \"\"\"        if cfg.use_ilp and graph.num_edges() > 0:
            _ilp_timeout = float(os.environ.get("BIOHUB_ILP_TIMEOUT", "60.0"))
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight,
                timeout=_ilp_timeout,
            )\"\"\"
if _ilp_old in _s:
    _s = _s.replace(_ilp_old, _ilp_new, 1)
    compile(_s, str(_ps), "exec")
    _ps.write_text(_s)
    print("ILPSolver timeout guard applied (timeout <= 60.0s)")
else:
    print("ILPSolver timeout guard: block not found or already patched")
"""

def patch_notebook(path: Path) -> bool:
    content = path.read_text(encoding="utf-8")
    nb = json.loads(content)
    cells = nb.get("cells", [])
    if len(cells) < 8:
        print(f"Skipping {path}: only {len(cells)} cells")
        return False

    # --- 1. Patch Cell 0 ---
    c0 = "".join(cells[0].get("source", []))
    c0_lines = c0.splitlines(keepends=True)
    new_c0_lines = []
    has_val = False
    has_ilp_timeout = False

    for line in c0_lines:
        if 'BIOHUB_VALIDATOR_ENABLE' in line:
            new_c0_lines.append('os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"  # NEVER run train validation in submission rerun\n')
            has_val = True
        elif 'BIOHUB_ILP_TIMEOUT' in line:
            new_c0_lines.append('os.environ["BIOHUB_ILP_TIMEOUT"] = "60.0"  # Strict timeout to avoid ILP hang\n')
            has_ilp_timeout = True
        elif 'BIOHUB_SAFE_DIV_MAX_UM' in line:
            new_c0_lines.append('os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "7.0"\n')
        elif 'BIOHUB_SAFE_DIV_SISTER_MAX_UM' in line:
            new_c0_lines.append('os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "12.0"\n')
        elif 'BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP' in line:
            new_c0_lines.append('os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"\n')
        elif 'BIOHUB_SAFE_DIV_FRAME_FRAC_CAP' in line:
            new_c0_lines.append('os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"\n')
        elif 'BIOHUB_ILP_DIVISION_WEIGHT' in line:
            new_c0_lines.append('os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.2"     # Bound division graph expansion\n')
        else:
            new_c0_lines.append(line)

    if not has_val or not has_ilp_timeout:
        insert_idx = len(new_c0_lines)
        for idx, l in enumerate(new_c0_lines):
            if l.startswith("print("):
                insert_idx = idx
                break
        if not has_ilp_timeout:
            new_c0_lines.insert(insert_idx, 'os.environ["BIOHUB_ILP_TIMEOUT"] = "60.0"  # Strict timeout to avoid ILP hang\n')
        if not has_val:
            new_c0_lines.insert(insert_idx, 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"  # NEVER run train validation in submission rerun\n')

    cells[0]["source"] = new_c0_lines

    # --- 2. Patch Cell 4 (ILP Solver Timeout Injection) ---
    c4 = "".join(cells[4].get("source", []))
    if "ILPSolver timeout guard applied" not in c4:
        hook_marker = 'print("Pre-ILP detector-coordinate manifest hook applied")\n'
        if hook_marker in c4:
            c4 = c4.replace(hook_marker, hook_marker + ILP_INJECTION)
            cells[4]["source"] = [line + "\n" for line in c4.splitlines()]
        else:
            # Append at end of cell 4 before execution
            c4 = c4 + "\n" + ILP_INJECTION
            cells[4]["source"] = [line + "\n" for line in c4.splitlines()]

    # --- 3. Patch Cell 7 (Validator Default Guard) ---
    c7 = "".join(cells[7].get("source", []))
    target_old = 'VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "1") != "0"'
    target_new = 'VALIDATOR_ENABLE = os.environ.get("BIOHUB_VALIDATOR_ENABLE", "0") == "1"'
    if target_old in c7:
        c7 = c7.replace(target_old, target_new)
        cells[7]["source"] = [line + "\n" for line in c7.splitlines()]

    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")
    print(f"Patched: {path.relative_to(REPO_ROOT)}")
    return True

def main():
    experiments = [
        "EXP-051", "EXP-052", "EXP-053", "EXP-055",
        "EXP-056", "EXP-057", "EXP-058", "EXP-059",
    ]
    count = 0
    for exp in experiments:
        matches = list(REPO_ROOT.glob(f"experiments/{exp}*/kaggle/submission_notebook.ipynb"))
        for p in matches:
            if patch_notebook(p):
                count += 1
    print(f"Done. Successfully patched {count} notebooks.")

if __name__ == "__main__":
    main()
