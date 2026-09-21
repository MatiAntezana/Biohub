"""Pair-specific lineage ILP on SCIP (the backend used by ilpy).

Fixed reference detections cannot be deleted. A second child is possible only
through one approved mother/daughter-pair variable, not a mother-wide discount.
Rescued nodes must have both an incoming and outgoing edge or are omitted.
"""
from __future__ import annotations
from collections import defaultdict
import math

import numpy as np

from core.nextgen.contracts import Lineage


def reference_divisions(edges):
    children = defaultdict(list)
    for u, v in edges:
        children[int(u)].append(int(v))
    return [(u, *sorted(vs), .5) for u, vs in sorted(children.items()) if len(vs) == 2]


def log_odds(probability):
    p = min(1 - 1e-6, max(1e-6, float(probability)))
    return math.log(p / (1 - p))


def solve_lineage(coords, pairs, probabilities, divisions, fixed_count, config):
    from pyscipopt import Model, quicksum
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    probabilities = np.asarray(probabilities, float)
    if probabilities.shape != (len(pairs),) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid learned edge probabilities")
    if len(set(map(tuple, pairs))) != len(pairs) or not 0 < fixed_count <= len(coords):
        raise ValueError("Duplicated candidate edges or invalid fixed-node boundary")
    if len(pairs) and ((pairs < 0).any() or (pairs >= len(coords)).any()
            or not (coords[pairs[:, 1], 0] == coords[pairs[:, 0], 0] + 1).all()):
        raise ValueError("Candidate endpoints must be consecutive and in bounds")
    incoming, outgoing = defaultdict(list), defaultdict(list)
    index = {tuple(edge): i for i, edge in enumerate(pairs)}
    model = Model("biohub_pair_hypotheses")
    model.hideOutput()
    model.setRealParam("limits/time", float(config["timeout_seconds"]))
    model.setIntParam("parallel/maxnthreads", 1)
    model.setIntParam("randomization/randomseedshift", int(config["seed"]))
    variables = [model.addVar(vtype="B", name=f"e{i}", obj=-log_odds(p)) for i, p in enumerate(probabilities)]
    for i, (u, v) in enumerate(pairs):
        outgoing[int(u)].append(variables[i])
        incoming[int(v)].append(variables[i])
    division_vars = defaultdict(list)
    seen = set()
    for mother, a, b, probability in divisions:
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid division evidence")
        key = (int(mother), *sorted((int(a), int(b))))
        mother, a, b = key
        if key in seen or a == b or mother >= fixed_count or a >= fixed_count or b >= fixed_count:
            raise ValueError("Invalid/duplicated division hypothesis, or a rescue used for mitosis")
        seen.add(key)
        if (mother, a) not in index or (mother, b) not in index:
            raise ValueError("Division children must both be candidate edges")
        cost = config["division_penalty"] - config["division_evidence_weight"] * log_odds(probability)
        z = model.addVar(vtype="B", name=f"d{len(seen)}", obj=float(cost))
        model.addCons(z <= variables[index[(mother, a)]])
        model.addCons(z <= variables[index[(mother, b)]])
        division_vars[mother].append(z)
    retained = {}
    last_frame = coords[:, 0].max()
    for node in range(len(coords)):
        inc, out = quicksum(incoming[node]), quicksum(outgoing[node])
        if node >= fixed_count:
            q = model.addVar(vtype="B", name=f"r{node}", obj=float(config["rescue_penalty"]))
            retained[node] = q
            model.addCons(inc == q)
            model.addCons(out == q)
        else:
            model.addCons(inc <= 1)
            forks = quicksum(division_vars[node])
            model.addCons(forks <= 1)
            model.addCons(out <= 1 + forks)
            # One selected pair forces exactly its two edges. Other pairs/third
            # children cannot hitchhike on a cheap mother-level division cost.
            death = model.addVar(vtype="B", name=f"end{node}",
                obj=0.0 if coords[node, 0] == last_frame else float(config["disappearance_penalty"]))
            model.addCons(out >= 1 - death)
            model.addCons(out <= 2 * (1 - death))
    model.setMinimize()
    model.optimize()
    status = str(model.getStatus())
    if model.getNSols() == 0:
        raise RuntimeError(f"SCIP returned no feasible incumbent ({status}); no silent LAP fallback")
    solution = model.getBestSol()
    selected = pairs[[model.getSolVal(solution, var) > .5 for var in variables]]
    keep = list(range(fixed_count)) + [i for i, var in retained.items() if model.getSolVal(solution, var) > .5]
    remap = {old: new for new, old in enumerate(keep)}
    result = Lineage(coords[keep].copy(), np.asarray([(remap[int(u)], remap[int(v)]) for u, v in selected], dtype=np.int64).reshape(-1, 2)).validate()
    report = {"status": status, "optimal": status == "optimal", "solver_seconds": float(model.getSolvingTime()),
              "mip_gap": float(model.getGap()), "objective": float(model.getObjVal()),
              "candidate_edges": len(pairs), "selected_edges": len(selected),
              "approved_division_pairs": len(seen), "selected_rescues": len(keep) - fixed_count}
    model.freeProb()
    return result, report
