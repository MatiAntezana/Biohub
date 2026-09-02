import json
from pathlib import Path

audit = json.loads(Path(".state/remote_experiments_audit.json").read_text())

for node, data in audit.items():
    print("=" * 80)
    print("NODE: " + node.upper())
    if "error" in data:
        print("  Error: " + str(data.get("error")))
        continue
    print("  GPU: " + str(data.get("gpu")))
    procs = data.get("procs", [])
    print("  Active Python Processes: " + str(len(procs)))
    for p in procs:
        print("    - " + p[:120])
    exps = data.get("experiments", {})
    print("  Experiments on Remote Disk (" + str(len(exps)) + "):")
    for exp_name, exp_info in exps.items():
        print("\n    📁 Experiment: " + exp_name)
        cps = exp_info.get("checkpoints", [])
        print("       Checkpoints (" + str(len(cps)) + "):")
        for cp in cps:
            print("         • " + Path(cp["path"]).name + " (" + str(cp["size_mb"]) + " MB)")
        arts = exp_info.get("artifacts", {})
        print("       Artifacts JSONs: " + str(list(arts.keys())))
        if "FAILED.json" in arts and isinstance(arts["FAILED.json"], dict):
            failed = arts["FAILED.json"]
            print("       ❌ FAILED Status: at " + str(failed.get("failed_at")))
            err = failed.get("error", "")
            err_line = err.splitlines()[-1] if err else "N/A"
            print("          Reason: " + err_line[:140])
        if "metrics_primary.json" in arts and isinstance(arts["metrics_primary.json"], list):
            epochs = arts["metrics_primary.json"]
            print("       📈 Primary Metrics: " + str(len(epochs)) + " epochs logged")
            if epochs:
                last_ep = epochs[-1]
                print("          Last Epoch " + str(last_ep.get("epoch")) + ": edge_loss=" + str(last_ep.get("edge_loss")) + ", det_loss=" + str(last_ep.get("detection_loss")))
        if "metrics_secondary.json" in arts and isinstance(arts["metrics_secondary.json"], list):
            epochs = arts["metrics_secondary.json"]
            print("       📈 Secondary Metrics: " + str(len(epochs)) + " epochs logged")
