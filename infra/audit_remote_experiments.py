#!/usr/bin/env python3
import subprocess
import json
import os
from pathlib import Path

nodes = ["a1552", "a1555", "a1557", "a1560", "a1558", "a1559", "a1563", "a1688"]
audit_results = {}

remote_probe = """
import os, glob, json, subprocess

res = {}

# 1. GPU Status
try:
    smi = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True)
    res["gpu"] = smi.strip()
except Exception as e:
    res["gpu"] = str(e)

# 2. Running Python Processes
try:
    ps = subprocess.check_output(["ps", "-eo", "pid,user,%cpu,%mem,etime,cmd"], text=True)
    py_procs = [l.strip() for l in ps.splitlines() if "python" in l and "camber" not in l and "grep" not in l]
    res["procs"] = py_procs
except Exception as e:
    res["procs"] = str(e)

# 3. Experiments directory inspection
exp_dir = os.path.expanduser("~/Documents/Biohub/experiments")
res["experiments"] = {}
if os.path.isdir(exp_dir):
    for item in sorted(os.listdir(exp_dir)):
        ipath = os.path.join(exp_dir, item)
        if os.path.isdir(ipath):
            info = {"checkpoints": [], "artifacts": {}, "logs": []}
            # Check checkpoints
            for cp in glob.glob(os.path.join(ipath, "**/*.pth"), recursive=True):
                size_mb = round(os.path.getsize(cp) / (1024*1024), 2)
                info["checkpoints"].append({"path": cp, "size_mb": size_mb, "mtime": os.path.getmtime(cp)})
            # Check artifacts
            for art in glob.glob(os.path.join(ipath, "artifacts/**"), recursive=True):
                if os.path.isfile(art):
                    fname = os.path.basename(art)
                    if fname.endswith(".json"):
                        try:
                            with open(art) as f:
                                info["artifacts"][fname] = json.load(f)
                        except Exception as e:
                            info["artifacts"][fname] = f"error: {e}"
            # Check logs
            for log in glob.glob(os.path.join(ipath, "logs/*.log")):
                info["logs"].append({"path": log, "size": os.path.getsize(log)})
            res["experiments"][item] = info

print("__JSON_START__")
print(json.dumps(res))
print("__JSON_END__")
"""

for node in nodes:
    print(f"Auditing {node}...")
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", node, "python3 -"]
    p = subprocess.run(cmd, input=remote_probe, capture_output=True, text=True)
    if p.returncode == 0 and "__JSON_START__" in p.stdout:
        try:
            payload_str = p.stdout.split("__JSON_START__")[1].split("__JSON_END__")[0].strip()
            audit_results[node] = json.loads(payload_str)
        except Exception as e:
            audit_results[node] = {"error": f"JSON parse error: {e}", "raw": p.stdout}
    else:
        audit_results[node] = {"error": (p.stderr or p.stdout).strip() or "SSH failed"}

Path(".state/remote_experiments_audit.json").write_text(json.dumps(audit_results, indent=2))
print("Audit saved to .state/remote_experiments_audit.json")
