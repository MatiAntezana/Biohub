"""Stage an explicitly reviewed HOCT revision in an isolated target directory."""
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.request

from core.artifacts import ROOT, digest, dump


def hoct_job(args, run):
    spec = json.loads((ROOT / "configs/hoct.json").read_text())
    if (spec.get("admissibility_reviewed") is not True or not spec.get("review_evidence_url")
        or not re.fullmatch(r"[0-9a-f]{40}", spec.get("source_revision") or "")
        or not re.fullmatch(r"[0-9a-f]{64}", spec.get("api_sha256") or "")):
        raise ValueError("HOCT needs an immutable reviewed revision, API hash and official rules evidence")
    if spec["source_url"] != "https://github.com/royerlab/hoct/releases/download/weights-v1/general_v1.pt":
        raise ValueError("Unexpected HOCT model source")
    assets = Path(args.artifact_root) / "assets"
    site = assets / "hoct-site"
    if site.exists() or (assets / "hoct.json").exists():
        raise FileExistsError("HOCT staging already exists; inspect prior receipt before any retry")
    wheel_dir = assets / "hoct-wheels"
    wheel_dir.mkdir(parents=True, exist_ok=False)
    source = "git+https://github.com/royerlab/hoct.git@" + spec["source_revision"]
    subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(wheel_dir), source], check=True)
    wheels = list(wheel_dir.glob("hoct-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected one HOCT wheel")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(site), str(wheels[0])], check=True)
    api_file = site / "hoct/_api.py"
    if digest(api_file) != spec["api_sha256"]:
        raise ValueError("Installed HOCT ABI source hash differs from reviewed source")
    checkpoint = assets / "hoct-general-v1.pt"
    with urllib.request.urlopen(spec["source_url"], timeout=60) as response, checkpoint.open("xb") as out:
        while chunk := response.read(1048576):
            out.write(chunk)
    if digest(checkpoint) != spec["sha256"]:
        raise ValueError("Downloaded HOCT checkpoint mismatch")
    dump(assets / "hoct.json", {**spec, "checkpoint": str(checkpoint),
        "wheel": str(wheels[0]), "wheel_sha256": digest(wheels[0])})
    return {"hoct_spec": str(assets / "hoct.json"), "installed_in_isolated_target": str(site)}
