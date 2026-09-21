"""Atomic receipts and content identities shared by local and remote commands."""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("core", "ops", "configs", "legacy", "vendor", "tests", "notebooks")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(clean(payload), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def code_manifest(root=ROOT):
    return {str(path.relative_to(root)): digest(path)
            for folder in CODE_DIRS for path in sorted((root / folder).rglob("*"))
            if path.is_file() and (path.suffix in {".py", ".json", ".ipynb", ".txt"} or path.name == "LICENSE")
            and "__pycache__" not in path.parts}


def code_hash(root=ROOT):
    return hashlib.sha256(json.dumps(code_manifest(root), sort_keys=True).encode()).hexdigest()
