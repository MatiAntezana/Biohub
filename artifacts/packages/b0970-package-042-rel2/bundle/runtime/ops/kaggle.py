"""Explicit Kaggle actions with exact run/version identity and uncertain-write stops."""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.cli_bootstrap import configure_cli_imports
configure_cli_imports(ROOT)
from core.artifacts import dump, digest, clean
from core.remote_executor import LOCAL_PYTHON
from core.submission_contract import validate_csv
from core.protocol import PROTOCOL

COMPETITION = "biohub-cell-tracking-during-development"


def ready(path):
    meta = json.loads((path / "publish-ready.json").read_text())
    if not all(meta["gates"][phase]["passed"] for phase in ("development", "confirmation")):
        raise ValueError("Measured promotion gates did not pass")
    for relative, key in (("kaggle/submission.ipynb", "notebook_sha256"), ("bundle/bundle.json", "bundle_sha256"),
                          ("dataset-upload/biohub_payload.zip", "archive_sha256")):
        if digest(path / relative) != meta[key]:
            raise ValueError("Publish package modified after validation")
    return meta


def api_submissions(api):
    return api.competition_submissions(COMPETITION, page_size=100) or []


def status_name(value):
    """Accept SDK enums and their textual representation, never substring matches."""
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


def require_complete_kernel(api, kernel):
    response = api.kernels_status(kernel)
    if status_name(response.status) != "COMPLETE" or response.failure_message:
        raise RuntimeError(f"Kernel is not successfully complete: {response.status}; {response.failure_message}")
    return response


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["upload", "push", "status", "download", "submit", "score"])
    p.add_argument("--package", required=True, help="Fetched complete package run directory")
    args = p.parse_args()
    if platform.system() != "Darwin" or Path(sys.executable).resolve() != Path(LOCAL_PYTHON).resolve():
        raise RuntimeError("Kaggle operations run only from the Mac AIEnv controller")
    package = Path(args.package).resolve()
    meta = ready(package)
    kernel_meta = json.loads((package / "kaggle/kernel-metadata.json").read_text())
    kernel = kernel_meta["id"]
    if kernel != "matiasantezana/biohub-0970-" + meta["run_id"]:
        raise ValueError("Unexpected shared/legacy Kaggle kernel target")
    upload_meta = json.loads((package / "dataset-upload/dataset-metadata.json").read_text())
    dataset_id = upload_meta.get("id")
    expected_dataset_ids = {"matiasantezana/biohub-pkg-" + meta["run_id"], kernel}
    if dataset_id not in expected_dataset_ids:
        raise ValueError("Unexpected Kaggle dataset target")
    if (kernel_meta.get("dataset_sources") != [dataset_id] or kernel_meta.get("is_private") is not True
        or kernel_meta.get("enable_internet") is not False or kernel_meta.get("enable_gpu") is not True
        or kernel_meta.get("competition_sources") != [COMPETITION]):
        raise ValueError("Generated Kaggle metadata was modified")
    state_root = ROOT / ".state/kaggle"
    state_root.mkdir(parents=True, exist_ok=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    with (state_root / "operator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = state_root / (meta["run_id"] + ".json")
        state = json.loads(state_path.read_text()) if state_path.exists() else {"run_id": meta["run_id"], "kernel": kernel}
        if args.action in {"upload", "push", "submit"}:
            marker = args.action + "_attempted"
            if state.get(marker):
                raise RuntimeError("This write was already attempted. Reconcile its actual Kaggle outcome; never retry blindly")
            if args.action == "push" and not state.get("upload_complete"):
                raise RuntimeError("Upload must complete before kernel push")
            if args.action == "push":
                dataset_status = api.dataset_status(dataset_id)
                if dataset_status.lower() not in {"ready", "dataset_creation_status_ready"}:
                    raise RuntimeError(f"Dataset is not ready yet ({dataset_status}); wait before push")
            if args.action == "submit":
                if not state.get("smoke_passed") or not state.get("version"):
                    raise RuntimeError("Need exact kernel version and downloaded runtime smoke validation")
                submissions = api_submissions(api)
                today = dt.datetime.now(dt.timezone.utc).date()
                dated = [s for s in submissions if getattr(s, "date", None) and s.date.date() == today]
                if len(dated) >= PROTOCOL["promotion"]["max_submissions_per_utc_day"]:
                    raise RuntimeError("Predeclared one-submission UTC-day budget exhausted; wait for next day")
                if any(str(getattr(s, "description", "")) == "BIOHUB0970 " + meta["run_id"] for s in submissions):
                    raise RuntimeError("Submission already exists for this exact run")
            state[marker] = dt.datetime.now(dt.timezone.utc).isoformat()
            dump(state_path, state)  # Persist BEFORE the network mutation, even if response is lost.
            if args.action == "upload":
                response = api.dataset_create_new(str(package / "dataset-upload"), public=False,
                    quiet=False, convert_to_csv=False, dir_mode="skip")
                if getattr(response, "error", None):
                    raise RuntimeError(str(response.error))
                state["upload_complete"] = True
            elif args.action == "push":
                response = api.kernels_push(str(package / "kaggle"))
                v_num = response.version_number or getattr(response, "versionNumber", None) or 1
                if response.error or response.ref != kernel:
                    raise RuntimeError(f"Kernel push identity/error: {response.error}")
                state.update(version=v_num, kernel=response.ref)
            else:
                response = api.competition_submit_code("submission.csv", "BIOHUB0970 " + meta["run_id"],
                    competition=COMPETITION, kernel=kernel, kernel_version=state["version"])
                if not response.ref:
                    raise RuntimeError("Submission response lacks exact reference; reconcile without retry")
                state["submission_ref"] = str(response.ref)
        elif args.action == "status":
            print(api.kernels_status(kernel))
        elif args.action == "download":
            if not state.get("version"):
                raise RuntimeError("No recorded kernel version")
            # Check before creating the immutable output directory, so an early
            # status query does not prevent the eventual completed download.
            require_complete_kernel(api, kernel)
            outputs = package / "kaggle-output"
            outputs.mkdir(exist_ok=True)
            api.kernels_output(kernel, path=str(outputs), file_pattern="(submission.csv|runtime_receipt.json)$", quiet=False)
            receipt = json.loads((outputs / "runtime_receipt.json").read_text())
            if (receipt.get("status") != "completed" or receipt.get("run_id") != meta["run_id"]
                or receipt.get("bundle_sha256") != meta["bundle_sha256"] or receipt.get("ground_truth_accessed") is not False):
                raise ValueError("Downloaded runtime is not this exact immutable experiment")
            contract = validate_csv(outputs / "submission.csv", expected_datasets=receipt["test_datasets"])
            if contract != receipt["contract"] or receipt["seconds"] > PROTOCOL["promotion"]["max_runtime_seconds"]:
                raise ValueError("Downloaded artifact contract/timing mismatch")
            exact_sha = (contract["csv_sha256"] == meta["visible_contract"]["csv_sha256"])
            structural_parity = (
                contract["nodes"] == meta["visible_contract"]["nodes"]
                and contract["divisions"] == meta["visible_contract"]["divisions"]
                and contract["datasets"] == meta["visible_contract"]["datasets"]
                and abs(contract["edges"] - meta["visible_contract"]["edges"]) <= 25
                and abs(contract["csv_rows"] - meta["visible_contract"]["csv_rows"]) <= 25
            )
            if not (exact_sha or structural_parity):
                raise ValueError("Kaggle visible output differs from remote visible output; return to design for parity analysis")
            for old_path in state_root.glob("*.json"):
                old = json.loads(old_path.read_text())
                if old.get("submission_ref") and old.get("csv_sha256") == contract["csv_sha256"]:
                    raise ValueError("This CSV has already been submitted")
            state.update(smoke_passed=True, csv_sha256=contract["csv_sha256"])
        else:
            rows = api_submissions(api)
            matches = [s for s in rows if str(s.ref) == state.get("submission_ref")]
            if len(matches) != 1:
                raise RuntimeError("Exact submission reference unavailable; no newest-submission fallback")
            row = matches[0]
            status = str(row.status)
            state["submission_status"] = status
            err = getattr(row, "error_description", None)
            if err:
                state["submission_error"] = str(err)
            if status_name(row.status) == "COMPLETE":
                if err:
                    state["submission_status"] = "ERROR"
                else:
                    value = getattr(row, "public_score", None)
                    if value is None or value == "":
                        raise RuntimeError("COMPLETE without score is unresolved")
                    score = float(value)
                    import math
                    if not math.isfinite(score):
                        raise ValueError("Non-finite Kaggle score")
                    state.update(public_score=score, target_reached=score >= .970)
        dump(state_path, state)
        print(json.dumps(clean(state), indent=2))


if __name__ == "__main__":
    main()
