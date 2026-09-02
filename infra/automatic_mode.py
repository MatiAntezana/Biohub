#!/usr/bin/env python3
"""Persistent operational autopilot for queued Biohub work.

It completes the declared EXP-031 -> EXP-030B bootstrap and may then create
variants through the bounded recipe policy.  It never edits arbitrary model
code or silently closes DOC_PENDING conclusions.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aienv import ensure_aienv
from experiment_scheduler import run_once as scheduler_run_once
from experimentlib import (
    atomic_write_json,
    exclusive_lock,
    experiment_directories,
    latest_events_by_run,
    load_manifest,
    load_queue,
    read_json,
    repo_root,
    utc_now,
)
from kaggle_pipeline import KagglePipeline, package_hash
from variant_planner import maintain_autonomous_queue
from variant_kaggle import prepare_ready_variants
from remote_gpu_monitor import compact_view as compact_gpu_view, scan_and_store as scan_remote_gpu_activity


def _state_path(root: Path) -> Path:
    return root / ".state" / "automatic-mode.json"


def _last_metrics(path: Path) -> dict[str, Any] | None:
    """Return only the latest compact row from an atomically written history."""

    try:
        payload = read_json(path, [])
    except Exception:
        return None
    if not isinstance(payload, list) or not payload or not isinstance(payload[-1], dict):
        return None
    row = payload[-1]
    keep = (
        "epoch",
        "edge_loss",
        "det_loss",
        "train_edge_loss",
        "train_detection_loss",
        "validation_acc",
        "validation_recall",
        "validation_score",
        "train_loss_diagnostic",
        "learning_rate",
    )
    compact = {key: row[key] for key in keep if key in row}
    return compact or row


def monitoring_snapshot(root: Path | None = None) -> dict[str, Any]:
    """Build a compact, token-safe snapshot for an attached agent session."""

    root = root or repo_root()
    state = read_json(_state_path(root), {"status": "STOPPED"})
    directories = experiment_directories(root)
    runs: list[dict[str, Any]] = []
    for item in load_queue(root).get("items", []):
        row: dict[str, Any] = {
            "experiment_id": item.get("experiment_id"),
            "status": item.get("status"),
            "run_id": item.get("run_id"),
            "node": item.get("node") or item.get("requested_node"),
        }
        experiment_id = item.get("experiment_id")
        run_id = item.get("run_id")
        directory = directories.get(experiment_id)
        if directory and run_id:
            run_dir = directory / "artifacts" / run_id
            progress: dict[str, Any] = {}
            for name, role in (
                ("metrics.json", "main"),
                ("metrics_primary.json", "primary"),
                ("metrics_secondary.json", "secondary"),
            ):
                metrics = _last_metrics(run_dir / name)
                if metrics:
                    progress[role] = metrics
            if progress:
                row["progress"] = progress
            heartbeat = read_json(run_dir / "HEARTBEAT.json", {})
            if heartbeat:
                row["heartbeat"] = {key: heartbeat.get(key) for key in ("status", "at", "pid")}
        runs.append(row)

    jobs_payload = read_json(root / ".state" / "kaggle-jobs.json", {"jobs": {}})
    jobs = [
        {
            "job_id": job.get("job_id"),
            "experiment_id": job.get("experiment_id"),
            "state": job.get("state"),
            "kernel_version": job.get("kernel_version"),
            "public_score": job.get("public_score"),
            "error": job.get("error"),
        }
        for job in jobs_payload.get("jobs", {}).values()
        if job.get("state") not in {"DRY_RUN"}
    ]
    errors = [
        {"at": event.get("at"), "action": event.get("action"), "error": event.get("error") or event.get("result")}
        for event in state.get("events", [])
        if event.get("action") in {"cycle-error", "kaggle-push-error"}
        or (event.get("action") == "variant-kaggle-prepare" and event.get("result", {}).get("errors"))
    ][-5:]
    gpu_activity = read_json(root / ".state" / "remote-gpu-activity.json", {"activities": [], "nodes": {}})
    return {
        "mode": state.get("status", "STOPPED"),
        "daemon_pid": state.get("pid"),
        "last_cycle_at": state.get("last_cycle_at"),
        "runs": runs,
        "kaggle_jobs": jobs,
        "recent_errors": errors,
        "remote_gpu_activity": compact_gpu_view(gpu_activity),
    }


def _snapshot_signature(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _record(state: dict[str, Any], action: str, **details: Any) -> None:
    state.setdefault("events", []).append({"at": utc_now(), "action": action, **details})
    state["events"] = state["events"][-500:]
    state["updated_at"] = utc_now()


def _run_reconcile(root: Path, state: dict[str, Any]) -> None:
    result = subprocess.run(
        [sys.executable, str(root / "infra" / "experimentctl.py"), "reconcile"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    line = (result.stdout or result.stderr).strip().splitlines()[-1:] or [""]
    _record(state, "reconcile", returncode=result.returncode, summary=line[0][-1000:])


def _watch_kaggle(root: Path, state: dict[str, Any]) -> None:
    pipeline = KagglePipeline(root)
    jobs = pipeline.store.read().get("jobs", {})
    for job_id, job in list(jobs.items()):
        if job.get("state") not in {"RUNNING", "QUEUED", "UNKNOWN", "SUBMITTED"}:
            continue
        updated = pipeline.watch_once(job_id)
        if updated.get("state") != job.get("state"):
            _record(state, "kaggle-transition", job_id=job_id, before=job.get("state"), after=updated.get("state"))


def _utc_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _latest_successful_runs(root: Path) -> dict[str, dict[str, Any]]:
    """Return the newest reconciled successful run for every experiment."""

    latest = latest_events_by_run(root)
    successful: dict[str, dict[str, Any]] = {}
    for run_id, event in latest.items():
        if event.get("status") not in {"SUCCEEDED", "DOC_PENDING", "CLOSED"}:
            continue
        experiment_id = event.get("experiment_id")
        if not experiment_id or str(run_id).startswith("queue-"):
            continue
        current = successful.get(experiment_id)
        if current is None or _utc_timestamp(event.get("timestamp_utc")) > _utc_timestamp(current.get("timestamp_utc")):
            successful[experiment_id] = {**event, "run_id": run_id}
    return successful


def _same_deployment(job: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if job.get("deployment_key") == candidate["deployment_key"]:
        return True
    # Jobs created before deployment_key/source_run_id existed remain
    # idempotent for the run that preceded them. A future retrain has a newer
    # success timestamp and therefore receives a new Kaggle version.
    return (
        not job.get("source_run_id")
        and job.get("experiment_id") == candidate["experiment_id"]
        and job.get("package_sha256") == candidate["package_sha256"]
        and _utc_timestamp(job.get("created_at")) >= _utc_timestamp(candidate.get("trained_at"))
    )


def _transient_push_failure(job: dict[str, Any]) -> bool:
    text = str(job.get("error") or "").lower()
    return any(
        token in text
        for token in (
            "maximum batch gpu session",
            "gpu session count",
            "429",
            "too many requests",
            "service unavailable",
            "temporarily unavailable",
            "connection reset",
            "gateway timeout",
        )
    )


def _push_new_successes(root: Path, state: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    """Automatically publish every trained, explicitly eligible deployment.

    This is deliberately broader than autonomous variants: normal experiments,
    bootstrap experiments and future variants all use the same gate. Publishing
    is authorized only by the experiment manifest and remains test-only.
    """

    policy = read_json(
        root / ".agents" / "kaggle-automation.json",
        {
            "enabled_in_automatic_mode": True,
            "max_concurrent_kaggle_jobs": 2,
            "max_push_attempts_per_deployment": 3,
            "push_retry_backoff_seconds": 600,
        },
    )
    summary: dict[str, Any] = {
        "dry_run": dry_run,
        "eligible": [],
        "pushed": [],
        "waiting": [],
        "blocked": [],
    }
    if policy.get("enabled_in_automatic_mode") is not True:
        summary["blocked"].append({"reason": "automatic Kaggle publishing disabled by policy"})
        return summary

    successful = _latest_successful_runs(root)
    pipeline = KagglePipeline(root)
    jobs = list(pipeline.store.read().get("jobs", {}).values())
    active_states = {"PUSHING", "RUNNING", "QUEUED", "UNKNOWN", "SUBMITTED"}
    active_count = sum(job.get("state") in active_states for job in jobs)
    max_concurrent = int(policy.get("max_concurrent_kaggle_jobs", 2))
    max_attempts = int(policy.get("max_push_attempts_per_deployment", 3))
    backoff_seconds = int(policy.get("push_retry_backoff_seconds", 600))

    candidates: list[dict[str, Any]] = []
    for experiment_id, event in successful.items():
        try:
            manifest_path, manifest = load_manifest(experiment_id, root)
        except Exception as error:
            summary["blocked"].append({"experiment_id": experiment_id, "reason": repr(error)})
            continue
        config = manifest.get("kaggle", {})
        if config.get("enabled") is not True:
            continue
        if config.get("auto_publish_after_train", config.get("auto_push")) is not True:
            continue
        if config.get("auto_push") is not True:
            summary["blocked"].append({"experiment_id": experiment_id, "reason": "kaggle.auto_push no autorizado"})
            continue
        if config.get("deployment_ready") is not True:
            summary["waiting"].append({"experiment_id": experiment_id, "run_id": event["run_id"], "reason": "deployment_ready=false"})
            continue
        kaggle_dir = manifest_path.parent / "kaggle"
        digest = package_hash(kaggle_dir)
        candidate = {
            "experiment_id": experiment_id,
            "run_id": event["run_id"],
            "trained_at": event.get("timestamp_utc"),
            "package_sha256": digest,
            "deployment_key": f"{experiment_id}:{event['run_id']}:{digest}",
            "priority": int(manifest.get("priority", 100)),
            "kernel_slug": config.get("kernel_slug"),
        }
        candidates.append(candidate)

    candidates.sort(key=lambda item: (item["priority"], item.get("trained_at") or "", item["experiment_id"]))
    now = time.time()
    for candidate in candidates:
        experiment_id = candidate["experiment_id"]
        summary["eligible"].append({key: candidate[key] for key in ("experiment_id", "run_id", "deployment_key")})
        matching = [job for job in jobs if _same_deployment(job, candidate)]
        active_for_kernel = [
            job for job in jobs
            if job.get("kernel_slug") == candidate.get("kernel_slug") and job.get("state") in active_states
        ]
        if active_for_kernel:
            summary["waiting"].append({"experiment_id": experiment_id, "reason": "kernel version already active", "job_id": active_for_kernel[-1].get("job_id")})
            continue
        handled = [job for job in matching if job.get("state") not in {"FAILED", "RETRY_WAIT"}]
        if handled:
            summary["waiting"].append({"experiment_id": experiment_id, "reason": "deployment already registered", "job_id": handled[-1].get("job_id"), "state": handled[-1].get("state")})
            continue
        failed = [job for job in matching if job.get("state") in {"FAILED", "RETRY_WAIT"}]
        if failed:
            latest_failure = max(failed, key=lambda job: _utc_timestamp(job.get("updated_at") or job.get("created_at")))
            if not _transient_push_failure(latest_failure):
                summary["blocked"].append({"experiment_id": experiment_id, "reason": "deterministic Kaggle push failure", "job_id": latest_failure.get("job_id")})
                continue
            if len(failed) >= max_attempts:
                summary["blocked"].append({"experiment_id": experiment_id, "reason": "automatic push retry budget exhausted", "attempts": len(failed)})
                continue
            age = now - _utc_timestamp(latest_failure.get("updated_at") or latest_failure.get("created_at"))
            if age < backoff_seconds:
                summary["waiting"].append({"experiment_id": experiment_id, "reason": "transient push backoff", "retry_in_seconds": max(0, int(backoff_seconds - age))})
                continue
        if active_count >= max_concurrent:
            summary["waiting"].append({"experiment_id": experiment_id, "reason": "Kaggle concurrency budget reached"})
            continue
        if dry_run:
            summary["waiting"].append({"experiment_id": experiment_id, "reason": "dry-run: would preflight and push"})
            continue
        try:
            job = pipeline.push(experiment_id)
            job.update(
                source_run_id=candidate["run_id"],
                deployment_key=candidate["deployment_key"],
                trained_at=candidate.get("trained_at"),
            )
            pipeline.store.save_job(job)
            jobs.append(job)
            if job.get("state") in active_states:
                active_count += 1
            summary["pushed"].append({"experiment_id": experiment_id, "run_id": candidate["run_id"], "job_id": job["job_id"], "state": job.get("state")})
            _record(state, "kaggle-push", experiment_id=experiment_id, job_id=job["job_id"], state=job["state"])
        except Exception as error:
            summary["blocked"].append({"experiment_id": experiment_id, "reason": repr(error)})
            _record(state, "kaggle-push-error", experiment_id=experiment_id, error=repr(error))
    return summary


def _sync_hf_and_git(root: Path, state: dict[str, Any], *, dry_run: bool = False) -> None:
    """Automatically back up heavy weights to Hugging Face and push code/ledgers to GitHub."""
    if dry_run:
        return

    # 1. Check if any new checkpoints need Hugging Face upload
    try:
        from hf_sync import upload_experiment_checkpoints
        for d in (root / "experiments").iterdir():
            if d.is_dir() and d.name.startswith("EXP-"):
                pths = list(d.rglob("*.pth"))
                if pths:
                    res = upload_experiment_checkpoints(d.name, root=root, purge=True)
                    if res.get("status") == "success":
                        _record(state, "hf-checkpoint-upload", experiment_id=d.name, bytes=res.get("total_bytes"))
    except Exception as e:
        _record(state, "hf-sync-error", error=repr(e))

    # 2. Check if git repository has modifications to commit and push
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
        if status.returncode == 0 and status.stdout.strip():
            subprocess.run(["git", "add", "."], cwd=root, capture_output=True, check=False)
            commit_msg = f"chore(auto-mode): update experiment ledgers, manifests, and states [{utc_now()}]"
            subprocess.run(["git", "commit", "-m", commit_msg], cwd=root, capture_output=True, check=False)
            push_res = subprocess.run(["git", "push", "origin", "main"], cwd=root, capture_output=True, text=True, check=False)
            if push_res.returncode == 0:
                _record(state, "git-auto-push", commit=commit_msg)
            else:
                _record(state, "git-push-error", error=push_res.stderr.strip()[:300])
    except Exception as e:
        _record(state, "git-sync-error", error=repr(e))


def cycle(root: Path, state: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    _run_reconcile(root, state)
    scheduled = scheduler_run_once(dry_run=dry_run, root=root)
    _record(state, "scheduler", result=scheduled)
    variants = maintain_autonomous_queue(root=root, dry_run=dry_run)
    _record(state, "variant-policy", result=variants)
    deployments = prepare_ready_variants(root=root, dry_run=dry_run)
    _record(state, "variant-kaggle-prepare", result=deployments)
    if not dry_run:
        gpu_activity = scan_remote_gpu_activity(root)
        gpu_summary = compact_gpu_view(gpu_activity)
        if gpu_summary["transitions"] or gpu_summary["node_errors"]:
            _record(state, "remote-gpu-activity", result=gpu_summary)
        _watch_kaggle(root, state)
    kaggle_automation = _push_new_successes(root, state, dry_run=dry_run)
    _record(state, "kaggle-auto-publish", result=kaggle_automation)
    _sync_hf_and_git(root, state, dry_run=dry_run)
    state["last_cycle_at"] = utc_now()
    atomic_write_json(_state_path(root), state)
    return state


def daemon(interval: int) -> None:
    root = repo_root()
    path = _state_path(root)
    stop = False

    def handle_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    with exclusive_lock(root / ".state" / "automatic-mode.lock", nonblocking=True):
        state = read_json(path, {})
        state.update(status="RUNNING", pid=os.getpid(), started_at=state.get("started_at") or utc_now())
        _record(state, "started", pid=os.getpid())
        atomic_write_json(path, state)
        while not stop:
            try:
                cycle(root, state)
            except Exception as error:
                _record(state, "cycle-error", error=repr(error))
                atomic_write_json(path, state)
            deadline = time.monotonic() + max(15, interval)
            while not stop and time.monotonic() < deadline:
                time.sleep(min(2, deadline - time.monotonic()))
        state.update(status="STOPPED", stopped_at=utc_now(), pid=None)
        _record(state, "stopped")
        atomic_write_json(path, state)


def start(interval: int) -> dict[str, Any]:
    root = repo_root()
    current = read_json(_state_path(root), {})
    if current.get("status") == "RUNNING" and _pid_alive(current.get("pid")):
        return {"already_running": True, "pid": current["pid"]}
    atomic_write_json(
        _state_path(root),
        {"schema_version": 1, "status": "STARTING", "pid": None, "started_at": utc_now(), "events": []},
    )
    log_path = root / ".state" / "automatic-mode.log"
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(root / "infra" / "automatic_mode.py"), "daemon", "--interval", str(interval)],
        cwd=root,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    state = read_json(_state_path(root), {})
    if state.get("status") == "STARTING":
        state["pid"] = process.pid
        atomic_write_json(_state_path(root), state)
    return {"already_running": False, "pid": process.pid, "log": str(log_path.relative_to(root))}


def supervise(interval: int, poll_seconds: int | None, heartbeat_seconds: int | None) -> None:
    """Attach the invoking agent to the daemon until an explicit stop.

    The daemon owns work. This foreground process owns visibility: it emits a
    compact JSON line whenever state/progress changes and a bounded heartbeat
    while nothing changes. An agent tool call therefore remains open instead
    of returning a final answer immediately after startup.
    """

    root = repo_root()
    policy = read_json(root / ".agents" / "monitoring.json", {})
    poll_seconds = int(poll_seconds or policy.get("poll_seconds", 10))
    heartbeat_seconds = int(heartbeat_seconds or policy.get("heartbeat_seconds", 60))
    attached = start(interval)
    print(json.dumps({"event": "AUTOMATIC_ATTACHED", **attached}, sort_keys=True, ensure_ascii=False), flush=True)
    previous = ""
    last_heartbeat = 0.0
    while True:
        state = read_json(_state_path(root), {"status": "STOPPED"})
        status = state.get("status", "STOPPED")
        pid = state.get("pid")
        if status in {"RUNNING", "STARTING"} and pid and not _pid_alive(pid):
            restarted = start(interval)
            print(json.dumps({"event": "AUTOMATIC_RESTARTED", **restarted}, sort_keys=True, ensure_ascii=False), flush=True)
            previous = ""
            time.sleep(max(2, min(poll_seconds, 10)))
            continue
        snapshot = monitoring_snapshot(root)
        signature = _snapshot_signature(snapshot)
        now = time.monotonic()
        if signature != previous:
            print(json.dumps({"event": "AUTOMATIC_UPDATE", "at": utc_now(), **snapshot}, sort_keys=True, ensure_ascii=False), flush=True)
            previous = signature
            last_heartbeat = now
        elif now - last_heartbeat >= heartbeat_seconds:
            print(json.dumps({"event": "AUTOMATIC_HEARTBEAT", "at": utc_now(), **snapshot}, sort_keys=True, ensure_ascii=False), flush=True)
            last_heartbeat = now
        if status == "STOPPED":
            print(json.dumps({"event": "AUTOMATIC_DETACHED", "at": utc_now(), **snapshot}, sort_keys=True, ensure_ascii=False), flush=True)
            return
        time.sleep(max(2, poll_seconds))


def stop() -> dict[str, Any]:
    root = repo_root()
    state = read_json(_state_path(root), {})
    pid = state.get("pid")
    was_running = _pid_alive(pid)
    state["stop_requested_at"] = utc_now()
    atomic_write_json(_state_path(root), state)
    if was_running:
        os.kill(int(pid), signal.SIGTERM)
        deadline = time.monotonic() + 10
        while _pid_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.2)
    final = read_json(_state_path(root), state)
    counts: dict[str, int] = {}
    for event in final.get("events", []):
        counts[event.get("action", "unknown")] = counts.get(event.get("action", "unknown"), 0) + 1
    return {
        "was_running": was_running,
        "status": final.get("status", "STOPPED"),
        "started_at": final.get("started_at"),
        "stopped_at": final.get("stopped_at"),
        "actions": counts,
        "question": "¿Cómo querés proseguir con los entrenamientos, jobs Kaggle o documentación que quedaron pendientes?",
    }


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start_parser = sub.add_parser("start")
    start_parser.add_argument("--interval", type=int, default=120)
    daemon_parser = sub.add_parser("daemon")
    daemon_parser.add_argument("--interval", type=int, default=120)
    supervise_parser = sub.add_parser("supervise")
    supervise_parser.add_argument("--interval", type=int, default=120)
    supervise_parser.add_argument("--poll-seconds", type=int)
    supervise_parser.add_argument("--heartbeat-seconds", type=int)
    sub.add_parser("stop")
    sub.add_parser("status")
    once = sub.add_parser("once")
    once.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = repo_root()
    if args.command == "daemon":
        daemon(args.interval)
        return
    if args.command == "supervise":
        supervise(args.interval, args.poll_seconds, args.heartbeat_seconds)
        return
    if args.command == "start":
        result = start(args.interval)
    elif args.command == "stop":
        result = stop()
    elif args.command == "once":
        state = read_json(_state_path(root), {"schema_version": 1, "status": "ONE_SHOT", "events": []})
        result = cycle(root, state, dry_run=args.dry_run)
        result["status"] = "STOPPED"
        result["pid"] = None
        atomic_write_json(_state_path(root), result)
    else:
        result = read_json(_state_path(root), {"status": "STOPPED"})
    print(json.dumps({"ok": True, "action": f"automatic-{args.command}", "result": result}, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
