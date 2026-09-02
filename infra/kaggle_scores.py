#!/usr/bin/env python3
"""Append-only Kaggle public-score ledger and generated leaderboard view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from aienv import ensure_aienv
from experimentlib import (
    atomic_write_text,
    canonical_json,
    exclusive_lock,
    repo_root,
    sha256_bytes,
    utc_now,
)


def ledger_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "experiments" / "kaggle_scores.jsonl"


def read_scores(root: Path | None = None) -> list[dict[str, Any]]:
    path = ledger_path(root)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"kaggle_scores.jsonl corrupto en línea {line_number}: {error}") from error
    return rows


def verify_scores(root: Path | None = None) -> list[str]:
    errors: list[str] = []
    previous_hash: str | None = None
    seen: set[str] = set()
    for index, row in enumerate(read_scores(root), 1):
        event_id = row.get("event_id")
        if not event_id or event_id in seen:
            errors.append(f"fila {index}: event_id ausente o duplicado")
        seen.add(str(event_id))
        if row.get("previous_event_hash") != previous_hash:
            errors.append(f"fila {index}: previous_event_hash inválido")
        stored = row.get("event_hash")
        computed = sha256_bytes(canonical_json({key: value for key, value in row.items() if key != "event_hash"}))
        if stored != computed:
            errors.append(f"fila {index}: event_hash inválido")
        previous_hash = stored
    return errors


def append_score(
    *,
    experiment_id: str,
    status: str,
    source: str,
    submission_ref: str | int | None = None,
    public_score: float | str | None = None,
    kernel_slug: str | None = None,
    kernel_version: int | None = None,
    kernel_version_id: str | int | None = None,
    submitted_at: str | None = None,
    job_id: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = root or repo_root()
    path = ledger_path(root)
    identity = str(submission_ref or job_id or f"{experiment_id}:{kernel_slug}:{kernel_version}")
    try:
        score = float(public_score) if public_score not in (None, "", "None") else None
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"public_score inválido: {public_score!r}") from error
    normalized_status = status.upper()
    if normalized_status == "COMPLETE" and score is None:
        normalized_status = "PENDING_SCORE"
    event_id = sha256_bytes(f"kaggle-score|{identity}|{normalized_status}|{score}".encode())[:24]
    with exclusive_lock(path.with_suffix(".lock")):
        rows = read_scores(root)
        existing = next((row for row in rows if row.get("event_id") == event_id), None)
        if existing is not None:
            # Pending and scored states have different IDs; identical events are immutable.
            return existing
        row: dict[str, Any] = {
            "schema_version": 1,
            "event_id": event_id,
            "recorded_at": utc_now(),
            "submitted_at": submitted_at,
            "experiment_id": experiment_id,
            "status": normalized_status,
            "public_score": score,
            "submission_ref": str(submission_ref) if submission_ref is not None else None,
            "kernel_slug": kernel_slug,
            "kernel_version": kernel_version,
            "kernel_version_id": str(kernel_version_id) if kernel_version_id is not None else None,
            "job_id": job_id,
            "source": source,
            "previous_event_hash": rows[-1].get("event_hash") if rows else None,
        }
        row["event_hash"] = sha256_bytes(canonical_json(row))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return row


def import_legacy_scores(root: Path | None = None) -> list[dict[str, Any]]:
    root = root or repo_root()
    source = "Librería-Registros/Scores_Kaggle_Leaderboard/00_Tabla_Maestra_Scores_Kaggle.md"
    rows = [
        dict(experiment_id="EXP-017", status="COMPLETE", public_score=0.6830, submission_ref="55804607", kernel_slug="matiasantezana/biohub-exp-017-champion-090", kernel_version=8, kernel_version_id="345224829", submitted_at="2026-08-27T02:04:00Z"),
        dict(experiment_id="EXP-019", status="COMPLETE", public_score=0.6430, submission_ref="55804613", kernel_slug="matiasantezana/biohub-exp-019-wide-resunet-3d", kernel_version=3, kernel_version_id="345225808", submitted_at="2026-08-27T02:04:00Z"),
        dict(experiment_id="EXP-022", status="PENDING", public_score=None, submission_ref="55807435", kernel_slug="matiasantezana/biohub-exp-022-stardist-3d", kernel_version=4, kernel_version_id="345224886", submitted_at="2026-08-27T02:04:00Z"),
        dict(experiment_id="EXP-000", status="COMPLETE", public_score=0.1430, submission_ref="55783941", kernel_slug=None, kernel_version=1, kernel_version_id=None, submitted_at="2026-08-25T18:30:00Z"),
    ]
    return [append_score(root=root, source=source, **row) for row in rows]


def best_by_experiment(root: Path | None = None) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in read_scores(root):
        score = row.get("public_score")
        if score is None or row.get("status") not in {"COMPLETE", "SCORED"}:
            continue
        current = best.get(row["experiment_id"])
        if current is None or float(score) > float(current["public_score"]):
            best[row["experiment_id"]] = row
    return sorted(best.values(), key=lambda row: (-float(row["public_score"]), row["experiment_id"]))


def render_leaderboard(root: Path | None = None) -> Path:
    root = root or repo_root()
    ranked = best_by_experiment(root)
    score_rows = read_scores(root)
    latest_submission_state: dict[str, dict[str, Any]] = {}
    for row in score_rows:
        identity = str(row.get("submission_ref") or row.get("job_id") or row.get("event_id"))
        latest_submission_state[identity] = row
    pending = [
        row
        for row in latest_submission_state.values()
        if row.get("public_score") is None
        and row.get("status") not in {"FAILED"}
    ]
    lines = [
        "# Ranking de experimentos por Public Score de Kaggle",
        "",
        "> Vista factual generada desde `experiments/kaggle_scores.jsonl`.",
        "> Este ranking es independiente del Promotion Board local y se ordena exclusivamente por el mejor Public Score oficial de Kaggle de cada experimento.",
        "",
        f"Actualizado: `{utc_now()}`",
        "",
        "| Puesto | Experimento | Mejor Public Score | Kernel | Versión | Submission Ref | Fecha UTC |",
        "|---:|---|---:|---|---:|---|---|",
    ]
    for rank, row in enumerate(ranked, 1):
        lines.append(
            f"| {rank} | **{row['experiment_id']}** | **{float(row['public_score']):.4f}** | "
            f"{row.get('kernel_slug') or '—'} | {row.get('kernel_version') or '—'} | "
            f"{row.get('submission_ref') or '—'} | {row.get('submitted_at') or row.get('recorded_at', '—')} |"
        )
    if not ranked:
        lines.append("| — | Sin scores registrados | — | — | — | — | — |")
    lines += ["", "## Submissions pendientes de score", "", "| Experimento | Estado | Versión | Submission Ref | Fecha UTC |", "|---|---|---:|---|---|"]
    for row in pending:
        lines.append(
            f"| {row['experiment_id']} | {row['status']} | {row.get('kernel_version') or '—'} | "
            f"{row.get('submission_ref') or '—'} | {row.get('submitted_at') or row.get('recorded_at', '—')} |"
        )
    if not pending:
        lines.append("| — | Sin pendientes | — | — | — |")
    lines += [
        "",
        "## Interpretación",
        "",
        "Un puesto alto aquí significa mejor resultado oficial en Kaggle. No implica necesariamente mejor robustez en el test local; ambos rankings deben consultarse por separado.",
        "",
    ]
    path = root / "experiments" / "KAGGLE_LEADERBOARD.md"
    atomic_write_text(path, "\n".join(lines))
    return path


def record_job_score(job: dict[str, Any], submission: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    submission_ref = submission.get("ref") or submission.get("id") or submission.get("submissionRef")
    row = append_score(
        experiment_id=job["experiment_id"],
        status="SCORED",
        public_score=job.get("public_score"),
        submission_ref=submission_ref,
        kernel_slug=job.get("kernel_slug"),
        kernel_version=job.get("kernel_version"),
        submitted_at=submission.get("date") or submission.get("dateSubmitted") or job.get("updated_at"),
        job_id=job.get("job_id"),
        source="infra/kaggle_pipeline.py",
        root=root,
    )
    render_leaderboard(root)
    return row


def main() -> None:
    ensure_aienv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("import-legacy", "render", "status", "verify"))
    args = parser.parse_args()
    if args.command == "import-legacy":
        imported = import_legacy_scores()
        path = render_leaderboard()
        payload = {"imported": len(imported), "view": str(path.relative_to(repo_root()))}
    elif args.command == "render":
        path = render_leaderboard()
        payload = {"view": str(path.relative_to(repo_root()))}
    elif args.command == "verify":
        errors = verify_scores()
        payload = {"valid": not errors, "errors": errors}
    else:
        payload = {"ranking": best_by_experiment(), "events": len(read_scores())}
    print(json.dumps({"ok": True, "action": f"kaggle-scores-{args.command}", **payload}, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
