#!/usr/bin/env python3
"""Deterministic state, registry and validation primitives for Biohub.

This module intentionally uses only the Python standard library so the local
controller and every remote worker share the same behavior without installing
or resolving orchestration dependencies.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA_VERSION = 1
EXPERIMENT_RE = re.compile(r"^EXP-\d{3}[A-Z]?$")
SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]
TERMINAL_SUCCESS = {"SUCCEEDED", "DOC_PENDING", "CLOSED"}
TERMINAL_STATES = TERMINAL_SUCCESS | {"FAILED", "ORPHANED", "IMPORTED"}
QUEUE_STATES = {"PLANNED", "QUEUED", "READY", "RUNNING", "DOC_PENDING", "CLOSED", "FAILED", "ORPHANED"}
EVENT_STATES = QUEUE_STATES | {"SUCCEEDED", "IMPORTED"}
ALLOWED_TRANSITIONS = {
    "PLANNED": {"QUEUED", "READY", "FAILED"},
    "QUEUED": {"READY", "FAILED"},
    "READY": {"RUNNING", "FAILED", "ORPHANED"},
    "RUNNING": {"SUCCEEDED", "FAILED", "ORPHANED"},
    "SUCCEEDED": {"DOC_PENDING", "CLOSED"},
    "DOC_PENDING": {"CLOSED"},
    "FAILED": {"QUEUED", "READY"},
    "ORPHANED": {"QUEUED", "READY", "FAILED"},
    "IMPORTED": set(),
    "CLOSED": set(),
}


class ExperimentError(RuntimeError):
    """An operator-facing contract or state error."""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    """Replace a text file atomically and durably on the same filesystem."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExperimentError(f"JSON inválido en {path}: {error}") from error


@contextlib.contextmanager
def exclusive_lock(path: Path, *, nonblocking: bool = False) -> Iterator[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as error:
            raise ExperimentError(f"lock ocupado: {path}") from error
        try:
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def registry_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "experiments" / "registry.jsonl"


def read_events(root: Path | None = None) -> list[dict[str, Any]]:
    path = registry_path(root)
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ExperimentError(f"ledger corrupto en línea {line_number}: {error}") from error
        events.append(event)
    return events


def verify_ledger(root: Path | None = None) -> list[str]:
    errors: list[str] = []
    previous_hash: str | None = None
    seen_ids: set[str] = set()
    for index, event in enumerate(read_events(root), 1):
        event_id = event.get("event_id")
        if not event_id or event_id in seen_ids:
            errors.append(f"evento {index}: event_id ausente o duplicado")
        seen_ids.add(str(event_id))
        if event.get("previous_event_hash") != previous_hash:
            errors.append(f"evento {index}: cadena previous_event_hash inválida")
        stored_hash = event.get("event_hash")
        unhashed = {key: value for key, value in event.items() if key != "event_hash"}
        computed = sha256_bytes(canonical_json(unhashed))
        if stored_hash != computed:
            errors.append(f"evento {index}: event_hash inválido")
        previous_hash = stored_hash
    return errors


def append_event(
    *,
    experiment_id: str,
    run_id: str,
    event_type: str,
    status: str,
    root: Path | None = None,
    idempotency_key: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    """Append one hash-chained event under an exclusive file lock."""

    if not EXPERIMENT_RE.fullmatch(experiment_id):
        raise ExperimentError(f"experiment_id inválido: {experiment_id}")
    if status not in EVENT_STATES:
        raise ExperimentError(f"estado de evento inválido: {status}")
    root = root or repo_root()
    path = registry_path(root)
    lock_path = path.with_suffix(".lock")
    identity = idempotency_key or str(uuid.uuid4())
    event_id = sha256_bytes(f"{experiment_id}|{run_id}|{event_type}|{identity}".encode())[:24]

    with exclusive_lock(lock_path):
        events = read_events(root)
        existing = next((event for event in events if event.get("event_id") == event_id), None)
        if existing is not None:
            return existing
        previous_run_event = next((event for event in reversed(events) if event.get("run_id") == run_id), None)
        if previous_run_event is not None:
            previous_status = previous_run_event.get("status")
            if status != previous_status and status not in ALLOWED_TRANSITIONS.get(previous_status, set()):
                raise ExperimentError(f"transición inválida para {run_id}: {previous_status} -> {status}")
        event: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_id": event_id,
            "timestamp_utc": utc_now(),
            "experiment_id": experiment_id,
            "run_id": run_id,
            "event_type": event_type,
            "status": status,
            "previous_event_hash": events[-1].get("event_hash") if events else None,
            **details,
        }
        event["event_hash"] = sha256_bytes(canonical_json(event))
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    return event


def latest_events_by_run(root: Path | None = None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in read_events(root):
        latest[event["run_id"]] = event
    return latest


def latest_status_for_experiment(experiment_id: str, root: Path | None = None) -> str | None:
    status = None
    for event in read_events(root):
        if event.get("experiment_id") == experiment_id:
            status = event.get("status")
    return status


def experiment_directories(root: Path | None = None) -> dict[str, Path]:
    root = root or repo_root()
    found: dict[str, Path] = {}
    for directory in sorted((root / "experiments").glob("EXP-*")):
        if not directory.is_dir():
            continue
        match = re.match(r"^(EXP-\d{3}[A-Z]?)", directory.name)
        if match:
            found[match.group(1)] = directory
    return found


def experiment_directory(experiment_id: str, root: Path | None = None) -> Path:
    try:
        return experiment_directories(root)[experiment_id]
    except KeyError as error:
        raise ExperimentError(f"no existe carpeta para {experiment_id}") from error


def load_manifest(experiment_id: str, root: Path | None = None) -> tuple[Path, dict[str, Any]]:
    directory = experiment_directory(experiment_id, root)
    path = directory / "experiment.json"
    if not path.is_file():
        raise ExperimentError(f"falta manifiesto: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ExperimentError(f"manifiesto debe ser un objeto: {path}")
    return path, payload


def load_nodes(root: Path | None = None) -> dict[str, dict[str, Any]]:
    root = root or repo_root()
    payload = read_json(root / ".agents" / "nodes.json", {})
    return payload.get("nodes", {}) if isinstance(payload, dict) else {}


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_manifest(experiment_id: str, root: Path | None = None) -> list[str]:
    root = root or repo_root()
    errors: list[str] = []
    try:
        path, manifest = load_manifest(experiment_id, root)
    except ExperimentError as error:
        return [str(error)]
    directory = path.parent
    required = {
        "schema_version",
        "experiment_id",
        "entrypoint",
        "arguments",
        "allowed_nodes",
        "default_node",
        "execution_policy",
        "dependencies",
        "data_policy",
        "artifact_contract",
        "documentation",
        "kaggle",
    }
    missing = sorted(required.difference(manifest))
    if missing:
        errors.append(f"campos requeridos ausentes: {', '.join(missing)}")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version debe ser {SCHEMA_VERSION}")
    if manifest.get("experiment_id") != experiment_id:
        errors.append("experiment_id no coincide con la carpeta")

    entrypoint = directory / str(manifest.get("entrypoint", ""))
    if not entrypoint.is_file() or not is_within(entrypoint, directory):
        errors.append("entrypoint inexistente o fuera de la carpeta experimental")
    nodes = load_nodes(root)
    allowed_nodes = manifest.get("allowed_nodes", [])
    if not isinstance(allowed_nodes, list) or not allowed_nodes:
        errors.append("allowed_nodes debe contener al menos un nodo")
    else:
        unknown = sorted(set(allowed_nodes).difference(nodes))
        if unknown:
            errors.append(f"nodos no registrados: {', '.join(unknown)}")
    if manifest.get("default_node") not in allowed_nodes:
        errors.append("default_node debe pertenecer a allowed_nodes")
    execution_policy = manifest.get("execution_policy", {})
    if execution_policy.get("node_selection") not in {"max_free_vram", "auto", "explicit"}:
        errors.append("execution_policy.node_selection inválido")
    minimum_vram = execution_policy.get("min_free_vram_mib")
    if not isinstance(minimum_vram, int) or minimum_vram < 0:
        errors.append("execution_policy.min_free_vram_mib debe ser entero >= 0")
    if not isinstance(execution_policy.get("auto_launch"), bool):
        errors.append("execution_policy.auto_launch debe ser booleano")
    max_parallel = execution_policy.get("max_parallel_runs")
    if not isinstance(max_parallel, int) or max_parallel < 1:
        errors.append("execution_policy.max_parallel_runs debe ser entero positivo")
    if not isinstance(manifest.get("arguments"), list) or not all(
        isinstance(item, str) for item in manifest.get("arguments", [])
    ):
        errors.append("arguments debe ser una lista de strings")
    dependencies = manifest.get("code_dependencies", [])
    if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
        errors.append("code_dependencies debe ser una lista de rutas strings")
    else:
        for relative in dependencies:
            source = (root / relative).resolve()
            if not is_within(source, root) or not source.is_file():
                errors.append(f"code_dependency inexistente o fuera del repositorio: {relative}")
    if not isinstance(manifest.get("dependencies"), list):
        errors.append("dependencies debe ser una lista")

    for name in ("artifacts", "logs", "checkpoints", "code_snapshot", "validation", "kaggle"):
        target = directory / name
        if not target.is_dir():
            errors.append(f"falta subcarpeta obligatoria: {name}/")

    serialized = json.dumps(manifest, sort_keys=True)
    entry_text = entrypoint.read_text(encoding="utf-8") if entrypoint.is_file() else ""
    forbidden = ("experiment_artifacts", "official_starter_kit/weights", "eval_tmp")
    for token in forbidden:
        if token in serialized or token in entry_text:
            errors.append(f"ruta legacy prohibida detectada: {token}")

    kaggle = manifest.get("kaggle", {})
    if kaggle.get("enabled"):
        if kaggle.get("training_allowed") is not False:
            errors.append("Kaggle debe declarar training_allowed=false")
        if kaggle.get("accelerator") != "NvidiaTeslaT4":
            errors.append("Kaggle GPU debe usar NvidiaTeslaT4")
        if kaggle.get("submission_columns") != SUBMISSION_COLUMNS:
            errors.append("contrato CSV Kaggle no coincide con las 10 columnas oficiales")
        if kaggle.get("test_only") is not True:
            errors.append("Kaggle debe declarar test_only=true")
        if kaggle.get("deployment_ready"):
            for field in ("competition", "kernel_slug", "output_file"):
                if not kaggle.get(field):
                    errors.append(f"kaggle.{field} es obligatorio cuando deployment_ready=true")
            if kaggle.get("auto_submit") is not True:
                errors.append("Kaggle deployment_ready debe declarar auto_submit=true")

    artifact_contract = manifest.get("artifact_contract", {})
    if not artifact_contract.get("terminal_receipt"):
        errors.append("artifact_contract.terminal_receipt es obligatorio")
    if not artifact_contract.get("checkpoints"):
        errors.append("artifact_contract.checkpoints es obligatorio")
    minimum = artifact_contract.get("minimum_checkpoints")
    maximum = artifact_contract.get("maximum_checkpoints")
    if not isinstance(minimum, int) or minimum < 1:
        errors.append("artifact_contract.minimum_checkpoints debe ser entero positivo")
    if not isinstance(maximum, int) or not isinstance(minimum, int) or maximum < minimum:
        errors.append("artifact_contract.maximum_checkpoints debe ser >= minimum_checkpoints")
    return errors


def validate_kaggle_assets(experiment_id: str, root: Path | None = None) -> list[str]:
    """Statically validate an existing Kaggle package without executing it."""

    root = root or repo_root()
    directory = experiment_directory(experiment_id, root)
    kaggle_dir = directory / "kaggle"
    metadata_path = kaggle_dir / "kernel-metadata.json"
    notebook_path = kaggle_dir / "submission_notebook.ipynb"
    inference_path = kaggle_dir / "inference.py"
    if not any(path.exists() for path in (metadata_path, notebook_path, inference_path)):
        return []
    errors: list[str] = []
    if not metadata_path.is_file():
        errors.append("paquete Kaggle incompleto: falta kernel-metadata.json")
    else:
        metadata = read_json(metadata_path, {})
        if str(metadata.get("enable_gpu")).lower() != "true":
            errors.append("kernel Kaggle debe habilitar GPU T4")
        if metadata.get("machine_shape") != "NvidiaTeslaT4":
            errors.append("kernel Kaggle no declara NvidiaTeslaT4")
    if not notebook_path.is_file():
        errors.append("paquete Kaggle incompleto: falta submission_notebook.ipynb")
    else:
        notebook = read_json(notebook_path, {})
        kernelspec = notebook.get("metadata", {}).get("kernelspec", {})
        if kernelspec.get("name") != "python3":
            errors.append("notebook Kaggle no declara kernelspec python3")
    if not inference_path.is_file():
        errors.append("paquete Kaggle incompleto: falta inference.py")
    else:
        source = inference_path.read_text(encoding="utf-8")
        if not all(column in source for column in SUBMISSION_COLUMNS):
            errors.append("inference.py no contiene el contrato de 10 columnas")
        forbidden_training = ("optimizer.step(", ".backward(", "loss.backward(")
        if any(token in source for token in forbidden_training):
            errors.append("Kaggle inference.py contiene operaciones de entrenamiento")
    return errors


def queue_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / ".state" / "queue.json"


def load_queue(root: Path | None = None) -> dict[str, Any]:
    payload = read_json(queue_path(root), {"schema_version": SCHEMA_VERSION, "items": []})
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ExperimentError(".state/queue.json inválido")
    return payload


def save_queue(payload: dict[str, Any], root: Path | None = None) -> None:
    atomic_write_json(queue_path(root), payload)


def dependencies_satisfied(item: dict[str, Any], root: Path | None = None) -> bool:
    events = read_events(root)
    for dependency in item.get("dependencies", []):
        if not any(
            event.get("experiment_id") == dependency and event.get("status") in TERMINAL_SUCCESS
            for event in events
        ):
            return False
    return True


def _refresh_queue_payload(queue: dict[str, Any], root: Path) -> dict[str, Any]:
    for item in queue["items"]:
        try:
            _, manifest = load_manifest(item["experiment_id"], root)
            policy = manifest.get("execution_policy", {})
            item["auto_launch"] = policy.get("auto_launch", False)
            item["requested_node"] = policy.get("node_selection", item.get("requested_node", manifest["default_node"]))
            item["allowed_nodes"] = manifest["allowed_nodes"]
        except ExperimentError:
            # Historical queue entries remain readable even if their source was archived.
            item.setdefault("auto_launch", False)
        if item.get("status") in {"PLANNED", "QUEUED"} and dependencies_satisfied(item, root):
            item["status"] = "READY"
            item["ready_at"] = item.get("ready_at") or utc_now()
        if item.get("status") not in QUEUE_STATES:
            raise ExperimentError(f"estado de cola inválido para {item.get('experiment_id')}")
    queue["updated_at"] = utc_now()
    return queue


def refresh_queue(root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    with exclusive_lock(queue_path(root).with_suffix(".lock")):
        queue = _refresh_queue_payload(load_queue(root), root)
        save_queue(queue, root)
    return queue


def queue_experiment(experiment_id: str, root: Path | None = None) -> dict[str, Any]:
    root = root or repo_root()
    errors = validate_manifest(experiment_id, root)
    if errors:
        raise ExperimentError("; ".join(errors))
    _, manifest = load_manifest(experiment_id, root)
    with exclusive_lock(queue_path(root).with_suffix(".lock")):
        queue = load_queue(root)
        item = next((entry for entry in queue["items"] if entry.get("experiment_id") == experiment_id), None)
        if item is None:
            item = {
                "experiment_id": experiment_id,
                "status": "QUEUED",
                "priority": manifest.get("priority", 100),
                "dependencies": manifest.get("dependencies", []),
                "allowed_nodes": manifest["allowed_nodes"],
                "requested_node": manifest.get("execution_policy", {}).get("node_selection", manifest["default_node"]),
                "auto_launch": manifest.get("execution_policy", {}).get("auto_launch", False),
                "queued_at": utc_now(),
            }
            queue["items"].append(item)
        elif item.get("status") in {"RUNNING", "DOC_PENDING"}:
            raise ExperimentError(f"{experiment_id} ya está {item['status']}")
        else:
            item["status"] = "QUEUED"
            item["queued_at"] = utc_now()
        queue = _refresh_queue_payload(queue, root)
        save_queue(queue, root)
        item = next(entry for entry in queue["items"] if entry["experiment_id"] == experiment_id)
    append_event(
        experiment_id=experiment_id,
        run_id=f"queue-{experiment_id}",
        event_type="QUEUED",
        status=item["status"],
        root=root,
        idempotency_key=f"queue:{experiment_id}:{item['queued_at']}",
        dependencies=item["dependencies"],
        requested_node=item["requested_node"],
    )
    render_views(root)
    return item


def set_queue_status(experiment_id: str, status: str, root: Path | None = None, **extra: Any) -> dict[str, Any]:
    if status not in QUEUE_STATES:
        raise ExperimentError(f"estado inválido: {status}")
    root = root or repo_root()
    with exclusive_lock(queue_path(root).with_suffix(".lock")):
        queue = load_queue(root)
        item = next((entry for entry in queue["items"] if entry.get("experiment_id") == experiment_id), None)
        if item is None:
            raise ExperimentError(f"{experiment_id} no está en la cola")
        item.update({"status": status, "updated_at": utc_now(), **extra})
        queue = _refresh_queue_payload(queue, root)
        save_queue(queue, root)
    return item


def _markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, sort_keys=True, ensure_ascii=False)}`"
    return str(value).replace("|", "\\|")


def render_views(root: Path | None = None) -> None:
    root = root or repo_root()
    queue = load_queue(root)
    events = read_events(root)
    latest = latest_events_by_run(root)

    queue_lines = [
        "# Cola de experimentos",
        "",
        "> Vista generada por `experimentctl`; no editar manualmente.",
        "",
        "| Prioridad | Experimento | Estado | Dependencias | Nodo |",
        "|---:|---|---|---|---|",
    ]
    for item in sorted(queue["items"], key=lambda row: (row.get("priority", 100), row["experiment_id"])):
        queue_lines.append(
            f"| {item.get('priority', 100)} | {item['experiment_id']} | **{item['status']}** | "
            f"{', '.join(item.get('dependencies', [])) or '—'} | {item.get('requested_node', '—')} |"
        )
    queue_lines += [
        "",
        "Un elemento `READY` solo se autolanza si su manifiesto declara `execution_policy.auto_launch=true`; ",
        "los demás requieren `experimentctl run` explícito.",
        "",
    ]
    atomic_write_text(root / ".state" / "QUEUE.md", "\n".join(queue_lines))

    state_lines = [
        "# Estado operativo de Biohub",
        "",
        "> Vista generada. Fuente canónica: `experiments/registry.jsonl`.",
        "",
        f"Actualizado: `{utc_now()}`",
        "",
        "## Resumen",
        "",
        f"- Eventos registrados: **{len(events)}**",
        f"- Corridas registradas: **{len(latest)}**",
        f"- Trabajos listos: **{sum(i.get('status') == 'READY' for i in queue['items'])}**",
        f"- Trabajos READY con autolanzamiento opt-in: **{sum(i.get('status') == 'READY' and i.get('auto_launch') is True for i in queue['items'])}**",
        f"- Documentación pendiente: **{sum(e.get('status') == 'DOC_PENDING' for e in latest.values())}**",
        "",
        "## Últimas corridas",
        "",
        "| Corrida | Experimento | Estado | Fecha | Nodo |",
        "|---|---|---|---|---|",
    ]
    for event in list(latest.values())[-15:][::-1]:
        state_lines.append(
            f"| {event['run_id']} | {event['experiment_id']} | **{event['status']}** | "
            f"{event.get('timestamp_utc', '—')} | {event.get('node', '—')} |"
        )
    state_lines.append("")
    atomic_write_text(root / ".state" / "STATE.md", "\n".join(state_lines))

    result_lines = [
        "# Resultados de experimentos",
        "",
        "> Vista factual generada desde el ledger append-only. No reemplaza las conclusiones narrativas.",
        "",
        "| Corrida | Experimento | Estado | Validación | Métricas | Reporte |",
        "|---|---|---|---|---|---|",
    ]
    for event in latest.values():
        if event.get("status") not in TERMINAL_STATES:
            continue
        result_lines.append(
            f"| {event['run_id']} | {event['experiment_id']} | **{event['status']}** | "
            f"{_markdown_value(event.get('validation', '—'))} | {_markdown_value(event.get('metrics', {}))} | "
            f"{_markdown_value(event.get('report', '—'))} |"
        )
    result_lines.append("")
    atomic_write_text(root / "experiments" / "RESULTS.md", "\n".join(result_lines))


def write_factual_report(run_id: str, root: Path | None = None) -> Path:
    root = root or repo_root()
    events = [event for event in read_events(root) if event.get("run_id") == run_id]
    if not events:
        raise ExperimentError(f"run_id desconocido: {run_id}")
    last = events[-1]
    directory = experiment_directory(last["experiment_id"], root)
    report_dir = directory / "validation" / run_id
    report = report_dir / "REPORT.md"
    lines = [
        f"# Reporte factual — {run_id}",
        "",
        "> Generado automáticamente. La interpretación científica debe documentarse aparte.",
        "",
        f"- Experimento: `{last['experiment_id']}`",
        f"- Estado: `{last['status']}`",
        f"- Nodo: `{last.get('node', '—')}`",
        f"- Entorno: `{last.get('conda_env', '—')}`",
        f"- Fecha: `{last.get('timestamp_utc', '—')}`",
        f"- Validación: `{last.get('validation', '—')}`",
        f"- Config SHA-256: `{last.get('config_sha256', '—')}`",
        "",
        "## Métricas",
        "",
        "```json",
        json.dumps(last.get("metrics", {}), indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
        "## Artefactos",
        "",
        "```json",
        json.dumps(last.get("artifacts", {}), indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "",
    ]
    atomic_write_text(report, "\n".join(lines))
    return report


def run_id_for(experiment_id: str, manifest: dict[str, Any]) -> tuple[str, str]:
    config_hash = sha256_bytes(canonical_json(manifest))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_id}_{stamp}_{config_hash[:8]}", config_hash


def validate_document_link(value: str, root: Path | None = None) -> str:
    root = root or repo_root()
    if value.startswith(("http://", "https://")):
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    if not candidate.is_file():
        raise ExperimentError(f"documentación narrativa inexistente: {candidate}")
    return str(candidate.resolve().relative_to(root.resolve()))
