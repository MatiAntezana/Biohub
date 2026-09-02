#!/usr/bin/env python3
"""
infra/hf_sync.py - Sincronización de checkpoints de Biohub con Hugging Face Hub (Privado)
y purga local automática de binarios pesados para conservar espacio en disco.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

infra_dir = Path(__file__).resolve().parent
if str(infra_dir) not in sys.path:
    sys.path.insert(0, str(infra_dir))

from aienv import ensure_aienv
ensure_aienv()

from huggingface_hub import HfApi, hf_hub_download, list_repo_files, upload_file, upload_folder


DEFAULT_REPO_ID = "Matycaba2016/biohub-checkpoints"
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".safetensors"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    cache_path = Path.home() / ".cache/huggingface/token"
    if cache_path.exists():
        return cache_path.read_text().strip()
    legacy_path = Path.home() / ".huggingface/token"
    if legacy_path.exists():
        return legacy_path.read_text().strip()
    raise RuntimeError("HF_TOKEN no encontrado en el entorno ni en ~/.cache/huggingface/token")


def get_repo_id() -> str:
    return os.environ.get("BIOHUB_HF_REPO", DEFAULT_REPO_ID)


def get_hf_api() -> HfApi:
    return HfApi(token=get_token())


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_checkpoint_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS
    ]


def upload_experiment_checkpoints(
    experiment_id: str,
    root: Path | None = None,
    purge: bool = True,
) -> dict[str, Any]:
    """Sube todos los checkpoints de un experimento a HF Hub y purga los locales si se solicita."""
    root = root or repo_root()
    api = get_hf_api()
    repo_id = get_repo_id()

    exp_dir = root / "experiments" / experiment_id
    if not exp_dir.exists():
        # Intentar coincidencia por prefijo
        candidates = list((root / "experiments").glob(f"{experiment_id}*"))
        if candidates and candidates[0].is_dir():
            exp_dir = candidates[0]
            experiment_id = exp_dir.name
        else:
            raise FileNotFoundError(f"No se encontró el directorio para {experiment_id}")

    checkpoints_dir = exp_dir / "checkpoints"
    files = find_checkpoint_files(exp_dir)
    if not files:
        remote_marker = checkpoints_dir / ".hf_remote"
        if remote_marker.exists():
            return {
                "experiment_id": experiment_id,
                "status": "already_synced",
                "hf_repo": repo_id,
                "marker": remote_marker.read_text().strip(),
            }
        return {
            "experiment_id": experiment_id,
            "status": "no_checkpoints",
            "files_uploaded": [],
        }

    uploaded_files_info = []
    total_bytes = 0

    print(f"[{experiment_id}] Subiendo {len(files)} archivos de checkpoint a Hugging Face ({repo_id})...")
    for f in files:
        rel_to_exp = f.relative_to(exp_dir)
        path_in_repo = f"experiments/{experiment_id}/{rel_to_exp}"
        file_size = f.stat().st_size
        total_bytes += file_size
        print(f"  -> Subiendo {f.name} ({file_size / (1024**2):.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=get_token(),
        )
        uploaded_files_info.append({
            "filename": f.name,
            "relative_path": str(rel_to_exp),
            "path_in_repo": path_in_repo,
            "size_bytes": file_size,
        })

    hf_uri = f"hf://{repo_id}/experiments/{experiment_id}/"
    
    # Crear marcador local ligero
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    marker_data = {
        "experiment_id": experiment_id,
        "hf_repo": repo_id,
        "hf_uri": hf_uri,
        "synced_at": utc_now(),
        "total_bytes": total_bytes,
        "files": uploaded_files_info,
    }
    (checkpoints_dir / ".hf_remote").write_text(json.dumps(marker_data, indent=2))

    if purge:
        print(f"[{experiment_id}] Purgando {len(files)} archivos pesados locales ({total_bytes / (1024**2):.2f} MB liberados)...")
        for f in files:
            try:
                f.unlink()
            except OSError as e:
                print(f"  Advertencia: no se pudo eliminar {f}: {e}")

    return {
        "experiment_id": experiment_id,
        "status": "success",
        "hf_uri": hf_uri,
        "total_bytes": total_bytes,
        "files_uploaded": uploaded_files_info,
        "purged": purge,
    }


def upload_champion_weights(root: Path | None = None, purge: bool = True) -> dict[str, Any]:
    """Sube los pesos de champion_weights/ a HF Hub y purga locales."""
    root = root or repo_root()
    api = get_hf_api()
    repo_id = get_repo_id()
    champ_dir = root / "champion_weights"
    if not champ_dir.exists():
        return {"status": "no_dir"}

    files = find_checkpoint_files(champ_dir)
    if not files:
        return {"status": "no_files"}

    print(f"[champion_weights] Subiendo {len(files)} checkpoints a HF ({repo_id})...")
    total_bytes = 0
    uploaded_files_info = []
    for f in files:
        rel = f.relative_to(champ_dir)
        path_in_repo = f"champion_weights/{rel}"
        sz = f.stat().st_size
        total_bytes += sz
        print(f"  -> Subiendo {rel} ({sz / (1024**2):.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=get_token(),
        )
        uploaded_files_info.append({"path": str(rel), "size_bytes": sz})

    marker = {
        "hf_repo": repo_id,
        "synced_at": utc_now(),
        "total_bytes": total_bytes,
        "files": uploaded_files_info,
    }
    (champ_dir / ".hf_remote").write_text(json.dumps(marker, indent=2))

    if purge:
        print(f"[champion_weights] Purgando copias pesadas locales ({total_bytes / (1024**2):.2f} MB)...")
        for f in files:
            try:
                f.unlink()
            except OSError as e:
                print(f"  Advertencia: no se pudo eliminar {f}: {e}")

    return {"status": "success", "total_bytes": total_bytes, "files": uploaded_files_info}


def upload_archive(root: Path | None = None, purge: bool = True) -> dict[str, Any]:
    """Sube los checkpoints en _archive/ a HF Hub y purga locales."""
    root = root or repo_root()
    api = get_hf_api()
    repo_id = get_repo_id()
    archive_dir = root / "_archive"
    if not archive_dir.exists():
        return {"status": "no_dir"}

    files = find_checkpoint_files(archive_dir)
    if not files:
        return {"status": "no_files"}

    print(f"[_archive] Subiendo {len(files)} checkpoints a HF ({repo_id})...")
    total_bytes = 0
    uploaded_files_info = []
    for f in files:
        rel = f.relative_to(archive_dir)
        path_in_repo = f"_archive/{rel}"
        sz = f.stat().st_size
        total_bytes += sz
        print(f"  -> Subiendo {rel} ({sz / (1024**2):.2f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=get_token(),
        )
        uploaded_files_info.append({"path": str(rel), "size_bytes": sz})

    marker = {
        "hf_repo": repo_id,
        "synced_at": utc_now(),
        "total_bytes": total_bytes,
        "files": uploaded_files_info,
    }
    (archive_dir / ".hf_remote").write_text(json.dumps(marker, indent=2))

    if purge:
        print(f"[_archive] Purgando copias pesadas locales ({total_bytes / (1024**2):.2f} MB)...")
        for f in files:
            try:
                f.unlink()
            except OSError as e:
                print(f"  Advertencia: no se pudo eliminar {f}: {e}")

    return {"status": "success", "total_bytes": total_bytes, "files": uploaded_files_info}


def download_experiment_checkpoint(
    experiment_id: str,
    root: Path | None = None,
    filename: str | None = None,
) -> Path:
    """Descarga un checkpoint desde HF Hub si no existe localmente."""
    root = root or repo_root()
    repo_id = get_repo_id()
    exp_dir = root / "experiments" / experiment_id
    if not exp_dir.exists():
        candidates = list((root / "experiments").glob(f"{experiment_id}*"))
        if candidates and candidates[0].is_dir():
            exp_dir = candidates[0]
            experiment_id = exp_dir.name

    # Comprobar si ya existe localmente
    local_files = find_checkpoint_files(exp_dir)
    if local_files:
        if filename:
            for f in local_files:
                if f.name == filename:
                    return f
        return local_files[0]

    # Descargar desde HF
    api = get_hf_api()
    all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    prefix = f"experiments/{experiment_id}/"
    matching = [f for f in all_files if f.startswith(prefix) and any(f.endswith(ext) for ext in CHECKPOINT_EXTENSIONS)]
    
    if not matching:
        raise FileNotFoundError(f"No se encontraron checkpoints remotos para {experiment_id} en {repo_id}")

    target_remote = matching[0]
    if filename:
        for m in matching:
            if m.endswith(f"/{filename}") or m.endswith(filename):
                target_remote = m
                break

    rel_path_in_exp = target_remote[len(prefix):]
    local_dest = exp_dir / rel_path_in_exp
    local_dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{experiment_id}] Descargando {target_remote} desde Hugging Face...")
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=target_remote,
        repo_type="model",
        token=get_token(),
        local_dir=str(root),
    )
    return Path(downloaded)


def sync_all(root: Path | None = None, purge: bool = True) -> None:
    """Sincroniza todos los experimentos, champion_weights y _archive a Hugging Face."""
    root = root or repo_root()
    exp_root = root / "experiments"
    
    print("=== Iniciando sincronización integral de checkpoints con Hugging Face ===")
    print(f"Repositorio destino: {get_repo_id()}")
    
    # 1. Experimentos
    exp_dirs = sorted([d for d in exp_root.iterdir() if d.is_dir() and d.name.startswith("EXP-")])
    print(f"\nDetectados {len(exp_dirs)} directorios de experimentos.")
    
    total_freed = 0
    synced_count = 0

    for d in exp_dirs:
        res = upload_experiment_checkpoints(d.name, root=root, purge=purge)
        if res.get("status") == "success":
            total_freed += res.get("total_bytes", 0)
            synced_count += 1

    # 2. Champion Weights
    champ_res = upload_champion_weights(root=root, purge=purge)
    if champ_res.get("status") == "success":
        total_freed += champ_res.get("total_bytes", 0)

    # 3. Archive
    arch_res = upload_archive(root=root, purge=purge)
    if arch_res.get("status") == "success":
        total_freed += arch_res.get("total_bytes", 0)

    print("\n========================================================")
    print(f"Sincronización completada exitosamente:")
    print(f" - Experimentos sincronizados: {synced_count}")
    print(f" - Espacio liberado en disco: {total_freed / (1024**3):.2f} GB ({total_freed / (1024**2):.1f} MB)")
    print(f" - Repositorio HF: https://huggingface.co/{get_repo_id()}")
    print("========================================================")


def print_status(root: Path | None = None) -> None:
    root = root or repo_root()
    repo_id = get_repo_id()
    local_files = find_checkpoint_files(root)
    local_bytes = sum(f.stat().st_size for f in local_files)
    
    print("=== Estado de Almacenamiento Biohub ===")
    print(f"Hugging Face Repo: {repo_id}")
    print(f"Checkpoints pesados locales: {len(local_files)} archivos ({local_bytes / (1024**2):.2f} MB)")
    
    # Listar marcadores .hf_remote
    markers = list(root.rglob(".hf_remote"))
    print(f"Directorios sincronizados (.hf_remote): {len(markers)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Biohub Hugging Face Checkpoints Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # upload
    upload_parser = subparsers.add_parser("upload", help="Subir checkpoints a HF")
    upload_parser.add_argument("--exp", type=str, help="ID o nombre del experimento (ej: EXP-030B)")
    upload_parser.add_argument("--all", action="store_true", help="Subir todos los experimentos y weights")
    upload_parser.add_argument("--no-purge", action="store_true", help="No eliminar archivos locales tras subir")

    # download
    download_parser = subparsers.add_parser("download", help="Descargar checkpoints desde HF")
    download_parser.add_argument("--exp", type=str, required=True, help="ID del experimento")
    download_parser.add_argument("--file", type=str, help="Nombre del archivo a descargar")

    # status
    subparsers.add_parser("status", help="Mostrar estado local y remoto")

    args = parser.parse_args()

    if args.command == "upload":
        purge = not args.no_purge
        if args.all:
            sync_all(purge=purge)
        elif args.exp:
            upload_experiment_checkpoints(args.exp, purge=purge)
        else:
            print("Error: Debe especificar --exp EXP-XXX o --all")
            sys.exit(1)
    elif args.command == "download":
        dest = download_experiment_checkpoint(args.exp, filename=args.file)
        print(f"Descargado exitosamente en: {dest}")
    elif args.command == "status":
        print_status()


if __name__ == "__main__":
    main()
