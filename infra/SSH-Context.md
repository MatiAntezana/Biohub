# SSH-Context.md — Acceso GPU remota vía UCBV Gateway (A1552, A1555, A1557, A1558, A1559, A1560, A1563, A1688)

Última actualización y auditoría de acceso: **2026-09-01** (Migración a arquitectura segura UCBV).

---

## 1. Arquitectura de Conexión UCBV

El acceso a todas las máquinas remotas del cluster se realiza exclusivamente a través del **Gateway UCBV local del Mac** (`LaunchAgent com.ucbv.access-gateway` con proxy SOCKS5 en `127.0.0.1:1055` e identidad `tag:ucbv-access`).

### Cómo conectarse desde el Mac:
Se debe utilizar siempre el **alias de host en minúsculas** definido en `~/.ssh/config`. OpenSSH enruta automáticamente a través del proxy SOCKS5 con la clave `~/.ssh/id_ed25519`:

```bash
ssh a1552
ssh a1555
ssh a1557
ssh a1558
ssh a1559
ssh a1560
ssh a1563
ssh a1688
```

Para transferencias con `rsync`:
```bash
rsync -az -e ssh LOCAL a1555:Documents/DESTINO
rsync -az --partial -e ssh a1555:Documents/ORIGEN LOCAL/
```

> [!WARNING]
> La conexión directa por IP Tailscale (`ssh -p 2222 alumno1@100.X.X.X`) está **bloqueada por ACL del tailnet**. Solo se permite el tráfico `tag:ucbv-access -> tag:ucbv-remote:tcp:2222`.

---

## 2. Nodos Habilitados en Biohub

| Nodo | Alias SSH | GPU | Entorno Conda | Directorio Remoto | Estado en Biohub |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A1552** | `ssh a1552` | NVIDIA RTX 4090 (24 GB) | `train-52` | `~/Documents/Biohub` | 🟢 `ONLINE` / `training_ready=true` |
| **A1555** | `ssh a1555` | NVIDIA RTX 4090 (24 GB) | **`train-55`** | `~/Documents/Biohub` | 🟢 **`ONLINE` / `training_ready=true`** |
| **A1557** | `ssh a1557` | NVIDIA RTX 4090 (24 GB) | `train-57` | `~/Documents/Biohub` | 🟢 `ONLINE` / `training_ready=true` |
| **A1560** | `ssh a1560` | NVIDIA RTX 4090 (24 GB) | `train-60` | `~/Documents/Biohub` | 🟢 `ONLINE` / `training_ready=true` |
| **A1558** | `ssh a1558` | NVIDIA RTX 4090 (24 GB) | `train-58` | `~/Documents/Biohub` | 🟢 `ONLINE` / `training_ready=true` |
| **A1559** | `ssh a1559` | NVIDIA RTX 4090 (24 GB) | `train-59` | `~/Documents/Biohub` | 🟢 `ONLINE` / `training_ready=true` |
| **A1563** | `ssh a1563` | NVIDIA RTX 4090 (24 GB) | `train-63` | `~/Documents/Biohub` | 🟢 `ONLINE` / `training_ready=true` |
| **A1688** | `ssh a1688` | NVIDIA RTX 4090 (24 GB) | `train-88` | `~/Documents/Biohub` | 🟢 `ONLINE` / `training_ready=true` |
| **A1556** | `ssh a1556` | NVIDIA RTX 4090 (24 GB) | `train-56` | `~/Documents/Biohub` | 🔒 `RESERVED` / `training_ready=false` |

---

## 3. Detalle de Nodos Específicos

### A1555 (`ssh a1555`)
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM).
- **Entorno**: `train-55` (`~/miniconda3/envs/train-55/bin/python`, Python 3.11, PyTorch 2.5.1 + CUDA 12.1, Zarr, Numcodecs, Scipy).
- **Verificación**: Totalmente migrada bajo UCBV, validada con `BIOHUB_REMOTE_READY`.

### A1552 (`ssh a1552`)
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM).
- **Entorno**: `train-52` (Python 3.11, PyTorch 2.13.0 + cu130).

### A1557 (`ssh a1557`)
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM).
- **Entorno**: `train-57` (Python 3.11, PyTorch 2.13.0 + cu130).

### A1556 (`ssh a1556`)
- **GPU**: NVIDIA GeForce RTX 4090 (24 GB VRAM).
- **Estado**: Reservada para otros procesos.

---

## 4. Diagnóstico Rápido

Verificar estado y GPU de cualquier nodo:
```bash
ssh a1555 "hostname && nvidia-smi -L"
```

Probar entorno Python remoto:
```bash
ssh a1555 "source ~/miniconda3/etc/profile.d/conda.sh && conda activate train-55 && python -c 'import torch; print(torch.cuda.is_available())'"
```
