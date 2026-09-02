# EXP-031 — carpeta operativa

La revisión vigente es full-fit (`100 % train / 0 % validación`). El snapshot anterior que se encuentra directamente en `code_snapshot/` corresponde al intento embryo-disjoint detenido y se conserva sólo como historial.

- Launcher canónico vigente: `train.py`.
- Manifiesto ejecutable: `experiment.json`.
- Registro técnico: `../../Librería-Registros/Experimentos/EXP-031_Detector_First_ILP_Preservado.md`.
- Manifiesto full-fit: `code_snapshot/fullfit-2026-08-27/manifest_sha256.json`.
- Recibos por corrida: `artifacts/<run_id>/`.
- Checkpoint resumible: `checkpoints/<run_id>/model/training_checkpoint.pth`.

El snapshot `fullfit-2026-08-27` se conserva como evidencia legacy. Preparar con `python infra/experimentctl.py run EXP-031 --node A1556 --dry-run` y lanzar solo mediante el mismo comando sin `--dry-run` después de una autorización explícita.
