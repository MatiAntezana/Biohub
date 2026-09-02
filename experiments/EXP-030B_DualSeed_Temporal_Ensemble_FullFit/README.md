# EXP-030B — carpeta operativa

Nueva versión full-fit de EXP-030. Esta carpeta concentra logs, snapshots, checkpoints copiados, artefactos, validación posterior y paquete Kaggle sin mezclarlos con EXP-030.

- Launcher canónico: `train.py`.
- Manifiesto ejecutable: `experiment.json`.
- Registro técnico: `../../Librería-Registros/Experimentos/EXP-030B_DualSeed_Temporal_Ensemble_FullFit.md`.
- Recibos por corrida: `artifacts/<run_id>/`.
- Checkpoints resumibles: `checkpoints/<run_id>/<role>/training_checkpoint.pth`.
- Directorios reservados: `logs/`, `code_snapshot/`, `checkpoints/`, `artifacts/`, `validation/`, `kaggle/`.

El lanzamiento se hace exclusivamente con `python infra/experimentctl.py run EXP-030B --node A1556`. El controlador crea el snapshot y sus SHA-256. Aunque EXP-031 termine, EXP-030B solo cambia a `READY`: nunca se autolanza.
