# EXP-032 — carpeta operativa

Versión mejorada full-fit del ensamble temporal dual-seed (EXP-030B). Incorpora una jerarquía de canales más amplia ([48, 96, 192], 48 out channels) y pesos de pérdida de detección calibrados (1.5 / 0.03) para mayor sensibilidad y resolución en mitosis complejas.

- Launcher canónico: `train.py`.
- Manifiesto ejecutable: `experiment.json`.
- Registro técnico: `../../Librería-Registros/Experimentos/EXP-032_DualSeed_Temporal_Ensemble_Enhanced.md`.
- Recibos por corrida: `artifacts/<run_id>/`.
- Checkpoints resumibles: `checkpoints/<run_id>/<role>/training_checkpoint.pth`.
- Directorios reservados: `logs/`, `code_snapshot/`, `checkpoints/`, `artifacts/`, `validation/`, `kaggle/`.
