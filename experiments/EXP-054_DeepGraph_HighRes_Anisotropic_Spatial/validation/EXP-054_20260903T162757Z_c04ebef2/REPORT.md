# Reporte factual — EXP-054_20260903T162757Z_c04ebef2

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-054`
- Estado: `DOC_PENDING`
- Nodo: `A1556`
- Entorno: `train-56`
- Fecha: `2026-09-03T20:30:59.308955Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `c04ebef221b06a5a016d6f51629ca6d20f7d999d05c99ed44f97452f1970b93f`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.018329686203350622,
    "edge_loss": 0.00034721600079198347,
    "epoch": 49,
    "learning_rate": 0.0001,
    "selection_score": null,
    "train_seconds": 777.6225683229932,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01647594530725231,
    "edge_loss": 0.00036707987735007917,
    "epoch": 49,
    "learning_rate": 0.0001,
    "selection_score": null,
    "train_seconds": 887.0788790190018,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  }
}
```

## Artefactos

```json
{
  "checkpoints": [
    {
      "bytes": 25079895,
      "path": "experiments/EXP-054_DeepGraph_HighRes_Anisotropic_Spatial/checkpoints/EXP-054_20260903T162757Z_c04ebef2/primary/training_checkpoint.pth",
      "sha256": "3a2af5761d8a5749a926ac048392985518b0a3bf0123249b4bbb6a9ec9499b11"
    },
    {
      "bytes": 25079959,
      "path": "experiments/EXP-054_DeepGraph_HighRes_Anisotropic_Spatial/checkpoints/EXP-054_20260903T162757Z_c04ebef2/secondary/training_checkpoint.pth",
      "sha256": "dfb73bc6a0996e4734cb99f090c9aaf14668ad24b2dbfa938be69ef6616accff"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-054_DeepGraph_HighRes_Anisotropic_Spatial/artifacts/EXP-054_20260903T162757Z_c04ebef2/TRAINING_COMPLETE.json",
    "sha256": "c164724ce97fce64a75615ca3d99d646957010eeb56ad76ba3e422c274695f3a"
  }
}
```
