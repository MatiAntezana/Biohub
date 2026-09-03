# Reporte factual — EXP-053_20260903T162626Z_af0cefb2

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-053`
- Estado: `DOC_PENDING`
- Nodo: `A1555`
- Entorno: `train-55`
- Fecha: `2026-09-03T22:21:45.081112Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `af0cefb2492efe8d94595743ee5b7d3d6f33b6a405b1de29a837906bc9125e69`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.018877724308520555,
    "edge_loss": 0.0005387665062504917,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 922.8823613060085,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.022361534575621286,
    "edge_loss": 0.00044602429734110654,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 836.2001553840018,
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
      "bytes": 25079666,
      "path": "experiments/EXP-053_DeepGraph_HighRes_Regularized_Cosine/checkpoints/EXP-053_20260903T162626Z_af0cefb2/primary/training_checkpoint.pth",
      "sha256": "a11c96e44fa38a4a696475fca18e7e5d4e1f43d2678d27d911aaea19ab4d72ce"
    },
    {
      "bytes": 25079730,
      "path": "experiments/EXP-053_DeepGraph_HighRes_Regularized_Cosine/checkpoints/EXP-053_20260903T162626Z_af0cefb2/secondary/training_checkpoint.pth",
      "sha256": "fc8b954db093b05d495fac57696f13489990888074efaf31012f26892cafcebd"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-053_DeepGraph_HighRes_Regularized_Cosine/artifacts/EXP-053_20260903T162626Z_af0cefb2/TRAINING_COMPLETE.json",
    "sha256": "1433ef68350837fadfae498ee41741c9c874b3b377fb26bfb38d6549234a491a"
  }
}
```
