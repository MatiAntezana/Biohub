# Reporte factual — EXP-055_20260903T001305Z_d1f9ffe0

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-055`
- Estado: `SUCCEEDED`
- Nodo: `A1553`
- Entorno: `train-53`
- Fecha: `2026-09-03T14:53:26.967074Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `d1f9ffe030cea565b33a771e5f6c1fb9eaa9a366b3d4188fc5e5969f858c9665`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01700184487675627,
    "edge_loss": 0.0002759767247621312,
    "epoch": 49,
    "learning_rate": 0.0001,
    "selection_score": null,
    "train_seconds": 1233.391454334007,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01669064625756194,
    "edge_loss": 0.0002769144099368835,
    "epoch": 49,
    "learning_rate": 0.0001,
    "selection_score": null,
    "train_seconds": 1309.8339375899814,
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
      "bytes": 33648151,
      "path": "experiments/EXP-055_DeepGraph_MultiScale_Temporal_Window/checkpoints/EXP-055_20260903T001305Z_d1f9ffe0/primary/training_checkpoint.pth",
      "sha256": "6c39352b410f8079ab1bafb07e2519357c4fd30d5fd243b1ada3ff7e15405990"
    },
    {
      "bytes": 33649367,
      "path": "experiments/EXP-055_DeepGraph_MultiScale_Temporal_Window/checkpoints/EXP-055_20260903T001305Z_d1f9ffe0/secondary/training_checkpoint.pth",
      "sha256": "69211473439b8c4bf7d95984856c4327eaca459141a5afa7b069eca5d3b2a8a0"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-055_DeepGraph_MultiScale_Temporal_Window/artifacts/EXP-055_20260903T001305Z_d1f9ffe0/TRAINING_COMPLETE.json",
    "sha256": "7fb9c41fb98f6f839642c375254dfaf2f7b566982f24aec4d84f97888ad791ac"
  }
}
```
