# Reporte factual — EXP-051_20260831T150521Z_477a84b9

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-051`
- Estado: `SUCCEEDED`
- Nodo: `A1557`
- Entorno: `train-57`
- Fecha: `2026-09-01T12:41:10.119645Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `477a84b9efcf87cc76bfd6ddd715c9610dfe0e02d81c9e0d7b7bfcdab4525c69`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01771104339354982,
    "edge_loss": 0.0003533268082016245,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 780.7742544599969,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01871057736687362,
    "edge_loss": 0.0003240218590023384,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 780.223652353001,
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
      "bytes": 25078551,
      "path": "experiments/EXP-051_DeepGraph_HighRes_Temporal/checkpoints/EXP-051_20260831T150521Z_477a84b9/primary/training_checkpoint.pth",
      "sha256": "d01dee871c5d8035fad50bf8947424ff57b2490c293d9134bd1f43d3b9facab1"
    },
    {
      "bytes": 25078615,
      "path": "experiments/EXP-051_DeepGraph_HighRes_Temporal/checkpoints/EXP-051_20260831T150521Z_477a84b9/secondary/training_checkpoint.pth",
      "sha256": "b949df27db9d7625fb54d4493477ac6ea6314653d914b6739bf2d2c68d85b959"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-051_DeepGraph_HighRes_Temporal/artifacts/EXP-051_20260831T150521Z_477a84b9/TRAINING_COMPLETE.json",
    "sha256": "ef3e433ea79acd66938b696dcf710161c53b93d0f5d639f41f4c82c0883f537d"
  }
}
```
