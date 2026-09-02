# Reporte factual — EXP-052_20260901T050028Z_33370396

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-052`
- Estado: `DOC_PENDING`
- Nodo: `A1552`
- Entorno: `train-52`
- Fecha: `2026-09-01T12:42:36.884286Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `33370396757a3e74818ce9905e3c97f423e5da0c5d1faea9b48da8a6d8031c3c`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.019329789189000926,
    "edge_loss": 0.0004229900327845826,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 269.5195199850132,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01854319062953194,
    "edge_loss": 0.0003649145105240071,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 276.4961177089717,
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
      "bytes": 47520727,
      "path": "experiments/EXP-052_DeepGraph_WideCapacity/checkpoints/EXP-052_20260901T050028Z_33370396/primary/training_checkpoint.pth",
      "sha256": "74d52aff8b407ec31f10b5a650785521954453660024bfb070eabd6022111421"
    },
    {
      "bytes": 47520791,
      "path": "experiments/EXP-052_DeepGraph_WideCapacity/checkpoints/EXP-052_20260901T050028Z_33370396/secondary/training_checkpoint.pth",
      "sha256": "de875c8155c23df93060d17346c252ca420b12d4139a20ea6414e64f180991bf"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-052_DeepGraph_WideCapacity/artifacts/EXP-052_20260901T050028Z_33370396/TRAINING_COMPLETE.json",
    "sha256": "47de5d84a9d952ae04511146193517219daa34d027813bd4564fbc051ce951b2"
  }
}
```
