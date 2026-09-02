# Reporte factual — EXP-050_20260830T172441Z_40702514

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-050`
- Estado: `SUCCEEDED`
- Nodo: `A1556`
- Entorno: `train-56`
- Fecha: `2026-08-30T22:53:03.684753Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `4070251448d0b88322743c28c6e213ffbfe5700cf22f738f85a644757fb30f71`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.020560162771803638,
    "edge_loss": 0.00042578409668112726,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 189.83455236488953,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.019291062491014598,
    "edge_loss": 0.00042649946418168836,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 197.55876497004647,
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
      "path": "experiments/EXP-050_DeepGraph_UNet_Transformer/checkpoints/EXP-050_20260830T172441Z_40702514/primary/training_checkpoint.pth",
      "sha256": "5eec6fd436e2d3aa682f393af0e97ee8d65c17ae01bb7bf8600b6e32df96c300"
    },
    {
      "bytes": 25078615,
      "path": "experiments/EXP-050_DeepGraph_UNet_Transformer/checkpoints/EXP-050_20260830T172441Z_40702514/secondary/training_checkpoint.pth",
      "sha256": "35a7aed2c15e8acabc89cd21d1638e0afaab9c9218cd5852422e57bbae027ff7"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-050_DeepGraph_UNet_Transformer/artifacts/EXP-050_20260830T172441Z_40702514/TRAINING_COMPLETE.json",
    "sha256": "41155dfd40398bcbb6ae3aa051abe3cfbf183f4790a42a22222835278d01fab3"
  }
}
```
