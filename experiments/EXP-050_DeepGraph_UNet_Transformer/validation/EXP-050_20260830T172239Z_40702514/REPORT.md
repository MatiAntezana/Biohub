# Reporte factual — EXP-050_20260830T172239Z_40702514

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-050`
- Estado: `DOC_PENDING`
- Nodo: `A1552`
- Entorno: `train-52`
- Fecha: `2026-08-30T22:52:39.618059Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `4070251448d0b88322743c28c6e213ffbfe5700cf22f738f85a644757fb30f71`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.021041728532873095,
    "edge_loss": 0.00043562394317207993,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 191.11046526400605,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.019606644437337915,
    "edge_loss": 0.0003858766335639909,
    "epoch": 49,
    "selection_score": null,
    "train_seconds": 199.2255985859956,
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
      "path": "experiments/EXP-050_DeepGraph_UNet_Transformer/checkpoints/EXP-050_20260830T172239Z_40702514/primary/training_checkpoint.pth",
      "sha256": "5c689a79565f41ffbbb6dfef403703fa20661c1f97a926ba856d5550d14044b8"
    },
    {
      "bytes": 25078615,
      "path": "experiments/EXP-050_DeepGraph_UNet_Transformer/checkpoints/EXP-050_20260830T172239Z_40702514/secondary/training_checkpoint.pth",
      "sha256": "40e84941842981cf4917274d3ea234b3fc827fea3de90f77b62173fbb99910ae"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-050_DeepGraph_UNet_Transformer/artifacts/EXP-050_20260830T172239Z_40702514/TRAINING_COMPLETE.json",
    "sha256": "5510363c8fcf552170710b980c5d410d7e19857519c805e78e5866cc8478d603"
  }
}
```
