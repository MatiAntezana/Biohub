# Reporte factual — EXP-057_20260903T001106Z_b04ac218

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-057`
- Estado: `SUCCEEDED`
- Nodo: `A1560`
- Entorno: `train-60`
- Fecha: `2026-09-04T01:43:23.676936Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `b04ac218a485543ce7bf26443a1dbae6e02b617ca07a9e457a20540c63aa286a`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.020101821384082237,
    "edge_loss": 0.00043279885067871266,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 1198.5090795839933,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01820058015920222,
    "edge_loss": 0.0005871251005604184,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 1164.6028393809975,
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
      "bytes": 33649559,
      "path": "experiments/EXP-057_DeepGraph_MultiScale_T5_Anisotropic_Cosine/checkpoints/EXP-057_20260903T001106Z_b04ac218/primary/training_checkpoint.pth",
      "sha256": "6b240f7cd143551ff2d136b2da3717fbf805da9a95d1166a88c2eb73ae9b9bc1"
    },
    {
      "bytes": 33648407,
      "path": "experiments/EXP-057_DeepGraph_MultiScale_T5_Anisotropic_Cosine/checkpoints/EXP-057_20260903T001106Z_b04ac218/secondary/training_checkpoint.pth",
      "sha256": "13a72394f77d21bedfb927f0cef6a6a3205478f6a06edc5e48a79afacb85b558"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-057_DeepGraph_MultiScale_T5_Anisotropic_Cosine/artifacts/EXP-057_20260903T001106Z_b04ac218/TRAINING_COMPLETE.json",
    "sha256": "18ed81902497af06f1684d8498adc3722fc32850c02506420a83b13bd19edfe0"
  }
}
```
