# Reporte factual — EXP-060_20260904T134411Z_a4dce311

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-060`
- Estado: `SUCCEEDED`
- Nodo: `A1688`
- Entorno: `train-88`
- Fecha: `2026-09-05T11:32:11.224703Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `a4dce311d62cc9fef2f1c6c4212f53e9f3731a2d599f72448c9f291916cbd2df`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01999659926413248,
    "edge_loss": 0.00043022373962647484,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 779.6790038149993,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01813715001009405,
    "edge_loss": 0.0005063506807167263,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 777.6869467479992,
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
      "bytes": 56102871,
      "path": "experiments/EXP-060_DeepGraph_Wide_Window5_Anisotropic/checkpoints/EXP-060_20260904T134411Z_a4dce311/primary/training_checkpoint.pth",
      "sha256": "dc66471ed59367818cafa95ab16727588354d03162d5a28810eae8f93ee15551"
    },
    {
      "bytes": 56102935,
      "path": "experiments/EXP-060_DeepGraph_Wide_Window5_Anisotropic/checkpoints/EXP-060_20260904T134411Z_a4dce311/secondary/training_checkpoint.pth",
      "sha256": "94bdba056f1cedc8fabeac5486bf7c5acdf38e39b212d09da5b8162f35eeb4f9"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-060_DeepGraph_Wide_Window5_Anisotropic/artifacts/EXP-060_20260904T134411Z_a4dce311/TRAINING_COMPLETE.json",
    "sha256": "2d402ebe0f01529274924aeb177781df41cd9df39e72d1dfdbb4bd1076903534"
  }
}
```
