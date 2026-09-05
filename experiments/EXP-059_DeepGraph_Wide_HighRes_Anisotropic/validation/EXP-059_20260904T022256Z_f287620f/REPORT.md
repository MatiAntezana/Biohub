# Reporte factual — EXP-059_20260904T022256Z_f287620f

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-059`
- Estado: `SUCCEEDED`
- Nodo: `A1560`
- Entorno: `train-60`
- Fecha: `2026-09-04T19:47:18.419462Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `f287620f602a5857a87875c90ef1ca7352d033ba4d7829572a92ddb9cae3a4c7`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.01743882784806192,
    "edge_loss": 0.0005541448954037757,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 620.8108137669915,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.019317125339371462,
    "edge_loss": 0.0005373696318724797,
    "epoch": 49,
    "learning_rate": 1e-05,
    "selection_score": null,
    "train_seconds": 621.0090804699867,
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
      "path": "experiments/EXP-059_DeepGraph_Wide_HighRes_Anisotropic/checkpoints/EXP-059_20260904T022256Z_f287620f/primary/training_checkpoint.pth",
      "sha256": "21874bbc4bb6d723584033a38e402ae470a1f271ac49f47d45bcbe8be8031032"
    },
    {
      "bytes": 56102935,
      "path": "experiments/EXP-059_DeepGraph_Wide_HighRes_Anisotropic/checkpoints/EXP-059_20260904T022256Z_f287620f/secondary/training_checkpoint.pth",
      "sha256": "df4b3dd369d6f943d5cbb162eb03f4eb070db0ea306ae2ba03dc999e81bf6edb"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-059_DeepGraph_Wide_HighRes_Anisotropic/artifacts/EXP-059_20260904T022256Z_f287620f/TRAINING_COMPLETE.json",
    "sha256": "bfeef675e3d270e7bab66f09eb82a21027cda87ecaf7246dd5c3f6600e173a9a"
  }
}
```
