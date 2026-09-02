# Reporte factual — EXP-032_20260829T062432Z_3ee7f30d

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-032`
- Estado: `DOC_PENDING`
- Nodo: `A1556`
- Entorno: `train-56`
- Fecha: `2026-08-29T14:34:58.507258Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `3ee7f30da607f33ea4906e5d48651cea5f4188e08025e888aae566de94e2f959`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.02180089669736723,
    "edge_loss": 0.00039439632409206144,
    "epoch": 54,
    "selection_score": null,
    "train_seconds": 263.684020836954,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.023805484442661205,
    "edge_loss": 0.00039656299356647653,
    "epoch": 54,
    "selection_score": null,
    "train_seconds": 264.1373934990261,
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
      "path": "experiments/EXP-032_DualSeed_Temporal_Ensemble_Enhanced/checkpoints/EXP-032_20260829T062432Z_3ee7f30d/primary/training_checkpoint.pth",
      "sha256": "7c115fbc02e533308f5cf1e53cdf09e37acc16a464ef4dcc55270ac835ae2b03"
    },
    {
      "bytes": 47520791,
      "path": "experiments/EXP-032_DualSeed_Temporal_Ensemble_Enhanced/checkpoints/EXP-032_20260829T062432Z_3ee7f30d/secondary/training_checkpoint.pth",
      "sha256": "fa2c04526aa05089dc27a0957e000330f4a68b4577568103a0ffd27c38eb95cd"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-032_DualSeed_Temporal_Ensemble_Enhanced/artifacts/EXP-032_20260829T062432Z_3ee7f30d/TRAINING_COMPLETE.json",
    "sha256": "35caa1194a925039a1e57bb8dd579167a9bd74b01a733b18593a7bed9db79d03"
  }
}
```
