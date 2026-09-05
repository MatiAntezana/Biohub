# Reporte factual — EXP-058_20260904T005209Z_e4295bfd

> Generado automáticamente. La interpretación científica debe documentarse aparte.

- Experimento: `EXP-058`
- Estado: `SUCCEEDED`
- Nodo: `A1555`
- Entorno: `train-55`
- Fecha: `2026-09-05T13:59:14.458243Z`
- Validación: `N/A — full-fit sin validación`
- Config SHA-256: `e4295bfda3bedaf6fa71af15bedc653ddc17e534f1a9c977442ba5733ccb65c4`

## Métricas

```json
{
  "primary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.018557576538374025,
    "edge_loss": 0.00041131317128262405,
    "epoch": 49,
    "learning_rate": 5e-05,
    "selection_score": null,
    "train_seconds": 1291.808026268991,
    "validation_accuracy": null,
    "validation_loss": null,
    "validation_recall": null
  },
  "secondary": {
    "checkpoint_selection": "last_epoch",
    "detection_loss": 0.0182170271469901,
    "edge_loss": 0.0004256713788684768,
    "epoch": 49,
    "learning_rate": 5e-05,
    "selection_score": null,
    "train_seconds": 1277.085161408002,
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
      "bytes": 33649138,
      "path": "experiments/EXP-058_DeepGraph_MitosisAware_HighRes_T5/checkpoints/EXP-058_20260904T005209Z_e4295bfd/primary/training_checkpoint.pth",
      "sha256": "3a135da9b7559769e60e491a773d6e1727a76fbae4655b41ec5fd543af60fb08"
    },
    {
      "bytes": 33647986,
      "path": "experiments/EXP-058_DeepGraph_MitosisAware_HighRes_T5/checkpoints/EXP-058_20260904T005209Z_e4295bfd/secondary/training_checkpoint.pth",
      "sha256": "de24a9872ce63450fa105b676ef4101636463b046ebc628dea992930e5a86d6f"
    }
  ],
  "terminal_receipt": {
    "path": "experiments/EXP-058_DeepGraph_MitosisAware_HighRes_T5/artifacts/EXP-058_20260904T005209Z_e4295bfd/TRAINING_COMPLETE.json",
    "sha256": "665fef459ca9bbbde95ff5f8fd4472b4f421ad3100bd54ea1c166529cfbecf7b"
  }
}
```
