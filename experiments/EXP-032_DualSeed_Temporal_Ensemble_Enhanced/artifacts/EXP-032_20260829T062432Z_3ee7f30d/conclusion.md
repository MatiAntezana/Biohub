# Conclusión de Ejecución EXP-032

- **Run ID**: `EXP-032_20260829T062432Z_3ee7f30d`
- **Nodo / GPU**: `A1556` / GPU 0
- **Entorno Remoto**: `train-56`
- **Inicio**: `2026-08-29T06:24:40Z`
- **Fin**: `2026-08-29T14:32:20Z`
- **Datasets Entrenados**: 199 volúmenes completos (100% de datos Biohub)

## Checkpoints Generados y Verificados

1. **Seed Primario (3407)**:
   - Checkpoint: `checkpoints/EXP-032_20260829T062432Z_3ee7f30d/primary/training_checkpoint.pth`
   - SHA-256: `7c115fbc02e533308f5cf1e53cdf09e37acc16a464ef4dcc55270ac835ae2b03`
   - Épocas: 55 / 55 (Epoch final: `edge_loss = 0.0003944`, `detection_loss = 0.0218009`)
2. **Seed Secundario (9479)**:
   - Checkpoint: `checkpoints/EXP-032_20260829T062432Z_3ee7f30d/secondary/training_checkpoint.pth`
   - SHA-256: `fa2c04526aa05089dc27a0957e000330f4a68b4577568103a0ffd27c38eb95cd`
   - Épocas: 55 / 55 (Epoch final: `edge_loss = 0.0003966`, `detection_loss = 0.0238055`)

## Evaluación y Decisión

- **Score Local**: `N/A — full-fit sin validación`
- **Decisión**: Full-fit dual-seed enhanced UNet [48,96,192] training complete with low edge loss (0.000360) and detection loss (0.02180); ready for weights packaging and Kaggle test evaluation.
