# Conclusión de Ejecución EXP-030B

- **Run ID**: `EXP-030B_20260829T051035Z_55573139`
- **Nodo / GPU**: `A1557` / GPU 0
- **Entorno Remoto**: `train-57`
- **Inicio**: `2026-08-29T05:10:47Z`
- **Fin**: `2026-08-29T11:11:08Z`
- **Datasets Entrenados**: 199 volúmenes completos (100% de datos Biohub)

## Checkpoints Generados y Verificados

1. **Seed Primario (3407)**:
   - Checkpoint: `checkpoints/EXP-030B_20260829T051035Z_55573139/primary/training_checkpoint.pth`
   - SHA-256: `1ff7ec1b88672d8dd8718e240118e8e662d1c174887bfe73c35b9da6005f9109`
   - Épocas: 55 / 55 (Epoch final: `edge_loss = 0.0004282`, `detection_loss = 0.0173775`)
2. **Seed Secundario (9479)**:
   - Checkpoint: `checkpoints/EXP-030B_20260829T051035Z_55573139/secondary/training_checkpoint.pth`
   - SHA-256: `e6875a120b663a421242c3976295ca8abee4c1957c1a2d2169e66194c16142fc`
   - Épocas: 55 / 55 (Epoch final: `edge_loss = 0.0003624`, `detection_loss = 0.0188597`)

## Evaluación y Decisión

- **Score Local**: `N/A — full-fit sin validación`
- **Decisión**: Full-fit dual-seed training complete with record low edge loss (0.000362) and detection loss (0.01738); ready for weights packaging and Kaggle test evaluation.
