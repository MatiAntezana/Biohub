# EXP-053: DeepGraph HighRes Regularized Cosine (WD 0.0003 + Cosine LR)

## 1. Resumen y Motivación
- **ID de Experimento**: `EXP-053`
- **Linaje Padre**: `EXP-051` (HighRes Temporal)
- **Objetivo**: Evaluar la regularización explícita (`weight_decay=0.0003`) combinada con un decaimiento suave de tasa de aprendizaje (`CosineAnnealingLR` de $10^{-3}$ a $10^{-5}$) sobre la arquitectura HighRes Temporal con ventana temporal $T=4$.
- **Configuración Clave**:
  - `unet_layers`: [32, 64, 128]
  - `unet_out_channels`: 32
  - `downsample`: (1, 2, 2)
  - `window_size`: 4
  - `pool_kernel_um`: 5.0
  - `lr_scheduler_type`: cosine
  - `weight_decay`: 0.0003
  - Doble semilla: `[3407, 9479]`
  - Épocas: 50 por semilla (Full-fit sobre 199 datasets)

## 2. Resultados de Entrenamiento
- **Semilla 1 (Primary, seed 3407)**:
  - `detection_loss`: 0.01888
  - `edge_loss`: 0.000539
- **Semilla 2 (Secondary, seed 9479)**:
  - `detection_loss`: 0.02236
  - `edge_loss`: 0.000446
- **Estado**: COMPLETADO con éxito al 100% de épocas en GPU RTX 4090 (`A1555`).
