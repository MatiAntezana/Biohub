# EXP-054: DeepGraph HighRes Anisotropic Spatial (Z-aware spatial pooling)

## 1. Resumen y Motivación
- **ID de Experimento**: `EXP-054`
- **Linaje Padre**: `EXP-051` (HighRes Temporal)
- **Objetivo**: Integrar pooling espacial anisótropo (`pool_kernel_um=5.0 µm` con corrección axial Z) y alta resolución temporal para mejorar la detección en tejidos biológicos multicapa sin perder coherencia temporal.
- **Configuración Clave**:
  - `unet_layers`: [32, 64, 128]
  - `unet_out_channels`: 32
  - `downsample`: (1, 2, 2)
  - `window_size`: 4
  - `pool_kernel_um`: 5.0
  - Doble semilla: `[3407, 9479]`
  - Épocas: 50 por semilla (Full-fit sobre 199 datasets)

## 2. Resultados de Entrenamiento
- **Semilla 1 (Primary, seed 3407)**:
  - `detection_loss`: 0.01833
  - `edge_loss`: 0.000347
- **Semilla 2 (Secondary, seed 9479)**:
  - `detection_loss`: 0.01648 (mínimo histórico del linaje)
  - `edge_loss`: 0.000367
- **Estado**: COMPLETADO con éxito al 100% de épocas en GPU RTX 4090 (`A1556`).
