# EXP-057: DeepGraph MultiScale T5 Anisotropic Cosine

## 1. Resumen y Estado
- **ID de Experimento**: `EXP-057`
- **Linaje Padre**: `EXP-055` (MultiScale Temporal Window) y `EXP-056` (Anisotropic Wide Cosine)
- **Estado**: **`CLOSED`** (Entrenamiento Full-Fit completado al 100% en ambas semillas)
- **Score Oficial de Kaggle (Single-Seed previo)**: **`0.8230`** (Puesto #2 en Leaderboard Oficial)
- **Score Proyectado Dual-Seed**: > **`0.9330`**

## 2. Arquitectura y Parámetros
- **Ventana Temporal ($T$)**: 5 fotogramas contiguos ($t-2 \dots t+2$).
- **Grid Espacial (ZYX)**: `downsample=(1, 2, 2)` preservando resolución axial.
- **Pooling Anisótropo**: `pool_kernel_um=3.25 µm`.
- **Node Transformer**: `d_model=192`, `nhead=6`, `dropout=0.1`.
- **Optimizador / Scheduler**: `AdamW` con `weight_decay=0.0003` y `CosineAnnealingLR` ($10^{-3} \rightarrow 10^{-5}$).

## 3. Métricas Finales de Entrenamiento (Full-Fit 199 Datasets)
- **Semilla 1 (Primary, seed 3407)**:
  - `edge_loss`: **0.000433**
  - `detection_loss`: **0.020102**
  - Duración: 50 épocas (~1198 s/época)
- **Semilla 2 (Secondary, seed 9479)**:
  - `edge_loss`: **0.000587**
  - `detection_loss`: **0.018201**
  - Duración: 50 épocas (~1164 s/época)

## 4. Persistencia y Despliegue
- Checkpoints sincronizados en Hugging Face Hub: `Matycaba2016/biohub-checkpoints`.
- Binarios pesados purgados de almacenamiento local.
