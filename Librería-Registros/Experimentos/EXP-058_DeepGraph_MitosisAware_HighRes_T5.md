# EXP-058: DeepGraph Mitosis-Aware HighRes Temporal T5

## 1. Resumen y Motivación
- **ID de Experimento**: `EXP-058`
- **Linaje Padre**: `EXP-055` (MultiScale T5) y `EXP-054` (HighRes Anisotropic Spatial)
- **Objetivo**: Superar el récord de 0.9338 / 0.934 integrando:
  1. **Resolución Espacial Z y XY Completa**: `downsample=(1, 2, 2)` para evitar compresión excesiva en el eje axial.
  2. **Horizonte Temporal Extendido**: Ventana temporal $T=5$ para seguimiento biológico continuo y captura de divisiones.
  3. **Pooling Anisótropo Físico**: `pool_kernel_um=3.75 µm` ajustado al diámetro promedio de células embrionarias.
  4. **Supervisión Mitosis-Aware**: Incremento en la ponderación de detección (`det_loss_weight=1.75`, `det_neg_weight=0.018`) para maximizar el recall en células hijas tras la mitosis.
  5. **Node Transformer Profundo**: `d_model=192`, `nhead=6`, `dropout=0.1`.
  6. **Optimización**: `AdamW` + `CosineAnnealingLR` ($10^{-3} \rightarrow 5\times 10^{-5}$) con `weight_decay=0.0002`.

## 2. Hardware y Ejecución
- **Nodo**: `A1555` (NVIDIA GeForce RTX 4090, 24 GB VRAM).
- **Consumo de Memoria Proyectado**: ~22 GB VRAM (90% de capacidad, sin riesgo de OOM).
- **Semillas**: Doble semilla estandarizada `[3407, 9479]`, 50 épocas por semilla en modo Full-Fit (199 datasets).
