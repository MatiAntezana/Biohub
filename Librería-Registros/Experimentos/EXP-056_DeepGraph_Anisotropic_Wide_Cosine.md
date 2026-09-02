# EXP-056: DeepGraph Anisotropic Wide Cosine (Wide UNet 48-192 + Z 3.25um + WD 0.0003 + Cosine LR)

## 1. Resumen y Motivación
- **ID de Experimento**: `EXP-056`
- **Linaje Padre**: `EXP-052` (Public Score **0.933**) y `EXP-051` (Public Score **0.9338**)
- **Objetivo**: Integrar las ventajas demostradas en la competencia en un único modelo de máxima capacidad:
  1. **Capacidad de Representación Visual**: UNet volumétrica amplia (`channels=[48, 96, 192]`, salida 48ch).
  2. **Resolución Espacial Z**: Pooling espacial anisótropo (`pool_kernel_um=3.25 µm`) para corregir la compresión axial en microscopía biológica 3D.
  3. **Transformer Temporal Robusto**: Node Transformer con `d_model=192`, `nhead=6`, `dim_feedforward=384`, `dropout=0.1`.
  4. **Optimización Avanzada**: `CosineAnnealingLR` ($10^{-3} \rightarrow 10^{-5}$) regularizado con `weight_decay=0.0003` para evitar sobreajuste en grafos densos.

## 2. Hardware y Ejecución
- **Nodo Asignado**: `A1556` (NVIDIA GeForce RTX 4090, 24,564 MiB VRAM).
- **Modo de Entrenamiento**: Full-fit (199 datasets pareados), doble seed estandarizada (`[3407, 9479]`), 50 épocas por seed.
- **Inferencia / Kaggle**: Pipeline automático con ensamble dual-seed y submission para Public Score.
