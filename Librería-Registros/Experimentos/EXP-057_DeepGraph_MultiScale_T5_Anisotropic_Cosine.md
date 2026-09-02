# EXP-057: DeepGraph MultiScale T5 Anisotropic Cosine (T=5 + Z 3.25um + WD 0.0003 + Cosine LR)

## 1. Resumen y Motivación
- **ID de Experimento**: `EXP-057`
- **Linajes Padres**: `EXP-055` (MultiScale $T=5$, récord de pérdida `0.000309`) y `EXP-054` (Anisotropic Z $3.25\ \mu	ext{m}$, `0.000325`).
- **Objetivo**: Sintetizar los dos mayores descubrimientos de modelado del proyecto en un único modelo de máxima precisión:
  1. **Ventana Temporal Extendida ($T=5$)**: Aumenta el horizonte de tracking para capturar mitosis lentas y trayectorias celulares con oclusión.
  2. **Resolución Espacial Anisótropa Z ($3.25\ \mu	ext{m}$)**: Spatial pooling calibrado para preservar la morfología axial 3D.
  3. **Node Transformer de Alta Capacidad**: `d_model=192`, `nhead=6`, `dim_feedforward=384`, `dropout=0.1`.
  4. **Optimización Suave Regularizada**: `CosineAnnealingLR` ($10^{-3} \rightarrow 10^{-5}$) regularizado con `weight_decay=0.0003`.

## 2. Hardware y Ejecución
- **Nodo Asignado**: `A1560` (NVIDIA GeForce RTX 4090, 24,564 MiB VRAM).
- **Modo de Entrenamiento**: Full-fit (199 datasets pareados), doble seed estandarizada (`[3407, 9479]`), 50 épocas por seed.
- **Inferencia / Kaggle**: Pipeline automático con ensamble dual-seed y submission para Public Score.
