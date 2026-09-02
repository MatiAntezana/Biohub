---
tags: [biohub, experimentos, tracking]
updated: 2026-08-25
---

# 📋 Registro Central de Experimentos

> [!TIP]
> Cada fila de esta tabla representa un experimento único. Al hacer click en el ID del experimento (ej: `[[EXP-000_Baseline_Nearest_Neighbor|EXP-000]]`), accederás al reporte técnico completo con hipótesis, configuración, logs y conclusiones.

---

## 📊 Tabla Comparativa de Rendimiento

| ID | Fecha | Modelo / Estrategia | % Split (Train / Val) | CV Score | Adj Edge Jaccard | Div Jaccard | LB Score | Estado | Acción |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| [[Experimentos/EXP-000_Baseline_Nearest_Neighbor\|EXP-000]] | 2026-08-25 | Baseline Heurístico 90% Percentil + Hungarian | `64% / 36%` (128 / 71 vols) | `0.0133` | `0.0133` | `0.0000` | `0.1430` | ⚪ Baseline | Superado |
| [[Experimentos/EXP-001A_3D_Anisotropic_UNet\|EXP-001]] | 2026-08-25 | 3D Anisotropic U-Net + Gaussian Heatmaps | `64% / 36%` (128 / 71 vols) | `0.5810` | `0.5810` | `0.0000` | - | 🟢 Exitoso | Superado por EXP-004 |
| [[Experimentos/EXP-002A_MPM_Flow_Mitosis\|EXP-002A]] | 2026-08-25 | MPM 3D Multi-Task (Heatmap + Flow 3D + Mitosis) | `64% / 36%` (128 / 71 vols) | `0.0321` | `0.0321` | `0.0000` | - | 🔴 Fallido | Autopsia documentada |
| [[Experimentos/EXP-004_3D_TTA_Gap_Closing\|EXP-004]] | 2026-08-25 | 3D Anisotropic U-Net + 4-Way 3D TTA + Gap Closing | `64% / 36%` (128 / 71 vols) | `0.6007` | `0.6007` | `0.0000` | - | 🟢 Exitoso | Superado por EXP-005 |
| [[Experimentos/EXP-005_FullFit_Anisotropic_UNet\|EXP-005]] | 2026-08-25 | Full-Fit 3D Anisotropic U-Net (100% Datos + 3D TTA) | `100% / 0%` (199 / 0 vols) | `0.6700` | `0.6700` | `0.0000` | *Enviando (v8)* | 🟢 Exitoso | Superado por EXP-009 |
| [[Experimentos/EXP-007_ResNet_SelfAttention\|EXP-007]] | 2026-08-25 | 3D ResNet-UNet + Spatial Self-Attention (100% Datos) | `100% / 0%` (199 vols) | `0.0000` | `0.0000` | `0.0000` | - | 🔴 Fallido | Autopsia documentada |
| [[Experimentos/EXP-008_SE_ResNet_80_20\|EXP-008]] | 2026-08-25 | 3D SE-ResNet-UNet (Channel Attention + Split 80/20) | `80% / 20%` (159 / 40 vols) | `0.0000` | `0.0000` | `0.0000` | - | 🔴 Fallido | Autopsia documentada |
| [[Experimentos/EXP-009_Anisotropic_80_20\|EXP-009]] | 2026-08-25 | 3D Anisotropic U-Net (Esquema 80/20 + Calibración $\tau=0.05$) | `80% / 20%` (159 / 40 vols) | `0.6903` | `0.6903` | `0.0000` | - | 🟢 Exitoso | Superado por EXP-011 |
| [[Experimentos/EXP-010_StarDist3D_Polyhedra_80_20\|EXP-010]] | 2026-08-25 | StarDist 3D Polyhedra (32 Rayos Radiales en Split 80/20) | `80% / 20%` (159 / 40 vols) | `0.6147` | `0.6147` | `0.0000` | - | 🟢 Exitoso | **Node Recall Récord (97.81%)** |
| [[Experimentos/EXP-011_Mitosis_Estimator_80_20\|EXP-011]] | 2026-08-25 | Spatio-Temporal Model + Multi-Frame Gap Closing (80/20) | `80% / 20%` (159 / 40 vols) | `0.7968` | `0.7968` | `0.0000` | - | 🟢 Exitoso | Superado por EXP-015 |
| [[Experimentos/EXP-012_ConvNeXt_3D_80_20\|EXP-012]] | 2026-08-25 | 3D ConvNeXt-UNet (Large 7x7 Kernels en Split 80/20) | `80% / 20%` (159 / 40 vols) | `0.0000` | `0.0000` | `0.0000` | - | 🔴 Fallido | Autopsia: Inestabilidad FP16 en GELU 3D |
| [[Experimentos/EXP-013_ResUNet3D_80_20\|EXP-013]] | 2026-08-25 | 3D Residual U-Net (ResUNet 3D en Split 80/20) | `80% / 20%` (159 / 40 vols) | `0.7900` | `0.7900` | `0.0000` | - | 🟢 Exitoso | **Node Recall Récord (98.00%)** |
| [[Experimentos/EXP-014_Wide_UNet3D_80_20\|EXP-014]] | 2026-08-25 | 3D Wide Anisotropic U-Net (64 Canales en Split 80/20) | `80% / 20%` (159 / 40 vols) | `0.7920` | `0.7920` | `0.0000` | - | 🟢 Exitoso | Pool de Ensamble |
| [[Experimentos/EXP-015_Anisotropic_90_10\|EXP-015]] | 2026-08-25 | 3D Anisotropic U-Net (Esquema 90/10 - 179 Train vols) | `90% / 10%` (179 / 20 vols) | **`0.8096`** | **`0.8096`** | `0.0000` | *Enviando* | 🏆 **NUEVO RÉCORD** | **Champion Actual (0.8096)** |
| [[Experimentos/EXP-016_DeepSupervision_80_20\|EXP-016]] | 2026-08-25 | 3D Deep Supervision U-Net (Dec3/Dec2/Dec1 Heads) | `80% / 20%` (159 / 40 vols) | `0.6114` | `0.6114` | `0.0000` | - | 🟢 Exitoso | Pool de Ensamble |
| [[Experimentos/EXP-017_Wide_UNet3D_90_10\|EXP-017]] | 2026-08-25 | 3D Wide U-Net 64ch (Esquema 90/10 - 179 Train vols) | `90% / 10%` (179 / 20 vols) | **`0.9010`** | **`0.9010`** | `0.0000` | **`0.6830`** | 🏆 **NUEVO RÉCORD** | **GRAND CHAMPION (LB 0.683 / CV 0.901)** |
| [[Experimentos/EXP-018_Boundary_Tversky_80_20\|EXP-018]] | 2026-08-25 | 3D Boundary Focal-Tversky U-Net (Esquema 80/20) | `80% / 20%` (159 / 40 vols) | `0.0566` | `0.0566` | `0.0000` | - | 🔴 Fallido | Autopsia: Tversky Incompatible con Heatmaps |
| [[Experimentos/EXP-019_Wide_ResUNet3D_90_10\|EXP-019]] | 2026-08-25 | 3D Wide ResUNet 64ch (Esquema 90/10 - 179 Train vols) | `90% / 10%` (179 / 20 vols) | `0.8867` | `0.8867` | `0.0000` | **`0.6430`** | 🟢 Exitoso | 🥈 Segundo Mejor Score LB |
| [[Experimentos/EXP-020_Inception3D_80_20\|EXP-020]] | 2026-08-25 | 3D Inception-UNet (Dilated Kernels en Split 80/20) | `80% / 20%` (159 / 40 vols) | `0.6150` | `0.6150` | `0.0000` | - | 🟢 Exitoso | Pool de Ensamble |
| [[Experimentos/EXP-021_Full_Volume_High_VRAM_80_20\|EXP-021]] | 2026-08-26 | 3D Anisotropic U-Net (Volumen Completo 64x256x256 High-VRAM) | `80% / 20%` (159 / 40 vols) | `0.0000` | `0.0000` | `0.0000` | - | 🔴 Fallido | Autopsia: BatchNorm inestable con BS=1 en Full Vol |
| [[Experimentos/EXP-022_StarDist3D_90_10\|EXP-022]] | 2026-08-26 | StarDist 3D Polyhedra 32 Rayos (Esquema 90/10 - 179 Train) | `90% / 10%` (179 / 20 vols) | `0.8601` | `0.8601` | `0.0000` | *Corriendo en Kaggle (StarDist 3D)* | 🟢 Exitoso | **Node Recall Récord (99.23%)** |
| [[Experimentos/EXP-023_Wide_DeepSupervision_90_10\|EXP-023]] | 2026-08-26 | 3D Wide Deep Supervision U-Net 64ch (Esquema 90/10) | `90% / 10%` (179 / 20 vols) | `0.8779` | `0.8779` | `0.0000` | - | 🟢 Exitoso | Pool de Ensamble |
| [[Experimentos/EXP-024_Wide_StarDist3D_90_10\|EXP-024]] | 2026-08-26 | 3D Wide StarDist Polyhedra 64ch (Esquema 90/10) | `90% / 10%` (179 / 20 vols) | `0.8544` | `0.8544` | `0.0000` | - | 🟢 Exitoso | Pool de Ensamble (Train Loss 0.000372) |
| [[Experimentos/EXP-025_Wide_Inception3D_90_10\|EXP-025]] | 2026-08-26 | 3D Wide Inception-UNet 64ch (Esquema 90/10) | `90% / 10%` (179 / 20 vols) | *Entrenando (Época 6/20)* | - | - | - | 🟡 En Progreso | Nodo A1552 (RTX 4090) |
| [[Experimentos/EXP-026_Wide_Full_Volume_90_10\|EXP-026]] | 2026-08-27 | 3D Wide Full-Volume (64x256x256) 64ch (Gradient Checkpointing) | `90% / 10%` (179 / 20 vols) | `0.0000` | `0.0000` | `0.0000` | - | 🔴 Fallido | Autopsia: BatchNorm NaN con BS=1 en Full Vol |
| [[Experimentos/EXP-027_Wide_UNet3D_Focal_Alpha20_90_10\|EXP-027]] | 2026-08-26 | 3D Wide UNet 64ch (Focal Hard Mining $\alpha=20$ en 90/10) | `90% / 10%` (179 / 20 vols) | `0.7940` | `0.7940` | `0.0000` | - | 🟢 Exitoso | Pool de Ensamble (Recall 96.9%) |
| [[Experimentos/EXP-028_Wide_Attention_UNet3D_90_10\|EXP-028]] | 2026-08-26 | 3D Wide Attention U-Net 64ch (Spatial Attention Gates 90/10) | `90% / 10%` (179 / 20 vols) | *Entrenando (Época 18/20)* | - | - | - | 🟡 En Progreso | Nodo A1560 (RTX 4090) |
| [[Experimentos/EXP-029_GroupNorm_Full_Volume_90_10\|EXP-029]] | 2026-08-26 | 3D Wide Full-Volume GroupNorm U-Net 64ch (Esquema 90/10) | `90% / 10%` (179 / 20 vols) | *Entrenando (Época 1/20)* | - | - | - | 🟡 En Progreso | Nodo A1557 (RTX 4090) |
| [[Experimentos/EXP-030_DualSeed_Temporal_Ensemble\|EXP-030]] | 2026-08-27 | Dual-Seed TemporalUNet3D + Node Transformer + TTA/ILP | `Embryo-disjoint` | *Entrenando Seed 9479 (Sec)* | - | - | - | 🟡 En Progreso | Nodo A1556 (RTX 4090) |
| [[Experimentos/EXP-030B_DualSeed_Temporal_Ensemble_FullFit\|EXP-030B]] | 2026-08-27 | Nueva versión Dual-Seed Temporal Ensemble Full-Fit | `100% / 0%` (conteo al iniciar) | `N/A — sin val` | - | - | - | 🟣 Código listo | Ejecutar después de EXP-031 |
| [[Experimentos/EXP-031_Detector_First_ILP_Preservado\|EXP-031]] | 2026-08-27 | TemporalUNet3D Detector-First + Hard Negatives + ILP Preservado Full-Fit | `100% / 0%` (conteo al iniciar) | `N/A — sin val` | - | - | - | 🟣 Código listo | Relanzar primero; revisión full-fit |

---

## Reevaluación comparable Promotion Board

El 2026-08-27 se reevaluaron EXP-017 y EXP-019 mediante inferencia real sobre las mismas 12 muestras holdout y la métrica oficial. EXP-017 obtuvo `0.8098`, EXP-019 `0.7783`; delta pareado `+0.0315`, IC 95% `[+0.0046, +0.0592]`. El orden coincide con Kaggle. Ver [[Validacion_y_Metricas/05_Piloto_Promotion_Board_EXP017_vs_EXP019|reporte completo]]. Estos valores no reemplazan los CV históricos porque corresponden a otro protocolo.

---

## 🏷️ Convenciones de Estado y Decisión

- **Estados:** `Planificado` 📝 | `En Progreso` ⏳ | `Finalizado` ✅ | `Descartado` ❌
- **Decisiones:**
  - `PROMOVIDO A PIPELINE`: Pasa a formar parte del ensemble o pipeline base.
  - `MEJORA PROMETEDORA`: Mostró delta positivo, requiere ajuste de hiperparámetros.
  - `RECHAZADO`: Delta negativo en CV o violación de restricciones de tiempo/memoria.
