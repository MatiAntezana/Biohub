---
id: EXP-017
tags: [experimento, wide_unet3d, 64_channels, split_90_10, grand_champion, score_0_9010]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.9010
score_lb: null
split: "90% / 10% (179 Train / 20 Val vols)"
estado: Finalizado (Grand Champion)
decision: NUEVO CHAMPION ABSOLUTO (Score 0.9010 / Edge Jaccard 0.8763 / Recall 95.91%)
---

# 🏆 EXP-017: 3D Wide Anisotropic U-Net (64 Canales) en Esquema 90/10 - GRAND CHAMPION

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Combinar la **capacidad cuádruple de canales base (64 canales)** con el dataset ampliado de **179 volúmenes de entrenamiento (17,021 parches 3D)** en el esquema 90/10.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1556-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 10%)

| Umbral ($\tau$) | **Node Recall** | Edge Jaccard | Adjusted Edge Jaccard | **Score Total Oficial** | Estado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.020$ | **`0.9700` (97.00%)** | `0.8778` | `0.8901` | `0.8901` | Altísima sensibilidad |
| **$\tau = 0.035$** | **`0.9591` (95.91%)** | **`0.8763`** | **`0.9010`** | 🏆 **`0.9010`** | 🥇 **NUEVO RÉCORD PROYECTO (ROMPE BARRERA 0.90)** |
| $\tau = 0.050$ | `0.9475` (94.75%) | `0.8655` | `0.8975` | `0.8975` | Altísima precisión |

---

## 🔍 3. Análisis Científico
1. **Ruptura de la Barrera de los 0.90:**
   * La combinación de 64 canales con 17,021 parches 3D logró la menor pérdida de entrenamiento de todo el proyecto (`Train Loss = 0.000364`).
   * En el holdout, el emparejamiento espacio-temporal (Edge Jaccard) saltó a **`0.8763`**, llevando el score final a **`0.9010`**.
2. **Rol en la Estrategia:**
   * Este modelo se convierte en el **Champion Absoluto** y el pilar central para los pesos de inferencia en Kaggle y el ensamble multimodelo.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **CORONADO NUEVO GRAND CHAMPION (0.9010)**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp017_wide_unet3d_90_10.pt` guardado en `A1556-UBU`.

## Reevaluación Promotion Board 2026-08-27

Sobre 12 muestras comunes del holdout seed 42, con inferencia real y métrica oficial, obtuvo `0.8098` (IC bootstrap 95% `[0.7431, 0.8727]`), recall `0.8676` y runtime `421.0 s`. Superó pareadamente a EXP-019 por `+0.0315` y fue marcado `SUBMIT_CANDIDATE`. Ver [[../Validacion_y_Metricas/05_Piloto_Promotion_Board_EXP017_vs_EXP019|reporte del piloto]].
