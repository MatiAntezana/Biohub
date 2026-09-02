---
id: EXP-019
tags: [experimento, wide_resunet3d, 64_channels, split_90_10, high_train_data, score_0_8867]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.8867
score_lb: null
split: "90% / 10% (179 Train / 20 Val vols)"
estado: Finalizado
decision: INTEGRAR AL NÚCLEO PRINCIPAL DE ENSAMBLE (Score 0.8867 / Edge Jaccard 0.8604 / Recall 96.31%)
---

# 🧪 EXP-019: 3D Wide Residual U-Net (64 Canales) en Esquema 90/10

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Combinar los **bloques residuales 3D (`ResBlock3D`)** que alcanzaron el récord de sensibilidad en `EXP-013` con una capacidad expandida a **64 canales base** y los **179 volúmenes de entrenamiento (17,021 parches 3D)**.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1552-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 10%)

| Umbral ($\tau$) | **Node Recall** | Edge Jaccard | Adjusted Edge Jaccard | **Score Total Oficial** | Estado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.010$ | **`0.9873` (98.73%)** | `0.8338` | `0.7877` | `0.7877` | Sensibilidad extrema |
| $\tau = 0.020$ | **`0.9761` (97.61%)** | `0.8556` | `0.8477` | `0.8477` | Gran equilibrio |
| $\tau = 0.035$ | **`0.9684` (96.84%)** | `0.8552` | `0.8702` | `0.8702` | Sólido rendimiento |
| **$\tau = 0.050$** | **`0.9631` (96.31%)** | **`0.8604`** | **`0.8867`** | 🟢 **`0.8867`** | **Score Standalone Óptimo (0.8867)** |

---

## 🔍 3. Análisis Científico
1. **Poder Residual de Alta Capacidad:**
   * La combinación de bloques residuales con 64 canales logró la menor pérdida de entrenamiento del proyecto (`Train Loss = 0.000343`).
   * A $\tau = 0.050$, el modelo alcanza un **Score de `0.8867`** con un **Edge Jaccard de `0.8604`**, posicionándose como el segundo modelo más potente del proyecto.
2. **Rol en el Ensamble Multimodelo:**
   * Al estar basado en residuos `ResBlock3D`, su patrón de activación es el complemento perfecto para `EXP-017` (Wide U-Net 64ch - `0.9010`) y `EXP-022` (StarDist 3D - `0.8601`).

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **APROBADO CON HONORES E INTEGRADO AL NÚCLEO DURO DEL ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp019_wide_resunet3d_90_10.pt` guardado en `A1552-UBU`.

## Reevaluación Promotion Board 2026-08-27

Sobre las mismas 12 muestras comunes del holdout seed 42, obtuvo `0.7783` (IC bootstrap 95% `[0.7005, 0.8494]`), recall `0.8593` y runtime `524.8 s`. Quedó `HOLD`; EXP-017 lo superó pareadamente por `+0.0315`. Ver [[../Validacion_y_Metricas/05_Piloto_Promotion_Board_EXP017_vs_EXP019|reporte del piloto]].
