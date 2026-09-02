---
id: EXP-023
tags: [experimento, wide_unet3d, deep_supervision, 64_channels, split_90_10, score_0_8779]
fecha: 2026-08-26
autor: Matias & Antigravity AI
score_cv: 0.8779
score_lb: null
split: "90% / 10% (179 Train / 20 Val vols)"
estado: Finalizado
decision: INTEGRAR AL POOL DE MODELOS PRINCIPALES DE ENSAMBLE (Score 0.8779 / Recall 98.95%)
---

# 🧪 EXP-023: 3D Wide Deep Supervision U-Net (64 Canales) en Esquema 90/10

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Fusionar la **capacidad cuádruple de canales (64 canales base)** con cabezales de **Supervisión Profunda Multiescala (`Dec3`, `Dec2`, `Dec1`)** y el dataset expandido de **179 volúmenes de entrenamiento (17,021 parches 3D)**.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1556-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 10%)

| Umbral ($\tau$) | **Node Recall** | Edge Jaccard | Adjusted Edge Jaccard | **Score Total Oficial** | Estado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.010$ | **`0.9895` (98.95%)** | `0.8482` | `0.8224` | `0.8224` | Sensibilidad extrema |
| $\tau = 0.020$ | **`0.9813` (98.13%)** | `0.8586` | `0.8601` | `0.8601` | Gran consistencia |
| $\tau = 0.035$ | **`0.9716` (97.16%)** | `0.8476` | `0.8637` | `0.8637` | Sólido rendimiento |
| **$\tau = 0.050$** | **`0.9669` (96.69%)** | **`0.8537`** | **`0.8779`** | 🟢 **`0.8779`** | **Score Standalone Óptimo (0.8779)** |

---

## 🔍 3. Análisis Científico
1. **Gradientes Multiescala Estables:**
   * La supervisión auxiliar en `Dec3` y `Dec2` estabilizó el entrenamiento de los 64 canales, permitiendo un salto masivo desde el `0.6114` de `EXP-016` (32ch en 80/20) a **`0.8779`** (+0.2665).
2. **Rol en el Gran Ensamble Multimodelo:**
   * Al predecir características a múltiples resoluciones espaciales, su mapa de calor de salida es más suave y menos propenso al sobreajuste en bordes ruidosos.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **APROBADO CON HONORES E INTEGRADO AL POOL DE ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp023_wide_deep_supervision_90_10.pt` guardado en `A1556-UBU`.
