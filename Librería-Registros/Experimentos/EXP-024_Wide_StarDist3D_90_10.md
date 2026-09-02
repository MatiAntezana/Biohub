---
id: EXP-024
tags: [experimento, wide_stardist3d, 64_channels, polyhedra_rays, split_90_10, score_0_8544]
fecha: 2026-08-26
autor: Matias & Antigravity AI
score_cv: 0.8544
score_lb: null
split: "90% / 10% (179 Train / 20 Val vols)"
estado: Finalizado
decision: INTEGRAR AL POOL DE MODELOS PRINCIPALES DE ENSAMBLE (Score 0.8544 / Recall 98.15%)
---

# 🧪 EXP-024: 3D Wide StarDist Polyhedra (64 Canales) en Esquema 90/10

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Combinar la **capacidad cuádruple de canales (64 canales base)** con la predicción de **32 rayos radiales polihédricos** sobre el dataset ampliado de **179 volúmenes de entrenamiento (17,021 parches 3D)**.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1560-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 10%)

| Umbral ($\tau$) | **Node Recall** | Edge Jaccard | Adjusted Edge Jaccard | **Score Total Oficial** | Estado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.010$ | **`0.9815` (98.15%)** | `0.7736` | `0.7589` | `0.7589` | Sensibilidad extrema |
| $\tau = 0.020$ | **`0.9735` (97.35%)** | `0.8059` | `0.8114` | `0.8114` | Sólido rendimiento |
| $\tau = 0.035$ | **`0.9664` (96.64%)** | `0.8250` | `0.8429` | `0.8429` | Gran equilibrio |
| **$\tau = 0.050$** | **`0.9554` (95.54%)** | **`0.8292`** | **`0.8544`** | 🟢 **`0.8544`** | **Score Standalone Óptimo (0.8544)** |

---

## 🔍 3. Análisis Científico
1. **Poder Geométrico de 64 Canales:**
   * La combinación de 64 canales base con 32 rayos radiales elevó el score de StarDist a **`0.8544`** con una pérdida de entrenamiento récord de `0.000372`.
2. **Rol en el Gran Ensamble Multimodelo:**
   * Su modelado no euclidiano de distancias polihédricas resuelve con gran precisión los cúmulos donde 3 o más células se tocan, reforzando la precisión de `EXP-017` (`0.9010`) y `EXP-019` (`0.8867`).

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **APROBADO CON HONORES E INTEGRADO AL NÚCLEO DURO DEL ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp024_wide_stardist3d_90_10.pt` guardado en `A1560-UBU`.
