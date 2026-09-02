---
id: EXP-001A
tags: [experimento, unet3d, anisotropica, heatmaps]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.0539
score_lb: null
estado: Finalizado
decision: MEJORA SIGNIFICATIVA (+305% vs Baseline)
---

# 🧪 EXP-001A: 3D Anisotropic U-Net + Gaussian Heatmap Regression

## 🎯 1. Hipótesis & Objetivo
- **Hipótesis:** Reemplazar el umbral heurístico por percentil global del baseline con una red neuronal volumétrica 3D diseñada con convoluciones anisotrópicas $(1, 3, 3)$ para preservar la resolución espacial en $X/Y$ ($0.40625\,\mu\text{m}$) antes de comprimir el eje $Z$ ($1.625\,\mu\text{m}$), entrenada con heatmaps gaussianos 3D y Masked Focal Loss.
- **Módulo:** [[Modulos_Tecnicos/02_Deteccion_3D_y_Segmentacion|Detección 3D]]

---

## ⚙️ 2. Configuración y Parámetros
- **Arquitectura:** `AnisotropicUNet3D` (Canales: $32 \to 64 \to 128 \to 256$, Convoluciones anisotrópicas en Stage 1).
- **Target:** Heatmaps gaussianos 3D con $\sigma_z=1.0, \sigma_y=2.0, \sigma_x=2.0$.
- **Loss:** Focal MSE ponderado ($1 + 10 \cdot Y$).
- **Optimizador:** AdamW ($\text{lr}=10^{-3}$, Cosine Annealing a $10^{-5}$ en 15 épocas).
- **Split:** Embryo-disjoint (Entrenado en 128 muestras del embrión `6bba`, validado en `44b6`).
- **Hardware:** NVIDIA RTX 4090 (24 GB) en `A1560-UBU`.

---

## 📊 3. Resultados Cuantitativos

| Métrica | Baseline (EXP-000) | **EXP-001A (3D Anisotropic U-Net)** | Delta ($\Delta$) |
|---|---|---|---|
| **Node Recall** | `0.0536` | **`0.1242`** | **$+131.7\%$** 🚀 |
| **Edge Jaccard** | `0.0122` | **`0.0492`** | **$+303.3\%$** 🚀 |
| **Adjusted Edge Jaccard** | `0.0133` | **`0.0539`** | **$+305.3\%$** 🚀 |
| **Division Jaccard** | `0.0000` | `0.0000` | `+0.0000` (Hungarian 1-to-1) |
| **CV Total Score** | **`0.0133`** | **`0.0539`** | **$+0.0406$ ($4\times$ superior)** |
| **Tiempo de Entrenamiento** | $0\text{ min}$ | $60\text{ min}$ (15 épocas sobre 12,540 crops 3D) | Muy eficiente en RTX 4090 |

---

## 🔍 4. Análisis Cualitativo & Modo de Falla
### ✅ Aciertos:
1. Las convoluciones anisotrópicas permitieron a la red aprender la forma real elipsoidal de las células de pez cebra sin aplanarlas en $Z$.
2. La pérdida Focal MSE convergió de `0.004147` a `0.000240` en validación sin sobreajuste.
3. El `Adjusted Edge Jaccard` subió de `0.0133` a `0.0539` (+305%).

### ❌ Limitaciones Detectadas (Vías de Mejora para EXP-001B / EXP-002):
1. **Node Recall al 12.4%:** `peak_thresh=0.25` con `min_distance=2` es demasiado conservador para células tenues o profundas. Bajar el umbral a `0.15` y calibrar con `estimated_number_of_nodes`.
2. **Matching ciego en tiempo:** La distancia euclidiana falla cuando las células experimentan flujos convectivos en la epibolia. Se requieren **vectores de movimiento 3D (MPM)**.
3. **Cero Mitosis:** Al usar asignación bipartita 1-a-1 estándar, la tasa de divisiones predichas es 0.

---

## 🏁 5. Conclusión & Próximos Pasos
- **Decisión:** **PROMOVIDO COMO NUEVO DETECTOR BASE**.
- **Siguiente Experimento (`EXP-001B` / `EXP-002`):** Implementar **MPM (Motion & Position Maps)** con predicción conjunta de vectores de velocidad 3D $(v_z, v_y, v_x)$ y el módulo de **Detección Geométrica de Mitosis** para desbloquear el término de `division_jaccard`.
