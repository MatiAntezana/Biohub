---
id: EXP-014
tags: [experimento, wide_unet3d, 64_channels, split_80_20, score_0_7920]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.7920
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: Finalizado
decision: INTEGRAR AL POOL PRINCIPAL DE ENSAMBLE (Score 0.7920 / Edge Jaccard 0.7756)
---

# 🧪 EXP-014: 3D Wide Anisotropic U-Net (64 Canales Base en Esquema 80/20)

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Duplicar el número de canales base de $32 \to 64$ en la arquitectura `AnisotropicUNet3D` para cuadruplicar ($4\times$) la capacidad de modelado de texturas celulares y gradientes de intensidad en el **Esquema 80% Train / 20% Val**.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1560-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 20%)

| Umbral ($\tau$) | **Node Recall** | Edge Jaccard | Adjusted Edge Jaccard | **Score Total Oficial** | Estado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.035$ | **`0.9611` (96.11%)** | `0.7742` | `0.7869` | **`0.7869`** | Gran sensibilidad |
| **$\tau = 0.050$** | `0.9455` (94.55%) | **`0.7756`** | **`0.7920`** | 🟢 **`0.7920`** | **Score Standalone Óptimo** |

---

## 🔍 3. Análisis Científico
1. **Gran Rendimiento Standalone (`0.7920`):**
   La capacidad expandida de 64 canales permitió una reconstrucción volumétrica superior de las intensidades celulares, logrando un Edge Jaccard consistente de `0.7756`.
2. **Rol en el Ensamble Multimodelo:**
   Su combinación con `EXP-013` (Residual) y `EXP-010` (StarDist) conformará el núcleo duro de modelos de alta capacidad para la generalización en el test oculto.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **APROBADO E INTEGRADO AL ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp014_wide_unet3d_80_20.pt` guardado en `A1560-UBU`.
