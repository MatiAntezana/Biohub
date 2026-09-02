---
id: EXP-027
tags: [experimento, wide_unet3d, 64_channels, focal_loss, alpha_20, split_90_10]
fecha: 2026-08-26
autor: Matias & Antigravity AI
score_cv: null
score_lb: null
split: "90% / 10% (179 Train / 20 Val vols)"
estado: En Progreso
decision: EN ENTRENAMIENTO EN NODO A1560 (RTX 4090)
---

# 🧪 EXP-027: 3D Wide Anisotropic U-Net (64 Canales + Focal Alpha=20) en Esquema 90/10

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Elevar el factor de ponderación focal a $\alpha=20$ ($\mathcal{L} = (1 + 20 \cdot y)(p - y)^2$) para forzar al modelo a priorizar fuertemente las células tenues y los centros en cúmulos celulares sobre el fondo vacío.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1560-UBU`.

---

## ⚙️ 2. Configuración y Parámetros
- **Arquitectura:** `AnisotropicUNet3D` con `base_channels=64`.
- **Pérdida:** Focal Weighted MSE con $\alpha = 20.0$.
- **Dataset:** 179 volúmenes de entrenamiento y 20 volúmenes de validación (Esquema 90/10).
- **Optimizador:** AdamW con Cosine Annealing (20 épocas).

---

## 📊 3. Resultados de Validación (En Progreso)
*(Se actualizarán automáticamente ni bien concluyan las 20 épocas)*
