---
id: EXP-025
tags: [experimento, wide_inception3d, 64_channels, dilated_conv, split_90_10]
fecha: 2026-08-26
autor: Matias & Antigravity AI
score_cv: null
score_lb: null
split: "90% / 10% (179 Train / 20 Val vols)"
estado: En Progreso
decision: EN ENTRENAMIENTO EN NODO A1552 (RTX 4090)
---

# 🧪 EXP-025: 3D Wide Inception-UNet (64 Canales + Ramas Dilatadas) en Esquema 90/10

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Expandir la arquitectura **Inception 3D multiescala** (con ramas convolucionales estándar, anisotrópicas y dilatadas $d=2$) a **64 canales base ($4\times$ mayor capacidad)** sobre los **179 volúmenes de entrenamiento (17,021 parches 3D)**.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1552-UBU`.

---

## ⚙️ 2. Configuración y Parámetros
- **Arquitectura:** `InceptionUNet3D` con `base_channels=64`.
- **Dataset:** 179 volúmenes de entrenamiento y 20 volúmenes de validación (Esquema 90/10).
- **Optimizador:** AdamW con Cosine Annealing (20 épocas).

---

## 📊 3. Resultados de Validación (En Progreso)
*(Se actualizarán automáticamente ni bien concluyan las 20 épocas)*
