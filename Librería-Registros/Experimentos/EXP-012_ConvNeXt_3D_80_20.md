---
id: EXP-012
tags: [experimento, convnext3d, large_kernels, split_80_20]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: null
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: En Progreso
decision: EN ENTRENAMIENTO EN NODO A1560 (RTX 4090)
---

# 🧪 EXP-012: 3D ConvNeXt-UNet (Large 7x7 Depthwise Kernels en Esquema 80/20)

## 🎯 1. Hipótesis & Estrategia
- **Hipótesis:** Incorporar bloques **ConvNeXt 3D con convoluciones depthwise $7 \times 7 \times 7$ y $(3 \times 7 \times 7)$** combinadas con cuellos de botella invertidos $4\times$ y activaciones GELU.
- **Ventaja Única:** Otorga un **campo receptivo volumétrico masivo** capaz de integrar contexto espacial amplio para rescatar blastómeros tenues en planos $Z$ profundos sin inestabilidad matemática.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo `A1560-UBU`.

---

## ⚙️ 2. Configuración y Parámetros
- **Arquitectura:** `ConvNeXtUNet3D` con concatenación U-Net estándar en el decoder.
- **Dataset:** 159 volúmenes de entrenamiento (15,117 parches 3D) y 40 volúmenes de validación holdout (Esquema 80/20).
- **Optimizador:** AdamW con Cosine Annealing (20 épocas).

---

## 📊 3. Resultados de Validación (En Progreso)
*(Se actualizarán automáticamente ni bien concluyan las 20 épocas)*
