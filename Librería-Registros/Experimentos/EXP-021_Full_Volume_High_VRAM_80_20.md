---
id: EXP-021
tags: [experimento, full_volume, high_vram, no_cropping, split_80_20]
fecha: 2026-08-26
autor: Matias & Antigravity AI
score_cv: null
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: Preparado para Lanzamiento
decision: RAMA INDEPENDIENTE DE ALTA VRAM (Volumen Completo 64x256x256)
---

# 🧪 EXP-021: 3D Anisotropic U-Net sobre Volumen Completo (64, 256, 256) Sin Recortes

## 🎯 1. Hipótesis & Estrategia
- **Fundamento Biológico y Matemático:** Los parches convencionales de $(32, 128, 128)$ solo capturan el $12.5\%$ del embrión, cortando células en los bordes y generando ruido en la detección de centroides.
- **Innovación High-VRAM:** Entrenar directamente con el **volumen completo del embrión $(64, 256, 256)$**, utilizando entre $14$ y $18\text{ GiB}$ de VRAM en la RTX 4090 con `batch_size=2` y precisión mixta FP16.
- **Estatus Estratégico:** Tratada como una **variante independiente** que se subirá a Kaggle para medir su impacto en el Leaderboard sin bloquear las demás líneas de investigación arquitectónica.

---

## ⚙️ 2. Configuración y Parámetros
- **Arquitectura:** `AnisotropicUNet3D` (Full-Volume Receptive Field).
- **Entrada:** Tensores 3D completos $(1, 64, 256, 256)$ sin crop espacial.
- **Dataset:** `FullVolumeBiohubDataset` (159 Train vols / 40 Val vols).
- **Optimizador:** AdamW con Cosine Annealing (20 épocas).

---

## 📊 3. Resultados de Validación (En Espera de Lanzamiento)
*(Se completarán automáticamente en cuanto concluya su entrenamiento y evaluación en holdout)*
