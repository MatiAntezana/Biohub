---
id: EXP-018
tags: [experimento, tversky_loss, fallido, autopsia, split_80_20]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.0566
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: Fallido (Autopsia Documentada)
decision: RECHAZADO (Tversky Loss Incompatible con Heatmaps Continuos)
---

# 🧪 EXP-018: 3D Boundary Focal-Tversky U-Net (Autopsia de Pérdida Continua)

## 🎯 1. Hipótesis & Estrategia
- **Hipótesis:** Utilizar Focal-Tversky Loss para penalizar fuertemente las fronteras celulares y evitar la fusión de blastómeros contiguos.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1557-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 20%)

| Métrica | Valor Obtenido | Estado |
|---|:---:|:---:|
| **Score Total (Adjusted Edge Jaccard)** | **`0.0566`** | 🔴 Fallido |
| **Node Recall** | **`40.47%`** | Caída masiva de detección |
| **Edge Jaccard** | **`0.0659`** | Ruptura de trayectorias |

---

## 🔍 3. Autopsia Científica & Causa Raíz
1. **Incompatibilidad Matemática con Heatmaps Gaussianos Continuos:**
   * Las pérdidas de tipo Tversky y Dice fueron diseñadas para **máscaras binarias $\{0, 1\}$**.
   * En nuestra tarea, el target es un **campo de calor Gaussiano continuo $[0.0, 1.0]$** con colas exponenciales $e^{-r^2 / 2\sigma^2}$.
   * Al evaluar $\text{FP} = \text{pred} \times (1 - \text{target})$, Tversky castiga las colas suaves del gradiente celular como si fueran falsos positivos, aplanando la curvatura de los picos celulares y provocando que el extractor de máximos locales (`peak_local_max`) no detecte el $60\%$ de las células.
2. **Lección Aprendida:**
   * Para predicción de centroides continuos en 3D, **Focal MSE Loss Ponderada ($\mathcal{L} = (1 + 10 \cdot y) (p - y)^2$) es matemáticamente superior**.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **RECHAZADO Y DOCUMENTADO EN EL CEMENTERIO DE IDEAS FALLIDAS**.
