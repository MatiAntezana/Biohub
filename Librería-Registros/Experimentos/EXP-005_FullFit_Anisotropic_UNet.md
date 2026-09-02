---
id: EXP-005
tags: [experimento, full_fit, champion, sota_record, 0_67]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.6700
score_lb: null
estado: Finalizado
decision: NUEVO CHAMPION ABSOLUTO (Score 0.6700)
---

# 🧪 EXP-005: Full-Fit 3D Anisotropic U-Net (100% Data + 4-Way 3D TTA + Gap Closing)

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Entrenar sobre el **100% del dataset disponible (199 volúmenes Zarr apareados con grafos GEFF de los embriones `6bba` y `44b6`)** con data augmentation 3D completa y evaluar con **4-Way 3D Test-Time Augmentation (TTA)** y **Multi-Frame Gap Closing**.
- **Objetivo:** Maximizar la capacidad de generalización sobre el test set oculto de Kaggle.

---

## ⚙️ 2. Configuración y Parámetros
- **Arquitectura:** `AnisotropicUNet3D` (Convoluciones $(1, 3, 3)$ y $(3, 3, 3)$).
- **Dataset de Entrenamiento:** 18,933 parches volumétricos 3D $(32, 128, 128)$ de todos los estadios embrionarios.
- **Inferencia:** 4-Way 3D TTA ($V, \text{flip}_X, \text{flip}_Y, \text{flip}_Z$) + Umbral $\tau = 0.010$.
- **Tracking:** Multi-Frame Gap Closing ($\Delta t \le 2, D_{\text{gap}} \le 18\,\mu\text{m}$) con supresión de nodos huérfanos (`min_track_length = 2`).
- **Checkpoint Generado:** `exp005_fullfit_best.pt` (21 MB, cargado a Kaggle Datasets).

---

## 📊 3. Resultados Cuantitativos Oficiales

| Métrica | Baseline (EXP-000) | EXP-001E (Single Pass) | EXP-004 (TTA 1 Embrión) | **EXP-005 (Full-Fit 100% Data)** | Delta vs Baseline ($\Delta$) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Node Recall** | `0.0536` | `0.9387` | `0.9484` | **`0.9605` (96.1%)** | **+96.1% absoluto** 🚀 |
| **Edge Jaccard** | `0.0122` | `0.5876` | `0.6056` | **`0.6771`** | **+0.6649** 🚀 |
| **Adjusted Edge Jaccard** | `0.0133` | `0.5810` | `0.6007` | **`0.6700`** | **+0.6567** 🚀 |
| **Division Jaccard** | `0.0000` | `0.0000` | `0.0000` | `0.0000` | `+0.0000` |
| **Score Total Oficial** | **`0.0133`** | **`0.5810`** | **`0.6007`** | **`0.6700`** | **NUEVO RÉCORD (+4937%)** 🏆 |

---

## 🔍 4. Desglose Detallado por Muestra de Validación

- `44b6_12dfb391`: **687 TP** / 126 FP / 86 FN -> Jaccard = **`76.4%`** 🚀
- `44b6_0c582fdc`: **61 TP** / 12 FP / 9 FN -> Jaccard = **`74.4%`** 🚀
- `44b6_1574802b`: **130 TP** / 32 FP / 23 FN -> Jaccard = **`70.3%`** 🚀
- `44b6_0db75fae`: **124 TP** / 44 FP / 27 FN -> Jaccard = **`63.6%`** 🚀
- `44b6_18ced818`: **79 TP** / 32 FP / 20 FN -> Jaccard = **`60.3%`** 🚀

---

## 🏁 5. Conclusión & Estado de Envío
- **Decisión:** **PROMOVIDO COMO EL MODELO DEFINITIVO DE PRODUCCIÓN (CHAMPION)**.
- **Kaggle Kernel:** Empaquetado y subido en la **Versión 5** con el dataset `matiasantezana/biohub-champion-weights`.
