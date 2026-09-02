---
id: EXP-004
tags: [experimento, tta, gap_closing, sota, nuevo_record]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.6007
score_lb: null
estado: Finalizado
decision: NUEVO SOTA RÉCORD (Rompe la barrera de 0.60)
---

# 🧪 EXP-004: 3D Anisotropic U-Net + 4-Way 3D TTA + Multi-Frame Gap Closing

## 🎯 1. Hipótesis & Fundamento Físico-Biológico
- **Hipótesis:** 
  1. En microscopía de fluorescencia 3D, el tejido embrionario posee simetrías espaciales en $X, Y, Z$. Promediar las predicciones de la U-Net 3D evaluando el volumen original y sus 3 reflexiones espaciales (4-Way 3D TTA) suprime artefactos de frontera y ruido espurio, agudizando la localización sub-vóxel de los centroides.
  2. El motor de **Multi-Frame Gap-Closing** ($\Delta t \le 2$) repara trayectorias rotas por fluorescencia tenue temporal y poda falsos positivos de longitud 1.
- **Módulos:** [[Modulos_Tecnicos/02_Deteccion_3D_y_Segmentacion|Detección 3D]] + [[Modulos_Tecnicos/03_Asociacion_Temporal_Tracking|Tracking]]

---

## ⚙️ 2. Configuración y Parámetros
- **Detector:** `AnisotropicUNet3D` con pesos `exp001a_unet3d_best.pt`.
- **Inferencia 3D TTA:**
  $$H_{\text{TTA}} = \frac{1}{4} \left[ H(V) + \text{flip}_X(H(\text{flip}_X(V))) + \text{flip}_Y(H(\text{flip}_Y(V))) + \text{flip}_Z(H(\text{flip}_Z(V))) \right]$$
- **Umbral de Picos:** $\tau = 0.010$.
- **Gap-Closing:** $\Delta t \le 2$ con $D_{\text{gap}} \le 18.0\,\mu\text{m}$ y poda de huérfanos (`min_track_length = 2`).
- **Split:** Embryo-disjoint estricto (Validado en embrión `44b6`).

---

## 📊 3. Resultados Cuantitativos

| Métrica | Baseline (EXP-000) | EXP-001E (Single Pass) | **EXP-004 (4-Way 3D TTA + Gap Closing)** | Delta ($\Delta$) |
|---|:---:|:---:|:---:|:---:|
| **Node Recall** | `0.0536` | `0.9387` | **`0.9484`** | **+94.8% absoluto** 🚀 |
| **Edge Jaccard** | `0.0122` | `0.5876` | **`0.6056`** | **+0.0180** 🚀 |
| **Adjusted Edge Jaccard** | `0.0133` | `0.5810` | **`0.6007`** | **+0.0197** 🚀 |
| **Division Jaccard** | `0.0000` | `0.0000` | `0.0000` | `+0.0000` |
| **Score Total Oficial** | **`0.0133`** | **`0.5810`** | **`0.6007`** | **NUEVO RÉCORD SOTA (+4416%)** |

---

## 🔍 4. Análisis Detallado por Muestra

- `44b6_12dfb391`: **652 TP** / 167 FP / 121 FN (Jaccard = **`69.4%`**).
- `44b6_0db75fae`: **115 TP** / 58 FP / 36 FN (Jaccard = **`55.0%`**).
- `44b6_18ced818`: **75 TP** / 35 FP / 24 FN (Jaccard = **`56.0%`**).
- `44b6_0c582fdc`: **53 TP** / 26 FP / 17 FN (Jaccard = **`55.2%`**).

---

## 🏁 5. Conclusión & Próximo Salto
- **Decisión:** **PROMOVIDO COMO EL NUEVO PIPELINE CHAMPION**.
- **Próximo Experimento (`EXP-005`):** Entrenar el modelo en el **100% de los datos (Full Fit sobre `6bba` + `44b6` con 199 volúmenes)** para maximizar la generalización y transferir esta configuración de 3D TTA a la submission de Kaggle.
