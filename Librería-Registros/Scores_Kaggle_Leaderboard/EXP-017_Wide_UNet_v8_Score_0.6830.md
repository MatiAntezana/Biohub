---
tags: [score_kaggle, exp017, leaderboard, champion]
experimento: EXP-017
fecha_evaluacion: 2026-08-27
submission_ref: 55804607
script_version_id: 345224829
kernel_name: biohub-exp-017-champion-090
kernel_version: v8
public_score: 0.6830
cv_score_local: 0.9010
acelerador: GPU Tesla T4
tiempo_inferencia: 903.9s (~15.0 min)
predicciones_generadas: 185017
tamano_csv: 9.47 MB
---

# 🥇 Registro Oficial de Score en Kaggle: EXP-017 (Score `0.6830`)

## 📌 Datos de la Evaluación en Kaggle
* **Competencia:** `biohub-cell-tracking-during-development`
* **Submission Reference:** `55804607`
* **Cuaderno Kaggle:** `biohub-exp-017-champion-090` (Versión 8)
* **Acelerador de Ejecución:** GPU Tesla T4 (`machine_shape: NvidiaTeslaT4`)
* **Estado de la Evaluación:** 🟢 `SubmissionStatus.COMPLETE`
* **🏆 Public Score Oficial en Leaderboard:** **`0.6830`**

---

## 🧪 Experimento Utilizado
* **ID de Experimento:** [[Experimentos/EXP-017_Wide_UNet3D_90_10|EXP-017]]
* **Modelo / Arquitectura:** `3D Wide Anisotropic U-Net` (64 canales base).
* **Pesos del Modelo:** `champion_weights/exp017_wide_unet3d_90_10.pt` (85 MB).
* **Esquema de Entrenamiento:** Split 90/10 Disjoint (179 volúmenes de train, 20 volúmenes de val).
* **Validación Local (CV):** `0.9010`

---

## 📈 Análisis de Rendimiento y Comparativa
* **Delta vs Baseline:** $+0.5400$ ($4.78\times$ superior a `0.1430`).
* **Estabilidad:** 185,017 filas de nodos y aristas espacio-temporales 3D generadas en 15 minutos en Kaggle sin advertencias de memoria ni violaciones de formato.
