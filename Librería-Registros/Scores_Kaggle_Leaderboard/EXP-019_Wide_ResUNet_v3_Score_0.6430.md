---
tags: [score_kaggle, exp019, leaderboard, resunet]
experimento: EXP-019
fecha_evaluacion: 2026-08-27
submission_ref: 55804613
script_version_id: 345225808
kernel_name: biohub-exp-019-wide-resunet-3d
kernel_version: v3
public_score: 0.6430
cv_score_local: 0.8867
acelerador: GPU Tesla T4
tiempo_inferencia: 1087.0s (~18.1 min)
predicciones_generadas: 201065
tamano_csv: 10.32 MB
---

# 🥈 Registro Oficial de Score en Kaggle: EXP-019 (Score `0.6430`)

## 📌 Datos de la Evaluación en Kaggle
* **Competencia:** `biohub-cell-tracking-during-development`
* **Submission Reference:** `55804613`
* **Cuaderno Kaggle:** `biohub-exp-019-wide-resunet-3d` (Versión 3)
* **Acelerador de Ejecución:** GPU Tesla T4 (`machine_shape: NvidiaTeslaT4`)
* **Estado de la Evaluación:** 🟢 `SubmissionStatus.COMPLETE`
* **🏆 Public Score Oficial en Leaderboard:** **`0.6430`**

---

## 🧪 Experimento Utilizado
* **ID de Experimento:** [[Experimentos/EXP-019_Wide_ResUNet3D_90_10|EXP-019]]
* **Modelo / Arquitectura:** `3D Wide ResUNet` (Conexiones Residuales 64ch base).
* **Pesos del Modelo:** `champion_weights/exp019_wide_resunet3d_90_10.pt` (95 MB).
* **Esquema de Entrenamiento:** Split 90/10 Disjoint (179 volúmenes de train, 20 volúmenes de val).
* **Validación Local (CV):** `0.8867`

---

## 📈 Análisis de Rendimiento y Comparativa
* **Delta vs Baseline:** $+0.5000$ ($4.50\times$ superior a `0.1430`).
* **Comparativa vs EXP-017:** `EXP-017` puro (`0.6830`) supera a `EXP-019` (`0.6430`) por $+0.0400$, confirmando que la U-Net Anisotrópica clásica retiene bordes celulares ligeramente más nítidos que la variante con bloques residuales densos, pero ambos modelos son excelentes candidatos complementarios para ensamble.
