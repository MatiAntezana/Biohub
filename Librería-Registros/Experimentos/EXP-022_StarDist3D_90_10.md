---
id: EXP-022
tags: [experimento, stardist3d, polyhedra_rays, split_90_10, recall_record_99_2, score_0_8601]
fecha: 2026-08-26
autor: Matias & Antigravity AI
score_cv: 0.8601
score_lb: null
split: "90% / 10% (179 Train / 20 Val vols)"
estado: Finalizado
decision: INTEGRAR AL POOL DE MODELOS PRINCIPALES DE ENSAMBLE (Score 0.8601 / Node Recall Récord 99.23%)
---

# 🧪 EXP-022: StarDist 3D Polyhedra (32 Rayos) en Esquema 90/10 - RÉCORD DE RECALL

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Extender el modelado de **distancias polihédricas radiales** sobre 32 direcciones de Fibonacci al dataset expandido de **179 volúmenes de entrenamiento (17,021 parches 3D)** para maximizar la separación de células adyacentes.
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1560-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 10%)

| Umbral ($\tau$) | **Node Recall** | Edge Jaccard | Adjusted Edge Jaccard | **Score Total Oficial** | Estado |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.010$ | 🏆 **`0.9923` (99.23%)** | `0.8336` | `0.7958` | `0.7958` | 🥇 **RÉCORD ABSOLUTO DE SENSIBILIDAD DEL PROYECTO (99.23%)** |
| $\tau = 0.020$ | **`0.9878` (98.78%)** | `0.8543` | `0.8448` | `0.8448` | Gran equilibrio |
| **$\tau = 0.035$** | **`0.9817` (98.17%)** | **`0.8525`** | **`0.8601`** | 🟢 **`0.8601`** | **Score Standalone Óptimo (0.8601)** |

---

## 🔍 3. Análisis Científico
1. **Sensibilidad Récord del 99.23%:**
   * La parametrización de 32 rayos polihédricos sobre 179 volúmenes de entrenamiento permitió a la red detectar prácticamente la totalidad de las células en el volumen tridimensional.
2. **Rol en el Gran Ensamble Multimodelo:**
   * La combinación de **`EXP-017` (Wide U-Net 64ch - Score 0.9010)** con **`EXP-022` (StarDist 3D - Recall 99.23% / Score 0.8601)** conforma una dupla de modelos de máxima complementariedad y potencia para la inferencia final.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **APROBADO CON HONORES E INTEGRADO AL NÚCLEO DURO DEL ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp022_stardist3d_90_10.pt` guardado en `A1560-UBU`.
