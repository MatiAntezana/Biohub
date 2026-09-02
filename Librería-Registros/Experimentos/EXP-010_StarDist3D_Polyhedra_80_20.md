---
id: EXP-010
tags: [experimento, stardist3d, polyhedra, radial_rays, split_80_20, recall_97_8]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.6147
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: Finalizado
decision: INTEGRAR AL GRAN ENSAMBLE MULTIMODELO (Node Recall Récord 97.81%)
---

# 🧪 EXP-010: StarDist 3D Polyhedra Multi-Ray (Esquema 80/20)

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Implementar la arquitectura `StarDist3D` con cabezal de 32 rayos radiales sobre la esfera de Fibonacci, entrenada en el **Esquema 80% Train (159 volúmenes) / 20% Val (40 volúmenes)** sobre la RTX 4090 de `A1556-UBU`.
- **Especialidad:** Separar y despegar células en cúmulos densos donde las U-Nets clásicas tienden a fusionar centroides contiguos.

---

## 📊 2. Resultados de Validación Oficial (Holdout 20%)

| Umbral ($\tau$) | **Node Recall** | Edge Jaccard | Adjusted Edge Jaccard | **Score Total Oficial** | Comentarios |
|:---:|:---:|:---:|:---:|:---:|:---:|
| $\tau = 0.020$ | 🏆 **`0.9781` (97.81%)** | `0.5674` | `0.5665` | **`0.5665`** | **RÉCORD ABSOLUTO DE RECALL** |
| $\tau = 0.035$ | `0.9644` (96.44%) | `0.5978` | `0.6033` | **`0.6033`** | Balance intermedio |
| **$\tau = 0.050$** | `0.9534` (95.34%) | **`0.6054`** | **`0.6147`** | 🟢 **`0.6147`** | **Score Standalone Óptimo** |

---

## 🔍 3. Análisis Científico
1. **Récord Histórico de Node Recall (`97.81%`):**
   StarDist 3D logró la mayor sensibilidad de detección de células observada en el proyecto, detectando blastómeros en estadios densos de alta aglomeración.
2. **Rol en el Gran Ensamble Multimodelo:**
   Su combinación ponderada con `EXP-009` (`CV 0.6903`) aportará resolución de cúmulos densos sin sacrificar la precisión de tracking de la U-Net.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **APROBADO E INTEGRADO AL ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp010_stardist3d_80_20.pt` guardado en `A1556-UBU`.
