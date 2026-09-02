---
id: EXP-016
tags: [experimento, deep_supervision, multiscale_loss, split_80_20]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.6114
score_lb: null
split: "80% / 20% (159 Train / 40 Val vols)"
estado: Finalizado
decision: INTEGRAR AL POOL DE ARQUITECTURAS PARA EL ENSAMBLE (Score 0.6114)
---

# 🧪 EXP-016: 3D Anisotropic U-Net con Multi-Scale Deep Supervision (Esquema 80/20)

## 🎯 1. Hipótesis & Estrategia
- **Fundamento:** Incorporar cabezales auxiliares de supervisión profunda en las capas intermedias del decoder (`Dec3` y `Dec2`) además de la salida principal (`Dec1`):
  $$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{Main}} + 0.3 \mathcal{L}_{\text{Aux2}} + 0.1 \mathcal{L}_{\text{Aux3}}$$
- **Hardware:** NVIDIA GeForce RTX 4090 (24 GB VRAM) en nodo remoto `A1557-UBU`.

---

## 📊 2. Resultados de Validación Oficial (Holdout 20%)

| Métrica | Valor Oficial | Comentarios |
|---|:---:|---|
| **Score Total (Adjusted Edge Jaccard)** | **`0.6114`** | Puntuación sólida standalone |
| **Node Recall** | **`0.9107` (91.07%)** | Detección limpia de células |
| **Edge Jaccard** | **`0.6039`** | Consistencia de trayectoria temporal |
| **Division Penalties (FP)** | **`0`** | Política limpia de cero divisiones espurias |

---

## 🔍 3. Análisis Científico
1. **Velocidad y Estabilidad de Gradientes:**
   La supervisión profunda aceleró el tiempo de cómputo a solo 236 segundos por época sin degradación de gradientes.
2. **Rol en el Ensamble:**
   Al haber aprendido mapas de características a múltiples escalas espaciales (`Dec3`, `Dec2`), sus activaciones proveen una señal de probabilidad suave y complementaria a la de `EXP-011` y `EXP-015`.

---

## 🏁 4. Conclusión & Acción
- **Decisión:** **FINALIZADO E INCORPORADO AL POOL DE ENSAMBLE**.
- **Checkpoint:** `/home/alumno1/Documents/Biohub/checkpoints/exp016_deep_supervision_80_20.pt` guardado en `A1557-UBU`.
