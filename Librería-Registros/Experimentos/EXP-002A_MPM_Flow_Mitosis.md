---
id: EXP-002A
tags: [experimento, mpm, multi_task, autopsia]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.0321
score_lb: null
estado: Finalizado
decision: DESCARTADO (Interferencia de gradientes multi-tarea)
---

# 🧪 EXP-002A: 3D Motion and Position Maps (MPM Multi-Task)

## 🎯 1. Hipótesis & Configuración
- **Hipótesis:** Entrenar una red multi-tarea que prediga simultáneamente el heatmap gaussiano de centroides, los vectores de desplazamiento 3D $\mathbf{v}_{t \to t+1}$ y la probabilidad de mitosis en un único decoder compartido.
- **Configuración:** `MPMUNet3D` con loss $\mathcal{L} = \mathcal{L}_{\text{heat}} + 0.3 \cdot \mathcal{L}_{\text{flow}} + 2.0 \cdot \mathcal{L}_{\text{div}}$, entrenado en 12,416 parches 3D en la RTX 4090 de `A1560-UBU`.

---

## 📊 2. Resultados Cuantitativos

| Métrica | EXP-001E (3D U-Net Single-Task) | **EXP-002A (MPM Multi-Task)** | Delta ($\Delta$) |
|---|:---:|:---:|:---:|
| **Node Recall** | **`0.9387`** | `0.1011` | $-83.7\%$ ❌ |
| **Edge Jaccard** | **`0.5876`** | `0.0302` | $-94.8\%$ ❌ |
| **Adjusted Edge Jaccard** | **`0.5810`** | `0.0321` | $-94.5\%$ ❌ |
| **CV Score Total** | **`0.5810`** | **`0.0321`** | $-0.5489$ |

---

## 🔍 3. Autopsia del Modo de Falla (Causa Raíz)

### 🔴 ¿Por qué falló el enfoque multi-tarea end-to-end?
1. **Interferencia de Gradientes Multi-Tarea:**
   - Los gradientes de regresión no acotada de la pérdida de flujo ($\mathcal{L}_{\text{flow}}$) y división sobrepasaron la señal de los heatmaps en el decoder compartido.
2. **Elevación del Ruido de Fondo:**
   - La red aprendió a predecir valores residuales de $\approx 0.02 - 0.04$ en vóxeles de fondo (que en la U-Net single-task eran $0.0001$).
   - Al evaluar con umbrales sensibles ($0.015$), `peak_local_max` extrajo **más de 218,000 falsos positivos por muestra** (inundando el matching húngaro y destruyendo la precisión).

---

## 💡 4. Aprendizaje Clave para la Estrategia Ganadora
- **Regla de Diseño:** **Desacoplar la Detección 3D del Tracking Temporal (Two-Stage Architecture)**.
- El detector 3D debe mantenerse como una red pura single-task de alta precisión (`AnisotropicUNet3D`, que ya demostró score `0.5810`).
- La asociación temporal y la detección de mitosis deben realizarse sobre las coordenadas y embeddings de los centroides ya detectados (mediante Transformers o Gap-Closing ILP), sin ensuciar la salida del detector volumétrico.
