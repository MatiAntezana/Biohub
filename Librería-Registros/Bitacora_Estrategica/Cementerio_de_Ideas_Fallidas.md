---
tags: [bitacora, cementerio, fallos, lecciones]
---

# 🪦 Cementerio de Ideas Fallidas & Lecciones Aprendidas

> [!WARNING]
> *"El que no conoce los errores del pasado está condenado a re-entrenarlos."*
> Esta sección documenta todas las hipótesis que dieron score negativo, sobreajustaron o fueron computacionalmente inviables, para evitar repetir caminos muertos.

---

## ⚰️ Registro de Ideas Descartadas

### 2. Multi-Task Simultaneous Flow & Mitosis Gradient Interference (EXP-002A)
- **Fecha:** 2026-08-25
- **Hipótesis:** Predecir conjuntamente en una sola U-Net: Heatmap de centroides, Campo de Vectores de Movimiento 3D y Probabilidad de Mitosis.
- **Resultado:** Score se desplomó de `0.5810` a `0.0321`.
- **Causa Raíz:** Interferencia de gradientes. Las pérdidas de mitosis y flujo distorsionaron las primeras capas convolucionales, elevando el ruido de fondo por encima de 0.03 y generando más de 200,000 picos falsos.
- **Solución / Decisión:** **Desacoplamiento Modular Estricto**. La detección 3D debe ser un modelo puro y especializado.

---

### 3. Dense Spatial Self-Attention en Volúmenes 3D (EXP-007)
- **Fecha:** 2026-08-25
- **Hipótesis:** Agregar matrices de auto-atención densa spatial $(N \times N)$ en el cuello de botella de la 3D ResNet para capturar contexto global en Z.
- **Resultado:** Inestabilidad numérica en FP16 AMP. El cuello de botella 3D $(8 \times 32 \times 32 = 8,192 \text{ tokens})$ generó matrices de $67.1 \text{ millones de elementos}$ que causaron *underflow* en el softmax.
- **Causa Raíz:** Complejidad cuadrática $O(N^2)$ inviable para tensores volumétricos 3D en precisión mixta.
- **Solución / Decisión:** Descartar capas de atención espacial densa en 3D.

---

### 6. ConvNeXt 3D con Large 7x7 Kernels y GELU en FP16 (EXP-012)
- **Fecha:** 2026-08-25
- **Hipótesis:** Utilizar bloques ConvNeXt 3D con depthwise $7 \times 7 \times 7$ y GELU para aumentar el campo receptivo volumétrico.
- **Resultado:** Desbordamiento numérico (NaNs en FP16 autocast) produciendo predicciones vacías.
- **Causa Raíz:** Las convoluciones 3D con kernel $7^3=343$ multiplicadas por la expansión $4\times$ de canales en inverted bottlenecks con GELU generan gradientes que exceden el rango dinámico de FP16 ($\pm 65,504$).
- **Solución / Decisión:** **Retener bloques convolucionales $3 \times 3 \times 3$ con ReLU y BatchNorm 3D**, que han demostrado estabilidad numérica absoluta y convergencia récord (como en EXP-009 con `CV 0.6903`).

### 5. Inferencia con GPU Tesla P100 para Submission Oficial
- **Fecha:** 2026-08-25
- **Problema:** El modal de envío oficial de Kaggle bloquea la submission con el mensaje: *"Your Notebook cannot use P100 GPUs in this competition."*
- **Causa Raíz:** Restricción explícita de hardware impuesta por los anfitriones del concurso Biohub (la P100 está inhabilitada para envíos al Leaderboard).
- **Solución / Decisión:** **Configurar explícitamente CPU Multi-Hilo (`enable_gpu: false`) o GPU Tesla T4x2 / TPU**. En CPU con nuestro motor optimizado, la inferencia completa corre en $\approx 3$ minutos y está 100% admitida sin límite de cuota.

### 7. Focal-Tversky Loss para Mapas de Calor Gaussianos Continuos (EXP-018)
- **Fecha:** 2026-08-26
- **Hipótesis:** Utilizar Focal-Tversky Loss para penalizar fuertemente las fronteras celulares y evitar la fusión de blastómeros contiguos.
- **Resultado:** Score colapsó a `0.0566` con una caída catastrófica de Node Recall al `40.47%`.
- **Causa Raíz:** Tversky/Dice loss penaliza las colas continuas gaussianas $e^{-r^2/2\sigma^2}$ como si fueran falsos positivos, destruyendo la curvatura de los picos celulares.
- **Solución / Decisión:** **Retener exclusivamente Focal MSE Loss Ponderada ($\mathcal{L} = (1 + 10 \cdot y)(p - y)^2$)**, que preserva la suavidad del gradiente y el 98% de sensibilidad.

### 8. BatchNorm3D en Volúmenes 3D Completos con Batch Size = 1 (EXP-021 / EXP-026)
- **Fecha:** 2026-08-27
- **Hipótesis:** Entrenar volúmenes masivos no recortados ($64 \times 256 \times 256$) con 64 canales usando `BatchNorm3d` y `batch_size = 1` con Gradient Checkpointing para optimizar la visión global sin costuras de parches.
- **Resultado:** En entrenamiento la pérdida cayó a récord de `0.000089`, pero en inferencia (`model.eval()`) la red predijo `NaN` en todos los vóxeles, generando `0 nodes, 0 edges`.
- **Causa Raíz:** Con `batch_size = 1` sobre volúmenes masivos donde el 98% son vóxeles de fondo (ceros), la estimación de varianza acumuló divisiones por cero en FP16, contaminando `running_mean` y `running_var` con `NaN`. En `eval()`, PyTorch activa estas estadísticas móviles dañadas y destruye las activaciones.
- **Solución / Decisión:** **Prohibir estrictamente `BatchNorm3d` en pipelines de volumen completo con `batch_size = 1`**. En su lugar:
  1. Para volúmenes completos usar **`GroupNorm3d` (8 grupos)** como en `EXP-029` (que no mantiene estadísticas móviles inter-batch y es 100% estable).
  2. Para `BatchNorm3d`, usar exclusivamente entrenamiento por parches ($64^3$) con `batch_size >= 8` (como en los campeones `EXP-017` con CV `0.9010` y `EXP-019` con CV `0.8867`).

---
*(Se irán agregando las ideas descartadas con su respectiva autopsia técnica).*
