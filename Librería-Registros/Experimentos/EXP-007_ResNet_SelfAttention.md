---
id: EXP-007
tags: [experimento, resnet3d, self_attention, fallido, autopsia]
fecha: 2026-08-25
autor: Matias & Antigravity AI
score_cv: 0.0000
score_lb: null
split: "100% / 0% (199 vols)"
estado: Descartado
decision: DESCARTADO - REEMPLAZADO POR EXP-008 (SE-Channel Attention)
---

# 🧪 EXP-007: 3D ResNet-UNet con Bottleneck Spatial Self-Attention

## 🎯 1. Hipótesis & Estrategia
- **Hipótesis:** Introducir auto-atención espacial densa $N \times N$ en el cuello de botella de la 3D ResNet-UNet para capturar dependencias de largo alcance en el eje Z (profundidad del microscopio) y mejorar la detección en planos profundos tenues.
- **Configuración:** Arquitectura `ResNetUNet3D` con stem anisotrópico, bloques residuales y capa de Self-Attention densa en el bottleneck.

---

## 🔬 2. Resultados de la Evaluación
- **Score CV:** `0.0000` (Predicciones vacías / sub-umbral $\tau < 0.010$).
- **Node Recall:** `0.0000`
- **Edge Jaccard:** `0.0000`

---

## 🔍 3. Autopsia de la Causa Raíz
1. **Explosión Combinatoria Volumétrica:**
   En 2D, el grid espacial en el bottleneck es de $16 \times 16 = 256$ tokens. En 3D, el tensor $(8, 32, 32)$ genera **$N = 8,192$ tokens espaciales**.
2. **Inestabilidad Numérica en Softmax FP16:**
   Calcular $\text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right)$ sobre una matriz de $8192 \times 8192 = \mathbf{67.1 \text{ millones de elementos}}$ en precisión mixta causó un *underflow* severo.
3. **Anulación de Gradientes:**
   A partir de la época 2, la función de pérdida no pudo propagar gradientes útiles y los pesos del cabezal quedaron inactivados.

---

## 🏁 4. Lección Aprendida y Transferencia
- **Conclusión:** La auto-atención espacial densa $O(N^2)$ es inviable para tensores volumétricos 3D continuos.
- **Acción Inmediata:** Reemplazar por **Squeeze-and-Excitation (SE-Block 3D)** con atención a nivel de canales $O(C)$, implementado en **`EXP-008`**.
