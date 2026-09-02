---
tags: [experimento, full_volume, 90_10, wide_unet, gradient_checkpointing, descartado]
fecha: 2026-08-27
score_cv: 0.0000 (NaN en running_var)
edge_jaccard: 0.0000
node_recall: 0.00%
loss_final: 0.000089
estado: 🔴 Descartado (Autopsia Documentada)
---

# 🧪 EXP-026: 3D Wide Full-Volume Anisotropic U-Net (64ch) en Esquema 90/10

## 🎯 Objetivo e Hipótesis
Entrenar la arquitectura Wide Anisotropic U-Net (64 canales base) directamente sobre los **volúmenes 3D completos sin recortes** ($64 \times 256 \times 256$) utilizando el esquema ampliado 90/10 (179 volúmenes de train = 17,021 volúmenes 3D) y **Gradient Checkpointing** para eliminar las costuras de parches y optimizar la detección global.

---

## ⚙️ Configuración Experimental
* **Arquitectura:** `AnisotropicUNet3D` (64 base channels) con `BatchNorm3d` y `torch.utils.checkpoint.checkpoint` en codificador.
* **Tamaño de Entrada:** $1 \times 64 \times 256 \times 256$ (Volumen completo del embrión, `batch_size = 1`).
* **Esquema:** 90/10 Disjoint (179 volúmenes train / 20 volúmenes val).
* **Hardware:** Servidor `A1556-UBU` (GPU NVIDIA GeForce RTX 4090 de 24 GB).
* **VRAM Utilizada:** ~17.6 GB con Gradient Checkpointing.
* **Épocas:** 20 épocas completadas con cadencia de ~31.8 min por época.
* **Checkpoint:** `champion_weights/exp026_wide_full_volume_90_10.pt`.

---

## 📊 Resultados de Entrenamiento y Evaluación
* **Convergencia en Entrenamiento:** Récord mínimo histórico de pérdida: **`Train Loss = 0.000089`**.
* **Resultado en Inferencia (`model.eval()`):** **`0 nodes, 0 edges`** (Salida `NaN` en todas las predicciones).

---

## 🔬 Autopsia Técnica y Causa Raíz
1. **Subdesbordamiento Numérico en BatchNorm con `batch_size = 1`:**  
   Al procesar volúmenes masivos de $64 \times 256 \times 256$ con solo 1 elemento por batch y 98% de vóxeles de fondo (ceros), el cálculo de varianza muestral sufrió división por cero y subdesbordamiento aritmético en FP16 AMP.
2. **Divergencia entre `train()` y `eval()`:**  
   * En `model.train()`, PyTorch utilizó las estadísticas del batch actual y los gradientes fueron recortados (`clip_grad_norm_`), permitiendo que el entrenamiento avanzara hasta pérdida mínima.
   * En `model.eval()`, PyTorch conmuta a las estadísticas móviles acumuladas (`running_mean` y `running_var`), las cuales quedaron contaminadas con valores `NaN`.
   * En consecuencia, cualquier pase hacia adelante en modo de inferencia evaluó `NaN >= 0.035 = False`, produciendo matrices vacías.

---

## 💡 Decisión Estratégica y Regla de Oro
* **Regla Establecida:** Prohibido el uso de `BatchNorm3d` en modelos con `batch_size = 1` en volúmenes completos.
* **Alternativa Validada:** Usar **`GroupNorm3d` (8 grupos)** como en `EXP-029` ([unet3d_groupnorm.py](file:///Users/matias/Proyects/Biohub/src/models/unet3d_groupnorm.py)), el cual no depende de estadísticas inter-batch y es 100% inmune a este fallo.
