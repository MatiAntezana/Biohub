---
tags: [modulo, deteccion, segmentacion, 3d]
---

# 🔬 Módulo 02: Detección 3D y Segmentación de Células

## 🎯 Objetivo
Detectar las coordenadas $(z, y, x)$ de los centroides celulares en cada timepoint $t$ con la máxima precisión sub-vóxel posible y mínimo ratio de falsos positivos/negativos.

---

## 💡 Enfoques a Probar

### 1. 3D U-Net Heatmap Regression
- **Arquitectura:** U-Net volumétrica con encoder liviano (ej: ResNet3D-18 / ConvNeXt3D).
- **Target:** Heatmap continuo con gaussianas 3D anisotrópicas centradas en los ground-truth nodes:
  $$H(z,y,x) = \exp\left( -\left[ rac{(z-z_0)^2}{2\sigma_z^2} + rac{(y-y_0)^2}{2\sigma_y^2} + rac{(x-x_0)^2}{2\sigma_x^2} ight] ight)$$
- **Loss:** Focal Loss continua o Weighted MSE para compensar el desbalance extremo de fondo vs células.
- **Extracción de Picos:** `skimage.feature.peak_local_max` con umbral adaptativo.

### 2. StarDist 3D ([[Literatura_y_Referencias/Mapeo_Papers_a_Implementaciones#StarDist-3D|Ver Paper]])
- Predice probabilidad de objeto + distancias radiales a la superficie en direcciones poliedrales predefinidas.
- Excelente para células esféricas u ovoides.

### 3. MPM (Motion & Position Maps) ([[Literatura_y_Referencias/Mapeo_Papers_a_Implementaciones#MPM|Ver Paper]])
- Predice simultáneamente el mapa de probabilidad de centroide y el desplazamiento vectorial hacia el frame anterior y posterior.
