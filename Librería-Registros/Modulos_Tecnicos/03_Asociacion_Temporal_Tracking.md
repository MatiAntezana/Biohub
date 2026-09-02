---
tags: [modulo, tracking, asociacion]
---

# 🔗 Módulo 03: Asociación Temporal y Tracking de Células

## 🎯 Objetivo
Unir los centroides detectados en $t$ con sus correspondientes en $t+1$ (aristas $1 	o 1$), respetando límites biológicos de migración celular.

---

## 💡 Estrategias de Asociación

### 1. Hungarian Matching con Distancia Física Escalada (Baseline)
- Matriz de costos: Distancia Euclidiana en micrones:
  $$D_{ij} = \sqrt{(1.625 \cdot \Delta z)^2 + (0.40625 \cdot \Delta y)^2 + (0.40625 \cdot \Delta x)^2}$$
- `scipy.optimize.linear_sum_assignment(D)` con cota máxima $D_{ij} \le D_{\max} pprox 10	ext{--}15\,\mu	ext{m}$.

### 2. Matching Asistido por Vectores de Flujo (MPM)
- En lugar de buscar en vecindad isotrópica, desplazar la posición esperada en $t+1$:
  $$\hat{P}_{t+1} = P_t + \mathbf{v}_{	ext{forward}}(P_t)$$
- Minimizar $\|\hat{P}_{t+1} - P_{t+1}\|$.

### 3. Trackastra / Graph Transformers ([[Literatura_y_Referencias/Mapeo_Papers_a_Implementaciones#Trackastra|Ver Paper]])
- Construir un grafo espacial en cada ventana de frames $[t-k, t+k]$ y usar Transformers para predecir probabilidades de aristas $P(e_{u 	o v})$.
