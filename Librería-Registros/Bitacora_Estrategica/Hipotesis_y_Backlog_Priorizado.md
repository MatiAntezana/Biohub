---
tags: [bitacora, backlog, ideas]
---

# 💡 Backlog Priorizado de Hipótesis y Experimentos

A continuación se listan las ideas de alto potencial clasificadas por Retorno de Inversión (ROI) estimado para el score final.

---

## 🔥 Prioridad Alta (Impacto Inmediato en CV)
- [ ] **HIP-01: Detector 3D U-Net de Heatmaps con Loss Focal:** Reemplazar el percentil 90% con una red neuronal 3D entrenada con las anotaciones de centroides `.geff`.
- [ ] **HIP-02: Algoritmo de Detección de Mitosis por Reglas Geométricas:** Añadir a cualquier tracker una rutina que detecte pares de células hijas en $t+1$ que divergen simétricamente desde un padre en $t$ ($+0.05$ a $+0.10$ score directo).
- [ ] **HIP-03: Remuestreo Isótropo vs Convoluciones Anisotrópicas:** Evaluar si interpolar el eje Z a $0.40625\,\mu	ext{m}$ mejora la detección en comparación con trabajar directamente en $1.625\,\mu	ext{m}$.

---

## ⚡ Prioridad Media (Refinamiento y Asociación SOTA)
- [ ] **HIP-04: Implementar Motile (ILP Solver):** Reemplazar Hungarian matching voraz frame-a-frame por un grafo global resuelto con ILP (`motile` / `ultrack`).
- [ ] **HIP-05: Vectores de Movimiento MPM (Hayashida et al.):** Entrenar el modelo 3D para predecir no solo posición sino también el vector de velocidad $(v_z, v_y, v_x)$ hacia $t-1$ y $t+1$.
- [ ] **HIP-06: Filtrado de Ramas Huérfanas:** Eliminar cualquier trayectoria que dure menos de 3 frames continuos (reducción drástica de falsos positivos en Edge Jaccard).

---

## 🧪 Prioridad Baja / Ensembling (Fase Final)
- [ ] **HIP-07: Test-Time Augmentation (TTA):** Promediar predicciones con 8 variantes de volteo 3D ($X, Y, Z$).
- [ ] **HIP-08: Ensembling StarDist 3D + 3D U-Net:** Fusionar centroides mediante Weighted Box/Point Fusion 3D.
