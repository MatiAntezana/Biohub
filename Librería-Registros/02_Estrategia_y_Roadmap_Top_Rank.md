---
tags: [biohub, roadmap, estrategia]
updated: 2026-08-25
---

# 🚀 Estrategia Maestra y Roadmap para Top Rank

Para ganar una competencia de tracking 3D biológico a gran escala como **Biohub**, la solución no se basa en un único modelo pesado, sino en la integración sinérgica de **3 pilares**:

```
[ Volúmenes 3D+t Zarr ] 
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  PILAR 1: Detección 3D Sub-Vóxel de Alta Precisión    │
│  (Heatmaps 3D U-Net / StarDist 3D / Cellpose)          │
└────────────────────────────────────────────────────────┘
          │  Centroides (t, z, y, x) con Score
          ▼
┌────────────────────────────────────────────────────────┐
│  PILAR 2: Asociación Temporal & Predicción de Mitosis  │
│  (MPM Flow Vectors / Trackastra / LapTrack Feature)    │
└────────────────────────────────────────────────────────┘
          │  Aristas candidatas y probabilidades de split
          ▼
┌────────────────────────────────────────────────────────┐
│  PILAR 3: Optimización Global del DAG & Post-Proc      │
│  (ILP con Motile / Ultrack + Pruning de ramas muertas) │
└────────────────────────────────────────────────────────┘
          │
          ▼
[ submission.csv (Nodes + Edges Formateados) ]
```

---

## 📅 Roadmap por Fases

### Fase 1: Infraestructura y Validación Local (Días 1-3)
- [x] Extracción y comprensión de especificaciones (Zarr v3, escala anisotrópica $Z=1.625, Y=X=0.40625$).
- [x] Descarga y catalogación de literatura SOTA oficial.
- [ ] Implementación de la métrica oficial exacta en Python local (`metrics.py` que calcula `adjusted_edge_jaccard + 0.1 * division_jaccard`).
- [ ] Construcción del pipeline de Cross-Validation GroupKFold por `embryo_id`.

### Fase 2: Detectores 3D de Centroides (Días 4-10)
- [ ] **Baseline DL 1:** 3D U-Net liviano entrenado para predecir heatmaps gaussianos de centroides con pérdida MSE / Focal Loss ponderada.
- [ ] **Baseline DL 2:** StarDist 3D (regresión de distancias poliedrales para segmentar formas convexas).
- [ ] Manejo de la anisotropía en Z mediante convolutions anisotrópicas o remuestreo isótropo.
- [ ] Extracción de centroides locales (`skimage.feature.peak_local_max`).

### Fase 3: Asociación Temporal y Detección de Mitosis (Días 11-20)
- [ ] Enlace temporal con distancias euclidianas escaladas físicamente + LAP.
- [ ] Incorporación de vectores de flujo/movimiento (MPM - *Hayashida et al.*).
- [ ] Implementación de detector explícito de mitosis (1 nodo en $t 	o$ 2 nodos en $t+1$).
- [ ] Evaluación de Trackastra (Transformers de grafos temporales).

### Fase 4: Optimización Global por ILP (Días 21-30)
- [ ] Formulación de flujo de costos con `motile` o `ultrack`:
  - Costo de aparición de nodo (nacimiento / entrada al FoV).
  - Costo de desaparición de nodo (muerte / salida del FoV).
  - Costo de enlace $1 	o 1$.
  - Costo de división $1 	o 2$ (mitosis).
- [ ] Resolución óptima exacta mediante solucionadores ILP (SCIP / Gurobi / HiGHS).

### Fase 5: Ensembling, TTA y Postprocesamiento (Días 31-final)
- [ ] Test-Time Augmentation (flips 3D en X, Y, Z).
- [ ] Fusión de detecciones de múltiples modelos (Non-Maximum Suppression 3D ponderado).
- [ ] Poda de componentes conexas huérfanas o trayectorias de un solo frame.
- [ ] Verificación de submission contra el pipeline de inferencia dentro del límite de 12 horas.
