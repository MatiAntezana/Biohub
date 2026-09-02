---
id: EXP-000
tags: [experimento, baseline, nearest_neighbor]
fecha: 2026-08-25
autor: Kaggle Staff (inversion)
score_cv: 0.1430
score_lb: 0.143
estado: Finalizado
decision: BASELINE DE REFERENCIA
---

# 🧪 EXP-000: Baseline Heurístico con Nearest Neighbor (Hungarian LAP)

## 🎯 1. Hipótesis & Objetivo
Establecer un pipeline mínimo funcional de extremo a extremo que valide la lectura de Zarr v3, segmentación clásica simple, cálculo de centroides, enlace temporal básico y exportación al formato exigido por la competencia.

---

## ⚙️ 2. Configuración del Experimento
- **Preprocesamiento:** Downsampling volumétrico $	imes 4$ en $Z, Y, X$.
- **Filtrado:** `scipy.ndimage.uniform_filter(size=3)`.
- **Segmentación:** Umbralización por percentil 90% + etiquetado de componentes conexas (`scipy.ndimage.label`).
- **Centroides:** Promedio de coordenadas de cada componente reescalado por factor $	imes 4$.
- **Asociación Temporal:** Hungarian LAP (`scipy.optimize.linear_sum_assignment`) sobre distancia euclidiana en escala física ($Z=1.625, Y=X=0.40625\,\mu	ext{m}$) con cota $\le 15.0\,\mu	ext{m}$.
- **Mitosis:** No implementada (0 divisiones predichas).

---

## 📊 3. Resultados Obtenidos
- **Edge Jaccard:** `0.1430`
- **Division Jaccard:** `0.0000`
- **Score Total:** **`0.1430`**
- **Public LB Score:** `0.143`
- **Tiempo de Inferencia:** $pprox 1.5	ext{ minutos}$ en CPU.

---

## 🔍 4. Análisis de Fallas y Puntos Fuertes

### ✅ Lo que funcionó bien:
- Pipeline liviano, sin dependencias complejas de GPU.
- Correcto formateo de `submission.csv` con nodos y aristas indexadas.
- La escala física en micrones funciona correctamente para el matching húngaro.

### ❌ Lo que falló / Limitaciones:
1. **Downsampling excesivo ($	imes 4$):** Fusiona células densas vecinas en un solo centroide promedio, perdiendo detecciones críticas.
2. **Umbral global por percentil:** Muy sensible al ruido de fondo variable entre timepoints.
3. **Sin detección de mitosis:** Pierde automáticamente todo el término $0.1 	imes 	ext{division\_jaccard}$.
4. **Sin persistencia temporal:** No maneja oclusiones, gaps temporales ni bifurcaciones.

---

## 🚀 5. Próximos Pasos Derivados
- Reemplazar la segmentación clásica con un detector 3D basado en Deep Learning ([[Modulos_Tecnicos/02_Deteccion_3D_y_Segmentacion|3D U-Net]]).
- Implementar extractor explícito de eventos de mitosis.
