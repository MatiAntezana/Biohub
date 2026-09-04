# 🧠 MEMORIA CIENTÍFICA DE FALLOS Y APRENDIZAJES (MEMORY_FAILURES.md)

> Este archivo registra de forma acumulativa todas las hipótesis que no superaron el benchmark, junto con su análisis post-mortem y la causa biológica/computacional de su fallo.
> **Regla Invariante**: Antes de proponer una nueva mutación en `core/graph_builder.py`, el agente debe revisar este archivo para no reincidir en estrategias desmentidas.

---

## Heurísticas y Anti-Patrones Extraídos

1. **Anti-patrón 1: Umbrales de confianza excesivamente permisivos (`CONF_THRESHOLD < 0.30`)**:
   - Causa: Genera una avalancha de picos de ruido que sobrecargan el paso de asignación lineal (LAP) y provocan `wrong_association_edges` espurios.
2. **Anti-patrón 2: `SAFE_DIV_FRAME_FRAC_CAP` desmedido (> 0.05)**:
   - Causa: En un embrión real en desarrollo, solo una fracción muy pequeña (1-2%) de las células entra en telofase/mitosis por frame temporal. Elevar el cap genera bifurcaciones artificiales donde dos células migrantes cercanas son erróneamente etiquetadas como hermanas, aumentando `div_fp` y destruyendo `division_jaccard`.
3. **Anti-patrón 3: Ignorar la anisotropía en Z**:
   - Causa: Si la distancia en Z no se pondera con factor $\ge 4.0$, el linker salta indiscriminadamente entre cortes focales contiguos en vez de seguir el flujo laminar XY natural de la capa celular.

---

## Historial de Experimentos Fallidos

*(Nuevos fallos se agregarán automáticamente aquí por `fast_loop/orchestrator.py`)*

### ❌ Run FAST-002 (2026-09-04 18:57:35 UTC)
- **Hipótesis**: Integration test hypothesis
- **Sub-métrica objetivo**: `total_score`
- **Resultado**: Score `0.8380` vs Mejor `0.8380` (Delta: `+0.0000`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: 0.7978 (Unadjusted: 0.7978)
  - `Node Penalty Multiplier`: 1.0 (T_pred: 2728 / T_true: 2871)
  - `Division Jaccard`: 0.4028 (TP: 29 | FP: 0 | FN: 43)
  - `Fragmented Edges`: 106 | `Wrong Associations`: 197
- **Post-Mortem / Aprendizaje**:
  Hypothesis 'Integration test hypothesis' resulted in score 0.8380 (delta: 0.0000). Reverting.

---

### ❌ Run FAST-003 (2026-09-04 18:57:57 UTC)
- **Hipótesis**: Initial verified baseline
- **Sub-métrica objetivo**: `total_score`
- **Resultado**: Score `0.8380` vs Mejor `0.8380` (Delta: `+0.0000`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: 0.7978 (Unadjusted: 0.7978)
  - `Node Penalty Multiplier`: 1.0 (T_pred: 2728 / T_true: 2871)
  - `Division Jaccard`: 0.4028 (TP: 29 | FP: 0 | FN: 43)
  - `Fragmented Edges`: 106 | `Wrong Associations`: 197
- **Post-Mortem / Aprendizaje**:
  Hypothesis 'Initial verified baseline' resulted in score 0.8380 (delta: 0.0000). Reverting.

---

### ❌ Run FAST-005 (2026-09-04 19:06:43 UTC)
- **Hipótesis**: Verify default remote cluster execution
- **Sub-métrica objetivo**: `total_score`
- **Resultado**: Score `0.8598` vs Mejor `0.8598` (Delta: `+0.0000`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: 0.8015 (Unadjusted: 0.8015)
  - `Node Penalty Multiplier`: 1.0 (T_pred: 2728 / T_true: 2871)
  - `Division Jaccard`: 0.5833 (TP: 42 | FP: 0 | FN: 30)
  - `Fragmented Edges`: 94 | `Wrong Associations`: 198
- **Post-Mortem / Aprendizaje**:
  Hypothesis 'Verify default remote cluster execution' resulted in score 0.8598 (delta: 0.0000). Reverting.

---

### ❌ Run FAST-006 (2026-09-04 19:07:10 UTC)
- **Hipótesis**: Integration test hypothesis
- **Sub-métrica objetivo**: `total_score`
- **Resultado**: Score `0.8598` vs Mejor `0.8598` (Delta: `+0.0000`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: 0.8015 (Unadjusted: 0.8015)
  - `Node Penalty Multiplier`: 1.0 (T_pred: 2728 / T_true: 2871)
  - `Division Jaccard`: 0.5833 (TP: 42 | FP: 0 | FN: 30)
  - `Fragmented Edges`: 94 | `Wrong Associations`: 198
- **Post-Mortem / Aprendizaje**:
  Hypothesis 'Integration test hypothesis' resulted in score 0.8598 (delta: 0.0000). Reverting.

---

### ❌ Run FAST-007 (2026-09-04 19:11:19 UTC)
- **Hipótesis**: Verify exclusive remote execution on A1557
- **Sub-métrica objetivo**: `total_score`
- **Resultado**: Score `0.0000` vs Mejor `0.8598` (Delta: `-0.8598`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: None (Unadjusted: None)
  - `Node Penalty Multiplier`: None (T_pred: None / T_true: None)
  - `Division Jaccard`: None (TP: None | FP: None | FN: None)
  - `Fragmented Edges`: None | `Wrong Associations`: None
- **Post-Mortem / Aprendizaje**:
  Hypothesis 'Verify exclusive remote execution on A1557' resulted in score 0.0000 (delta: -0.8598). Reverting.

---

### ❌ Run FAST-008 (2026-09-04 19:11:34 UTC)
- **Hipótesis**: Verify exclusive remote execution on A1557
- **Sub-métrica objetivo**: `total_score`
- **Resultado**: Score `0.8598` vs Mejor `0.8598` (Delta: `+0.0000`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: 0.8015 (Unadjusted: 0.8015)
  - `Node Penalty Multiplier`: 1.0 (T_pred: 2728 / T_true: 2871)
  - `Division Jaccard`: 0.5833 (TP: 42 | FP: 0 | FN: 30)
  - `Fragmented Edges`: 94 | `Wrong Associations`: 198
- **Post-Mortem / Aprendizaje**:
  Hypothesis 'Verify exclusive remote execution on A1557' resulted in score 0.8598 (delta: 0.0000). Reverting.

---

### ❌ Run REAL-003 (2026-09-04 19:37:13 UTC)
- **Hipótesis**: Raise CONF_THRESHOLD from 0.42 to 0.70 to eliminate low-confidence background noise
- **Sub-métrica objetivo**: `adj_edge_jaccard`
- **Resultado**: Score `0.7383` vs Mejor `0.7383` (Delta: `+0.0000`)
- **Telemetría**:
  - `Adjusted Edge Jaccard`: 0.7383 (Unadjusted: 0.7383)
  - `Node Penalty Multiplier`: 1.0 (T_pred: 55711 / T_true: 58700)
  - `Division Jaccard`: 0.0 (TP: 0 | FP: 0 | FN: 1)
  - `Fragmented Edges`: 69 | `Wrong Associations`: 0
- **Post-Mortem / Aprendizaje**:
  Hypothesis 'Raise CONF_THRESHOLD from 0.42 to 0.70 to eliminate low-confidence background noise' resulted in score 0.7383 (delta: 0.0000). Reverting.

---
