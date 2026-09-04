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
