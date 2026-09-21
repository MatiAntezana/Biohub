# Informe Oficial de Evaluación — Campaña 034: Drift-Compensated Spatial Proximity Invariant Regularization (DC-SPI)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub)  
**Run Evaluado**: `b0970-temporal-034-rel1`  
**Run Fuente de Control**: `b0970-selector-corpus-011` (score: `0.901274`)  
**Suite de Tests**: `b0970-tests-034-rel1` (425/425 tests aprobados en 6.74s)  
**Código Hash Congelado**: `f301348f064fbf7d52a15fa009aaaf7ca676b63b06ba94af0dbda3248e3c9dac`  
**Nodo de Ejecución**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta, wall-clock: 324.2s)  
**Disposición Oficial**: `rejected_no_kaggle_submission` (No promovida; no publicada en Kaggle)  
**Campeona de Confirmación Vigente**: Campaña 032 Variante 2 (DDKS-SPI-Cytokinesis-V2, score: `0.955029`, delta: `+0.002776`)

---

## 1. Resumen Ejecutivo y Veredicto

La Campaña 034 evaluó la hipótesis de **Sinergia entre Suavizado Axial Compensado por Deriva Global y Poda por Invarianza de Proximidad Espacial (DC-SPI)** con supresión de reasignación mitótica no condicionada.

### Veredicto de Gate de Desarrollo
- **Variante 1 (DC-SPI-Pure: Reasignación Desactivada)**: $\text{Score} = 0.903063$ (ref: $0.901274$, $\mathbf{\Delta = +0.001788} > 0$), **RECHAZADO** (inferior al umbral de desarrollo $+0.003000$).
- **Variante 2 (DC-SPI-Reinforced: $w_z = 0.85$)**: $\text{Score} = 0.903063$ (ref: $0.901274$, $\mathbf{\Delta = +0.001788} > 0$), **RECHAZADO**.
- **Variante 3 (DC-SPI-UltraGated: $p \ge 0.85, \cos\theta \le -0.60$)**: $\text{Score} = 0.901116$ (ref: $0.901274$, $\mathbf{\Delta = -0.000159}$), **RECHAZADO**.

**Disposición Oficial**: `rejected_no_kaggle_submission`. La campeona vigente e indiscutida de publicación permanece en **Campaña 032 Variante 2** (Confirmación: `0.955029`, $\Delta = +0.002776$).

---

## 2. Métricas Pareadas por Variante

| Métrica | Referencia (`corpus-011`) | Variante 1 (DC-SPI Pure) | Variante 2 (Reinforced) | Variante 3 (UltraGated) |
| :--- | :---: | :---: | :---: | :---: |
| **Score Oficial Agregado** | **0.901274** | **0.903063** ($\mathbf{+0.001788}$) | **0.903063** ($\mathbf{+0.001788}$) | 0.901116 ($\mathbf{-0.000159}$) |
| **Jaccard Aristas Ajustadas** | **0.888231** | **0.890019** ($\mathbf{+0.001788}$) | **0.890019** ($\mathbf{+0.001788}$) | **0.889577** ($\mathbf{+0.001346}$) |
| **Jaccard de División** | **0.130435** | **0.130435** ($\pm 0.000000$) | **0.130435** ($\pm 0.000000$) | 0.115385 ($\mathbf{-0.015050}$) |
| **Divisiones TP** | 3 / 15 | 3 / 15 (100% intactas) | 3 / 15 (100% intactas) | 3 / 15 (100% intactas) |
| **Divisiones FP** | 8 | **8** (Cero inflación) | **8** (Cero inflación) | 11 ($+3$) |
| **Divisiones FN** | 12 | 12 | 12 | 12 |
| **Aristas Reasignadas** | 0 | **0** | **0** | **116** (vs 980 en 033) |
| **Tiempo de Cómputo** | — | ~108s | ~108s | ~108s |

---

## 3. Síntesis Científica Mayor

1. **Eliminación Total de la Inflación de Falsos Positivos de División**:
   - Al desactivar la reasignación de mitosis no condicionada en las Variantes 1 y 2, los falsos positivos de división se mantuvieron estrictamente en 8 (cero inflación respecto al control).
   - En la Variante 3, el filtrado estricto ($p \ge 0.85$, $\cos\theta \le -0.60$) redujo las reasignaciones en un **$88.2\%$** (de 980 en 033 a solo 116 en 034), logrando una ganancia local en `6bba_48816121` ($\Delta = +0.00205$). Sin embargo, las 3 divisiones espurias residuales en otros datasets causaron una ligera caída neta ($\Delta = -0.000159$).
2. **Robustez de la Campeona 032**:
   - La arquitectura de **Campaña 032 Variante 2** (DDKS-SPI con selección fotométrica pura sobre grafo de enlace directo, sin reasignación ad-hoc de mitosis) representa el equilibrio óptimo probado en desarrollo ($\Delta = +0.004006$) y confirmado en holdout ciego ($\Delta = +0.002776$).
   - Por tanto, la línea de acción prioritaria para la competencia es la **publicación formal del paquete 032** en Kaggle.
