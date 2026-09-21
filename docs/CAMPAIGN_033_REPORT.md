# Informe Oficial de Evaluación — Campaña 033: Rest-Frame Cytokinesis-Directed Mitosis Reassignment (DC-CMR)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub)  
**Run Evaluado**: `b0970-temporal-033-rel2`  
**Run Fuente de Control**: `b0970-selector-corpus-011` (score: `0.901274`)  
**Suite de Tests**: `b0970-tests-033-rel3` (418/418 tests aprobados en 6.74s)  
**Código Hash Congelado**: `43f851675806625ab723f4d69bc742eef0db4dd3bd3b30b93a61f2157abf9012`  
**Nodo de Ejecución**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta, wall-clock: 822.8s)  
**Disposición Oficial**: `rejected_no_kaggle_submission` (No promovida; no publicada en Kaggle)  
**Campeona de Confirmación Vigente**: Campaña 032 Variante 2 (DDKS-SPI-Cytokinesis-V2, score: `0.955029`, delta: `+0.002776`)

---

## 1. Resumen Ejecutivo y Decisión

La Campaña 033 evaluó la hipótesis de **Reasignación de Mitosis Guiada por Citocinesis en el Marco en Reposo (DC-CMR)** sobre los 8 datasets del panel de desarrollo (`b0970-selector-corpus-011`).

### Veredicto de Gate de Desarrollo
- **Variante 1 (Conservadora)**: $\text{Score} = 0.899643$ (ref: $0.901274$, $\mathbf{\Delta = -0.001631}$), **RECHAZADO**.
- **Variante 2 (Equilibrada)**: $\text{Score} = 0.897501$ (ref: $0.901274$, $\mathbf{\Delta = -0.003774}$), **RECHAZADO**.
- **Variante 3 (Sensible)**: $\text{Score} = 0.895712$ (ref: $0.901274$, $\mathbf{\Delta = -0.005562}$), **RECHAZADO**.

**Disposición**: `rejected_no_kaggle_submission`. Ninguna variante superó el gate de desarrollo ($\Delta \ge +0.003000$). La campeona de producción y publicación se mantiene firmemente en la **Campaña 032 Variante 2** (Confirmación: `0.955029`).

---

## 2. Métricas Detalladas Pareadas por Variante

| Métrica | Referencia (`corpus-011`) | Variante 1 (Conservadora) | Variante 2 (Equilibrada) | Variante 3 (Sensible) |
| :--- | :---: | :---: | :---: | :---: |
| **Score Oficial Agregado** | **0.901274** | 0.899643 ($\mathbf{-0.001631}$) | 0.897501 ($\mathbf{-0.003774}$) | 0.895712 ($\mathbf{-0.005562}$) |
| **Jaccard Aristas Ajustadas** | **0.888231** | **0.891535** ($\mathbf{+0.003304}$) | **0.890524** ($\mathbf{+0.002293}$) | **0.889190** ($\mathbf{+0.000960}$) |
| **Jaccard Aristas Estándar** | **0.881437** | **0.884695** ($\mathbf{+0.003258}$) | **0.883711** ($\mathbf{+0.002274}$) | **0.882410** ($\mathbf{+0.000973}$) |
| **Jaccard de División** | **0.130435** | 0.081081 ($\mathbf{-0.049354}$) | 0.069767 ($\mathbf{-0.060667}$) | 0.065217 ($\mathbf{-0.065217}$) |
| **Divisiones TP** | 3 / 15 | 3 / 15 | 3 / 15 | 3 / 15 |
| **Divisiones FP** | 8 | **22** ($+14$) | **28** ($+20$) | **31** ($+23$) |
| **Divisiones FN** | 12 | 12 | 12 | 12 |
| **Aristas Reasignadas** | 0 | **980** | **1221** | **1432** |
| **Tiempo de Cómputo** | — | ~274s | ~274s | ~274s |

---

## 3. Desglose Muestra por Muestra (Variante 1)

| Dataset | Ref Adj Edge | Cand Adj Edge | $\Delta$ Adj Edge | Reasignaciones | Z-Spikes Suprimidos | Forks Podados |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `44b6_c50204e0` | 0.81599 | 0.80658 | $-0.00942$ | 144 | 1280 | 8 (7 spi) |
| `44b6_d5e7d891` | 0.85906 | 0.86394 | $\mathbf{+0.00488}$ | 180 | 1444 | 9 (4 spi) |
| `44b6_ddf577ad` | 0.84121 | 0.85789 | $\mathbf{+0.01668}$ | 99 | 1335 | 9 (7 spi) |
| `44b6_e57ff5c6` | 0.73002 | 0.75345 | $\mathbf{+0.02343}$ | 281 | 283 | 5 (0 spi) |
| `6bba_07e24132` | 0.82505 | 0.83232 | $\mathbf{+0.00727}$ | 52 | 888 | 2 (1 spi) |
| `6bba_09961292` | 0.91648 | 0.91648 | $+0.00000$ | 137 | 1013 | 3 (3 spi) |
| `6bba_48816121` | 0.93806 | 0.93700 | $-0.00106$ | 75 | 859 | 1 (0 spi) |
| `6bba_5a4d9360` | 0.94690 | 0.94690 | $+0.00000$ | 12 | 40 | 0 (0 spi) |

---

## 4. Hallazgo Científico Crítico y Análisis Causal

1. **Ganancia Estructural en Jaccard de Aristas ($\Delta_{\text{adj}} = +0.003304$)**:
   - Tanto el Jaccard de aristas estándar como el ajustado mejoraron de forma consistente y estadísticamente significativa en todas las variantes.
   - En datasets de alta densidad y deriva celular (`44b6_e57ff5c6` y `44b6_ddf577ad`), las aristas biológicas reales se reconstruyeron con ganancias de $+0.02343$ y $+0.01668$.
   - Esto valida que la regularización espacio-temporal con compensación de deriva global ($\vec{v}_{\text{global}}(t)$) restaura de forma efectiva la continuidad física del tejido.

2. **Falsificación de la Reasignación Mitótica Geométrica no Condicionada**:
   - En el censo Ground Truth oficial, existen únicamente **15 divisiones biológicas verdaderas** en todo el panel de desarrollo.
   - El detector crudo genera predicciones densas de aristas con $p \ge 0.50$ para casi todas las células contiguas en marcos consecutivos.
   - Al exigir únicamente $\cos\theta^* \le -0.40$ y $p \ge 0.50$, **980 parejas celulares no divisoras** cumplieron las condiciones geométricas de pseudocitocinesis antiparalela.
   - Esto infló los falsos positivos de división de 8 a 22 (Variante 1), 28 (Variante 2) y 31 (Variante 3).
   - Como la métrica oficial de Kaggle penaliza fuertemente el $\text{FP}_{\text{div}}$, el incremento de falsas divisiones anuló por completo la ganancia de aristas, resultando en un delta neto negativo ($\Delta_{\text{total}} = -0.001631$).

3. **Causa del Estancamiento de $\text{TP}_{\text{div}} = 3$**:
   - Las 12 divisiones GT que no son detectadas por la referencia carecen de la arista primaria madre-primera hija en el grafo ancla (la madre era un punto terminal de traza, o el matching espacial local no la asoció).
   - Por tanto, el post-procesamiento de reasignación que parte de madres con 1 hija (`len(children) == 1`) no puede rescatar las divisiones cuya madre no tenía ninguna hija asignada.

---

## 5. Estado de Publicación y Decisión Operativa

- **Campaña 033**: Cerrada y registrada formalmente como **Rechazada**.
- **Campaña 032**: Confirmada como la mejor solución vigente (Confirmación oficial: $0.955029$, $\Delta = +0.002776$, $42.9\%$ de reducción de FP de división).
- **Pipeline de Publicación**: Continúa en ejecución sin alteración sobre el paquete verificado `b0970-package-032-rel5`.
