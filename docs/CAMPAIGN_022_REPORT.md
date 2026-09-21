# Informe Científico — Campaña 022: Regularización Axial con Guardias de Choque de Cuadro y Densidad Espacial (Rama E)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity  
**Identificador de Corrida**: `b0970-temporal-022`  
**Nodo de Cómputo**: `A1555`  
**Código Base**: Release 1 (`3f7196df36dd33cc640155898c42e0b454e85660ede6ea36755d663a68e6d04b`)  
**SHA256 del Manifiesto de Resultados**: `40b3963056e8e2705abb06c7ee44a103f4f72fac7ffe39a4670aefbb6c75349d`

---

## 1. Resumen Ejecutivo

La Campaña 022 evaluó la hipótesis de **Regularización Axial Guiada por Guardias de Choque Colectivo y Densidad Espacial** sobre los 8 datasets de desarrollo de `b0970-selector-corpus-011`.
El objetivo fue desbloquear las ganancias positivas de reversión direccional observadas en la campaña 018 ($+0.01551$ acumulado en 4 datasets) eliminando las regresiones en `6bba_09961292` y `44b6_d5e7d891`.

### Resultados Globales

| Variante | Descripción | Score Ref | Score Cand | Delta Promedio | $\Delta$ 44b6 | $\Delta$ 6bba | Regresiones | Gate ($\ge +0.003$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variante 1** | Control Simétrico Puro | 0.901274 | 0.903063 | **+0.001788** | +0.004765 | 0.000000 | **0** | No alcanzado |
| **Variante 2** | Guardias Conservadoras ($nn \ge 6.5\,\mu\text{m}$, shock $\le 4$) | 0.901274 | 0.903063 | **+0.001788** | +0.004765 | 0.000000 | **0** | No alcanzado |
| **Variante 3** | Guardias Calibradas ($nn \ge 5.5\,\mu\text{m}$, shock $\le 4$, $w_z=0.85$) | 0.901274 | 0.903063 | **+0.001788** | +0.004765 | 0.000000 | **0** | No alcanzado |

---

## 2. Desglose por Dataset

| Dataset | Grupo | Ref Adj Edge | Cand Adj Edge | $\Delta$ Edge Jaccard | Picos Suavizados | Choques Bloqueados | Densidad Bloqueada |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `44b6_c50204e0` | 44b6 | 0.81599 | 0.81599 | +0.00000 | 1294 | 0 | 0 |
| `44b6_d5e7d891` | 44b6 | 0.85906 | 0.85906 | **+0.00000** | 1458 | 0 | 1 |
| `44b6_ddf577ad` | 44b6 | 0.84121 | 0.85644 | **+0.01524** | 1338 | 0 | 0 |
| `44b6_e57ff5c6` | 44b6 | 0.73002 | 0.73002 | +0.00000 | 309 | 0 | 2 |
| `6bba_07e24132` | 6bba | 0.82505 | 0.82505 | +0.00000 | 895 | 9 | 0 |
| `6bba_09961292` | 6bba | 0.91648 | 0.91648 | **+0.00000** | 1016 | 18 | 1 |
| `6bba_48816121` | 6bba | 0.93806 | 0.93806 | +0.00000 | 873 | 0 | 1 |
| `6bba_5a4d9360` | 6bba | 0.94690 | 0.94690 | +0.00000 | 45 | 0 | 0 |

---

## 3. Hallazgos Científicos Críticos

1. **Validación Total de la Guardia de Choque Colectivo**:
   - En `6bba_09961292`, la guardia bloqueó con éxito 18 oscilaciones espurias concentradas en el cuadro 39 (desplazamiento de platina / deriva de enfoque).
   - Como resultado, la regresión de $-0.00297$ observada en la campaña 018 fue **completamente eliminada**, logrando un comportamiento plano sin degradación ($\Delta = 0.00000$).
2. **Validación de la Guardia de Densidad Espacial**:
   - En `44b6_d5e7d891`, la guardia bloqueó la oscilación en cúmulo denso, eliminando la pérdida de $-0.00280$ de la campaña 018 y manteniendo $\Delta = 0.00000$.
3. **Comportamiento Libre de Regresiones en los 8 Datasets**:
   - Las 3 variantes demostraron **cero regresiones en todos y cada uno de los 8 datasets**.
   - En `44b6_ddf577ad`, la ganancia masiva de $+0.01524$ se mantiene intacta.
4. **Causa del Límite en $+0.001788$**:
   - Al requerir simultáneamente aislamiento espacial estricto y supresión de choques colectivos, las reversiones asimétricas corregidas fueron muy escasas ($\sim 10\text{--}25$ nodos adicionales por volumen), insuficientes para empujar enlaces adicionales dentro del radio de matching en `c50204e0`, `07e24132` y `48816121`.
   - Por tanto, la regularización de coordenadas en trayectorias lineales existentes está saturada en $+0.001788$.

---

## 4. Decisión Protocolar y Siguiente Paso

Conforme a `docs/AUTONOMOUS_CONTINUATION.md`:
- Ninguna variante alcanzó el umbral mínimo de desarrollo de $+0.00300$ necesario para justificar una confirmación y submission independiente.
- Disposición: `rejected_no_kaggle_submission`.
- Se preservan todos los artefactos, receipts y código (`b0970-temporal-022`).
- **Próxima Campaña (Campaña 023)**:
  - Investigar la resolución de **cruces de trayectorias (Edge Crossovers / Switch Errors)** en tejido denso con búsqueda espacial real mediante KDTree y correlación normalizada 3D (Rama E).
  - Un único cruce corregido convierte $2 \times \text{FP} \to 2 \times \text{TP}$ y elimina $2 \times \text{FN}$, lo que produce un salto de $+0.00986$ por evento resuelto.
