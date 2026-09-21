# Predeclaración Formal — Campaña 019: Regularización Axial Calibrada en Z y Cierre Conservador de Huecos (Rama E + Rama D)

Predeclaración formal dentro del protocolo autónomo de Biohub (`docs/AUTONOMOUS_CONTINUATION.md` y `AGENTS.md`).

---

## 1. Identidad y punto de partida

- **Campaña / Fecha / Responsable**: Campaña 019 / 2026-09-14 / Antigravity (autoridad de continuidad autónoma).
- **Hipótesis**: Rama E (Regularización Axial Invariante en Z con Guarda de Reversión Direccional y Rango Dinámico Calibrado) combinada con Rama D (Cierre Conservador de Huecos de 1 frame con Validación por Similitud de Parche 3D).
- **Run fuente**: `b0970-selector-corpus-011` en `A1555`.
- **Hashes de artefactos fuente**:
  - `receipt.json`: `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`
  - `corpus.json`: `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`
  - 53 tensores `*.evidence.npz` verificados.
- **Nodo de ejecución**: `A1555` (RTX 4090 libre, 29.2 GiB RAM disponibles, 168.2 GiB disco libre).
- **Perfil de recursos**: `cpu_evaluation_v1`. Tarea CPU acotada a 16 threads, CUDA oculta, sin GPU.
- **Release de control**: `b0970-selector-corpus-011` (grafo de referencia V19 congelado).
- **Limitaciones de exposición**: Los 8 datasets de desarrollo (`44b6_c50204e0`, `44b6_d5e7d891`, `44b6_ddf577ad`, `44b6_e57ff5c6`, `6bba_07e24132`, `6bba_09961292`, `6bba_48816121`, `6bba_5a4d9360`) se evalúan de forma pareada; la generación de candidatos es 100% label-free y nunca lee GT.

---

## 2. Hipótesis falsable

- **Cuello de botella medido**:
  1. En la Campaña 018 Release 4, la regularización con guarda de reversión (`reversal_only=true`) demostró **cero regresiones en los 8 datasets**, logrando un salto de $+0.015235$ en `44b6_ddf577ad` y un promedio de $+0.001788$.
  2. En Release 4, la corrección máxima estuvo acotada artificialmente a $0.6\,\mu\text{m}$ ($0.37$ cortes focales frente al paso discreto axial de $s_z = 1.625\,\mu\text{m}$). Al estar restringida a $0.37$ de corte, las oscilaciones genuinas entre secciones focales no alcanzaron a cruzar el umbral euclidiano de matching ($7.0\,\mu\text{m}$) en datasets con márgenes más estrechos.
  3. En Rama D, 5 huecos de 1 frame en tracks maduros presentan alta correlación de parche 3D ($\ge 0.60$) y proximidad espacial ($\le 4.5\,\mu\text{m}$), pero no están conectados en el grafo de referencia.
- **Mecanismo propuesto**:
  - Calibrar el rango de corrección axial para reversiones genuinas ($dz_1 \cdot dz_2 < 0$) permitiendo un desplazamiento de $1.0 - 1.5\,\mu\text{m}$ ($0.62 - 0.92$ cortes), restaurando la posición axial real sin cortar curvas naturales.
  - Integrar el puente conservador de huecos de 1 frame con conservación estricta de grados ($\text{in\_deg} \le 1, \text{out\_deg} \le 1$) y exclusión espacial ($\ge 2.0\,\mu\text{m}$).
- **Criterio de falsación**:
  - Si una variante produce regresión en algún grupo ($\Delta \text{score}_{\text{group}} < 0.0$) o regresión individual $\Delta J_{\text{adj}} < -0.015$, es rechazada.
  - Si ninguna variante alcanza $\Delta \text{score}_{\text{dev}} \ge +0.00300$, se declara `rejected_no_kaggle_submission`.
- **Incertidumbre que no resuelve**: No reentrena la segmentación 3D ni modifica detecciones crudas de la red neuronal.

---

## 3. Cambio acotado

- **Nuevos módulos**:
  - `core/temporal_association_019.py`
  - `configs/temporal_association_019.json`
  - `tests/test_temporal_association_019.py`
  - `ops/audit_temporal_inputs_019.py`
- **Drivers modificados**: `ops/biohub.py`, `ops/verify.py`.
- **Archivos protegidos inmutables**: Todos los pipelines 001-015, evaluador oficial, pesos de modelos, tensores de evidencia NPZ.
- **Definición de Variantes**:
  - **Control**: Grafo de referencia V19 congelado.
  - **Variante 1**: Regularización axial con guarda de reversión direccional y clamp medio:
    `reversal_only=true`, `jitter_threshold_um=0.8`, `max_correction_um=1.0`, `weight_z=0.80`, `weight_xy=0.0`, `gaps=disabled`.
  - **Variante 2**: Regularización axial con guarda de reversión direccional y restauración casi completa de corte focal:
    `reversal_only=true`, `jitter_threshold_um=0.8`, `max_correction_um=1.4`, `weight_z=0.85`, `weight_xy=0.0`, `gaps=disabled`.
  - **Variante 3**: Regularización axial con guarda de reversión (`max_correction_um=1.2`, `weight_z=0.80`) + Cierre Conservador de Huecos de 1 frame (`max_distance_um=4.5`, `min_sim=0.60`, `max_bridges=50`).

---

## 4. Gates de Decisión Predeclarados

- **Gate de Desarrollo**:
  - $\Delta \text{score}_{\text{dev}} \ge +0.00300$
  - $\Delta \text{score}_{44b6} \ge 0.0$
  - $\Delta \text{score}_{6bba} \ge 0.0$
  - $\min_{i} \Delta J_{\text{adj}, i} \ge -0.015$
  - $\Delta J_{\text{div}} \ge 0.0$
- **Gate de Confirmación** (si pasa desarrollo):
  - Evaluación sobre los 8 datasets de confirmación predeclarados.
  - $\Delta \text{score}_{\text{conf}} \ge +0.00200$
  - Sin regresión en ningún grupo.
- **Presupuesto de Filas de Kaggle**: Estrictamente $234{,}000 < \text{filas} < 242{,}000$.
