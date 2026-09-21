# Predeclaración Formal — Campaña 037: Pure Biophysical Cytokinesis Symmetry Gating (BP-CSG)

## Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaña 037 / 2026-09-15 / Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub).
- **Hipótesis**: Reasignación de segundas hijas robadas mediante compuertas biológicas puras de simetría de citocinesis en marco libre de deriva (BP-CSG: Biophysical Cytokinesis Symmetry Gating). Se elimina la dependencia espuria de probabilidades soft de ILP ($p(m \to d_2)$ y cocientes de softmax) que colapsaban por proximidad local de células competidoras no emparentadas.
- **Run fuente**: `b0970-selector-corpus-011` (A1555), receipt sha256 `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`, corpus sha256 `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`. Score de control: `0.901274`.
- **Nodo de ejecución**: `A1555` bajo `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta, tiempo máximo 1800s).
- **Campeona de confirmación vigente**: Campaña 032 Variante 2 (`b0970-confirm-032-rel5`, confirmación: `0.955029`).
- **Exposiciones previas**: El panel de desarrollo comprende los 8 datasets oficiales de `b0970-selector-corpus-011` (`44b6_c50204e0`, `44b6_d5e7d891`, `44b6_ddf577ad`, `44b6_e57ff5c6`, `6bba_07e24132`, `6bba_09961292`, `6bba_48816121`, `6bba_5a4d9360`). GT reservado y exclusivo para la evaluación métrica final; nunca accesible durante la generación de candidatos ni asignación de aristas.

---

## Hipótesis falsable

- **Cuello de botella medido**:
  - En la Campaña 036 (`b0970-temporal-036-rel1`), la reasignación de mitosis produjo 0 reasignaciones efectivas (`reassignments: 0`).
  - La auditoría de trazabilidad reveló que la Capa 6 exigía $p(m \to d_2) \ge 0.70$ y $\frac{p(m \to d_2)}{p(c \to d_2)} \ge 1.15$.
  - En los casos reales de citocinesis donde una segunda hija fue "robada" por un competidor $c$, la distancia espacial $c \to d_2$ es típicamente de $0.4\ \mu\text{m} - 1.8\ \mu\text{m}$, mientras que la madre $m \to d_2$ está a $4.0\ \mu\text{m} - 10.0\ \mu\text{m}$ debido al empuje mecánico del surco de división celular.
  - La normalización por softmax en el ILP lineal asigna $\approx 100\%$ de probabilidad a $c$, reduciendo la probabilidad aparente $p(m \to d_2)$ a $\approx 0.0000$. En consecuencia, el filtro de probabilidad soft bloqueaba el 100% de las divisiones biológicas genuinas.
- **Mecanismo propuesto**:
  - La verdadera señal invariante de mitosis es la **conservación física de momento y la simetría geométrica de citocinesis**:
    1. Divergencia angular en marco corregido por deriva: $\cos\theta^* = \frac{\vec{u}_1^* \cdot \vec{u}_2^*}{\|\vec{u}_1^*\| \|\vec{u}_2^*\|} \le -0.40$ (antiparalelismo del huso mitótico).
    2. Proximidad de hermanas recién separadas: $\|\vec{r}_{d_1} - \vec{r}_{d_2}\| \le 16.0\ \mu\text{m}$.
    3. Conservación del centro de masas: $\cos(\vec{v}_{\text{mid}}^*, \vec{v}_{\text{in}}^*) \ge 0.30$.
    4. Historia temporal de la célula madre $\ge 2$ cuadros temporales y persistencia de ambas hijas $\ge 2$ cuadros temporales.
    5. Consistencia morfológica (similitud de embeddings $\ge 0.50$, cociente de escala espacial $\le 2.0$).
  - En evaluación preliminar sobre los paneles de desarrollo (`scratch/mitosis_reassignment_eval.py`), esta regla recuperó **+8 divisiones verdaderas** con solo 1 falso positivo, aportando un delta métrico directo de **+0.035821**.
- **Qué observación concreta lo rechazaría**:
  - $\Delta_{\text{score}} < +0.003000$ en desarrollo frente a `b0970-selector-corpus-011`.
  - Si la reasignación incrementa los falsos positivos de división en más de 2 divisiones espurias.
- **Qué incertidumbre NO resuelve el ensayo**:
  - No resuelve casos donde el detector segmentó una sola hija o no detectó a la célula madre.

---

## Cambio acotado

- **Módulos nuevos y drivers**:
  - `core/temporal_association_037.py`: Implementa el pipeline con BP-CSG sin filtro de probabilidad soft.
  - `configs/temporal_association_037.json`: Parámetros de variantes 1 y 2.
  - `tests/test_temporal_association_037.py`: Suite de verificación unitaria e invariantes físicas.
  - `ops/biohub.py`: Tareas `associate-temporal-037` y `confirm-temporal-037`.
  - `ops/verify.py`: Registro en la suite canónica de verificación.
- **Archivos protegidos que no cambian**:
  - Todos los módulos de campañas 001–036 permanecen intactos y congelados.
  - Caches de candidatos, bases de datos Zarr y checkpoints permanecen de solo lectura.
- **Control**:
  - `b0970-selector-corpus-011` (score: `0.901274`).
- **Variantes predeclaradas (máximo 2)**:
  - **Variante 1 (BP-CSG Standard Symmetry)**:
    - $\cos\theta^* \le -0.40$
    - Distancia entre hermanas $\le 16.0\ \mu\text{m}$
    - Forward drift alignment $\ge 0.30$
    - Cosine similarity $\ge 0.50$
    - Scale ratio $\le 2.0$
  - **Variante 2 (BP-CSG Strict Cleavage Angle)**:
    - $\cos\theta^* \le -0.60$
    - Distancia entre hermanas $\le 12.0\ \mu\text{m}$
    - Forward drift alignment $\ge 0.45$
    - Cosine similarity $\ge 0.60$
    - Scale ratio $\le 1.6$

---

## Datos y ejecución

- **Listas exactas**:
  - Desarrollo: 8 datasets de `b0970-selector-corpus-011`.
  - Confirmación (si califica en desarrollo): 3 paneles no expuestos de confirmación (`0f845d07`, `137456d9`, `e1c51478`).
- **Perfil de recursos**: `cpu_evaluation_v1` en `A1555` (2 vCPUs, 8 GiB RAM límite, CUDA oculta, tiempo máximo 1800s).
- **IDs de corrida**:
  - Tests: `b0970-tests-037-rel1`
  - Desarrollo: `b0970-temporal-037-rel1`
- **Comandos**:
  - Tests: `python ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-037-rel1`
  - Desarrollo: `python ops/biohub.py launch --node A1555 --task associate-temporal-037 --source-run b0970-selector-corpus-011 --run-id b0970-temporal-037-rel1 --tests-run b0970-tests-037-rel1`

---

## Verificación y decisión predeclarada

- **Criterio de desarrollo**:
  - Si Variante 1 o 2 alcanza $\Delta_{\text{score}} \ge +0.003000$ frente al control (`0.901274`) y mejora la recuperación de divisiones sin inflar falsos positivos, se seleccionará la mejor variante para confirmación.
  - Si ninguna variante alcanza $\Delta_{\text{score}} \ge +0.003000$, la hipótesis se declarará formalmente rechazada y se conservarán íntegros los receipts y logs para el análisis retrospectivo.
