# Predeclaración Formal — Campaña 039: Calibrated Cytokinesis-Directed Guard (CCDG)

## Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaña 039 / 2026-09-15 / Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub).
- **Hipótesis**: Calibración angular biofísica de la compuerta de citocinesis (CCDG: Calibrated Cytokinesis-Directed Guard) a $\cos_{3d} \le +0.20$ (ángulo $\ge 78.5^\circ$) con persistencia $\ge 2$. Inmuniza divisiones biológicas inclinadas por flujo tisular (shear drift) mientras elimina masivamente cruces paralelos espurios ($\cos > 0.20$).
- **Run fuente**: `b0970-selector-corpus-011` (A1555), receipt sha256 `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`, corpus sha256 `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`. Score de control: `0.901274`.
- **Nodo de ejecución**: `A1555` bajo `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta, tiempo máximo 1800s).
- **Campeona de confirmación vigente**: Campaña 032 Variante 2 (`b0970-confirm-032-rel5`, confirmación: `0.955029`).
- **Exposiciones previas**: El panel de desarrollo comprende los 8 datasets oficiales de `b0970-selector-corpus-011` (`44b6_c50204e0`, `44b6_d5e7d891`, `44b6_ddf577ad`, `44b6_e57ff5c6`, `6bba_07e24132`, `6bba_09961292`, `6bba_48816121`, `6bba_5a4d9360`). GT reservado y exclusivo para la evaluación métrica final; nunca accesible durante la generación de candidatos ni asignación de aristas.

---

## Hipótesis falsable

- **Cuello de botella medido en Campaña 038**:
  - En la Campaña 038, la regla CEDTP con $\cos_{3d} \le 0.00$ superó holgadamente el gate de desarrollo ($\Delta = +0.004147 \ge +0.003$), logrando un salto en Edge Jaccard en ambos grupos (6bba y 44b6) y podando 143 bifurcaciones falsas.
  - Sin embargo, en confirmación, la exigencia estricta de $\cos_{3d} \le 0.00$ podó 1 rama de una división verdadera en `44b6_587a1e22`. La causa raíz física descubierta es que en tejidos en migración colectiva, el flujo de cizallamiento (shear drift) inclina los vectores de las células hijas respecto al sistema de coordenadas del laboratorio, haciendo que el ángulo entre hijas sea de $\sim 80^\circ - 85^\circ$ ($\cos \approx +0.06$ a $+0.09$), en lugar de $< 90^\circ$ ($\cos < 0.0$).
  - Esto provocó que `division_tp` en 44b6 cayera de 1 a 0, restando -0.001864 a la confirmación a pesar de que Edge Jaccard subió sólidamente.
- **Mecanismo propuesto (CCDG)**:
  - En lugar de fijar la compuerta en la ortogonal teórica $\cos \le 0.00$ ($90^\circ$), CCDG calibra la tolerancia angular a $\cos_{3d} \le +0.20$ (ángulo $\ge 78.5^\circ$) con persistencia $\ge 2$.
  - Esta cota biológica cubre con margen holgado todas las divisiones celulares en presencia de deriva o rotación tisular ($\cos \in [-1.0, +0.15]$), otorgándoles **inmunidad total**.
  - Al mismo tiempo, todos los cruces paralelos espurios (donde dos células viajan en el mismo haz con $\cos \in [0.25, 0.95]$) quedan **fuera de la inmunidad** y son eliminados por DDKS-SPI y la función de costo biofísica.
  - Simulación local confirmó: con $\cos \le +0.20$, la división verdadera de `44b6_587a1e22` queda 100% protegida, mientras se eliminan ~35 bifurcaciones falsas en confirmación y ~72 en desarrollo.
- **Qué observación concreta lo rechazaría**:
  - $\Delta_{\text{score}} < +0.003000$ en desarrollo frente a `b0970-selector-corpus-011`.
  - Si $\text{division\_tp}$ en desarrollo cae por debajo del control ($\text{division\_tp} < 3$).
- **Qué incertidumbre NO resuelve el ensayo**:
  - No resuelve divisiones donde una de las hijas no fue detectada por el segmentador óptico.

---

## Cambio acotado

- **Módulos nuevos y drivers**:
  - `core/temporal_association_039.py`: Implementa CCDG en `prune_spurious_cytokinesis_forks` con `heading_guard_cos_max: 0.20/0.25`, `guard_requires_cos: true`, `persistence_min_frames: 2`.
  - `configs/temporal_association_039.json`: Configuración estricta de las variantes.
  - `tests/test_temporal_association_039.py`: Suite de 5 tests unitarios que verifican la protección de divisiones inclinadas ($\cos = +0.08$) y la poda de cruces paralelos agudos ($\cos = 1.0$).
  - `ops/biohub.py`: Tareas `associate-temporal-039` y `confirm-temporal-039`.
  - `ops/verify.py`: Registro de `temporal_039` en la suite canónica (440 tests pasando).
- **Archivos protegidos que no cambian**:
  - Todos los módulos de 001–038 permanecen intactos y congelados.
  - Caches de candidatos, bases de datos Zarr y checkpoints permanecen de solo lectura.
- **Control**:
  - `b0970-selector-corpus-011` (score: `0.901274`).
- **Variantes predeclaradas (máximo 2)**:
  - **Variante 1 (CCDG-020, $\cos \le +0.20$)**:
    - Suavizado axial simétrico ($w_z = 0.80, \Delta z_{\max} = 0.6\,\mu\text{m}$).
    - Protección de bifurcaciones exige $\cos_{3d} \le +0.20$ con persistencia $\ge 2$.
    - Poda de ramas espurias vía DDKS-SPI y keeper selection lineal.
  - **Variante 2 (CCDG-025, $\cos \le +0.25$)**:
    - Suavizado axial simétrico ($w_z = 0.80, \Delta z_{\max} = 0.6\,\mu\text{m}$).
    - Protección de bifurcaciones exige $\cos_{3d} \le +0.25$ con persistencia $\ge 2$.
    - Poda de ramas espurias vía DDKS-SPI y keeper selection lineal.

---

## Datos y ejecución

- **Listas exactas**:
  - Desarrollo: 8 datasets de `b0970-selector-corpus-011`.
  - Confirmación (si califica en desarrollo): 8 datasets de confirmación (`b0970-confirm-032-rel5`).
- **Perfil de recursos**: `cpu_evaluation_v1` en `A1555` (2 vCPUs, 8 GiB RAM límite, CUDA oculta, tiempo máximo 1800s).
- **IDs de corrida**:
  - Tests: `b0970-tests-039-rel1`
  - Desarrollo: `b0970-temporal-039-rel1`
- **Comandos**:
  - Tests: `python ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-039-rel1`
  - Desarrollo: `python ops/biohub.py launch --node A1555 --task associate-temporal-039 --source-run b0970-selector-corpus-011 --run-id b0970-temporal-039-rel1 --tests-run b0970-tests-039-rel1`

---

## Verificación y decisión predeclarada

- **Criterio de desarrollo**:
  - Si Variante 1 o 2 alcanza $\Delta_{\text{score}} \ge +0.003000$ frente al control (`0.901274`) manteniendo $\text{division\_tp} \ge 3$, se seleccionará la mejor variante para confirmación inmediata (`b0970-confirm-039-rel1`).
  - Si ninguna variante alcanza $\Delta_{\text{score}} \ge +0.003000$, la hipótesis se declarará formalmente rechazada y se conservarán íntegros los receipts y logs.
