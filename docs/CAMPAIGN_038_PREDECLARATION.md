# Predeclaración Formal — Campaña 038: Cytokinesis-Enforced Dual-Track Protection (CEDTP)

## Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaña 038 / 2026-09-15 / Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub).
- **Hipótesis**: Eliminación de bifurcaciones espurias paralelas ($\cos_{3d} > 0.0$) mediante la compuerta de protección biológica de doble trayectoria forzada por citocinesis (CEDTP: Cytokinesis-Enforced Dual-Track Protection).
- **Run fuente**: `b0970-selector-corpus-011` (A1555), receipt sha256 `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`, corpus sha256 `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`. Score de control: `0.901274`.
- **Nodo de ejecución**: `A1555` bajo `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta, tiempo máximo 1800s).
- **Campeona de confirmación vigente**: Campaña 032 Variante 2 (`b0970-confirm-032-rel5`, confirmación: `0.955029`).
- **Exposiciones previas**: El panel de desarrollo comprende los 8 datasets oficiales de `b0970-selector-corpus-011` (`44b6_c50204e0`, `44b6_d5e7d891`, `44b6_ddf577ad`, `44b6_e57ff5c6`, `6bba_07e24132`, `6bba_09961292`, `6bba_48816121`, `6bba_5a4d9360`). GT reservado y exclusivo para la evaluación métrica final; nunca accesible durante la generación de candidatos ni asignación de aristas.

---

## Hipótesis falsable

- **Cuello de botella medido**:
  - En la Campaña 032 (y subsecuentes 033–037), la regla de protección de doble trayectoria contenía una inmunización incondicional:
    `if (len1 >= 5 and len2 >= 5): continue`
  - Esto provocaba que cualquier bifurcación espuria donde dos células que viajan en paralelo ($\cos > 0.0$) persistieran al menos 5 cuadros quedara **100% protegida contra la poda**, impidiendo la aplicación de DDKS-SPI o keeper selection.
  - La auditoría topológica en confirmación reveló **365 bifurcaciones supervivientes** frente a solo **16 divisiones verdaderas en GT** (más de 340 bifurcaciones espurias no podadas). En particular, en `44b6_c15fded2` (donde GT tiene 0 divisiones), sobrevivieron 59 bifurcaciones espurias produciendo 13 aristas FP que reducen el $J_{\text{adj}}$ a 0.7959.
  - En los 8 datasets de desarrollo, existen 591 bifurcaciones de las cuales solo 27 eran podadas por 032.
- **Mecanismo propuesto (CEDTP)**:
  - En la biología de la citocinesis, el anillo contráctil secciona las dos células hijas a lo largo del huso mitótico divergente ($\cos_{3d} \le 0.0$, típicamente $\le -0.40$).
  - Dos células que se mueven hacia adelante en la misma dirección ($\cos_{3d} > 0.0$) **NUNCA son hijas de una citocinesis biológica**; representan colisiones o cruces espurios donde una célula saltó a la trayectoria de la otra.
  - **Regla CEDTP**: La protección de doble trayectoria exige obligatoriamente divergencia angular:
    `if cos_3d <= heading_guard_cos_max and len1 >= 3 and len2 >= 3: continue`
  - Si $\cos_{3d} > \text{heading_guard_cos_max}$, la bifurcación no goza de inmunidad biológica: se somete a DDKS-SPI (designación de la hija local/estacionaria si hay asimetría extrema) y selección de keeper por costo de distancia y alineación de velocidad, podando la rama espuria que salta.
- **Simulación preliminar**:
  - En desarrollo, CEDTP poda **137 bifurcaciones espurias** (frente a solo 27 en 032), de las cuales 135 tienen $\cos > 0.0$.
  - En confirmación, CEDTP poda **92 bifurcaciones espurias** (frente a 23 en 032), podando 19 bifurcaciones espurias con $\cos > 0$ en `44b6_c15fded2` y atacando directamente las 13 aristas FP.
- **Qué observación concreta lo rechazaría**:
  - $\Delta_{\text{score}} < +0.003000$ en desarrollo frente a `b0970-selector-corpus-011`.
  - Si CEDTP incrementa los falsos negativos de división ($\text{division\_fn} > 12$) o degrada la recuperación de divisiones verdaderas.
- **Qué incertidumbre NO resuelve el ensayo**:
  - No resuelve divisiones donde una de las hijas no fue detectada en la segmentación.

---

## Cambio acotado

- **Módulos nuevos y drivers**:
  - `core/temporal_association_038.py`: Implementa CEDTP con `guard_requires_cos: true` en `prune_spurious_cytokinesis_forks`.
  - `configs/temporal_association_038.json`: Configuración estricta de las dos variantes.
  - `tests/test_temporal_association_038.py`: Suite de 5 unit tests que verifican la invariante biológica.
  - `ops/biohub.py`: Tareas `associate-temporal-038` y `confirm-temporal-038`.
  - `ops/verify.py`: Registro en la suite canónica de verificación.
- **Archivos protegidos que no cambian**:
  - Todos los módulos de 001–037 permanecen intactos y congelados.
  - Caches de candidatos, bases de datos Zarr y checkpoints permanecen de solo lectura.
- **Control**:
  - `b0970-selector-corpus-011` (score: `0.901274`).
- **Variantes predeclaradas (máximo 2)**:
  - **Variante 1 (CEDTP Standard, $\cos \le 0.0$)**:
    - Suavizado axial simétrico ($w_z = 0.80, \Delta z_{\max} = 0.6\,\mu\text{m}$).
    - Protección de bifurcaciones exige $\cos_{3d} \le 0.0$ con persistencia $\ge 3$.
    - Poda de ramas espurias vía DDKS-SPI y keeper selection lineal.
  - **Variante 2 (CEDTP Strict Divergence, $\cos \le -0.20$)**:
    - Suavizado axial simétrico ($w_z = 0.80, \Delta z_{\max} = 0.6\,\mu\text{m}$).
    - Protección de bifurcaciones exige divergencia estricta $\cos_{3d} \le -0.20$ con persistencia $\ge 3$.
    - Cualquier bifurcación con $\cos_{3d} > -0.20$ (incluso ortogonales) es podada vía keeper selection.

---

## Datos y ejecución

- **Listas exactas**:
  - Desarrollo: 8 datasets de `b0970-selector-corpus-011`.
  - Confirmación (si califica en desarrollo): 8 datasets de confirmación (`b0970-confirm-032-rel5`).
- **Perfil de recursos**: `cpu_evaluation_v1` en `A1555` (2 vCPUs, 8 GiB RAM límite, CUDA oculta, tiempo máximo 1800s).
- **IDs de corrida**:
  - Tests: `b0970-tests-038-rel1`
  - Desarrollo: `b0970-temporal-038-rel1`
- **Comandos**:
  - Tests: `python ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-038-rel1`
  - Desarrollo: `python ops/biohub.py launch --node A1555 --task associate-temporal-038 --source-run b0970-selector-corpus-011 --run-id b0970-temporal-038-rel1 --tests-run b0970-tests-038-rel1`

---

## Verificación y decisión predeclarada

- **Criterio de desarrollo**:
  - Si Variante 1 o 2 alcanza $\Delta_{\text{score}} \ge +0.003000$ frente al control (`0.901274`) sin degradar la métrica de división ($\text{division\_fn} \le 12$), se seleccionará la mejor variante para confirmación inmediata.
  - Si ninguna variante alcanza $\Delta_{\text{score}} \ge +0.003000$, la hipótesis se declarará formalmente rechazada y se conservarán íntegros los receipts y logs.
