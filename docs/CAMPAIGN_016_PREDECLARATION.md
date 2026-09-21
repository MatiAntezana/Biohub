# Predeclaración formal — Campaña 016: Asociación Temporal y Regularización Cinemática Anisótropa

Ficha de predeclaración formal redactada según `docs/NEXT_EXPERIMENT_TEMPLATE.md` y autorizada bajo el contrato de continuidad de Biohub (`AGENTS.md` y `docs/AUTONOMOUS_CONTINUATION.md`).

---

## 1. Identidad y punto de partida

- **Campaña / Fecha / Responsable**: Campaña 016 / 2026-09-14 / Antigravity.
- **Hipótesis científica**: Rama E + Rama D + Regularización Cinemática Anisótropa (`AUTONOMOUS_CONTINUATION.md`, Sección 6.E, 6.D).
- **Run fuente y nodo de origen**: `b0970-selector-corpus-011` en A1555 (receipt SHA `7016ff9486c91a47738222a07c91fa04b3164a27fc2cfa8aee20b92e7372d8fc`).
- **Nodo de ejecución**: A1555 (168 GiB libres, RTX 4090 ociosa, conda `train-55`). Perfil de recursos: `cpu_evaluation_v1`.
- **Release y control de partida**: Release 015 (`182748fa31d8dc721803362ba3cf1092980ac7a1c44b04f492bc6401bf46a657`). Control pareado idéntico a la campeona de referencia V19 (score desarrollo: `0.9012742110283266`).
- **Exposiciones previas**: 8 datasets de desarrollo congelados (`44b6_d5e7d891`, `44b6_c50204e0`, `44b6_e57ff5c6`, `44b6_ddf577ad`, `6bba_48816121`, `6bba_09961292`, `6bba_5a4d9360`, `6bba_07e24132`), 35 datasets FIT de entrenamiento de priors, y 10 datasets de calibración. Confirmación y Visible permanecen como conjuntos no alterados.

---

## 2. Hipótesis falsable

- **Cuello de botella medido**:
  1. En la evaluación oficial, los enlaces determinan el **90% del score final** ($	ext{Score} = J_{\text{adj\_edge}} + 0.1 \times J_{\text{div}}$).
  2. En `44b6`, el Jaccard de enlaces es de solo **0.8181** (frente a 0.9173 en `6bba`).
  3. Las células presentan un paso de vóxel anisótropo de $1.625\,\mu\text{m}$ en $Z$ (4x más grueso que lateral $0.40625\,\mu\text{m}$). El jitter por ruido de detección en $Z$ alcanza segundas diferencias de hasta $6.50\,\mu\text{m}$, empujando los centroides al límite del radio de matching oficial ($7.0\,\mu\text{m}$). Al fallar el matching de un único nodo, se generan simultáneamente 2 Falsos Positivos y 2 Falsos Negativos de enlaces incidentes.
  4. En zonas de alta densidad celular ($\le 6.0\,\mu\text{m}$), las trayectorias paralelas sufren cruces de identidad (switch errors) que fragmentan los linajes y degradan el edge recall.
- **Mecanismo propuesto**:
  1. **Regularización Cinemática Anisótropa**: Desacoplar el suavizado de trayectoria en $Z$ frente a $XY$. Aplicar un peso de regularización mayor en el eje axial ($w_z = 0.85$) para amortiguar el jitter discreto de cortes, y un peso moderado en lateral ($w_{xy} = 0.65$) sobre ventana temporal $W=3$, verificando estrictamente la no-colisión de centroides.
  2. **Resolución de Cruces de Identidad (Rama E)**: Transacciones 2x2 de intercambio de enlaces entre tracks vecinos ($\le 6.0\,\mu\text{m}$) evaluadas mediante ganancia combinada de continuidad direccional de velocidad ($\Delta \text{res} \ge 1.0\,\mu\text{m}$) y correlación de parches 3D normalizados ($\text{NCC} \ge 0.50$). Esta operación conserva estrictamente la cardinalidad ($\Delta \text{nodes} = 0, \Delta \text{edges} = 0$).
  3. **Variante 2 (Rama D)**: Cierre conservador de huecos de 1 frame entre endpoints libres y starts libres ($\le 4.5\,\mu\text{m}$) con soporte estricto de imagen (SNR $\ge 4.0$, NCC $\ge 0.50$, exclusión espacial $\ge 2.0\,\mu\text{m}$) y confirmación de calor de DeepCenter ($\ge 0.25$).
- **Qué observación lo rechazaría**:
  - Si la regularización de coordenadas incrementa el error de asignación de centroides, reduciendo el node recall o el edge Jaccard en el desarrollo global o en cualquiera de los dos prefijos biológicos ($\Delta \text{group} < 0.0$).
  - Si alguna muestra individual sufre una regresión de Jaccard ajustado mayor a $-0.015$.
  - Si el score delta pareado en desarrollo no supera el umbral obligatorio de $+0.003$.
- **Qué incertidumbre NO resuelve el ensayo**:
  - No sustituye el detector base (UNet + Transformer full-fit).
  - No entrena una nueva arquitectura profunda; optimiza la cinemática, consistencia de parches y topología local de enlaces.

---

## 3. Cambio acotado

- **Módulos y configuraciones nuevos**:
  - `core/temporal_association_016.py`: Motor de asociación temporal, regularización anisótropa, resolución de switches y evaluación métrica pareada.
  - `configs/temporal_association_016.json`: Configuración congelada de hiperparámetros (pesos $w_z, w_{xy}$, ventana $W$, umbrales de switch y gap).
  - `tests/test_temporal_association_016.py`: Suite de pruebas unitarias y de integración para la cinemática, invariantes topológicos y paridad métrica.
  - `ops/audit_temporal_inputs_016.py`: Auditor de procedencia de entradas y hashes previo al lanzamiento.
- **Drivers modificados**:
  - `ops/biohub.py`: Registro de la tarea `associate-temporal-016` en `CPU_TASKS` y despacho de ejecución.
  - `ops/verify.py`: Registro de verificación de fuentes y de release.
- **Archivos y modelos protegidos que NO cambian**:
  - Checkpoints originales DeepCenter y UNet.
  - Tensores de evidencia de `b0970-selector-corpus-011`.
  - Evaluador métrico oficial de `vendor/official_metric`.
  - Protocolo de evaluación `configs/protocol.json`.
- **Variantes predeclaradas**:
  - **Control**: Grafo de referencia original V19 congelado (score desarrollo: `0.9012742110283266`).
  - **Variante 1 (Anisotropic Trajectory Regularization + Switch Resolution)**:
    - $w_z = 0.85$, $w_{xy} = 0.65$, ventana $W=3$.
    - Radio de búsqueda de switch $\le 6.0\,\mu\text{m}$, $\Delta \text{res} \ge 1.0\,\mu\text{m}$, $\text{NCC} \ge 0.50$.
    - Gap closing desactivado.
  - **Variante 2 (Kinematic Regularization + Switch Resolution + Rama D 1-Frame Gap Recovery)**:
    - Todos los componentes de la Variante 1.
    - Cierre de huecos de 1 frame con $\text{dist} \le 4.5\,\mu\text{m}$, $\text{SNR} \ge 4.0$, $\text{NCC} \ge 0.50$, DeepCenter heatmap $\ge 0.25$, exclusión $\ge 2.0\,\mu\text{m}$.

---

## 4. Datos y ejecución

- **Entradas**: Los 53 archivos `.evidence.npz` precomputados en `artifacts/A1555/b0970-selector-corpus-011/`.
- **Aislamiento de inferencia**: Las transformaciones cinemáticas y topológicas operan exclusivamente sobre coordenadas predichas, velocidades y parches de imagen. El Ground Truth **nunca** se consulta durante la inferencia o propuesta de cambios; se abre únicamente en la etapa final de cálculo de la métrica oficial.
- **Recursos requeridos**: Perfil `cpu_evaluation_v1` (CUDA oculta, 2 CPUs lógicas, RLIMIT_AS 8 GiB, RAM libre $\ge 16$ GiB).
- **Identificadores previstos**:
  - Tests: `b0970-tests-016`
  - Ejecución: `b0970-temporal-016`
- **Comandos canónicos**:
  ```bash
  python ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-016
  python ops/biohub.py launch --node A1555 --task associate-temporal-016 --run-id b0970-temporal-016 --source-run b0970-selector-corpus-011 --tests-run b0970-tests-016
  ```

---

## 5. Verificación y decisión predeclarada

- **Criterios de validación (Development Gates)**:
  - $\Delta \text{score} \ge +0.003$ sobre el control pareado V19 ($0.90127$).
  - $\Delta J_{\text{edge}} \ge 0.0$ y $\Delta J_{\text{div}} \ge 0.0$.
  - $\Delta \text{score}_{44b6} \ge 0.0$ y $\Delta \text{score}_{6bba} \ge 0.0$ (sin regresión de grupo).
  - Regresión máxima por muestra: $\Delta J_{\text{adj\_edge}} \ge -0.015$ para todas las 8 muestras.
  - Cardinalidad visible: estrictamente $234{,}000 < N < 242{,}000$ filas.
- **Criterio de selección de variante para confirmación**:
  - Si ambas variantes superan los gates de desarrollo, se seleccionará para confirmación aquella con mayor $\Delta \text{score}$ global. En caso de empate técnico ($|\Delta| < 10^{-4}$), se elegirá la Variante 1 por mayor parsimonia.
- **Acción tras el resultado**:
  - Si la candidata seleccionada aprueba confirmación ($\Delta \text{score} \ge +0.002$): Se empaqueta inmediatamente el offline bundle, se ejecuta smoke test Kaggle en destino aislado y se procede a la publicación autónoma en Kaggle mediante `ops/kaggle.py`.
  - Si la candidata no aprueba los gates: Se registra formalmente el rechazo en `STATUS.md$, se conservan intactos todos los artefactos y receipts, y se avanza hacia la siguiente hipótesis en la hoja de ruta sin relanzar mutaciones no comprobadas.
