# Predeclaración de Experimento — Campaña 015

Ficha canónica inmutable de acuerdo con `docs/NEXT_EXPERIMENT_TEMPLATE.md` y `docs/AUTONOMOUS_CONTINUATION.md` (Sección 6.A). Redactada y congelada antes de ejecutar el fitting y selección de variantes.

---

## 1. Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaña 015 / 2026-09-14 / Antigravity (continuidad autónoma autorizada).
- **Hipótesis**: **Rama A — Aprendizaje de Ranking con Supervisión Deduplicada y Balanceada por Evento**.
- **Evidencia nueva habilitante**:
  - En la Campaña 013, el ranker se detuvo (`insufficient_supervision_no_submission`) debido a que el prefijo `44b6` contaba con solo 2 divisiones biológicas utilizables (< 3 requerido por `configs/event_ranker.json`).
  - La Campaña 014 auditó 38 datasets y confirmó que la muestra de supervisión autorizada `44b6_a21120c2:12535` (madre 179) aporta exactamente **1 nueva división biológica utilizable con 2 comparaciones confirmadas**, alcanzando el mínimo innegociable de **3 eventos biológicos utilizables en 3 volúmenes para `44b6`** (y 16 eventos en 11 volúmenes para `6bba`).
  - La Campaña 014 minó **217 nuevas preferencias directas únicas** (`new_direct`), completamente deduplicadas frente a la base 012.
- **Runs fuente y hashes de origen**:
  - `b0970-coverage-014` (A1555): SHA-256 `8fa9be728e2ed153d00e5578755ac46f98695d06191760d86578e0817cb16817`.
  - `b0970-coverage-corpus-014` (A1555): SHA-256 `95fab8963f5a95c21bc4b95d4763b456e7bf5822e597eabf19786aaaf608746b`.
  - `b0970-ranking-012` (A1555): Preferencias directas históricas (518 en `44b6`, 928 en `6bba`).
- **Nodos de ejecución autorizados**:
  - Primario: **A1555** (histórico continuo, 168 GB libres, RTX 4090 ociosa).
  - Reemplazo autorizado por el usuario para A1557: **A1688** (verificado con suite completa 285/285 aprobada en `b0970-tests-audit-1688`, dataset completo de 199 `.zarr` y 199 `.geff` presente, RTX 4090 ociosa, 101.5 GB libres).
- **Limitaciones de independencia**: Las 3 muestras añadidas (`44b6_12dfb391`, `44b6_a21120c2`, `44b6_e28840c6`) fueron previamente expuestas y están autorizadas **exclusivamente como supervisión de entrenamiento**, jamás como holdout, desarrollo o confirmación.

---

## 2. Hipótesis falsable

- **Cuello de botella medido**:
  - El generador actual sufre de baja precisión en bifurcaciones mitóticas debido a un ordenamiento de candidatos dominado por falsas alarmas geométricas.
  - La escasez de divisiones mitóticas en `44b6` impedía ajustar un potencial discriminativo sin incurrir en degeneración de clases o sobreajuste a una única muestra.
- **Mecanismo propuesto**:
  - Integrar el tensor de supervisión ampliado y deduplicado (`new-supervision-44b6.npz` + base 012) mediante una función de pérdida convexa que penalice las violaciones de orden de pares de candidatos relativos a la misma madre celular.
  - Ponderar por evento biológico individual (madre celular) en lugar de por número bruto de alternativas, evitando que madres con 15 alternativas sesguen el gradiente respecto a madres con 2 alternativas.
- **Qué observación concreta lo rechazaría (criterio de parada / rechazo)**:
  - Si en calibración fuente no se seleccionan al menos el 50% de las propuestas beneficiosas o se selecciona alguna propuesta perjudicial (`recall < 0.5` o `harmful > 0`).
  - Si en desarrollo cruzado no se obtiene al menos **+1 nuevo fork unique-top1** frente al control histórico sin regresión en AUROC confirmed-only.
  - Si la política de margen colapsa en abstención total ($\Delta \text{score} = 0.0$) o si introduce degradación de tracking ($\Delta \text{score} < +0.003$ en desarrollo).
- **Qué incertidumbre NO resuelve el ensayo**:
  - No resuelve el 29.7% de pérdidas por células no anotadas en el GT (`covered_without_confirmed_contrast`), ya que no inventa negativos artificiales.
  - No amplía el radio de búsqueda espacial (congelado en 12 µm) ni modifica el corte $K=6$.

---

## 3. Cambio acotado y diseño de variantes

- **Archivos/modelos protegidos que NO cambian**:
  - Checkpoints del detector (`assets/champion/primary` y `secondary`).
  - Checkpoint de DeepCenter (`assets/champion/deepcenter`).
  - Matching oficial de evaluación (radio 7 µm).
  - Reglas de filtrado de candidatos y geometría de división.
- **Control (Baseline)**:
  - Referencia congelada V19 / champion 012 (sin ediciones transaccionales, score de desarrollo 0.901274).
- **Variante 1 (Principal — Event-Balanced L-BFGS-B)**:
  - Función de potencial lineal sobre 34 diferencias de features normalizadas.
  - Regularización L2: $\lambda = 1.0$.
  - Ponderación: Inversa al número de comparaciones por evento biológico ($w_i = 1 / N_{\text{comp}(e)}$).
  - Optimización: L-BFGS-B sin intercepto (potencial nulo para abstención).
- **Variante 2 (Contraste — Stratified Event + Margin Scaling)**:
  - Misma arquitectura de potencial, pero con regularización adaptativa por estrato ($\lambda_{\text{direct}} = 0.5$, $\lambda_{\text{event}} = 2.0$) para balancear la masa de 217 preferencias directas contra los 3 eventos biológicos en `44b6`.
- **No habrá Variante 3**.

---

## 4. Datos y ejecución

- **Particiones inmutables**:
  - **Supervisión/Entrenamiento**: 35 muestras FIT históricas (12 en `44b6`, 23 en `6bba`) + 3 muestras de supervisión autorizadas (`44b6_12dfb391`, `44b6_a21120c2`, `44b6_e28840c6`).
  - **Calibración diagnóstica**: Calibración fuente fija (4 muestras).
  - **Desarrollo**: Desarrollo cruzado prefijo opuesto (4 muestras).
  - **Confirmación / Visible**: Estrictamente BLOQUEADOS para selección de hiperparámetros.
- **Aislamiento**:
  - Inferencia pura sin acceso a GT.
  - Tareas CPU en `cpu_evaluation_v1` (CUDA oculta, 2 hilos, 8 GiB límite de dirección).
  - MLflow estrictamente prohibido.
- **IDs de la campaña**:
  - `b0970-tests-015`
  - `b0970-events-015`

---

## 5. Criterios y Gates de Decisión Predeclarados

| Gate | Métrica / Criterio Requerido | Acción si Falla |
|---|---|---|
| **Gate de Admisión** | CPU aislada, $\ge 24$ GiB disco libre, tests 015 100% aprobados | Abortar antes de I/O |
| **Control de Degeneración** | $\ge 3$ eventos, $\ge 3$ madres benef., $\ge 10$ perjud. por prefijo | `insufficient_supervision` -> Rechazo |
| **Calibración (Fuente)** | Recall benef. $\ge 0.5$, Perjudiciales seleccionadas $= 0$ | `diagnostic_margin = null` -> Abstención |
| **Discriminación (Desarrollo)** | $\Delta \text{AUROC} \ge 0$, $\Delta \text{top1} \ge +1$, Perjudiciales $= 0$ | Rechazar variante, no aplicar grafos |
| **Tracking (Desarrollo)** | $\Delta \text{score} \ge +0.003$, caída local máx $\le 0.015$ | Rechazar variante para confirmación |
| **Confirmación** | Solo candidata ganadora; $\Delta \text{score} \ge +0.002$ | Rechazo de publicación Kaggle |

Cualquier fallo en estos gates documenta el resultado y detiene el avance hacia Kaggle de forma fail-closed.
