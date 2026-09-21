# Predeclaración Formal — Campaña 020: Regularización Axial Pura en Z y Poda de Bifurcaciones Mitóticas Espurias de Alta Precisión (Doble Brazo)

Predeclaración formal dentro del protocolo autónomo de Biohub (`docs/AUTONOMOUS_CONTINUATION.md` y `AGENTS.md`), redactada de acuerdo con la plantilla canónica `docs/NEXT_EXPERIMENT_TEMPLATE.md`.

Congelada **ANTES** de generar resultados de selección o ejecutar transformaciones de grafo en destino. Cualquier modificación ulterior exige una nueva versión o release.

---

## 1. Identidad y punto de partida

- **Campaña / Fecha / Responsable**: Campaña 020 / 2026-09-14 / Antigravity (autoridad de continuidad autónoma, conforme a instrucción vigente del usuario del 2026-09-13).
- **Hipótesis**: **Doble Brazo (Dual-Arm)**:
  - **Brazo 1 (Base congelada)**: Regularización Axial Pura y Simétrica en Z (`reversal_only=true`, `symmetric_only=true`, $\text{clamp}=0.6\,\mu\text{m}$, $\text{thresh}=0.8\,\mu\text{m}$, $w_z=0.80$, $w_{xy}=0.0$). Aprobada en Campaña 019 Release 4 como base con **cero regresiones en los 8 datasets de desarrollo** y ganancia masiva de $+0.015235$ en `44b6_ddf577ad` ($+0.001788$ micro-promedio).
  - **Brazo 2 (Intervención topológica)**: Poda de Bifurcaciones Mitóticas Espurias de Alta Precisión guiada por el Ranker Convexo de Eventos 015 (`event-ranker-44b6.json` y `event-ranker-6bba.json`) para erradicar las 8 divisiones falsas positivas que colapsan el Jaccard de división ($J_{\text{div}}$) y penalizan el Jaccard de aristas ajustado ($J_{\text{adj\_edge}}$).
- **Run fuente y hashes de origen**:
  - Run fuente principal: `b0970-selector-corpus-011` en nodo `A1555`.
  - Hashes de artefactos fuente verificados:
    - `receipt.json`: `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`
    - `corpus.json`: `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`
    - 53 tensores `*.evidence.npz` verificados byte a byte.
  - Artefactos del Ranker de Eventos 015 (`artifacts/A1555/b0970-events-015/`):
    - `event-ranker-44b6.json`: SHA-256 `3b9ccbe5bbbeea96ffb75d55eba67a985ef6cb7434396f5580139e2751ab96d3`
    - `event-ranker-6bba.json`: SHA-256 `2fffe5ac5e06b7c08b6dfba2deec7735c38f3ccc9d71277f64df9a40def65135`
    - `models.json`: SHA-256 `d87e07a3c3328e1d74659b854a85ba46a06655c65f97bc4a7d656910cb990799`
    - `receipt.json` (015): SHA-256 `e9ffbb25a3d7d45f3c64c12658a5e8c187be09a06141ae645719e83ec9e8770c`
- **Nodo de ejecución y perfil de admisión**:
  - Nodo de ejecución: `A1555` (nodo preferente con cadena histórica continua, RTX 4090 ociosa, 29.2 GiB RAM disponibles, 168.2 GiB disco libre).
  - Nodo secundario para tareas CPU/tests: `A1557` podrá utilizarse exclusivamente tras la verificación completa de dependencias, inventario y release operativa estipulada en `AUTONOMOUS_CONTINUATION.md`, Sección 3. No se ejecutará ningún trabajo científico en A1557 hasta validar su suite completa y receipts de staging.
  - Perfil de recursos: `cpu_evaluation_v1`. Tarea acotada a 2 threads lógicos, `nice >= 10`, `RLIMIT_AS` 8 GiB, CUDA oculta antes de imports, sin uso de GPU ajena ni modificación de procesos del sistema.
- **Release y control archivados**:
  - Control de referencia: Grafo de referencia V19 congelado de `b0970-selector-corpus-011` (score oficial de desarrollo = $0.901274$).
  - Campeona histórica de referencia: Score de test Kaggle documentado de $0.933$. Meta final: submission en estado COMPLETE con score $\ge 0.970$.
- **Exposiciones previas y limitaciones de independencia**:
  - Conjunto de desarrollo (8 datasets pareados): `44b6_c50204e0`, `44b6_d5e7d891`, `44b6_ddf577ad`, `44b6_e57ff5c6`, `6bba_07e24132`, `6bba_09961292`, `6bba_48816121`, `6bba_5a4d9360`.
  - Conjunto de confirmación (8 datasets reservados): `44b6_144b256d`, `44b6_267148e4`, `44b6_5f15d135`, `44b6_7a302da0`, `6bba_20852818`, `6bba_4aa14c89`, `6bba_61dd1e0d`, `6bba_74686d6a` (estratificados según split formal).
  - Limitación de independencia: Las muestras de supervisión 014 (`44b6_12dfb391`, `44b6_a21120c2`, `44b6_e28840c6`) fueron utilizadas para el entrenamiento del ranker 015 y permanecen estrictamente excluidas de desarrollo y confirmación.
  - Inferencia 100% label-free: Ni las coordenadas ni la topología leen GT durante la inferencia o la toma de decisiones de poda. GT ingresa única y exclusivamente al final a través de `tracking_cellmot.metrics.evaluate()`.

---

## 2. Hipótesis falsable

### 2.1 Cuello de botella medido y análisis matemático de la métrica oficial

1. **Estructura de la Métrica Oficial**:
   $$\text{Score} = J_{\text{adj\_edge}} + 0.1 \times J_{\text{div}}$$
   donde:
   $$J_{\text{div}} = \frac{TP_{\text{div}}}{TP_{\text{div}} + FP_{\text{div}} + FN_{\text{div}}}$$
   y $J_{\text{adj\_edge}}$ es el Jaccard de aristas ponderado por el factor de recall de nodos.

2. **Diagnóstico Cuantitativo en el Panel de Desarrollo (8 datasets)**:
   - **Divisiones en la Referencia V19**:
     - Verdaderos Positivos ($TP_{\text{div}}$): **3** (`6bba_09961292`: 2, `6bba_48816121`: 1).
     - Falsos Positivos ($FP_{\text{div}}$): **8** (`44b6_c50204e0`: 2, `44b6_ddf577ad`: 3, `44b6_e57ff5c6`: 1, `6bba_09961292`: 2).
     - Falsos Negativos ($FN_{\text{div}}$): **12** (`44b6_c50204e0`: 2, `44b6_d5e7d891`: 2, `6bba_07e24132`: 2, `6bba_09961292`: 2, `6bba_48816121`: 4).
     - **Predicciones Totales de División**: $TP + FP = 3 + 8 = 11$.
     - **Tasa de Falsa Alarma**: ¡**$8 / 11 = 72.7\%$** de las divisiones predichas por la referencia son falsos positivos!
     - **Jaccard de División de la Referencia**:
       $$J_{\text{div}}^{\text{ref}} = \frac{3}{3 + 8 + 12} = \frac{3}{23} \approx 0.130435 \implies 0.1 \times J_{\text{div}}^{\text{ref}} = 0.013043$$
   - **Penalización Pura en Datasets sin Divisiones**:
     - En `44b6_ddf577ad`, el GT tiene exactamente 0 divisiones verdaderas ($FN=0, TP=0$), pero la referencia predice 3 divisiones ($FP=3$).
     - En `44b6_e57ff5c6`, el GT tiene 0 divisiones verdaderas, pero la referencia predice 1 división ($FP=1$).
     - En ambos datasets, toda bifurcación predicha es una pura penalización matemática que reduce $J_{\text{div}}$ y añade aristas falsas hacia células erróneas o ruido ($FP_{\text{edge}}$).

3. **Limitación de la Campaña 019**:
   - En la Campaña 019 Release 4, la regularización axial simétrica pura (`reversal_only=true`, `symmetric_only=true`, $\text{clamp}=0.6\,\mu\text{m}$, $w_z=0.80$) logró $+0.015235$ en `44b6_ddf577ad` y exactamente $0.000000$ en los otros 7 datasets (+0.001788 micro-promedio).
   - Sin embargo, al mantener la topología 100% fija ($\Delta \text{forks} = 0$), el aporte de divisiones permaneció en $\Delta J_{\text{div}} = 0.0$, quedando a las puertas del gate de desarrollo ($+0.001788 < +0.003000$).

### 2.2 Mecanismo propuesto y por qué estos datos lo apoyan

- **Brazo 1 (Preservación de Ganancia Axial)**: Fijar la Regularización Axial Simétrica Pura como base inmutable del pipeline, asegurando $+0.001788$ sin riesgo de regresión en ningún dataset.
- **Brazo 2 (Poda Mitótica de Alta Precisión con Ranker 015)**:
  - En la Campaña 015 se entrenó un ranker convexo de eventos L2 regularizado sobre 34 diferencias de potenciales del grafo, evaluando características de parches 3D, contrastes, velocidades parentales y coherencia geométrica.
  - En `6bba`, el ranker 015 alcanzó **$0.9983$ AUROC** con 0 propuestas dañinas seleccionadas a margen estricto.
  - Para cada bifurcación $(m \to \{d_1, d_2\})$ en el grafo de referencia, se evalúa la transacción de poda a nivel de rama:
    - $T_1 = (m \to [d_1])$ (desconectar la rama espuria hacia $d_2$, dejando $d_1$ como continuación lineal).
    - $T_2 = (m \to [d_2])$ (desconectar la rama espuria hacia $d_1$, dejando $d_2$ como continuación lineal).
  - La referencia intacta posee por definición potencial nulo ($\Delta \Phi_{\text{ref}} = 0.0$).
  - Si la desconexión de una rama produce una ganancia de potencial $\Delta \Phi > \tau$ (donde $\tau$ es el umbral de margen calibrado por encima del máximo confirmed harmful), se poda la rama espuria y se convierte la bifurcación en una trayectoria lineal simple.
  - **Protección Natural de Divisiones Verdaderas**: En una división verdadera, ambas células hijas presentan alta correlación de parche, simetría de distancia y vectores coherentes; intentar podar cualquiera de ellas destruye potencial ($\Delta \Phi \le 0$). En cambio, en bifurcaciones espurias, una rama es el track genuino y la otra es una captura errónea de baja correlación o distancia excesiva; podar la rama falsa incrementa significativamente el potencial ($\Delta \Phi \gg \tau$).
- **Impacto Proyectado en la Métrica**:
  - Si se erradican 4 divisiones falsas positivas ($FP_{\text{div}}: 8 \to 4$) sin alterar los 3 TP:
    $$J_{\text{div}} = \frac{3}{3 + 4 + 12} = \frac{3}{19} \approx 0.157895 \implies \Delta J_{\text{div}} \approx +0.027460 \implies \Delta \text{score}_{\text{div}} = +0.002746$$
    Sumado a $+0.001788$ de regularización axial y $+0.0010$ de reducción de $FP_{\text{edge}}$, el delta alcanza $+0.0055 \gg +0.00300$.
  - Si se erradican las 8 divisiones falsas positivas ($FP_{\text{div}}: 8 \to 0$):
    $$J_{\text{div}} = \frac{3}{3 + 0 + 12} = \frac{3}{15} = 0.200000 \implies \Delta J_{\text{div}} \approx +0.069565 \implies \Delta \text{score}_{\text{div}} = +0.006957$$
    Delta global total proyectado: $+0.001788 + 0.006957 + 0.002000 \approx \mathbf{+0.010745}$.

### 2.3 Qué observación concreta lo rechazaría (Criterios de falsación)

1. Si la poda elimina cualquiera de las 3 divisiones verdaderas en desarrollo ($\Delta TP_{\text{div}} < 0$, es decir, si $TP_{\text{div}}$ cae de 3 a $< 3$).
2. Si alguna variante induce regresión neta en cualquiera de los dos grupos de linaje: $\Delta \text{score}_{44b6} < 0.0000$ o $\Delta \text{score}_{6bba} < 0.0000$.
3. Si ninguna variante alcanza el gate cuantitativo de desarrollo: $\Delta \text{score}_{\text{dev}} < +0.00300$.
4. Si el Jaccard de división colapsa ($\Delta J_{\text{div}} < 0.0$) o el Jaccard de aristas ajustado sufre una regresión local mayor a $-0.015$ en cualquier dataset individual.

### 2.4 Qué incertidumbre NO resuelve el ensayo

- No resuelve los 12 falsos negativos de división ($FN_{\text{div}} = 12$) derivados de células hijas no detectadas en la segmentación base o fuera del radio de búsqueda de 12 µm.
- No reentrena la segmentación 3D ni modifica los pesos de los checkpoints primario/secundario del detector congelado.

---

## 3. Cambio acotado y diseño de variantes

### 3.1 Módulos y drivers

- **Nuevos módulos versionados**:
  - `core/temporal_association_020.py`
  - `configs/temporal_association_020.json`
  - `tests/test_temporal_association_020.py`
  - `ops/audit_temporal_inputs_020.py`
- **Drivers modificados**:
  - `ops/biohub.py`: Incorporación de la tarea `associate-temporal-020` en `CPU_TASKS`, verificación de admisión en A1555 y validación de la suite de 315 tests.
  - `ops/verify.py`: Registro del task y validación de dependencias.
- **Archivos y modelos protegidos inmutables**:
  - Todo el pipeline 001-015, evaluador oficial `vendor/official_metric`, checkpoints de red y tensores `.evidence.npz`.

### 3.2 Definición estricta de variantes predeclaradas

- **Control (Baseline inmutable)**:
  - Grafo de referencia V19 congelado de `b0970-selector-corpus-011` (score de desarrollo = $0.901274$, $TP_{\text{div}} = 3$, $FP_{\text{div}} = 8$).

- **Variante 1 (Arm 1 Benchmark — Regularización Axial Pura y Simétrica solamente)**:
  - *Regularización en Z*: `enabled=true`, `reversal_only=true` ($dz_1 \cdot dz_2 < 0$), `symmetric_only=true` ($z_{\text{prev}} == z_{\text{next}}$), `jitter_threshold_um=0.8`, `max_correction_um=0.6`, `weight_z=0.80`, `weight_xy=0.0`.
  - *Poda de bifurcaciones*: Desactivada (`pruning=false`, $\Delta \text{forks} = 0$, $\Delta \text{edges} = 0$).
  - *Función*: Benchmark de control pareado que garantiza la reproducción exacta de la ganancia de $+0.001788$ sin regresiones observada en la Campaña 019 Release 4.

- **Variante 2 (Dual-Arm Conservador — Regularización Simétrica + Poda de Bifurcaciones a Margen Estricto)**:
  - *Regularización en Z*: Idéntica a Variante 1.
  - *Poda de bifurcaciones*: `enabled=true`.
  - *Modelos de Ranker*: `event-ranker-44b6.json` y `event-ranker-6bba.json` de la Campaña 015.
  - *Umbral de Margen Conservador*:
    - $\tau_{44b6} = 0.50$ (estrictamente por encima del máximo confirmed harmful de calibración).
    - $\tau_{6bba} = 0.40$ (estrictamente por encima del margen diagnóstico $0.2723$).
  - *Margen de separación entre ramas*: $\Delta \Phi(d_{\text{keep}}) - \Delta \Phi(d_{\text{prune}}) \ge 0.05$.
  - *Regla de decisión*: Para cada bifurcación $(m \to \{d_1, d_2\})$, se calcula la ganancia de potencial de desconectar $d_1$ frente a $d_2$. Se poda la rama espuria si y solo si la mejor opción supera $\tau$ con una separación clara sobre la alternativa. En caso contrario, abstención total (se preserva la bifurcación intacta).

- **Variante 3 (Dual-Arm Margen Medio — Regularización Simétrica + Poda de Bifurcaciones a Margen Medio)**:
  - *Regularización en Z*: Idéntica a Variante 1.
  - *Poda de bifurcaciones*: `enabled=true`.
  - *Modelos de Ranker*: `event-ranker-44b6.json` y `event-ranker-6bba.json` de la Campaña 015.
  - *Umbral de Margen Medio*:
    - $\tau_{44b6} = 0.20$
    - $\tau_{6bba} = 0.25$
  - *Margen de separación entre ramas*: $\Delta \Phi(d_{\text{keep}}) - \Delta \Phi(d_{\text{prune}}) \ge 0.02$.
  - *Función*: Evaluar la captura de bifurcaciones espurias con ganancia moderada en datasets con alta densidad, preservando las guardias contra falsas podas.

*No habrá Variante 4 ni búsquedas de grilla abiertas.*

### 3.3 Reglas de Candidatos, Features y Criterios de Abstención

- **Extracción de Features de Poda**:
  - Se utilizan exactamente los 34 factores lineales del grafo definidos en `core/nextgen/repair_ranker.py` (deltas de aristas, deltas de forks, diferencias de correlación de parche, simetría axial y lateral, velocidades normalizadas, delta de contador de aristas, delta de forks y delta de huérfanos).
  - Normalización y estandarización idéntica a la congelada en los JSON del ranker 015.
- **Criterio de Abstención**:
  - Si para una bifurcación dada, ninguna de las dos opciones de poda alcanza el umbral $\tau$, la bifurcación se conserva íntegra.
  - Si ambas opciones están empatadas ($|\Delta \Phi_1 - \Delta \Phi_2| < \text{separation\_margin}$), el algoritmo se abstiene de intervenir.

---

## 4. Datos y ejecución

### 4.1 Particiones y hashes

- **Particiones oficiales de evaluación (`configs/protocol.json`)**:
  - **Desarrollo (8 datasets pareados)**:
    - `44b6`: `44b6_c50204e0`, `44b6_d5e7d891`, `44b6_ddf577ad`, `44b6_e57ff5c6`
    - `6bba`: `6bba_07e24132`, `6bba_09961292`, `6bba_48816121`, `6bba_5a4d9360`
  - **Confirmación (8 datasets reservados)**:
    - `44b6`: `44b6_144b256d`, `44b6_267148e4`, `44b6_5f15d135`, `44b6_7a302da0`
    - `6bba`: `6bba_20852818`, `6bba_4aa14c89`, `6bba_61dd1e0d`, `6bba_74686d6a`
- **Garantía de ausencia de GT en inferencia**:
  - La inferencia ejecuta sobre tensores de entrada `.evidence.npz` y el grafo de referencia de `corpus.json`.
  - No existe lectura, carga ni dependencia de archivos `.geff` durante la regularización de coordenadas ni durante el cálculo de scores del ranker.
  - El evaluador oficial solo recibe el grafo final postprocesado mediante `vendor/official_metric/src/tracking_cellmot/metrics.py`.

### 4.2 Recursos, presupuesto y aislamiento

- **Perfil de Admisión**: `cpu_evaluation_v1`.
  - CPU: Máximo 2 threads lógicos concurrentes.
  - Memoria: `RLIMIT_AS` configurado a 8 GiB por proceso.
  - Aislamiento CUDA: `CUDA_VISIBLE_DEVICES=""` forzado en el entorno antes de cualquier import de PyTorch o NumPy.
  - Disco requerido: $\ge 15$ GiB libres en partición de ejecución.
- **Presupuesto de Filas de Kaggle**:
  - Rango admisible estricto: **$234{,}000 < \text{filas} < 242{,}000$** (sin contar cabecera).
  - Referencia base V19: 235,674 filas.
  - Impacto de la poda: Cada rama podada elimina exactamente 1 arista (-1 fila en el CSV). Para una poda proyectada de entre 4 y 15 bifurcaciones, el CSV resultante tendrá entre 235,659 y 235,670 filas, perfectamente centrado en el presupuesto válido.

### 4.3 Identificadores de corrida y dependencias

- **Corrida de Tests**: `b0970-tests-020` (ejecución de la suite completa de tests en A1555; meta: $\ge 315$ tests pasados, 0 fallos, 0 errores, 0 omitidos).
- **Corrida Científica**: `b0970-temporal-020` (evaluación pareada oficial de las 3 variantes sobre los 8 datasets de desarrollo en A1555).
- **Comandos canónicos ejecutables**:
  ```bash
  # 1. Auditoría estática local de inputs sin GPU ni GT:
  /Users/matias/miniforge3/envs/AIEnv/bin/python ops/audit_temporal_inputs_020.py --runs artifacts/A1555

  # 2. Ejecución remota de tests de la Campaña 020 en A1555:
  /Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --task tests --run-id b0970-tests-020 --node A1555 --profile cpu_evaluation_v1

  # 3. Ejecución remota de evaluación pareada de la Campaña 020 en A1555:
  /Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py launch --task associate-temporal-020 --run-id b0970-temporal-020 --node A1555 --profile cpu_evaluation_v1

  # 4. Monitoreo de estado y recuperación íntegra de artefactos:
  /Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py status --task associate-temporal-020 --run-id b0970-temporal-020 --node A1555
  /Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py fetch --task associate-temporal-020 --run-id b0970-temporal-020 --node A1555
  ```

---

## 5. Verificación y decisión predeclarada

### 5.1 Gates cuantitativos de decisión

1. **Gate de Desarrollo (Fase 1)**:
   - $\Delta \text{score}_{\text{dev}} \ge +0.00300$ sobre el control de referencia pareado.
   - $\Delta \text{score}_{44b6} \ge 0.0000$ (cero regresión en grupo 44b6).
   - $\Delta \text{score}_{6bba} \ge 0.0000$ (cero regresión en grupo 6bba).
   - $\Delta \text{division\_tp} \ge 0$ (cero pérdida de divisiones verdaderas, preservando $TP \ge 3$).
   - $\Delta J_{\text{div}} \ge 0.0$ (no regresión en Jaccard de divisiones).
   - $\Delta J_{\text{edge}} \ge 0.0$ (no regresión en Jaccard de aristas).
   - $\min_{i} \Delta J_{\text{adj\_edge}, i} \ge -0.015$ (ningún dataset individual sufre regresión excesiva).

2. **Gate de Confirmación (Fase 2, condicionado al éxito de Fase 1)**:
   - Se selecciona **únicamente la mejor variante de desarrollo** según criterio predeclarado ($\max \Delta \text{score}_{\text{dev}}$).
   - Evaluación sobre el panel de confirmación (8 datasets reservados).
   - $\Delta \text{score}_{\text{conf}} \ge +0.00200$.
   - Sin regresión en ningún grupo: $\Delta \text{score}_{\text{conf}, 44b6} \ge 0.0$ y $\Delta \text{score}_{\text{conf}, 6bba} \ge 0.0$.
   - $\Delta \text{division\_tp}_{\text{conf}} \ge 0$.

3. **Criterio de Presupuesto y Formato**:
   - Conteo de filas del CSV de submission: estrictamente $234{,}000 < \text{filas} < 242{,}000$.
   - Tiempo de ejecución end-to-end: $< 7200$ segundos.

### 5.2 Árbol de decisión y contingencias

- **Si una variante pasa el gate de desarrollo**:
  - Congelar la receta ganadora en `configs/temporal_association_020.json`.
  - Evaluar dicha variante única en el panel de confirmación.
  - Si supera el gate de confirmación: Proceder al empaquetado del bundle autónomo y smoke test visible con miras a la submission diaria autorizada hacia Kaggle.
- **Si todas las variantes fallan el gate de desarrollo ($\Delta < +0.00300$)**:
  - Declarar formalmente `rejected_no_kaggle_submission`.
  - Conservar íntegramente los artefactos y logs en `artifacts/A1555/b0970-temporal-020/`.
  - Extraer la auditoría detallada de bifurcaciones podadas (distribución de $\Delta \Phi$, falsos positivos extirpados vs falsos positivos remanentes).
  - No relanzar con hiperparámetros modificados sobre el mismo run. Proceder a la siguiente hipótesis de la hoja de ruta en `docs/AUTONOMOUS_CONTINUATION.md`.

---

## 6. Resultado (sección posterior; se completará tras la corrida)

*Esta sección se completará una vez finalizada la corrida `b0970-temporal-020` en A1555 y recuperados los artefactos correspondientes. La predeclaración previa permanecerá inmutable.*

- **Receipt / Hash / Estado**: *[Pendiente de ejecución]*
- **Métricas Oficiales Obtenidas**:
  - Score Control V19: *[0.901274]*
  - Score Variante 1: *[Pendiente]*
  - Score Variante 2: *[Pendiente]*
  - Score Variante 3: *[Pendiente]*
- **Deltas por Grupo y Muestra**: *[Pendiente]*
- **Auditoría de Poda de Bifurcaciones ($TP_{\text{div}}, FP_{\text{div}}, FN_{\text{div}}$)**: *[Pendiente]*
- **Disposición Final**: *[Pendiente]*
- **Próximo Paso / Comando**: *[Pendiente]*
