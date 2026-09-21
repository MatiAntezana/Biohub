# Predeclaración Oficial — Campaña 033: Rest-Frame Cytokinesis-Directed Mitosis Reassignment (DC-CMR)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub)  
**Código Hash Predeclarado**: `43f851675806625ab723f4d69bc742eef0db4dd3bd3b30b93a61f2157abf9012` (b0970-tests-033-rel3: 418/418 PASS)  
**Plantilla Base**: `docs/NEXT_EXPERIMENT_TEMPLATE.md`  
**Nodo de Ejecución Autorizado**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta)  
**Control Base de Desarrollo**: `b0970-selector-corpus-011` (score: `0.901274`)  
**Holdout Base de Confirmación**: `b0970-prepare-005` (`cache-confirmation`)  
**Campeona Vigente (Confirmación 032)**: Variante 2 (DDKS-SPI-Cytokinesis-V2, score: `0.955029`, delta: `+0.002776`)

---

## 1. Identidad y Punto de Partida

- **Campaña/fecha/responsable**: Campaña 033 / 2026-09-14 / Antigravity.
- **Hipótesis Científica**: Rest-Frame Cytokinesis-Directed Mitosis Reassignment (DC-CMR). Integración sinérgica de compensación de deriva global macroscópica de tejido $\vec{v}_{\text{global}}(t)$ y reasignación determinista de hijas robadas bajo simetría estricta de citocinesis en reposo, con exclusión total de inserción de nodos sintéticos (interpolación de gaps).
- **Evidencia Experimental Previa Directa**:
  1. **Descubrimiento del Robo de Hijas en el Solver ILP (`fae51bc0`, `4dc80dee`)**:
     - En el censo oficial de las 8 muestras de desarrollo (`cache-gt/*.geff`), existen **15 divisiones Ground Truth**. La referencia recupera solo **3 (TP=3)** y pierde **12 (FN=12)**.
     - El solver global ILP (`tracksdata.solvers.ILPSolver`) asigna peso de aparición $0.0$ y penalización de división $1.20$. La adición de una segunda hija siempre incurre en costo marginal $\Delta = 1.20 - p_2 > 0$ ($p_2 \le 1.0$), por lo que el ILP nunca selecciona divisiones; todas las divisiones del baseline provinieron del heurístico greedy `add_safe_divisions_postlink`.
     - De las 12 divisiones omitidas, **8 de ellas tuvieron a la segunda hija $d_2$ robada por un competidor lineal cercano no divisor $c$**: para $c$, enlazar $d_2$ tiene costo $-p(c \to d_2)$ con penalización $0.0$, mientras que para la madre $m$ cuesta $1.20 - p(m \to d_2)$. El detector ya había predicho la arista $p(m \to d_2) \ge 0.40$–$0.45$.
     - La reasignación determinista de $d_2$ desde $c$ hacia $m$ eleva el score de desarrollo de **$0.901274 \to 0.937095$ ($\Delta = +0.035821$)**, recuperando $+8$ divisiones verdaderas ($3 \to 11 / 15$) e incrementando el Jaccard de división de $0.1304 \to 0.4583$.
  2. **Deriva Global vs Grillas Espaciales (`4ceb66b7`)**:
     - El tejido embrionario presenta una deriva convectiva masiva de $1.5$–$2.7\ \mu\text{m/frame}$ (picos de $4.86\ \mu\text{m/frame}$), enmascarando los vectores celulares de citocinesis en el marco del laboratorio.
     - Grillas espaciales gruesas (2x2, 3x3) fallan severamente debido a la resolución axial gruesa de light-sheet ($\Delta z = 1.625\ \mu\text{m/slice}$): la mediana local por cuadrante salta erráticamente entre planos ("axial discretization snapping"). En la división GT de `6bba_48816121` en $t=60$, la grilla 2x2 alteró $\cos\theta$ de $-0.9285$ a $-0.0951$, causando su poda errónea.
     - La compensación por **Mediana Global de Cuadro ($\vec{v}_{\text{global}}(t)$)** elimina completamente el snapping axial, reduce la varianza de desplazamiento 3D en hasta $38.0\%$, y proporciona vectores puros de citocinesis en reposo.
  3. **Límites Calibrados de Simetría de Citocinesis (`cfb16db3`)**:
     - En las 15 divisiones biológicas verdaderas, el **$86.67\%$ ($13/15$)** cumple simultáneamente: $\cos\theta^* \le -0.30$, $d_{\text{sister}} \in [3.0, 16.0]\ \mu\text{m}$, y $\text{distance\_ratio} \ge 0.15$. El $100\%$ cumple la razón $\ge 0.15$.
  4. **Falsificación y Exclusión de Interpolación de Gaps (`827cc575`)**:
     - La inserción de nodos sintéticos para puentear gaps de 1 frame fue auditada sobre los datos crudos: arrojó **0% de precisión en GT** (las detecciones candidatas son ruido/muerte celular, ya que V19 ya cerró los gaps reales). Genera $+18$ falsos positivos de aristas, degradando el score en $-0.0025$. Queda terminantemente prohibida.
- **Run Fuente**: `b0970-selector-corpus-011` (desarrollo) y `b0970-prepare-005` (confirmación) en `A1555`.
  - Recibo selector: `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`.
  - Recibo confirmación 032 superada: `ae8b7a790355c4a64ae2ccd48b32b129526cac965c1ed09147024b40556b5d1e`.
- **Nodo de Ejecución Autorizado**: `A1555` (RTX 4090, 29.2 GiB RAM, 168 GiB disco libre). Perfil `cpu_evaluation_v1`.
- **Release y Campeona Vigente**:
  - Campeona de confirmación vigente: Campaña 032 Variante 2 (DDKS-SPI-Cytokinesis-V2), score $0.955029$ ($\Delta = +0.002776$).
  - Campeona de desarrollo vigente: Campaña 032 Variante 2, score $0.905280$ ($\Delta = +0.004006$).
- **Exposiciones Previas y Limitaciones de Independencia**:
  - Panel de desarrollo (8 datasets): Expuesto en campañas 016 a 032. Se utiliza para calibración y selección de la variante campeona.
  - Panel de confirmación (8 datasets holdout estricto): `44b6_2a2eff9f`, `44b6_587a1e22`, `44b6_c15fded2`, `44b6_d2f34f90`, `6bba_062c8d37`, `6bba_afb141ff`, `6bba_bb9f20c3`, `6bba_df673a83`. Rigurosamente aislado de todo tuning, diseño o ajuste de umbrales. Evaluado pareado únicamente en la fase final de confirmación.

---

## 2. Hipótesis Falsable y Formulación Matemática

### Cuello de Botella Medido
1. **Asimetría de Costo ILP**: El ILP penaliza divisiones con $+1.20$, mientras que un competidor lineal contiguo $c$ captura a la segunda hija $d_2$ con costo $-p(c \to d_2) + 0.0$. Esto causa que 8 divisiones verdaderas pierdan a su hija a manos de competidores lineales espurios.
2. **Deriva Convectiva**: El desplazamiento masivo del fluido altera la orientación espacial en el sistema de laboratorio, enmascarando la simetría antiparalela ($\theta \approx 180^\circ$) de la citocinesis biológica.

### Formulación Matemática de DC-CMR

Para cada transición de fotogramas $t \to t+1$:
1. **Vector de Deriva Global**:
   $$\vec{v}_{\text{global}}(t) = \text{median}_{e \in E_t} \left( (\vec{x}_{\text{target}(e)} - \vec{x}_{\text{source}(e)}) \odot \mathbf{s} \right)$$
   donde $\mathbf{s} = [1.625, 0.40625, 0.40625]\ \mu\text{m/vox}$.
2. **Vectores de Citocinesis en el Marco en Reposo**:
   Para la madre $m$ (enlazada a su primera hija $d_1$ en $E$) y una candidata a segunda hija $d_2$:
   $$\vec{v}_1^* = (\vec{x}_{d_1} - \vec{x}_m) \odot \mathbf{s} - \vec{v}_{\text{global}}(t)$$
   $$\vec{v}_2^* = (\vec{x}_{d_2} - \vec{x}_m) \odot \mathbf{s} - \vec{v}_{\text{global}}(t)$$
   $$\cos\theta^* = \frac{\vec{v}_1^* \cdot \vec{v}_2^*}{\max(\|\vec{v}_1^*\| \|\vec{v}_2^*\|, 10^{-6})}$$
   $$d_{\text{sister}} = \|(\vec{x}_{d_1} - \vec{x}_{d_2}) \odot \mathbf{s}\|$$
   $$\rho = \frac{\min(\|\vec{v}_1^*\|, \|\vec{v}_2^*\|)}{\max(\|\vec{v}_1^*\|, \|\vec{v}_2^*\|)}$$
3. **Regla de Reasignación**:
   Para toda madre $m \in V_t$ con $\text{out-degree}(m) = 1$:
   Buscar en $V_{t+1}$ una candidata $d_2 \neq d_1$ tal que su progenitor actual $c = \text{pred}(d_2)$ satisfaga $c \neq m$ y:
   - $\cos\theta^* \le \cos_{\max}$
   - $d_{\text{sister}} \in [3.0, d_{\max\_\text{sister}}]\ \mu\text{m}$
   - $\rho \ge 0.15$
   - $p_{\text{cand}}(m \to d_2) \ge p_{\min}$
   - $\text{out-degree}(c) = 1$ (o $c$ posee una división espuria podable)
   
   Si múltiples candidatas califican para $m$, seleccionar:
   $$d_2^* = \arg\min_{d_2} \cos\theta^* \quad (\text{máxima divergencia antiparalela})$$
   
   Efectuar la reasignación de aristas:
   $$E \leftarrow (E \setminus \{(c, d_2^*)\}) \cup \{(m, d_2^*)\}$$
   
4. **Destino del Competidor $c$ e Invariantes de Grafo**:
   - La arista $(c, d_2^*)$ es removida; el out-degree de $c$ pasa de 1 a 0 (terminación biológica limpia / desaparición en $t$).
   - Si $c$ tenía out-degree 2 (división falsa), $c$ pasa a out-degree 1, corrigiendo un falso positivo de división a trayecto lineal continuo.
   - **Invariantes preservados**: $\text{in-degree} \le 1$, $\text{out-degree} \le 2$, cero ciclos dirigidos.
   - **Balance de filas**: $\Delta \text{aristas} = -1 + 1 = 0$; $\Delta \text{nodos} = 0$. Cero impacto en el presupuesto de filas.

### Qué Observación Concreta Rechazaría la Hipótesis
- Que el delta de score en desarrollo no alcance $\Delta \text{score}_{\text{dev}} \ge +0.00300$.
- Que se produzca alguna regresión en subgrupos ($\Delta_{44\text{b}6} < 0$ o $\Delta_{6\text{bba}} < 0$).
- Que se pierda alguna división biológica verdadera existente en el control ($TP_{\text{div}} < 3$).
- Que en el panel de confirmación el delta sea inferior a $+0.00200$.

### Qué Incertidumbre NO Resuelve el Ensayo
- No recupera las 4 divisiones del censo donde la segunda hija no fue segmentada por la UNet ($d_2$ ausente del conjunto de detecciones).
- No resuelve divisiones con probabilidades neuronales de arista extremadamente bajas ($p < 0.05$).

---

## 3. Cambio Acotado y Variantes Predeclaradas

### Módulos Nuevos y Protegidos
- **Nuevos**:
  - `configs/temporal_association_033.json`: configuración canónica de variantes y gates.
  - `core/temporal_association_033.py`: motor de reasignación DC-CMR con deriva global.
  - `tests/test_temporal_association_033.py`: suite de pruebas unitarias y de integridad.
  - `ops/audit_temporal_inputs_033.py`: auditoría de entradas de la campaña 033.
- **Protegidos (Invariables)**:
  - Pesos congelados del detector y Transformer de inferencia V19.
  - Suavizado axial Z verificado de Campaña 032 ($w_z = 0.8$, jitter $0.8\ \mu\text{m}$, corrección máxima $0.6\ \mu\text{m}$).
  - Poda por Invariante de Proximidad Espacial (SPI) de Campaña 032.
  - Invariantes de linaje celular: $\text{in-degree} \le 1$, $\text{out-degree} \le 2$.
  - Prohibición estricta de MLflow e interpolación de gaps sintéticos.

### Control y Variantes Predeclaradas (3 Variantes)

1. **Control**:
   - Referencia dual-seed DeepCenter congelada de `b0970-selector-corpus-011` (score desarrollo: $0.901274$).
2. **Variante 1 (Conservadora)**:
   - $\cos\theta^* \le -0.40$
   - $d_{\text{sister}} \le 14.0\ \mu\text{m}$ (rango $[3.0, 14.0]\ \mu\text{m}$)
   - $p_{\text{cand}}(m \to d_2) \ge 0.50$
   - $\rho \ge 0.15$
   - Deriva: Mediana global por cuadro.
3. **Variante 2 (Campeona Balanceada - DC-CMR Champion)**:
   - $\cos\theta^* \le -0.30$
   - $d_{\text{sister}} \le 16.0\ \mu\text{m}$ (rango $[3.0, 16.0]\ \mu\text{m}$)
   - $p_{\text{cand}}(m \to d_2) \ge 0.45$
   - $\rho \ge 0.15$
   - Deriva: Mediana global por cuadro.
4. **Variante 3 (Sensible)**:
   - $\cos\theta^* \le -0.25$
   - $d_{\text{sister}} \le 16.0\ \mu\text{m}$ (rango $[3.0, 16.0]\ \mu\text{m}$)
   - $p_{\text{cand}}(m \to d_2) \ge 0.40$
   - $\rho \ge 0.15$
   - Deriva: Mediana global por cuadro.

### Criterios de Suficiencia y Abstención
- Si ninguna candidata cumple simultáneamente los 4 umbrales biológicos ($\cos, d_{\text{sister}}, \rho, p_{\text{cand}}$), la madre $m$ se abstiene y el grafo se preserva intacto.
- Si una candidata $d_2$ es pretendida por múltiples madres, se asigna a la que minimiza $\cos\theta^*$.

---

## 4. Datos y Ejecución

- **Particiones y Datos**:
  - FIT: 35 muestras de FIT011 históricas.
  - Desarrollo (8 datasets): `44b6_d5e7d891`, `44b6_c50204e0`, `44b6_e57ff5c6`, `44b6_ddf577ad`, `6bba_48816121`, `6bba_09961292`, `6bba_5a4d9360`, `6bba_07e24132`.
  - Confirmación Holdout (8 datasets): `44b6_2a2eff9f`, `44b6_587a1e22`, `44b6_c15fded2`, `44b6_d2f34f90`, `6bba_062c8d37`, `6bba_afb141ff`, `6bba_bb9f20c3`, `6bba_df673a83`.
- **Prueba de Inferencia Ciega a Etiquetas**:
  - DC-CMR utiliza exclusivamente coordenadas $3\text{D}+t$, la topología del grafo previo y probabilidades del detector (`*_candidates.npz`).
  - No accede a ninguna etiqueta ni almacén `.geff`.
  - El Ground Truth se consulta únicamente en la rutina oficial de evaluación pareada `assess`.
- **Recursos**: Perfil `cpu_evaluation_v1`. 2 hilos CPU, 8 GiB memoria virtual límite, CUDA oculta, tiempo máximo de ejecución 180s. Cero inferencia GPU nueva.
- **Identificadores Predeclarados**:
  - Suite de pruebas remota: `b0970-tests-033-rel1`
  - Desarrollo remoto: `b0970-temporal-033-rel1`
  - Confirmación remota: `b0970-confirm-033-rel1`
- **Comandos Oficiales de Ejecución**:
  - Auditoría local: `/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py audit_temporal_inputs_033`
  - Tests en A1555: `/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py execute_remote_tests --node A1555 --run-id b0970-tests-033-rel1`
  - Desarrollo en A1555: `/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py execute_temporal_033 --node A1555 --run-id b0970-temporal-033-rel1`
  - Confirmación en A1555: `/Users/matias/miniforge3/envs/AIEnv/bin/python ops/biohub.py execute_confirm_033 --node A1555 --run-id b0970-confirm-033-rel1`

---

## 5. Verificación y Gates Predeclarados

### Gate de Desarrollo (protocol.json)
Para que una variante sea declarada ganadora en desarrollo y elegible para confirmación:
- $\Delta \text{score}_{\text{dev}} \ge \mathbf{+0.00300}$
- Cero regresiones de grupo: $\Delta \text{score}_{44\text{b}6} \ge 0.00000$ y $\Delta \text{score}_{6\text{bba}} \ge 0.00000$
- Cota de peor delta individual por muestra: $\min_i \Delta_i \ge \mathbf{-0.01500}$
- Conservación estricta de divisiones biológicas: $TP_{\text{div}} \ge 3$ ($0\%$ de pérdida de divisiones verdaderas del control)

### Gate de Confirmación Holdout
Para que la candidata seleccionada sea declarada apta para empaquetado final y publicación a Kaggle:
- $\Delta \text{score}_{\text{conf}} \ge \mathbf{+0.00200}$
- Cero regresiones de grupo: $\Delta \text{score}_{44\text{b}6} \ge 0.00000$ y $\Delta \text{score}_{6\text{bba}} \ge 0.00000$
- Cota de peor delta individual: $\min_i \Delta_i \ge \mathbf{-0.01500}$
- Cero pérdidas de divisiones biológicas verdaderas: $TP_{\text{div}} \ge 4$ (sobre el holdout de 8 muestras)

### Presupuesto de Filas de Sumisión
- La sumisión final en Kaggle debe cumplir estrictamente:
  $$234{,}000 < \text{filas} < 242{,}000$$
  Dado que DC-CMR realiza reasignación 1 a 1 de aristas sin inserción sintética de nodos, el recuento de filas es idéntico al de la campeona 032 ($235{,}659$ filas), manteniéndose perfectamente en rango.

### Protocolo de Fallo y Éxito
- **Si falla**: Se preservan íntegros todos los recibos y registros de ejecución. No se eluden gates. Se documenta la causa raíz en STATUS.md y se formula la siguiente hipótesis dentro del protocolo.
- **Si pasa**: La variante campeona pasa de inmediato a empaquetado final de bundle offline (`ops/bundle_build.py`) y publicación oficial en Kaggle conforme a `ops/kaggle.py`.
