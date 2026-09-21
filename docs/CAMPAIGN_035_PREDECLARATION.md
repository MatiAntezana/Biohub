# Predeclaración Formal — Campaña 035: Drift-Compensated Linear Track Extension (DC-LTE) & Bio-Conserved Mitosis Gating (BC-MG)

## Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaña 035 / 2026-09-14 / Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub).
- **Hipótesis (D/E)**: Extensión y cierre de brechas temporales en trayectorias celulares lineales mediante compensación de deriva convectiva macroscópica $\vec{v}_{\text{global}}(t)$ (DC-LTE) sin riesgo mitótico ($\text{out\_degree} \le 1$), complementada opcionalmente con compuerta de conservación biofísica de 6 capas (BC-MG) para mitosis genuinas.
- **Run fuente**: `b0970-selector-corpus-011` (A1555), receipt sha256 `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`, corpus sha256 `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`. Score de control: `0.901274`.
- **Nodo de ejecución**: `A1555` bajo `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta, tiempo máximo 1800s).
- **Campeona de confirmación vigente**: Campaña 032 Variante 2 (DDKS-SPI-Cytokinesis-V2, confirmación: `0.955029`, $\Delta = +0.002776$).
- **Exposiciones previas**: El panel de desarrollo comprende 8 datasets oficiales de `b0970-selector-corpus-011` (`44b6_c50204e0`, `44b6_d5e7d891`, `44b6_ddf577ad`, `44b6_e57ff5c6`, `6bba_07e24132`, `6bba_09961292`, `6bba_48816121`, `6bba_5a4d9360`). GT reservado y exclusivo para la evaluación métrica final; nunca accesible durante la generación de candidatos.

---

## Hipótesis falsable

- **Cuello de botella medido**:
  - En las Campañas 033 y 034 se demostró matemáticamente que la reasignación libre de mitosis entre madres con 1 hija y segundas hijas robadas induce entre +3 y +23 falsos positivos de división debido al cruce antiparalelo de células no emparentadas en tejidos densos ($\cos\theta^* \le -0.95, p \ge 0.85$).
  - Por otro lado, la compensación de deriva convectiva $\vec{v}_{\text{global}}(t)$ probó restaurar la continuidad física del tejido embrionario (+0.003304 en Jaccard de aristas ajustadas en 033, y +0.001788 limpio en 034 Variante 1 con cero inflación de divisiones).
  - Existen miles de trayectorias celulares que se interrumpen prematuramente a tiempo $t$ ($\text{out\_degree} = 0$) mientras una célula huérfana comienza a tiempo $t+1$ ($\text{in\_degree} = 0$) a corta distancia física en el marco en reposo de la deriva del tejido.
  - Al unir un nodo terminal ($\text{out\_degree} = 0$) con un nodo raíz huérfano ($\text{in\_degree} = 0$), la cardinalidad de hijas permanece estrictamente $\le 1$. **El riesgo de falsos positivos de división es CERO**.
- **Mecanismo propuesto**:
  1. **DC-SPI Backbone**: Suavizado axial simétrico con compensación de deriva global ($w_z = 0.80$) + Poda DDKS-SPI de bifurcaciones espurias (idéntico a la campeona 032).
  2. **DC-LTE (Drift-Compensated Linear Track Extension)**:
     - Identificar nodos terminales $u$ a tiempo $t$ con $\text{out\_degree}(u) == 0$.
     - Identificar nodos huérfanos $v$ a tiempo $t+1$ con $\text{in\_degree}(v) == 0$.
     - Computar desplazamiento en marco en reposo: $\vec{d}^* = (\vec{r}_v - \vec{r}_u) - \vec{v}_{\text{drift}}(t)$.
     - Si $\|\vec{d}^*\| \le d_{\max}$ (e.g. $2.0\ \mu\text{m}$) y similitud de apariencia $\frac{\vec{q}_u \cdot \vec{q}_v}{\|\vec{q}_u\| \|\vec{q}_v\|} \ge \tau_{\text{sim}}$ (o $p(u \to v) \ge p_{\min}$), asignar arista $(u, v)$ de forma unívoca (emparejamiento codicioso 1-a-1 por distancia mínima).
  3. **BC-MG (Bio-Conserved Mitosis Gate - Variante 2)**:
     - Si se evalúa rescate de segunda hija de citocinesis, exigir compuerta biofísica de 6 capas:
       - Historia temporal de madre $\ge 3$ cuadros.
       - Continuidad futura de ambas hijas $\ge 2$ cuadros.
       - Alineación de momento del centro de masas $\cos(\vec{v}_{\text{mid}}, \vec{v}_{\text{in}}) \ge 0.45$.
       - Similitud morfológica de hermanas $\vec{q}_{d_1} \cdot \vec{q}_{d_2} \ge 0.75$.
       - Conservación de masa/contraste fluoróforo $|(s_{d_1} + s_{d_2}) - s_m| \le 0.35$.
       - Margen sobre competidor $\frac{p(m \to d_2)}{p(c \to d_2)} \ge 1.30$.
- **Qué observación concreta lo rechazaría**:
  - $\Delta_{\text{score}} < +0.003000$ en desarrollo frente a `b0970-selector-corpus-011`.
  - Si DC-LTE introduce discontinuidades que empeoren el Jaccard de aristas ajustadas.
- **Qué incertidumbre NO resuelve el ensayo**:
  - No resuelve divisiones donde el detector omitió ambas hijas simultáneamente.

---

## Cambio acotado

- **Módulos nuevos y drivers**:
  - `core/temporal_association_035.py`: Implementa DC-SPI + DC-LTE + BC-MG.
  - `configs/temporal_association_035.json`: Configuración estricta de las 3 variantes.
  - `ops/audit_temporal_inputs_035.py`: Verificación de integridad de fuentes de entrada.
  - `tests/test_temporal_association_035.py`: Suite de tests unitarios y propiedades invariantes.
  - Registro en `ops/biohub.py`: Tareas `associate-temporal-035` y `confirm-temporal-035`.
- **Archivos protegidos que no cambian**:
  - Todos los módulos de 001–034 permanecen intactos y congelados.
  - Pesos, caches y checkpoints permanecen de solo lectura.
- **Control**:
  - `b0970-selector-corpus-011` (score: `0.901274`).
- **Variantes predeclaradas (máximo 3)**:
  - **Variante 1 (DC-LTE Pure: Zero Division Risk)**:
    - DC-SPI ($w_z = 0.80$) + DDKS-SPI.
    - DC-LTE: $d_{\max} = 2.0\ \mu\text{m}$, $p_{\min} = 0.65$, $\text{sim}_{\min} = 0.70$.
    - Mitosis reassignment: `enabled: false`.
  - **Variante 2 (DC-LTE + Bio-Conserved Mitosis Gate [BC-MG])**:
    - Variante 1 + compuerta biológica de 6 capas para hijas de citocinesis.
  - **Variante 3 (DC-LTE Conservative: High Precision Extension)**:
    - DC-SPI ($w_z = 0.80$) + DDKS-SPI.
    - DC-LTE: $d_{\max} = 1.5\ \mu\text{m}$, $p_{\min} = 0.75$, $\text{sim}_{\min} = 0.80$.
    - Mitosis reassignment: `enabled: false`.

---

## Datos y ejecución

- **Panel de Desarrollo**: 8 datasets de `b0970-selector-corpus-011`.
- **Inferencia**: Cero acceso a GT durante candidate generation. GT se carga exclusivamente para `evaluate_association_panel`.
- **Perfil de Recursos**: `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta).
- **Tiempo límite**: 1800s.

---

## Verificación y decisión predeclarada

- **Gate de desarrollo**: $\Delta_{\text{score}} \ge +0.003000$ sobre el control `b0970-selector-corpus-011` (`0.901274`).
- **Gate de confirmación posterior**: $\Delta_{\text{score}} \ge +0.002000$ en holdout ciego de confirmación.
- **Si falla**: Documentar veredicto `rejected_no_kaggle_submission`, conservar artefactos en `artifacts/A1555/`.
- **Campeona de publicación actual**: No se altera; sigue siendo Campaña 032 Variante 2.
