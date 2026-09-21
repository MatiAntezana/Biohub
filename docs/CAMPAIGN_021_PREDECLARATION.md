# Campaña 021: Acoplamiento Dual de Regularización Axial Simétrica Pura y Adición Guiada por Ranker de Divisiones Mitóticas (Rama E + Rama A/C)

- **Campaña/Fecha/Responsable**: Campaña 021 (`b0970-temporal-021`), 2026-09-14, Antigravity (autoridad de diseño incremental y ejecución autónoma según `AGENTS.md` y `docs/AUTONOMOUS_CONTINUATION.md`).
- **Hipótesis**: Acoplamiento Dual: Rama E (Regularización Axial Simétrica Pura) + Rama A/C (Adición Guiada por Potencial de Ranker de Divisiones Mitóticas Bipartitas con Divergencia Cinética hacia Hijas Huérfanas).
- **Run Fuente**: `b0970-selector-corpus-011` (SHA256 receipt: `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`).
- **Modelos de Ranking Fuente**: `b0970-events-015` (`event-ranker-44b6.json` SHA256: `3b9ccbe5bbbeea96ffb75d55eba67a985ef6cb7434396f5580139e2751ab96d3`, `event-ranker-6bba.json` SHA256: `2fffe5ac5e06b7c08b6dfba2deec7735c38f3ccc9d71277f64df9a40def65135`).
- **Nodo de Ejecución**: `A1555` (RTX 4090 libre, CPU libre, 168 GiB disco libre, perfil `cpu_evaluation_v1`).
- **Release/Control Archivados**: Baseline de referencia congelado V19 con paridad estricta y guardias de colisión. Score de referencia en los 8 datasets de desarrollo: **0.9012742110**.

---

## 1. Hipótesis Falsable

### Cuello de Botella Medido
1. **Regularización Axial Pura (Arm 1)**: Demostró de forma reproducible un incremento de 1.015235$ en `44b6_ddf577ad` con **cero regresiones** en los otros 7 datasets de desarrollo, fijando un micro-average de 1.001788$.
2. **Cuello de Botella Mitótico (Arm 2)**:
   - En la referencia, el Jaccard mitótico es de apenas zsh.130435$ (=3, FP=8, FN=12$). Hay 12 eventos de división verdaderos no detectados (=12$).
   - La auditoría de cobertura de la Campaña 014 demostró que el 100% de las mitosis no enlazadas poseen cinemática asimétrica: la primera hija permanece cercana a la madre ( \le 3.0\,\mu	ext{m}$, conectada en el grafo de referencia), mientras que la segunda hija ($) es una detección huérfana en el frame +1$ que inicia una pista madura divergente.
   - El evaluador de potenciales de ranker 015 demostró que las transacciones de adición de mitosis genuinas poseen un delta de potencial fuertemente positivo ($\Delta \Phi > +0.40$ a 1.95$), debido al aporte de correlación de parche (1.117$), contraste (1.125$) y eliminación de huérfana (569X(-0.083) = +0.083$).

### Mecanismo Propuesto
1. **Brazo 1 (Base Invariable)**: Suavizado Z condicional exclusivo para reversiones de dirección simétricas puras (`reversal_only=True, symmetric_only=True, clamp_um=0.6, jitter_threshold_um=0.8, weight_z=0.80`).
2. **Brazo 2 (Adición Mitótica Guiada por Ranker)**:
   - Identifica madres con exactamente un hijo $ dentro de radio intermitótico {	ext{child}} \le 3.5\,\mu	ext{m}$.
   - Busca candidatas a segunda hija $ en el frame +1$ que sean huérfanas legítimas (in-degree == 0, out-degree == 1) y cumplan distancia entre hermanas (.0\,\mu	ext{m} \le d(d_1, d_2) \le 14.0\,\mu	ext{m}$).
   - **Filtro de Divergencia Post-Citoquinesis**: Requiere que en +2$ las nietas $ y $ se separen activamente ((gc_1, gc_2) > d(d_1, d_2) 	imes (1 + \epsilon_{	ext{div}})$).
   - **Evaluación por Ranker 015**: Evalúa la transacción de adición $\Delta \Phi$ con el modelo 015 correspondiente al grupo (`44b6` o `6bba`).
   - Admite solo adiciones cuyo score supere el umbral calibrado $	au$, con cota estricta de adiciones por volumen ( \le 10$ a $) para preservar holgadamente el presupuesto de filas ({,}000 < 	ext{filas} < 242{,}000$).

### Qué Observación Rechazaría la Hipótesis
- Que la adición de aristas mitóticas aumente los falsos positivos de división ({	ext{div}}$) degradando el Jaccard mitótico o el Jaccard de aristas.
- Que ninguna variante alcance $\Delta 	ext{score}_{	ext{dev}} \ge +0.00300$.

---

## 2. Predeclaración de Variantes

- **Variant 1 (Control de Suavizado Puro)**:
  - Suavizado axial simétrico puro (`reversal_only=True, symmetric_only=True, clamp=0.6, thresh=0.8, w_z=0.80`).
  - `fork_addition: enabled=false`.
  - Desempeño establecido: 1.001788$, 0 regresiones.

- **Variant 2 (Acoplamiento Conservador)**:
  - Suavizado axial simétrico puro.
  - `fork_addition`: `enabled=true`, `min_gain=0.80`, `max_additions=10`, `min_divergence_ratio=1.10`, `max_sister_um=12.0`, `max_d2_um=10.0`.

- **Variant 3 (Acoplamiento Calibrado)**:
  - Suavizado axial simétrico puro.
  - `fork_addition`: `enabled=true`, `min_gain=0.65`, `max_additions=25`, `min_divergence_ratio=1.05`, `max_sister_um=14.0`, `max_d2_um=12.0`.

---

## 3. Gates Cuantitativos Predeclarados

1. **Gate de Desarrollo**:
   - $\Delta 	ext{score}_{	ext{dev}} \ge \mathbf{+0.00300}$ sobre el control (zsh.901274$).
   - Cero regresiones en grupos: $\Delta 	ext{score}_{	ext{44b6}} \ge 0.00000$ y $\Delta 	ext{score}_{	ext{6bba}} \ge 0.00000$.
   - Jaccard de aristas micro-average no decreciente: $\Delta J_{	ext{edge}} \ge 0.00000$.
2. **Gate de Confirmación**:
   - Para la mejor variante de desarrollo: $\Delta 	ext{score}_{	ext{conf}} \ge \mathbf{+0.00200}$ sobre confirmación pareada.
3. **Cota de Cardinalidad de Filas Kaggle**:
   - Estrictamente {,}000 < 	ext{filas} < 242{,}000$.
4. **Seguridad y Paridad**:
   - Inferencia estrictamente label-free (GT jamás se abre durante la generación o filtrado).
   - Manifiesto inmutable de código y hashes reproducibles.

---

## 4. Cadena de Ejecución

1. Implementación en `core/temporal_association_021.py`, `configs/temporal_association_021.json`, `tests/test_temporal_association_021.py`.
2. Verificación estática y suite local completa: `/Users/matias/miniforge3/envs/AIEnv/bin/python ops/verify.py`.
3. Ejecución de suite remota en A1555: `ops/biohub.py launch --node A1555 --task tests --run-id b0970-tests-021-rel1`.
4. Lanzamiento de asociación 021 en A1555: `ops/biohub.py launch --node A1555 --task associate-temporal-021 --source-run b0970-selector-corpus-011 --tests-run b0970-tests-021-rel1 --run-id b0970-temporal-021`.
5. Fetch completo y evaluación de gates.
6. Si pasa: empaquetado con `ops/bundle_build.py` y envío a Kaggle con `ops/kaggle.py`.
