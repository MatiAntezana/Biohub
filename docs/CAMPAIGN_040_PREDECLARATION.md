# Predeclaración — Campaña 040: Bounded Divergence Persistence Gating (BDPG)

## 1. Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaign 040, 2026-09-15, Antigravity (autoridad de continuidad autónoma bajo `AGENTS.md`).
- **Hipótesis**: Bounded Divergence Persistence Gating (BDPG).
  - En la Campaña 039 (CCDG), la relajación de la guarda a $\cos \le 0.20$ con persistencia $\ge 2$ logró proteger el 100% de las divisiones biológicas verdaderas (`division_tp = 4/4`), resolviendo la regresión de 038.
  - Sin embargo, en el panel `44b6_d2f34f90`, se retuvo una bifurcación espuria efímera ($m=245$, donde la hija 2 da un salto de $4.143\,\mu\text{m}$ y muere a los 3 cuadros, mientras la hija 1 persiste 96 cuadros), generando un falso positivo de división y un falso positivo de arista respecto a Campaign 032.
  - La hipótesis falsable de BDPG establece que las citocinesis verdaderas bajo arrastre convectivo fuerte ($0.0 < \cos_{3d} \le 0.20$) producen células hijas que persisten a lo largo del linaje (longitud de pista $\ge 5$ cuadros) y nacen dentro de un radio acotado ($\le 6.0\,\mu\text{m}$), mientras que las bifurcaciones falsas por crossover transitorio involucran pistas cortas/efímeras ($L_2 \le 3$).
- **Run fuente, nodo de origen, hashes de receipts y artefactos**:
  - `source_run`: `b0970-selector-corpus-011` en `A1555` (hash de corpus: `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`).
  - `source_confirmation`: `b0970-prepare-005` en `A1555`.
- **Nodo de ejecución**: `A1555` exclusivamente. Perfil CPU `cpu_evaluation_v1`.
- **Campeona vigente**: Campaign 032 Variante 2 (`b0970-confirm-032-rel5`, score $0.955029$).

---

## 2. Hipótesis falsable

- **Cuello de botella medido**:
  - En Campaign 039 confirmación: `division_tp = 4/4` (óptimo), pero score $= 0.953897$ ($\Delta = +0.001644 < +0.002000$).
  - En `44b6_d2f34f90`: horquilla espuria $m=245$ retiene arista $(245, 359)$, con $\cos = +0.021$, $d_2 = 4.143\,\mu\text{m}$, longitud de hija $2 = 3$ cuadros.
- **Mecanismo propuesto**:
  - Segmentación de la guarda en 3 casos biofísicos:
    1. **Citocinesis obtusa divergente** ($\cos_{3d} \le 0.0$):
       Divergencia natural opuesta; protegida si ambas hijas persisten ($L_1, L_2 \ge 3$) o tienen forward heading positivo ($\text{head} > 0.3$).
    2. **Citocinesis inclinada por arrastre convectivo** ($0.0 < \cos_{3d} \le \cos_{\max}$):
       Protegida **únicamente si** ambas hijas exhiben persistencia robusta ($L_1, L_2 \ge 5$) **y** distancia de nacimiento acotada ($\max(d_1, d_2) \le 6.0\,\mu\text{m}$).
       Esto protege a la citocinesis verdadera en `44b6_587a1e22` ($L_1=14, L_2=77, d_{\max}=3.1\,\mu\text{m}$) y poda a la horquilla espuria 245 en `44b6_d2f34f90` ($L_2=3 < 5$).
    3. **Crossovers paralelos agudos** ($\cos_{3d} > \cos_{\max}$):
       Nunca inmunizados por persistencia; podados por DDKS-SPI.
- **Qué observación concreta lo rechazaría**:
  - Si en desarrollo la poda de divisiones verdaderas reduce `division_tp < 3`.
  - Si el delta de desarrollo no supera $+0.003$.
  - Si en confirmación se pierde alguna de las 4 divisiones verdaderas (`division_tp < 4`) o el delta no supera $+0.002000$.

---

## 3. Cambio acotado

- **Módulos/configs nuevos**:
  - `configs/temporal_association_040.json`
  - `core/temporal_association_040.py`
  - `tests/test_temporal_association_040.py`
- **Módulos modificados**:
  - `ops/verify.py` (registro de tests 040)
  - `ops/biohub.py` (registro de tareas `associate-temporal-040`, `confirm-temporal-040`)
- **Variantes declaradas**:
  - **Variante 1**: BDPG con $\cos_{\max} = 0.20$, `divergent_pers_min = 3`, `tilted_pers_min = 5`, `tilted_max_birth_dist_um = 6.0`.
  - **Variante 2**: BDPG con $\cos_{\max} = 0.25$, `divergent_pers_min = 3`, `tilted_pers_min = 5`, `tilted_max_birth_dist_um = 6.0`.

---

## 4. Datos y ejecución

- **Splits y paneles**:
  - **Desarrollo**: 3 paneles (`2a2eff9f`, `6bba_07e24132`, `6bba_df673a83`).
  - **Confirmación**: 3 paneles (`0f845d07`, `137456d9`, `e1c51478`).
- **Garantía Label-Free**: Candidatos generados estrictamente sin acceso a GT ni conocimiento de divisiones de oráculo. GT solo se abre en evaluación oficial pareada.
- **Recursos**: CPU exclusiva con límites `cpu_evaluation_v1` (2 hilos, 8 GiB memoria).
- **IDs de corrida**:
  - Tests: `b0970-tests-040-rel1`
  - Desarrollo: `b0970-temporal-040-rel1`
  - Confirmación (si dev pasa): `b0970-confirm-040-rel1`

---

## 5. Criterios de decisión predeclarados

- **Gate de Desarrollo**:
  - $\Delta \text{Score} \ge +0.003000$ vs referencia.
  - $\Delta \text{Score}_{44b6} \ge 0.0$.
  - $\Delta \text{Score}_{6bba} \ge 0.0$.
  - $\Delta \text{Score}_{\text{worst}} \ge -0.015$.
  - `division_tp >= 3`.
- **Gate de Confirmación**:
  - $\Delta \text{Score} \ge +0.002000$ vs referencia.
  - `division_tp >= 4` (100% de divisiones verdaderas conservadas).
  - Superar el score de confirmación de Campaign 032 ($0.955029$).
