# Predeclaración Formal de Campaña — Campaña 042: Adaptive Allometric Cytokinesis Gating (AACG)

## 1. Identidad y Autoridad de la Campaña

- **Campaña**: Campaign 042 (AACG).
- **Fecha**: 2026-09-15.
- **Autoridad de Ejecución**: Antigravity (bajo gobernanza de `AGENTS.md` y `docs/AUTONOMOUS_CONTINUATION.md`).
- **Nodo Autorizado**: `A1555` exclusivamente.
- **Code Hash**: `1b601815023722f3447eca246288e976474d1721231b3531bf3b6de018bdaf38`.
- **Source Run**: `b0970-selector-corpus-011`.
- **Runs Predeclarados**:
  - Tests: `b0970-tests-042-rel1`.
  - Desarrollo: `b0970-temporal-042-rel1`.
  - Confirmación (si desarrollo aprueba): `b0970-confirm-042-rel1`.

---

## 2. Evidencia Forense y Motivación Biofísica

1. **Récord Histórico Vigente (Campaign 040 BDPG)**:
   - Campaña 040 demostró que la estratificación biofísica en regímenes de divergencia angular (obtusa $\cos_{3d} \le 0.0$, inclinada por arrastre $0.0 < \cos_{3d} \le 0.20$ y paralela aguda $\cos_{3d} > 0.20$) protege el 100% de divisiones celulares reales (`division_tp = 4/4`) y reduce falsos positivos de 7 a 4, alcanzando el puntaje récord absoluto de confirmación de **`0.955208`** (+0.002956 vs ref).
   - En el grupo `6bba`, el puntaje de confirmación alcanzó **`0.964938`** con 3/3 divisiones verdaderas retenidas.
   - En el grupo `44b6`, el puntaje de confirmación fue de **`0.922451`** (4.2% menor que `6bba`), con 3 divisiones falsas positivas no podadas.

2. **Lección Cuantitativa de Campaña 041**:
   - Imponer un límite geométrico euclidiano uniforme o rígido ($d_{\text{daughters}} \le 10-11\,\mu\text{m}$) dañó al grupo `6bba` porque las células mesenquimales de motilidad rápida separan sus hijas a distancias de $11.5 - 14.0\,\mu\text{m}$ con desplazamientos respecto a la madre de hasta $7.0 - 7.5\,\mu\text{m}$.
   - Por el contrario, en `44b6` (epitelio de baja motilidad), las células hijas se separan estrictamente a $d \le 7.0\,\mu\text{m}$ y nacen a $d \le 6.0\,\mu\text{m}$.

3. **Hipótesis Científica de Campaña 042 (AACG)**:
   - Estratificar los límites de envoltura física de citocinesis de acuerdo con la alometría tisular de cada linaje celular:
     - **Grupo `44b6` (Epitelial)**:
       - $d_{\text{daughters}} \le 7.2\,\mu\text{m}$ (V1) / $7.5\,\mu\text{m}$ (V2).
       - $\max(d_1, d_2) \le 6.0\,\mu\text{m}$ (V1) / $6.2\,\mu\text{m}$ (V2).
       - $\text{mother\_track\_len} \ge 1 \lor t_m \le 0.0$.
     - **Grupo `6bba` (Mesenquimal)**:
       - $d_{\text{daughters}} \le 13.5\,\mu\text{m}$ (V1) / $14.5\,\mu\text{m}$ (V2).
       - $\max(d_1, d_2) \le 7.5\,\mu\text{m}$ (V1) / $8.0\,\mu\text{m}$ (V2).
       - $\text{mother\_track\_len} \ge 1 \lor t_m \le 0.0$.
   - **Invariante de Protección Biológica**:
     Una bifurcación es inmunizada de la poda **estrictamente si**:
     1. Satisface la maduración materna: $\text{mlen} \ge 1 \lor t_m \le 0.0$.
     2. Satisface la separación máxima alométrica: $d_{\text{daughters}} \le d_{\max}(\text{grupo})$.
     3. Satisface la distancia de nacimiento alométrica: $\max(d_1, d_2) \le d_{\text{birth}}(\text{grupo})$.
     4. Cumple los criterios de persistencia angular BDPG:
        - Obtusa ($\cos_{3d} \le 0.0$): persistencia $\ge 3$ o forward heading $> 0.3$.
        - Inclinada por arrastre ($0.0 < \cos_{3d} \le \cos_{\max}$): persistencia $\ge 5$.
        - Aguda ($\cos_{3d} > \cos_{\max}$): nunca inmunizada.
   - Si no cumple todos los invariantes alométricos y BDPG, la horquilla es espuria y se poda mediante DDKS-SPI (Stationary Persistence Induction) o por scoring cinemático/fotométrico.

---

## 3. Variantes Predeclaradas

### Variante 1: AACG-Exact
- `max_daughter_separation_44b6_um`: $7.2\,\mu\text{m}$
- `max_birth_dist_44b6_um`: $6.0\,\mu\text{m}$
- `max_daughter_separation_6bba_um`: $13.5\,\mu\text{m}$
- `max_birth_dist_6bba_um`: $7.5\,\mu\text{m}$
- `min_mother_track_len`: $1$ (o $t_m \le 0.0$)
- `heading_guard_cos_max`: $0.20$
- `divergent_pers_min`: $3$, `tilted_pers_min`: $5$

### Variante 2: AACG-Permissive
- `max_daughter_separation_44b6_um`: $7.5\,\mu\text{m}$
- `max_birth_dist_44b6_um`: $6.2\,\mu\text{m}$
- `max_daughter_separation_6bba_um`: $14.5\,\mu\text{m}$
- `max_birth_dist_6bba_um`: $8.0\,\mu\text{m}$
- `min_mother_track_len`: $1$ (o $t_m \le 0.0$)
- `heading_guard_cos_max`: $0.22$
- `divergent_pers_min`: $3$, `tilted_pers_min`: $5$

---

## 4. Gates de Promoción Contractuales

### Gate de Desarrollo (`b0970-temporal-042-rel1`):
- $\Delta_{\text{score}} \ge +0.003000$ vs referencia histórica.
- $\Delta_{\text{44b6}} \ge 0.0$.
- $\Delta_{\text{6bba}} \ge 0.0$.
- Peor muestra $\ge -0.015000$.
- $\text{division\_true\_positives} \ge 3$.

### Gate de Confirmación (`b0970-confirm-042-rel1`):
- $\Delta_{\text{score}} \ge +0.002000$ vs referencia histórica.
- $\Delta_{\text{44b6}} \ge 0.0$.
- $\Delta_{\text{6bba}} \ge 0.0$.
- Peor muestra $\ge -0.015000$.
- $\text{division\_true\_positives} \ge 4$.

---

## 5. Garantía de Seguridad e Invariantes

1. Generación de candidatos estrictamente libre de etiquetas (`label_free_candidate_generation: true`).
2. El ground truth permanece sellado y no se utiliza para generar, podar o guiar aristas.
3. El cómputo científico se despacha exclusivamente a la máquina remota autorizada `A1555` bajo el perfil `cpu_evaluation_v1`.
