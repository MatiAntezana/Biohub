# Predeclaración Formal de Confirmación — Campaña 042: Adaptive Allometric Cytokinesis Gating (AACG)

## 1. Identidad de la Corrida de Confirmación

- **Campaña**: Campaign 042 (AACG).
- **Fecha**: 2026-09-15.
- **Autoridad de Ejecución**: Antigravity (bajo gobernanza de `AGENTS.md` y `docs/AUTONOMOUS_CONTINUATION.md`).
- **Nodo Autorizado**: `A1555` exclusivamente.
- **Run ID Confirmación**: `b0970-confirm-042-rel1`.
- **Source Run**: `b0970-selector-corpus-011`.
- **Tests Run**: `b0970-tests-042-rel1` (474/474 tests pasados en 6.85s).
- **Code Hash**: `1b601815023722f3447eca246288e976474d1721231b3531bf3b6de018bdaf38`.
- **Variante Objetivo (Target Variant)**: `variant_1` (AACG-Exact).

---

## 2. Resultados Aprobados de Desarrollo

- **Run Desarrollo**: `b0970-temporal-042-rel1` (APROBADO).
- **Score Desarrollo**: $0.901274 \to \mathbf{0.905280}$ ($\mathbf{\Delta = +0.004006} \ge +0.003000$).
- $\Delta_{\text{44b6}} = +0.004679$, $\Delta_{\text{6bba}} = +0.002381$.
- `division_tp = 3/3`, `division_fp = 5` (reducido de 8 en referencia).
- Poda de bifurcaciones espurias en desarrollo: 149 bifurcaciones podadas (75 vía SPI).
- Gate de desarrollo superado con holgura y sin regresiones por muestra o por grupo.

---

## 3. Especificación Biofísica de la Variante a Confirmar (`variant_1`)

- **Grupo `44b6` (Células epiteliales lentas)**:
  - $d_{\text{daughters}} \le 7.2\,\mu\text{m}$.
  - $\max(d_1, d_2) \le 6.0\,\mu\text{m}$.
  - $\text{mother\_track\_len} \ge 1 \lor t_m \le 0.0$.
- **Grupo `6bba` (Células mesenquimales de alta motilidad)**:
  - $d_{\text{daughters}} \le 13.5\,\mu\text{m}$.
  - $\max(d_1, d_2) \le 7.5\,\mu\text{m}$.
  - $\text{mother\_track\_len} \ge 1 \lor t_m \le 0.0$.
- **Invariante de Protección Biológica (BDPG)**:
  - Divergencia obtusa ($\cos_{3d} \le 0.0$): persistencia $\ge 3$ o forward heading $> 0.3$.
  - Inclinada por arrastre ($0.0 < \cos_{3d} \le 0.20$): persistencia $\ge 5$.
  - Cruces paralelos agudos ($\cos_{3d} > 0.20$): nunca inmunizados.
- **Poda DDKS-SPI**:
  - Poda determinista de ramas lejanas en asimetría extrema ($d_{\text{close}} \le 1.5\,\mu\text{m}, d_{\text{far}} \ge 2.5\,\mu\text{m}$).

---

## 4. Paneles de Evaluación de Confirmación (Holdout Interno Reservado)

Los 8 paneles congelados del split de confirmación serán evaluados oficialmente tras la generación de candidatos estrictamente libre de etiquetas:
1. `44b6_2a2eff9f`
2. `44b6_587a1e22`
3. `44b6_c15fded2`
4. `44b6_d2f34f90`
5. `6bba_062c8d37`
6. `6bba_afb141ff`
7. `6bba_bb9f20c3`
8. `6bba_df673a83`

---

## 5. Gate Contractual de Confirmación

- $\Delta_{\text{score}} \ge +0.002000$ vs referencia histórica ($0.952252$).
- $\Delta_{\text{44b6}} \ge 0.0$.
- $\Delta_{\text{6bba}} \ge 0.0$.
- Peor muestra $\ge -0.015000$.
- $\text{division\_true\_positives} \ge 4$.
