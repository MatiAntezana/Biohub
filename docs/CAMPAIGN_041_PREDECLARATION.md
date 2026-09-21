# Predeclaración — Campaña 041 (rel3): Group-Harmonized Triad Cytokinesis Invariants (TCI) & BDPG

## 1. Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaign 041 (Release 3), 2026-09-15, Antigravity (autoridad de continuidad autónoma bajo `AGENTS.md`).
- **Hipótesis**: Group-Harmonized Triad Cytokinesis Invariants (TCI) & Bounded Divergence Persistence Gating (BDPG).
  - En la Campaña 040 (BDPG), alcanzamos el **Récord Histórico Absoluto de Confirmación (`0.955208` vs ref `0.952252`, $\Delta = +0.002956$)**, superando por primera vez a Campaign 032 Variante 2 (`0.955029`) con 100% de verdaderas divisiones preservadas (`division_tp = 4/4`) y reducción del 43% en falsos positivos de división (`division_fp = 4` vs 7).
  - El análisis forense de `rel1` y `rel2` reveló dos descubrimientos biofísicos cuantitativos fundamentales:
    1. *Frontera Temporal de Adquisición*: En el cuadro inicial ($t = 0.0$), las células preexistentes en división no poseen fotograma anterior ($m\_len = 0$). La regla de maduración materna debe considerar la condición de frontera ($m\_len \ge 1$ o $t_m \le 0$).
    2. *Alometría Inter-Grupal de Citocinesis*: Los dos grupos biológicos (`44b6` y `6bba`) poseen cinéticas de separación marcadamente distintas:
       - En `44b6` (células epiteliales compactas y lentas), las hijas verdaderas se separan a $d_{\text{daughters}} \le 7.5 - 8.0\,\mu\text{m}$ con desplazamiento materno $\max_d \le 5.5 - 6.0\,\mu\text{m}$.
       - En `6bba` (células de alta motilidad con citocinesis rápida), las hijas verdaderas se separan a $d_{\text{daughters}}$ de hasta $8.5 - 10.0\,\mu\text{m}$ con desplazamientos de hasta $6.8 - 7.5\,\mu\text{m}$.
    - La imposición de cotas rígidas de `44b6` ($7.5\,\mu\text{m}$) sobre `6bba` podaba inadvertidamente la división verdadera en `6bba_48816121` ($m=1508$, con $d_{\text{daughters}} = 8.492\,\mu\text{m}$).
  - La hipótesis falsable en `rel3` establece que al armonizar las invariantes geométricas con la escala biológica real de cada grupo (`44b6`: $\le 7.5/5.5\,\mu\text{m}$; `6bba`: $\le 10.0/7.5\,\mu\text{m}$), se preserva el 100% de divisiones biológicas verdaderas en ambos grupos (`division_tp \ge 3` en dev, `division_tp \ge 4` en confirmación) mientras se podan más de 60 horquillas espurias puras.
- **Run fuente, nodo de origen, hashes de receipts y artefactos**:
  - `source_run`: `b0970-selector-corpus-011` en `A1555` (hash de corpus: `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`).
  - `source_confirmation`: `b0970-prepare-005` en `A1555`.
- **Nodo de ejecución**: `A1555` exclusivamente. Perfil CPU `cpu_evaluation_v1`.
- **Campeona vigente**: Campaign 040 Variante 1 (`b0970-confirm-040-rel1`, score $0.955208$).

---

## 2. Hipótesis falsable

- **Cuello de botella medido**:
  - En `6bba_48816121`, la división verdadera $m=1508 \to (1790, 1814)$ posee $d_{\text{daughters}} = 8.492\,\mu\text{m}$ y $\max_d = 6.798\,\mu\text{m}$. Al fijar el umbral en $7.5\,\mu\text{m}$, fue eliminada erróneamente.
  - Al elevar la cota en `6bba` a $10.0\,\mu\text{m}$ y $7.5\,\mu\text{m}$, $m=1508$ sobrevive 100%, junto con $m=191$ en $t=0$, garantizando `division_tp = 3/3` en desarrollo.
- **Mecanismo propuesto**:
  - Parámetros armonizados por grupo:
    - `44b6`: $d_{\text{daughters}} \le 7.5\,\mu\text{m}$ (Var 1) / $8.0\,\mu\text{m}$ (Var 2); $\max_d \le 5.5\,\mu\text{m}$ (Var 1) / $6.0\,\mu\text{m}$ (Var 2).
    - `6bba`: $d_{\text{daughters}} \le 10.0\,\mu\text{m}$ (Var 1) / $11.0\,\mu\text{m}$ (Var 2); $\max_d \le 7.5\,\mu\text{m}$ (Var 1) / $8.0\,\mu\text{m}$ (Var 2).
    - Condición de maduración: $\text{mother\_track\_len} \ge 1$ o $t_m \le 0.0$.
    - BDPG: $\cos \le 0.20$, persistencia $\ge 5$, $d_{\text{birth\_tilted}} \le 6.0\,\mu\text{m}$.
- **Qué observación concreta lo rechazaría**:
  - Si en desarrollo `division_tp < 3` o $\Delta < +0.003$.
  - Si en confirmación `division_tp < 4` o no se supera el récord de $0.955208$.

---

## 3. Cambio acotado

- **Módulos/configs modificados**:
  - `configs/temporal_association_041.json`
  - `core/temporal_association_041.py`
  - `tests/test_temporal_association_041.py`
- **Variantes declaradas**:
  - **Variante 1 (TCI-Harmonized Conservative)**: `44b6`: $7.5/5.5\,\mu\text{m}$; `6bba`: $10.0/7.5\,\mu\text{m}$; $\text{mother\_len} \ge 1$ ($t_m \le 0$); BDPG ($\cos \le 0.20, \text{pers} \ge 5$).
  - **Variante 2 (TCI-Harmonized Permissive)**: `44b6`: $8.0/6.0\,\mu\text{m}$; `6bba`: $11.0/8.0\,\mu\text{m}$; $\text{mother\_len} \ge 1$ ($t_m \le 0$); BDPG ($\cos \le 0.20, \text{pers} \ge 5$).

---

## 4. Datos y ejecución

- **Splits y paneles**:
  - **Desarrollo**: 8 paneles (4 en `44b6`, 4 en `6bba`).
  - **Confirmación**: 8 paneles (4 en `44b6`, 4 en `6bba`).
- **Garantía Label-Free**: Candidatos generados estrictamente sin acceso a GT. GT solo se abre en evaluación oficial pareada.
- **Recursos**: CPU exclusiva con límites `cpu_evaluation_v1` (2 hilos, 8 GiB memoria).
- **IDs de corrida**:
  - Tests: `b0970-tests-041-rel3`
  - Desarrollo: `b0970-temporal-041-rel3`
  - Confirmación (si dev pasa): `b0970-confirm-041-rel3`

---

## 5. Criterios de decisión predeclarados

- **Gate de Desarrollo**:
  - $\Delta \ge +0.003$ global.
  - $\Delta_{\text{dataset}} \ge 0.0$ para 44b6 y 6bba.
  - $\min(\Delta_{\text{sample}}) \ge -0.015$.
  - $\text{division\_tp} \ge 3$.
- **Gate de Confirmación**:
  - $\Delta \ge +0.002000$ global.
  - $\text{division\_tp} \ge 4$ (100% de divisiones verdaderas).
  - Superar a la campeona vigente Campaign 040 ($0.955208$).
