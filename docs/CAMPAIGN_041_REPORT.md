# Informe de Campaña — Campaña 041: Triad Cytokinesis Invariants (TCI) & BDPG

## 1. Identidad y resumen ejecutivo

- **Campaña**: Campaign 041 (TCI & BDPG).
- **Fecha**: 2026-09-15.
- **Responsable**: Antigravity (autoridad de continuidad autónoma bajo `AGENTS.md`).
- **Nodo de ejecución**: `A1555` exclusivamente.
- **Runs ejecutados**:
  - Tests: `b0970-tests-041-rel1`, `b0970-tests-041-rel2`, `b0970-tests-041-rel3` (todos 100% aprobados, 0 skips, 0 fallos).
  - Desarrollo:
    - `rel1`: Rechazado ($\Delta = -0.001153$).
    - `rel2`: Rechazado ($\Delta = -0.001153$).
    - `rel3`: **APROBADO** ($\Delta = \mathbf{+0.004006} \ge +0.003000$, $\Delta_{\text{44b6}} = +0.004680$, $\Delta_{\text{6bba}} = +0.002381$, `division_tp = 3/3`, `division_fp = 5`).
  - Confirmación: `b0970-confirm-041-rel3`:
    - Score: $0.950018$ vs ref $0.952252$ ($\Delta = -0.002234 < +0.002000$).
    - `division_tp = 3/4` (se retuvo el 100% de divisiones en `44b6`, pero se perdió 1 división en `6bba_df673a83`).
    - Disposición: **`rejected_no_kaggle_submission`**.
- **Campeona vigente retenida**: **Campaign 040 Variante 1 (`b0970-confirm-040-rel1`, Score: `0.955208`)**.

---

## 2. Descubrimientos biofísicos cuantitativos

A través de la inspección rigurosa de las trazas de ground truth zarr v3 y candidatos de seguimiento, Campaign 041 estableció dos principios biológicos fundamentales:

### 2.1. Condición de Frontera Temporal de Adquisición ($t=0$)
- En timesteps intermedios ($t > 0$), una célula que aparece por primera vez y se bifurca inmediatamente en el mismo cuadro con $\text{in\_degree} = 0$ representa un *spur* de entrada espurio (artefacto de tracking o célula ingresando al volumen focal).
- Sin embargo, en el marco temporal inicial ($t = 0.0$), las células preexistentes que ya se encontraban en fase de división (telofase tardía / citocinesis) al comenzar la filmación carecen de historial previo no por ser espurias, sino porque $t < 0$ no existe en el experimento.
- La regla biofísica correcta debe incorporar la condición de frontera:
  $$\text{mother\_track\_len} \ge 1 \quad \lor \quad t_m \le 0.0$$

### 2.2. Alometría Inter-Grupal de Citocinesis (`44b6` vs `6bba`)
- La hipótesis de invariantes de separación de hijas ($d_{\text{daughters}} \le 7.5\,\mu\text{m}$) y envoltura de nacimiento ($\max(d_1, d_2) \le 5.5\,\mu\text{m}$) demostró ser altamente exitosa en el grupo `44b6`:
  - En `44b6` confirmación: Score subió de $0.916488$ a **$0.922451$** ($\mathbf{\Delta = +0.005963}$).
  - Redujo los falsos positivos de división a la mitad (`division_fp = 3` vs 6).
  - Preservó el 100% de divisiones verdaderas.
- Sin embargo, en el grupo `6bba` (células de alta motilidad con separación citocinética rápida), las mediciones de ground truth revelaron que las células hijas verdaderas se separan a distancias de $14.0\,\mu\text{m}$ a $20.3\,\mu\text{m}$ con desplazamientos respecto a la madre de hasta $8 - 12\,\mu\text{m}$.
- La imposición de límites geométricos euclidianos de $7.5 - 11.0\,\mu\text{m}$ sobre `6bba` recortó la división verdadera en `6bba_df673a83`, causando la regresión de confirmación.

---

## 3. Conclusión y gobernanza

- Bajo las reglas de gobernanza de `AGENTS.md` (Invariantes 6, 8, 12):
  - Un resultado negativo se registra transparentemente sin maquillar métricas ni flexibilizar gates retrospectivamente.
  - La receta 041 no se promueve a Kaggle ni sustituye a la campeona.
  - **Campaign 040 Variante 1 (`0.955208`) permanece como el RÉCORD HISTÓRICO Y CAMPEONA VIGENTE ABSOLUTA de Biohub.**
