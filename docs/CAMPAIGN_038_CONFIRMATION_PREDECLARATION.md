# Predeclaración Formal de Confirmación — Campaña 038 (CEDTP)

## Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaña 038 Confirmación / 2026-09-15 / Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub).
- **Hipótesis**: La regla biológica CEDTP (protección de doble trayectoria condicionada a divergencia citocinética $\cos_{3d} \le \text{heading\_guard\_cos\_max}$) poda más de 70 bifurcaciones espurias en paralelo ($\cos > 0$) en el panel de confirmación, eliminando aristas falsas positivas sin alterar divisiones verdaderas, mejorando la métrica macro $\overline{J}_{\text{adj}}$ y superando el gate de confirmación ($\Delta \ge +0.002$).
- **Run fuente**: `b0970-prepare-005` (A1555), panel de confirmación de 8 datasets:
  `['44b6_2a2eff9f', '44b6_587a1e22', '44b6_c15fded2', '44b6_d2f34f90', '6bba_062c8d37', '6bba_afb141ff', '6bba_bb9f20c3', '6bba_df673a83']`.
- **Nodo de ejecución**: `A1555` bajo `cpu_evaluation_v1`.
- **Campeona previa de confirmación**: Campaña 032 Variante 2 (`b0970-confirm-032-rel5`, confirmación: `0.955029`, $\Delta = +0.002776$).
- **Resultado en Desarrollo (038)**:
  - Variante 2: Score $0.905422$ vs ref $0.901274$ ($\Delta = \mathbf{+0.004147} \ge +0.003$).
  - Reducción de 50% de divisiones FP en 6bba y eliminación de 2 divisiones FP en 44b6.
  - Poda de 143 bifurcaciones espurias (135 con $\cos > 0$).

---

## Hipótesis falsable

- **Mecanismo evaluado en confirmación**:
  - En la confirmación de 032, existían 365 bifurcaciones supervivientes frente a solo 16 divisiones verdaderas en GT (94 bifurcaciones con $\cos > 0$). En `44b6_c15fded2` (0 divisiones GT), sobrevivían 59 bifurcaciones espurias con 13 aristas FP.
  - CEDTP Variante 2 ($\cos \le -0.20$) poda las bifurcaciones paralelas y ortogonales, reduciendo directamente las aristas FP y mejorando la Jaccard ajustada.
- **Criterio de éxito**:
  - Superar el gate de confirmación: $\Delta_{\text{score}} \ge +0.002000$ frente a la referencia `b0970-prepare-005` (score base 0.952252).
  - Superar a la campeona vigente Campaña 032 V2 ($0.955029$).
  - Preservar al menos 4 divisiones verdaderas ($\text{division\_tp} \ge 4$).
- **Criterio de rechazo**:
  - $\Delta < +0.002000$ o $\text{score} < 0.955029$.
  - Caída en divisiones verdaderas ($\text{division\_tp} < 4$).

---

## Configuración y Ejecución

- **Config**: `configs/temporal_association_038.json`
- **Variantes evaluadas**: Variante 1 ($\cos \le 0.0$) y Variante 2 ($\cos \le -0.20$).
- **ID de corrida**: `b0970-confirm-038-rel1`
- **Tests requeridos**: `b0970-tests-038-rel1` (451/451 aprobados en A1555).
- **Comando**:
  `python ops/biohub.py launch --node A1555 --task confirm-temporal-038 --source-run b0970-prepare-005 --run-id b0970-confirm-038-rel1 --tests-run b0970-tests-038-rel1`
