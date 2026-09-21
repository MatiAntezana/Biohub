# Predeclaración Formal de Confirmación — Campaña 039 (CCDG)

## Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaña 039 Confirmación / 2026-09-15 / Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub).
- **Hipótesis**: La regla biológica calibrada CCDG (Calibrated Cytokinesis-Directed Guard: protección de bifurcación con $\cos_{3d} \le +0.20$ y persistencia $\ge 2$) protege al 100% las divisiones verdaderas inclinadas por deriva tisular en confirmación (`division_tp = 4`, incluyendo `44b6_587a1e22` con $\cos \approx +0.08$), mientras poda más de 35 cruces paralelos agudos ($\cos > 0.20$), reduciendo aristas FP y superando el récord de la campeona vigente Campaña 032 V2 ($0.955029$).
- **Run fuente**: `b0970-prepare-005` (A1555), panel de confirmación de 8 datasets:
  `['44b6_2a2eff9f', '44b6_587a1e22', '44b6_c15fded2', '44b6_d2f34f90', '6bba_062c8d37', '6bba_afb141ff', '6bba_bb9f20c3', '6bba_df673a83']`.
- **Nodo de ejecución**: `A1555` bajo `cpu_evaluation_v1`.
- **Campeona previa de confirmación**: Campaña 032 Variante 2 (`b0970-confirm-032-rel5`, confirmación: `0.955029`, $\Delta = +0.002776$).
- **Resultado en Desarrollo (039)**:
  - Variante 1: Score $0.904424$ vs ref $0.901274$ ($\Delta = \mathbf{+0.003150} \ge +0.003$).
  - Reducción de divisiones FP en 6bba de 2 a 1; $\text{division\_tp} = 3, \text{division\_fn} = 8$ (sin regresión).
  - Poda de 97 bifurcaciones espurias (68 por SPI).

---

## Hipótesis falsable

- **Mecanismo evaluado en confirmación**:
  - En la confirmación de 038, la compuerta $\cos \le 0.00$ mejoró el Edge Jaccard en ambos grupos pero podó 1 rama en `44b6_587a1e22` cuya citocinesis real tenía $\cos \approx +0.08$ por flujo tisular.
  - CCDG Variante 1 calibra la compuerta a $\cos \le +0.20$. La simulación local confirmó protección total para `44b6_587a1e22` manteniendo `division_tp = 4` y conservando la poda de más de 35 bifurcaciones espurias en paralelo ($\cos > 0.20$).
- **Criterio de éxito**:
  - Superar el gate de confirmación: $\Delta_{\text{score}} \ge +0.002000$ frente a la referencia `b0970-prepare-005` (score base 0.952252).
  - Superar a la campeona vigente Campaña 032 V2 ($0.955029$).
  - Preservar al menos 4 divisiones verdaderas ($\text{division\_tp} \ge 4$).
- **Criterio de rechazo**:
  - $\Delta < +0.002000$ o $\text{score} < 0.955029$.
  - Caída en divisiones verdaderas ($\text{division\_tp} < 4$).

---

## Configuración y Ejecución

- **Config**: `configs/temporal_association_039.json`
- **Variantes evaluadas**: Variante 1 ($\cos \le +0.20$) y Variante 2 ($\cos \le +0.25$).
- **ID de corrida**: `b0970-confirm-039-rel1`
- **Tests requeridos**: `b0970-tests-039-rel1` (456/456 aprobados en A1555).
- **Comando**:
  `python ops/biohub.py launch --node A1555 --task confirm-temporal-039 --source-run b0970-prepare-005 --run-id b0970-confirm-039-rel1 --tests-run b0970-tests-039-rel1`
