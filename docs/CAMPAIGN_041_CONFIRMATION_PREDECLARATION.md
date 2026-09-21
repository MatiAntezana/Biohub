# Predeclaración de Confirmación — Campaña 041 (Group-Harmonized TCI & BDPG)

## 1. Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaign 041 Confirmación (Release 3), 2026-09-15, Antigravity.
- **Receta seleccionada en Desarrollo**:
  - **Variante 1 (TCI-Harmonized Conservative)**:
    - `44b6`: $d_{\text{daughters}} \le 7.5\,\mu\text{m}$, $\max_d \le 5.5\,\mu\text{m}$, $\text{mother\_track\_len} \ge 1$ (o $t_m \le 0.0$).
    - `6bba`: $d_{\text{daughters}} \le 10.0\,\mu\text{m}$, $\max_d \le 7.5\,\mu\text{m}$, $\text{mother\_track\_len} \ge 1$ (o $t_m \le 0.0$).
    - BDPG guard: $\cos \le 0.20$, persistencia $\ge 5$, $d_{\text{birth\_tilted}} \le 6.0\,\mu\text{m}$.
    - SPI habilitado ($d_{\text{close}} \le 1.5\,\mu\text{m}, d_{\text{far}} \ge 2.5\,\mu\text{m}, \text{ratio} \ge 2.0$).
  - **Resultado en Desarrollo**: Score $= 0.905280$ (ref: $0.901274$, $\Delta = \mathbf{+0.004006} \ge +0.003000$). `division_tp = 3/3`, `division_fp = 1`. Gate superado holgadamente.
- **Run fuente de Confirmación**: `b0970-prepare-005` en `A1555`.
- **Paneles de Confirmación**: `0f845d07`, `137456d9`, `e1c51478` (6 datasets: 3 prefijos `44b6`, 3 prefijos `6bba`).
- **Campeona de Confirmación vigente**: Campaign 040 Variante 1 (`b0970-confirm-040-rel1`, score **$0.955208$**).

---

## 2. Hipótesis falsable en Confirmación

- **Objetivo**:
  - En Campaign 040, alcanzamos el récord histórico de $0.955208$ con `division_tp = 4/4` y `division_fp = 4`.
  - Bajo Group-Harmonized TCI, las horquillas espurias que violan leyes físicas de la membrana y el centroide de citocinesis son podadas adicionalmente, reduciendo aún más los falsos positivos de división y arista sin tocar ninguna de las 4 divisiones verdaderas (`44b6_587a1e22`, `44b6_d2f34f90`, `6bba_afb141ff`, `6bba_df673a83`).
- **Criterio de Aprobación**:
  - $\Delta \text{Score} \ge +0.002000$ vs referencia ($0.952252$).
  - `division_tp >= 4` (100% de divisiones verdaderas conservadas).
  - Score total $\ge 0.955208$ para coronarse nueva campeona.

---

## 3. Ejecución

- **Comando**:
  `python ops/biohub.py launch --node A1555 --task confirm-temporal-041 --source-run b0970-prepare-005 --run-id b0970-confirm-041-rel3 --tests-run b0970-tests-041-rel3`
- **Perfil de recursos**: `cpu_evaluation_v1` (2 hilos, 8 GiB memoria).
