# Predeclaración de Confirmación — Campaña 040 (BDPG)

## 1. Identidad y punto de partida

- **Campaña/fecha/responsable**: Campaign 040 Confirmación, 2026-09-15, Antigravity.
- **Receta seleccionada en Desarrollo**:
  - **Variante 1 (BDPG-020)**: $\cos_{\max} = 0.20$, `divergent_pers_min = 3`, `tilted_pers_min = 5`, `tilted_max_birth_dist_um = 6.0`, SPI habilitado ($d_{\text{close}} \le 1.5\,\mu\text{m}, d_{\text{far}} \ge 2.5\,\mu\text{m}, \text{ratio} \ge 2.0$).
  - **Resultado en Desarrollo**: Score $= 0.905280$ (ref: $0.901274$, $\Delta = \mathbf{+0.004006} \ge +0.003000$). `division_tp = 3/3`, `division_fp = 1`. Gate superado holgadamente.
- **Run fuente de Confirmación**: `b0970-prepare-005` en `A1555`.
- **Paneles de Confirmación**: `0f845d07`, `137456d9`, `e1c51478` (6 datasets: 3 prefijos `44b6`, 3 prefijos `6bba`).
- **Campeona de Confirmación vigente**: Campaign 032 Variante 2 (`b0970-confirm-032-rel5`, score **$0.955029$**).

---

## 2. Hipótesis falsable en Confirmación

- **Objetivo**:
  - En Campaign 039, la confirmación alcanzó score $0.953897$ con `division_tp = 4/4`, pero no superó el gate de $+0.002000$ debido al falso positivo de división y arista en $m=245$ (`44b6_d2f34f90`).
  - Bajo BDPG-020, la bifurcación espuria efímera $m=245$ (con hija $2$ de solo $3$ cuadros) es podada porque no cumple la cota de persistencia de citocinesis inclinada ($L \ge 5$). Al mismo tiempo, la citocinesis verdadera en `44b6_587a1e22` ($L_1=14, L_2=77$) se mantiene 100% protegida.
- **Criterio de Aprobación**:
  - $\Delta \text{Score} \ge +0.002000$ vs referencia ($0.952252$).
  - `division_tp >= 4` (100% de divisiones verdaderas conservadas).
  - Score total $> 0.955029$ para coronarse nueva campeona.

---

## 3. Ejecución

- **Comando**:
  `python ops/biohub.py launch --node A1555 --task confirm-temporal-040 --source-run b0970-prepare-005 --run-id b0970-confirm-040-rel1 --tests-run b0970-tests-040-rel1`
- **Perfil de recursos**: `cpu_evaluation_v1` (2 hilos, 8 GiB memoria).
