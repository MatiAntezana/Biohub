# Predeclaración Oficial — Campaña 044: Asymmetric Birth Ratio Gating (ABRG)

## 1. Identidad y Punto de Partida

- **Campaña/Fecha/Responsable**: Campaña 044, 2026-09-15, Antigravity (autoridad autónoma según `AGENTS.md`).
- **Hipótesis**: Hipótesis B (Refinamiento biofísico de topología de bifurcación celular).
- **Run Fuente**: `b0970-selector-corpus-011` en `A1555`.
- **Hashes de Fuente**:
  - `receipt.json`: `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`
  - `corpus.json`: `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`
- **Nodo de Ejecución**: `A1555` exclusivamente (driver `execute_remotely`, perfil `cpu_evaluation_v1`).
- **Campeona Vigente de Referencia**: Campaña 042 Variante 1 (AACG-Exact, Score: `0.955208`, Delta: `+0.002956`, 4/4 TP divisiones en confirmación, FP reducidos a 4).
- **Limitaciones de Exposición**: 8 paneles de desarrollo (corpus011) para búsqueda de hiperparámetros; 8 paneles de confirmación (corpus011) estrictamente ciegos e intactos. Inferencia sin etiquetas ni GT.

---

## 2. Hipótesis Falsable

- **Cuello de Botella Medido**:
  - En la confirmación de Campaña 042, aún sobreviven 4 falsos positivos de división:
    - `44b6_2a2eff9f`: 1 FP
    - `44b6_587a1e22`: 2 FP
    - `6bba_bb9f20c3`: 1 FP
  - En `44b6_587a1e22`, la celda madre en $t$ se vincula con la celda biológica continua ($d_1 \approx 0.8 - 1.2\,\mu\text{m}$) y simultáneamente con una celda que entra lateralmente o se cruza ($d_2 \approx 4.0 - 5.5\,\mu\text{m}$).
  - La razón de desplazamiento al nacimiento $r = \frac{\min(d_1, d_2)}{\max(d_1, d_2)}$ es extremadamente asimétrica ($r \approx 0.18 - 0.25$).
  - Sin embargo, debido a que ambas trayectorias son de células sanas con persistencia $> 3$ fotogramas y divergencia direccional ($\cos_{3D} \le 0$), el mecanismo BDPG de Campaña 040/042 les otorgaba inmunidad biológica incondicional.

- **Mecanismo Propuesto (ABRG - Asymmetric Birth Ratio Gating)**:
  - En la citocinesis física real, la escisión del surco de división separa las dos células hijas con desplazamientos espaciales casi simétricos con respecto al centroide de la célula madre:
    $\|\vec{v}_1\| \approx \|\vec{v}_2\|$.
    La razón de nacimiento $r$ en mitosis verdaderas es siempre $\ge 0.35$ (típicamente $\ge 0.50$).
  - Una razón $r < 0.35$ cuando el desplazamiento mayor supera $3.5\,\mu\text{m}$ es una firma física inequívoca de **cruce o salto espurio** (una célula se mueve $1\,\mu\text{m}$ y la otra se "mueve" $5\,\mu\text{m}$ en el mismo paso temporal).
  - ABRG revoca la protección biológica para estas horquillas asimétricas (`abrg_ok = False`), sometiéndolas a la selección dominante por distancia (DDKS-SPI). Esto preserva la trayectoria primaria continua y elimina limpiamente el falso positivo de división.

- **Criterio de Rechazo**:
  - Si ABRG poda alguna división verdadera (caída en `division_true_positives` $< 3$ en desarrollo o $< 4$ en confirmación), o si $\Delta_{\text{score}} < +0.003000$ en desarrollo, la hipótesis se rechaza inmediatamente.

---

## 3. Variantes Predeclaradas

- **Variante 1 (ABRG-Moderada)**:
  - `min_birth_ratio`: `0.35`
  - `asym_dist_thresh_44b6_um`: `3.5`
  - `asym_dist_thresh_6bba_um`: `4.5`
  - Conserva todas las cotas alométricas de AACG (`d_daughters <= 7.2` en 44b6, `13.5` en 6bba; `max_birth <= 6.0` en 44b6, `7.5` en 6bba; maduración madre $\ge 1$).

- **Variante 2 (ABRG-Estricta)**:
  - `min_birth_ratio`: `0.40`
  - `asym_dist_thresh_44b6_um`: `3.2`
  - `asym_dist_thresh_6bba_um`: `4.0`
  - Misma base alométrica y de suavizado axial.

---

## 4. Gates de Decisión Oficiales

- **Desarrollo**:
  - $\Delta_{\text{score}} \ge +0.003000$ vs referencia `0.901274`.
  - $\Delta_{\text{44b6}} \ge 0.000000$, $\Delta_{\text{6bba}} \ge 0.000000$.
  - $\text{division\_tp} \ge 3$.
  - $\text{worst\_sample\_delta} \ge -0.015000$.

- **Confirmación**:
  - $\Delta_{\text{score}} \ge +0.002000$ vs referencia `0.952252`.
  - $\Delta_{\text{44b6}} \ge 0.000000$, $\Delta_{\text{6bba}} \ge 0.000000$.
  - $\text{division\_tp} \ge 4$.
  - $\text{worst\_sample\_delta} \ge -0.015000$.
