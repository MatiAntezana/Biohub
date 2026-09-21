# Reporte Oficial — Campaña 040: Bounded Divergence Persistence Gating (BDPG)

## 1. Resumen Ejecutivo

- **Estado**: **APROBADA EN DESARROLLO Y CONFIRMACIÓN — NUEVA CAMPEONA OFICIAL DE BIOHUB**.
- **Run ID Desarrollo**: `b0970-temporal-040-rel1` en `A1555`.
  - Score Candidata: **`0.905280`** vs Referencia `0.901274` ($\mathbf{\Delta = +0.004006} \ge +0.003000$).
  - `division_tp = 3/3`, `division_fp = 1`. Gate de Desarrollo superado holgadamente.
- **Run ID Confirmación**: `b0970-confirm-040-rel1` en `A1555`.
  - Score Candidata: **`0.955208`** vs Referencia `0.952252` ($\mathbf{\Delta = +0.002956} \ge +0.002000$).
  - Record anterior de confirmación (Campaign 032 V2): `0.955029`.
  - **Nuevo récord histórico absoluto de Biohub: `0.955208`**.
  - `division_tp = 4/4` (100% de divisiones verdaderas conservadas).
  - `division_fp = 4` (reducido en un 43% respecto a 7 en referencia).
  - Gate de Confirmación: **`passed = True`**.
  - Disposición: **`eligible_for_final_package_and_submission`**.

---

## 2. Comparación Histórica de Confirmación

| Campaña | Método Clave | Score Conf | Delta vs Ref | Div TP | Div FP | Veredicto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **030** | Baseline con axial regularizer | 0.952252 | +0.000000 | 4 | 7 | Base |
| **032 V2** | SPI + DDKS cuadrático | 0.955029 | +0.002777 | 3 | 3 | Aprobada (ex-campeona) |
| **038** | CEDTP ($\cos \le 0.0$) | 0.952403 | +0.000151 | 3 | 4 | Rechazada |
| **039** | CCDG ($\cos \le 0.20$, pers $\ge 2$) | 0.953897 | +0.001644 | 4 | 5 | Rechazada ($\Delta < 0.002$) |
| **040 V1** | **BDPG ($\cos \le 0.20$, pers $\ge 5$, $d \le 6\,\mu\text{m}$)** | **0.955208** | **+0.002956** | **4** | **4** | **APROBADA (NUEVA CAMPEONA)** |

---

## 3. Desglose Detallado por Grupo en Confirmación

### Grupo 44b6 (4 datasets)
- **Score**: **`0.922451`** vs Ref `0.916488` ($\mathbf{\Delta = +0.005963}$).
- **Edge Jaccard**: `0.890944` vs Ref `0.889320`.
- **Division Jaccard**: `0.142857` vs Ref `0.100000`.
- **Division TP**: `1 / 1` (100% conservada).
- **Division FP**: **`3`** (reducido a la mitad vs 6 en referencia).

### Grupo 6bba (4 datasets)
- **Score**: **`0.964938`** vs Ref `0.964938` ($\Delta = +0.000000$).
- **Division TP**: `3 / 3` (100% conservada).
- **Division FP**: `1`.

---

## 4. Desglose por Dataset en Confirmación

| Dataset | Split | Ref Score | Cand Score | Delta | Spikes Z | Horquillas Podadas | SPI Activados |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `44b6_2a2eff9f` | Conf | 0.90463 | 0.90860 | **+0.00397** | 483 | 26 | 21 |
| `44b6_587a1e22` | Conf | 0.94165 | 0.94165 | +0.00000 | 344 | 9 | 3 |
| `44b6_c15fded2` | Conf | 0.91745 | 0.91745 | +0.00000 | 753 | 10 | 6 |
| `44b6_d2f34f90` | Conf | 0.90223 | 0.92211 | **+0.01988** | 352 | 15 | 12 |
| `6bba_062c8d37` | Conf | 0.97010 | 0.97010 | +0.00000 | 165 | 1 | 0 |
| `6bba_afb141ff` | Conf | 0.94062 | 0.94062 | +0.00000 | 106 | 0 | 0 |
| `6bba_bb9f20c3` | Conf | 0.91352 | 0.91352 | +0.00000 | 498 | 2 | 1 |
| `6bba_df673a83` | Conf | 0.96387 | 0.96387 | +0.00000 | 816 | 4 | 3 |

En `44b6_d2f34f90`, el salto de score es de **$+0.01988$**, podando la horquilla espuria $m=245$ mientras en `44b6_587a1e22` la división verdadera biológica queda intacta.

---

## 5. Próximos Pasos Operativos

1. **Cuota de Kaggle**:
   - La submission en curso `56244048` (`b0970-package-032-rel6`) continúa evaluándose en Kaggle (`PENDING`, toma ~6 horas).
   - La cuota diaria (1/1 submission por día UTC) está consumida para 2026-09-15 UTC.
   - En cuanto esté disponible el siguiente slot UTC o finalice la corrida actual, Campaign 040 (`b0970-confirm-040-rel1`) está plenamente autorizada y lista para ser empaquetada como `b0970-package-040-rel1` y enviada a Kaggle.
2. **Exploración de la Siguiente Campaña (041)**:
   - Continuar explorando de forma proactiva mejoras hacia el hito $\ge 0.970$.
