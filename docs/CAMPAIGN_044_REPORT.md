# Informe Oficial de Campaña — Campaña 044: Asymmetric Birth Ratio Gating (ABRG)

## 1. Resumen Ejecutivo

- **Estado**: **RECHAZADA EN DESARROLLO — CAMPEONA VIGENTE RETENIDA (CAMPAÑA 042, 0.955208)**.
- **Run ID Desarrollo**: `b0970-temporal-044-rel2` en `A1555`.
- **Métricas Registradas**:
  - **Variante 1 (ABRG-Moderada, $r \ge 0.35$, $d_{\text{asym}} \ge 3.5/4.5\,\mu\text{m}$)**:
    - Score Candidata: `0.900625` vs Referencia `0.901274` ($\mathbf{\Delta = -0.000649} < +0.003000$).
    - `division_tp` en 6bba: 2 (reducción respecto a 3 en referencia).
    - Gate de Desarrollo: **`gate_passed = False`**.
    - Disposición: **`rejected_no_kaggle_submission`**.
  - **Variante 2 (ABRG-Estricta, $r \ge 0.40$, $d_{\text{asym}} \ge 3.2/4.0\,\mu\text{m}$)**:
    - Score Candidata: `0.900625` vs Referencia `0.901274` ($\Delta = -0.000649$).
    - Gate de Desarrollo: **`gate_passed = False`**.
- **Invariante de Gobernanza**: Conforme a `AGENTS.md` (Invariantes 6, 8, 12), el resultado negativo se conserva, se interpreta con rigor biofísico, y la receta no se promueve a confirmación ni a empaquetado. **Campaign 042 Variante 1 (AACG-Exact, Score: `0.955208`) permanece como la CAMPEONA ABSOLUTA DE BIOHUB**.

---

## 2. Análisis Forense Cuantitativo: ¿Por Qué Falló ABRG en 6bba?

1. **La Dinámica de Alta Movilidad y Deriva Celular en 6bba**:
   - Mientras que en el linaje epitelial `44b6` las células se mueven lentamente ($< 1.0\,\mu\text{m/frame}$) y las hijas permanecen confinadas al plano de clivaje, en el grupo mesenquimal `6bba` las células exhiben una velocidad motil mucho mayor y sufren deriva direccional asimétrica.
   - En una de las mitosis genuinas de `6bba`, el vector de deriva del tejido empuja a una de las células hijas en la dirección del flujo mientras la otra se mueve contra el flujo.
   - Esto produce un desplazamiento aparente asimétrico desde el centroide materno:
     $d_1 \approx 1.5\,\mu\text{m}$, mientras $d_2 \approx 4.8\,\mu\text{m}$.
     La razón al nacimiento resultante es $r = 1.5 / 4.8 = 0.3125 < 0.35$.
   - Al exigir $r \ge 0.35$, ABRG clasificó erróneamente esta división verdadera como un cruce espurio y revocó su protección biológica, reduciendo `division_tp` de 3 a 2 en 6bba.

2. **Comportamiento en 44b6**:
   - En `44b6`, ABRG funcionó exactamente según lo predicho:
     - `44b6_e57ff5c6`: $\Delta_{\text{adj}} = +0.00340$.
     - Podó exitosamente horquillas de cruce espurio sin afectar divisiones verdaderas.
   - Sin embargo, un gate de desarrollo en Biohub exige estrictamente $\Delta_{\text{6bba}} \ge 0.000000$ y $\text{division\_tp} \ge 3$. La pérdida de un verdadero positivo en 6bba invalida la receta globalmente.

3. **Conclusión Científica Inequívoca**:
   - **No se puede imponer una cota estricta de simetría de nacimiento ($r \ge 0.35$) de manera uniforme o desacoplada de la corrección por deriva vectorial local en grupos celulares de alta motilidad**.
   - Por lo tanto, el diseño alométrico exacto de **Campaña 042 (AACG)** sigue siendo la frontera de Pareto óptima descubierta para la geometría de citocinesis.

---

## 3. Estado de la Campeona Vigente

| Métrica | Referencia | Campaña 042 (Campeona) | Delta Oficial |
| :--- | :--- | :--- | :--- |
| **Score de Confirmación** | 0.952252 | **0.955208** | **+0.002956** |
| **Recall de División (Conf)** | 100% (4/4) | **100% (4/4)** | $\pm 0$ |
| **Falsos Positivos de División** | 7 | **4** | **-42.9%** |
| **Score de Desarrollo** | 0.901274 | **0.905280** | **+0.004006** |
| **Contrato Visible** | 239,742 filas | **235,570 filas** | Válido |

El empaquetado de producción `b0970-package-042-rel1` contiene exactamente este modelo campeón con todos los fixes de tiempo, memoria y validación de contrato.
