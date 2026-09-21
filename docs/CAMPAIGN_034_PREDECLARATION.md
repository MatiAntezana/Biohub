# Predeclaración Oficial — Campaña 034: Drift-Compensated Spatial Proximity Invariant Regularization (DC-SPI)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Protocolo de Continuidad Autónoma / Contrato de Operación y Continuidad de Biohub)  
**Código Hash Predeclarado**: `f301348f064fbf7d52a15fa009aaaf7ca676b63b06ba94af0dbda3248e3c9dac`  
**Plantilla Base**: `docs/NEXT_EXPERIMENT_TEMPLATE.md`  
**Nodo de Ejecución Autorizado**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 vCPUs, 8 GiB memoria virtual límite, CUDA oculta)  
**Control Base de Desarrollo**: `b0970-selector-corpus-011` (score: `0.901274`)  
**Holdout Base de Confirmación**: `b0970-prepare-005` (`cache-confirmation`, score: `0.952252`)  
**Campeona de Confirmación Vigente**: Campaña 032 Variante 2 (DDKS-SPI-Cytokinesis-V2, score: `0.955029`, delta: `+0.002776`)

---

## 1. Identidad y Punto de Partida

- **Campaña/fecha/responsable**: Campaña 034 / 2026-09-14 / Antigravity.
- **Hipótesis Científica**: Sinergia de Suavizado Axial Compensado por Deriva Global y Poda por Invarianza de Proximidad Espacial (DC-SPI). La integración de la compensación de deriva convectiva macroscópica $\vec{v}_{\text{global}}(t)$ antes del suavizado axial elimina el snapping axial y restaura aristas continuas ($+0.003304$ medido en 033), mientras que la poda DDKS-SPI elimina bifurcaciones paralelas espurias sin alterar las divisiones biológicas genuinas ($-42.9\%$ FP en 032). Al desactivar la reasignación mitótica no condicionada que introdujo FP en 033, se logra una ganancia estrictamente positiva tanto en Jaccard de aristas como en Jaccard de división.
- **Evidencia Experimental Previa Directa**:
  1. **Campaña 032 (`b0970-confirm-032-rel5`)**: Superó el gate de confirmación en holdout ciego nunca visto con $\Delta = +0.002776$ ($0.952252 \to 0.955029$), preservando al 100% las divisiones verdaderas y podando el $42.9\%$ de falsas divisiones.
  2. **Campaña 033 (`b0970-temporal-033-rel2`)**: Demostró de forma incontrovertible que la compensación de deriva global $\vec{v}_{\text{global}}(t)$ produce un incremento uniforme y estadísticamente significativo en el Jaccard de aristas ajustadas ($\Delta_{\text{adj}} = +0.003304$, alcanzando $+0.02343$ en `44b6_e57ff5c6`).
  3. **Diagnóstico Causal de 033**: La degradación de score en 033 provino exclusivamente de la reasignación mitótica no condicionada, que generó 980 aristas candidatas que aumentaron $\text{FP}_{\text{div}}$ de 8 a 22. Al suprimir la reasignación libre y acoplar el suavizado DC con la poda DDKS-SPI, se conserva la ganancia de $+0.0033$ en aristas y la ganancia de $+0.0261$ en división.
- **Run Fuente**: `b0970-selector-corpus-011` (desarrollo) y `b0970-prepare-005` (confirmación) en `A1555`.
- **Nodo de Ejecución Autorizado**: `A1555` (RTX 4090, 29.2 GiB RAM, 168 GiB disco libre). Perfil `cpu_evaluation_v1`.
- **Exposiciones Previas y Limitaciones de Independencia**:
  - Panel de desarrollo (8 datasets): Expuesto en campañas 016 a 033. Se utiliza para calibración y selección de la mejor variante.
  - Panel de confirmación (8 datasets holdout estricto): `44b6_2a2eff9f`, `44b6_587a1e22`, `44b6_c15fded2`, `44b6_d2f34f90`, `6bba_062c8d37`, `6bba_afb141ff`, `6bba_bb9f20c3`, `6bba_df673a83`. Completamente aislado. Se evaluará pareado únicamente si el gate de desarrollo es superado.

---

## 2. Hipótesis Falsable y Formulación Matemática

### Cuello de Botella Medido
1. El suavizado axial estándar sobre coordenadas no corregidas por deriva snapped produce oscilaciones en $z$ que quiebran aristas biológicas reales. La compensación por $\vec{v}_{\text{global}}(t)$ recupera $+0.0033$ en Jaccard de aristas.
2. Los forks de citocinesis espurios generados por células adyacentes paralelas degradan el Jaccard de división en $0.1304 \to 0.0811$ si no se podan selectivamente mediante DDKS-SPI.

### Formulación Matemática de DC-SPI
1. **Deriva Convectiva Global**:
   $$\vec{v}_{\text{global}}(t) = \text{median}_{e \in E_t} \left( (\vec{x}_{\text{target}(e)} - \vec{x}_{\text{source}(e)}) \odot \mathbf{s} \right)$$
2. **Suavizado Axial en Marco en Reposo**:
   $$\vec{x}'(t) = \vec{x}(t) - \frac{1}{2} w_z \left[ (\vec{x}(t) - \vec{x}(t-1) - \vec{v}_{\text{global}}(t-1)) + (\vec{x}(t) - \vec{x}(t+1) + \vec{v}_{\text{global}}(t)) \right]_z$$
   con guardia de reversión simétrica estricta.
3. **Poda de Forks DDKS-SPI**:
   Para bifurcaciones donde $\min(d_1, d_2) \le 1.5\,\mu\text{m}$ y $\max(d_1, d_2) \ge 2.5\,\mu\text{m}$ con razón de proximidad $\ge 2.0$:
   podar la rama distante si $\cos\theta > 0.00$ (ramas paralelas). Si $\cos\theta \le 0.00$ (citocinesis antiparalela), aplicar guarda dual de persistencia.

---

## 3. Cambio Acotado y Variantes

- **Módulos Nuevos**:
  - `core/temporal_association_034.py`
  - `configs/temporal_association_034.json`
  - `ops/audit_temporal_inputs_034.py`
  - `tests/test_temporal_association_034.py`
- **Variantes Predeclaradas**:
  - **Variante 1 (DC-SPI-Pure)**: Suavizado axial con deriva global $\vec{v}_{\text{global}}(t)$ + Poda DDKS-SPI de 032. Reasignación de mitosis **desactivada** (`enabled = false`).
  - **Variante 2 (DC-SPI-Reinforced)**: Suavizado axial DC con ponderación $w_z = 0.85$, $w_{xy} = 0.00$, corrección máxima $0.6\,\mu\text{m}$ + Poda DDKS-SPI con guarda cinemática estricta.
  - **Variante 3 (DC-SPI-UltraGated)**: DC-SPI con reasignación mitótica ultra-restringida: requerimiento de $p_{\text{cand}} \ge 0.85$, $\cos\theta^* \le -0.60$, $d_{\text{sister}} \in [3.0, 12.0]\,\mu\text{m}$, ratio $\ge 0.35$.
- **Criterio de Selección en Desarrollo**: Mayor score agregado pareado en `b0970-selector-corpus-011`. Umbral mínimo: $\Delta \ge +0.003000$.

---

## 4. Datos y Ejecución

- Panel de desarrollo: 8 datasets `b0970-selector-corpus-011`.
- Perfil: `cpu_evaluation_v1`.
- Límite de tiempo: < 300 segundos (con pre-indexación temporal $O(N)$).
- Sin fugas de GT: candidate generation estrictamente label-free; evaluación GT aislada.
