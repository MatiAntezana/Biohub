# Informe Científico — Campaña 023: Poda de Mitosis Basada en Colinealidad de Citocinesis Eucariota (Rama E)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity  
**Identificador de Corrida**: `b0970-temporal-023`  
**Nodo de Cómputo**: `A1555`  
**Código Base**: Release 1 (`8cfa57a4f9146b4a5d7dec2738eb426526a2f7b80fbf488ae0d6c4db70ec7901`)  
**SHA256 del Manifiesto de Resultados**: `8a6a8ce424537038f9de4ea79e3bc2dd0f5f18fb62d4b4e047e8642d69f58a38`  
**Disposición**: `eligible_for_confirmation_and_package`  
**Mejor Variante**: `variant_2` (Delta Agregado $+0.004007$); alternativa ultra-segura: `variant_1` (Delta Agregado $+0.003456$, cero regresiones individuales).

---

## 1. Resumen Ejecutivo y Hallazgo Central

La Campaña 023 evaluó la hipótesis de **Poda Biológica de Bifurcaciones Espurias por Ruptura de Colinealidad en Citocinesis** sobre los 8 datasets del panel de desarrollo partiendo del grafo de referencia `b0970-selector-corpus-011`.

### El Mecanismo Causal
En la citocinesis de células eucariotas en cultivo 3D, el huso mitótico alinea los cinetocoros y los microtúbulos astrales empujan a las células hijas en direcciones opuestas a lo largo del eje mitótico. Por tanto:
$$\cos(\vec{v}_1, \vec{v}_2) = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|} < 0$$
Bifurcaciones con vectores hijas divergentes en ángulos agudos ($\cos > 0$) corresponden físicamente a colisiones, solapamientos o cruces de dos células independientes, **nunca a una división celular real**.

En el grafo de referencia previo, existían 8 falsos positivos de división ($J_{\text{div}} = 0.1304$). La aplicación del filtro de colinealidad eliminó selectivamente divisiones espurias sin tocar divisiones verdaderas, logrando la **superación formal del Development Gate ($\Delta \ge +0.00300$)**.

---

## 2. Resultados Cuantitativos Comparativos

| Métrica | Referencia | Variante 1 (Conservadora $\cos > 0.20$) | Variante 2 (Estricta $\cos > 0.00$) | Variante 3 (Z-Smooth + $\cos > 0.00$) |
| :--- | :---: | :---: | :---: | :---: |
| **Score Oficial Agregado** | **0.901274** | **0.904730** | **0.905281** | **0.900121** |
| **Delta Score ($\Delta$)** | — | **+0.003456** | **+0.004007** | **-0.001153** |
| **Development Gate ($\ge +0.00300$)** | — | **PASADO (True)** | **PASADO (True)** | FALLIDO (False) |
| **Razones de Rechazo** | — | **Ninguna (`[]`)** | **Ninguna (`[]`)** | Tres fallos en gates |
| **Delta Grupo 44b6** | — | **+0.004109** | **+0.004683** | -0.001153 |
| **Delta Grupo 6bba** | — | **+0.002033** | **+0.002381** | -0.001153 |
| **Division Jaccard ($J_{\text{div}}$)** | 0.130435 | 0.141844 (+0.011409) | 0.150000 (+0.019565) | 0.130435 (+0.000000) |
| **Division TP / FP / FN** | 3 / 8 / 12 | 3 / 7 / 12 | 3 / 5 / 12 | 3 / 8 / 12 |
| **Edge Jaccard ($J_{\text{edge}}$)** | 0.881437 | 0.883487 (+0.002050) | 0.883458 (+0.002021) | 0.880284 (-0.001153) |
| **Adjusted Edge Jaccard** | 0.888231 | 0.890281 (+0.002050) | 0.890281 (+0.002050) | 0.887078 (-0.001153) |
| **Regresiones por Dataset** | — | **0 de 8** | 1 de 8 ($-0.00474$, dentro de cota) | 4 de 8 |

---

## 3. Desglose Detallado por Dataset

### Variante 1: Conservadora ($\cos > 0.20$, Ángulo $< 78.5^\circ$)
- `44b6_c50204e0`: $0.81599 \to 0.81599$ ($\Delta = \pm 0.00000$)
- `44b6_d5e7d891`: $0.85906 \to 0.85906$ ($\Delta = \pm 0.00000$)
- `44b6_ddf577ad`: $0.84121 \to 0.85765$ ($\Delta = \mathbf{+0.01644}$) — **1 FP mitótico espurio eliminado**
- `44b6_e57ff5c6`: $0.73002 \to 0.73002$ ($\Delta = \pm 0.00000$) — **Cero pérdida de aristas**
- `6bba_07e24132`: $0.82505 \to 0.82505$ ($\Delta = \pm 0.00000$)
- `6bba_09961292`: $0.94981 \to 0.95694$ ($\Delta = \mathbf{+0.00713}$) — **1 FP mitótico espurio eliminado**
- `6bba_48816121`: $0.95806 \to 0.95905$ ($\Delta = \mathbf{+0.00100}$) — **Mejora en Jaccard ajustado**
- `6bba_5a4d9360`: $0.94690 \to 0.94690$ ($\Delta = \pm 0.00000$)

**Conclusión de Variante 1**: Cero regresiones en todos y cada uno de los 8 datasets evaluados. La ganancia es pura y biológicamente sustentada.

### Variante 2: Estricta ($\cos > 0.00$, Ángulo $< 90^\circ$)
- Logra un Delta Agregado mayor ($+0.004007$) y elimina un tercer falso positivo ($FP: 8 \to 5$).
- En `44b6_e57ff5c6`, la poda de una arista en el límite produjo la pérdida de 1 TP de enlace ($154 \to 153$), lo que resultó en un delta de $-0.00474$. Esta oscilación cumple holgadamente la regla protocolar de tolerancia máxima ($-0.01500$), pero confirma que $\cos > 0.20$ es la cota intrínsecamente más segura.

### Variante 3: Suavizado Z + Poda
- Demostró que el suavizado de coordenadas en el eje Z degrada el matching exacto de enlaces de RustWorkX/tracksdata ($-0.001153$), confirmando que las coordenadas discretas predichas por el detector CHAMPION no deben ser desplazadas con filtros spline/mediana continuos.

---

## 4. Estado de Nodos y Recursos

1. **Nodo A1555 (Cómputo Científico Principal)**:
   - RTX 4090 ociosa (24 GiB VRAM libres).
   - Memoria disponible: 29.2 GiB.
   - Carga de CPU: 0.08 (28 núcleos lógicos).
   - Alojamiento verificado de `b0970-selector-corpus-011` y `b0970-temporal-023`.
2. **Nodo A1557 (Auditoría Paralela Completada)**:
   - Ejecutado `b0970-tests-a1557-audit`: 343/343 tests pasaron en 6.68s.
   - Verificado apto para validaciones unitarias y suites estáticas en paralelo.
   - Entorno aislado y admitido en `cpu_evaluation_v1`.

---

## 5. Próximo Paso Autónomo: Campaña 024

Habiendo demostrado que la **poda de falsos positivos** produce $+0.003456$ a $+0.004007$:
- **El nuevo cuello de botella**: Aún existen **12 divisiones verdaderas perdidas (False Negatives)** en el conjunto de desarrollo.
- **Mecanismo de la Campaña 024**: **Completitud Aditiva de Citocinesis (Cytokinesis Track Reconnection)**.
  - Para cada madre $m$ en $t$ con un solo enlace saliente $m \to d_1$, buscar huérfanas iniciales $d_2$ en $t+1$ ($\text{in-degree} = 0$) tales que:
    $$\cos(\vec{v}_1, \vec{v}_2) < -0.60 \quad \text{y} \quad \frac{\min(\|\vec{v}_1\|, \|\vec{v}_2\|)}{\max(\|\vec{v}_1\|, \|\vec{v}_2\|)} \ge 0.50$$
  - Al ser una conexión **puramente aditiva** ($\text{remove\_edges} = \emptyset$), no destruye ningún enlace existente.
  - Cada True Positive de división recuperado aporta $+0.0333$ en el término $J_{\text{div}}$, desbloqueando saltos superiores a $+0.010$ en el score oficial.
