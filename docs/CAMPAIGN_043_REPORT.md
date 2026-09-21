# Informe Oficial de Campaña — Campaña 043: De Novo Daughter Pairing (DNDP)

## 1. Resumen Ejecutivo

- **Estado**: **RECHAZADA EN DESARROLLO — CAMPEONA VIGENTE RETENIDA (CAMPAÑA 042, 0.955208)**.
- **Run ID Desarrollo**: `b0970-temporal-043-rel1` en `A1555`.
- **Métricas Registradas**:
  - **Variante 1 (DNDP Estricto, $\cos \le -0.10$)**:
    - Score Candidata: `0.885362` vs Referencia `0.901274` ($\mathbf{\Delta = -0.015912} < +0.003000$).
    - Gate de Desarrollo: **`gate_passed = False`**.
    - Disposición: **`rejected_no_kaggle_submission`**.
  - **Variante 2 (DNDP Permisivo, $\cos \le 0.05$)**:
    - Score Candidata: `0.881204` vs Referencia `0.901274` ($\Delta = -0.020070$).
    - Gate de Desarrollo: **`gate_passed = False`**.
- **Invariante de Gobernanza**: Conforme a `AGENTS.md` (Invariantes 6, 8, 12), el resultado negativo se conserva, se interpreta con rigor físico, y la receta no se promueve a confirmación ni a empaquetado. **Campaign 042 Variante 1 (AACG-Exact, Score: `0.955208`) permanece como la CAMPEONA ABSOLUTA DE BIOHUB**.

---

## 2. Análisis Forense Cuantitativo: ¿Por Qué Falló DNDP?

El análisis de la auditoría de ejecución reveló un comportamiento crítico de la topología de grafos celulares:

1. **La Paradoja de la Frecuencia de Eventos (Event Rarity Paradox)**:
   - En una película microscópica típica con ~40,000 detecciones y miles de células, la citocinesis es un evento biológico **extremadamente infrecuente** (~1 a 3 divisiones por película completa).
   - En el grafo de seguimiento, prácticamente el **95% de los nodos** forman parte de una trayectoria lineal continua con $\text{out\_degree} == 1$.
   - Al mismo tiempo, debido a la dinámica de entrada/salida del plano focal y fluctuaciones de segmentación, existen cientos de trazas que comienzan en cada fotograma $t+1$ ($\text{in\_degree} == 0$).

2. **Explosión Masiva de Falsos Positivos de División**:
   - Al buscar pares madre-huérfana puramente por proximidad euclidiana ($8 - 15\,\mu\text{m}$) y ángulo de divergencia cinemática, el algoritmo evaluó más de **2,400 candidatos** y creó **más de 1,600 nuevas aristas de división** en los 8 paneles de desarrollo:
     - `44b6_d5e7d891`: +324 divisiones espurias.
     - `6bba_09961292`: +371 divisiones espurias.
     - `44b6_c50204e0`: +248 divisiones espurias.
     - `6bba_07e24132`: +304 divisiones espurias.
   - En lugar de 15 divisiones reales, el grafo resultante tenía más de 1,700 divisiones.
   - Esto redujo el `division_jaccard` drásticamente al inflar el denominador con cientos de `division_fp`, causando la caída del score de $0.901274$ a $0.885362$ ($\Delta = -0.0159$).

3. **Conclusión Científica Inequívoca**:
   - **No es posible descubrir divisiones de novo a partir de información cinemática y geométrica pura sin una señal biofísica/fotométrica directa de condensación cromosómica o escisión de membrana**.
   - Como establece la Invariante 9 de `AGENTS.md`: *"No inventar células, enlaces o coordenadas para optimizar artificialmente score/cardinalidad. Rescate nuevo exige evidencia real de imagen y tiempo"*.
   - Por lo tanto, el camino óptimo y matemáticamente seguro hacia la mejora del puntaje NO radica en forzar la creación de aristas de división no vistas por el detector, sino en:
     1. **Refinar la poda de falsos positivos** (como demostraron exitosamente Campaña 040 y 042).
     2. **Optimizar la fidelidad de aristas ($\text{adj\_edge\_jaccard}$)** mediante interpolación y suavizado axial.

---

## 3. Estado de la Campeona Vigente

| Métrica | Referencia | Campaña 042 (Campeona) | Delta Oficial |
| :--- | :--- | :--- | :--- |
| **Score de Confirmación** | 0.952252 | **0.955208** | **+0.002956** |
| **Recall de División (Conf)** | 100% (4/4) | **100% (4/4)** | $\pm 0$ |
| **Falsos Positivos de División** | 7 | **4** | **-42.9%** |
| **Score de Desarrollo** | 0.901274 | **0.905280** | **+0.004006** |
| **Contrato Visible** | 239,742 filas | **235,570 filas** | Válido |

El empaquetado de producción `b0970-package-042-rel1` contiene exactamente este modelo campeón con todos los fixes de tiempo y memoria validados.
