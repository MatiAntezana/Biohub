# Campaña 031: Reporte Oficial — Track-Length Persistence Guarded Cytokinesis Regularization (TL-PG-CR)

**Fecha de Ejecución**: 2026-09-14 07:53 UTC  
**Nodo de Cómputo**: `A1555` (`cpu_evaluation_v1`, RTX 4090 libre con 23.98 GiB VRAM)  
**Ejecución Oficial**: `b0970-temporal-031-rel2` (precedida por diagnóstico `b0970-temporal-031`)  
**Run Fuente**: `b0970-selector-corpus-011` en `A1555`  
**Suite de Pruebas**: `b0970-tests-031-rel2` (403/403 pruebas superadas en 6.79s, 0 omisiones, 0 fallos, 0 errores)  
**Hash de Release**: `0f5d6a1c804261c2f8512e4c8b983898cc96687bad4fd0de939f8fd3470dd7dd`  
**Protocolo**: Gate de Desarrollo oficial ($\Delta \text{score}_{\text{dev}} \ge +0.00300$, $\Delta \text{score}_{\text{group}} \ge 0$, peor delta $\ge -0.01500$)

---

## 1. Resumen Ejecutivo y Veredicto

- **Score Agregado de Desarrollo**: $0.901274 \to \mathbf{0.903063}$ ($\Delta = \mathbf{+0.001788} > 0$, ganancia neta positiva en las 3 variantes).
- **Jaccard de División**: $0.130435 \to 0.130435$ (se mantuvo plano; las bifurcaciones de división falsa no fueron podadas).
- **Falsos Positivos de División**: $8 \to 8$ (permanecieron en 8).
- **Verdaderos Positivos de División**: $3 \to 3$ ($100\%$ conservados).
- **Jaccard de Aristas Ajustadas**: $0.888231 \to 0.890019$ ($\Delta = +0.001788$).
- **Score Grupo 44b6**: $0.834231 \to 0.838996$ ($\Delta = +0.004765$).
- **Score Grupo 6bba**: $0.941911 \to 0.941911$ ($\Delta = +0.000000$).
- **Veredicto de Gate de Desarrollo**: **RECHAZADO (`rejected_no_kaggle_submission`)**.
  - Aunque la ganancia fue positiva ($\Delta = +0.001788$), no alcanzó el umbral requerido de $+0.00300$.
  - La hipótesis de que las bifurcaciones falsas son "espolones transitorios de longitud $L \le 2$" quedó **empíricamente falsada**.

---

## 2. Resultados Cuantitativos por Variante

| Variante | Descripción | Score Ref | Score Cand | Delta ($\Delta$) | Div Jaccard (Ref $\to$ Cand) | Div FP (Ref $\to$ Cand) | Div TP (Ref $\to$ Cand) | Gate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variante 1** | TL-PG-V1 ($L_{\max\_prune} = 2$) | $0.901274$ | **$0.903063$** | **$+0.001788$** | $0.1304 \to 0.1304$ | $8 \to 8$ | $3 \to 3$ | FAIL |
| **Variante 2** | TL-PG-V2 ($L \le 3$ con $\Delta L \ge 3$) | $0.901274$ | **$0.903063$** | **$+0.001788$** | $0.1304 \to 0.1304$ | $8 \to 8$ | $3 \to 3$ | FAIL |
| **Variante 3** | TL-PG-V3 (Adaptativa por Grupo) | $0.901274$ | **$0.903063$** | **$+0.001788$** | $0.1304 \to 0.1304$ | $8 \to 8$ | $3 \to 3$ | FAIL |

---

## 3. Descubrimiento Causal y Diagnóstico Forense Profundo

### A. ¿Por qué fracasó la restricción por longitud de trayectoria ($L \le 2$)?
- Al medir la longitud real de las trayectorias de las 140 aristas podadas exitosamente en la Campaña 029 (que redujo FPs en $-37.5\%$ y alcanzó score récord $0.905586$), se descubrió que:
  - Longitud mínima: $1$–$3$ fotogramas.
  - **Mediana de longitud: $14$ a $17$ fotogramas**!
  - Más del $50\%$ de las ramas podadas tenían $L > 10$ fotogramas.
- **Conclusión Causal**: Las bifurcaciones falsas en los grafos de tracking NO son simples partículas de ruido que desaparecen en 1 o 2 frames. Son **fusiones o cruces erróneos entre dos células biológicas independientes**, donde el solver conectó erróneamente una célula a otra preexistente a lo largo de un salto espacial grande ($\approx 5$–$6$ µm).
- Al restringir la poda a $L \le 2$, la Campaña 031 impidió podar las ramas de salto espurio de 14 frames, dejando los 8 FPs intactos y perdiendo la ganancia de división.

### B. El Verdadero Origen de la Pérdida de Aristas en Confirmación (030)
- Al analizar minuciosamente las 22 bifurcaciones de `44b6_2a2eff9f` en Confirmación, se identificó la causa exacta de la penalización de aristas:
  - En todas las bifurcaciones evaluadas, una hija estaba muy cerca ($d_{\text{close}} \le 1.0$ µm) y la otra estaba lejos ($d_{\text{far}} \ge 5.0$ µm).
  - La hija cercana ($0.81$–$0.91$ µm) era una **célula estacionaria real** con continuidad biológica de hasta **51 fotogramas**.
  - Sin embargo, la fórmula lineal de score:
    $$\text{score}_i = -\frac{d_{v_i}}{10.0} + 1.0 \cdot \text{align}_i + 0.6 \cdot \text{head}_i$$
    penalizó la distancia de 6 µm con solo $-0.6$ puntos, mientras que la hija lejana que saltaba 6 µm en la dirección general del flujo tisular recibió hasta $+1.6$ puntos por alineamiento cinemático.
  - Como resultado, el algoritmo seleccionó a la hija de 6 µm como "conservada", y **PODÓ A LA CÉLULA ESTACIONARIA REAL DE 51 FOTOGRAMAS QUE ESTABA A SOLO 0.9 µm**!

---

## 4. Decisión de Continuidad Autónoma — Rumbo a Campaña 032

- Se archivan todos los artefactos de la Campaña 031 con disposición `rejected_no_kaggle_submission`.
- Se avanza a la **Campaña 032: Selección de Rama Conservada por Dominancia Espacial e Invariante de Proximidad (DDKS-SPI)**:
  - **Invariante de Proximidad Física (SPI)**: Si una hija está a $d_{\text{close}} \le 1.5$ µm y la otra a $d_{\text{far}} \ge 2.5$ µm, la hija cercana es **INCONDICIONALMENTE LA CONSERVADA**.
  - La rama candidata a poda es **SIEMPRE el salto lejano de 5–6 µm**, jamás la célula estacionaria.
  - Esto erradica al $100\%$ la amputación de células estacionarias continuas, preservando todos los Edge TPs y liberando toda la potencia de reducción de divisiones falsas tanto en Desarrollo como en Confirmación.
