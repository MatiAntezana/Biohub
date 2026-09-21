# Predeclaración — Campaña 017: Poda de Bifurcaciones Mitóticas Espurias y Supresión Condicional de Jitter Axial

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Continuidad Autónoma)  
**Hipótesis**: Rama E / Mitosis Pruning — Invariantes Biológicas y Regularización Condicional  
**Run Fuente**: `b0970-selector-corpus-011` en A1555 (53 tensores `.evidence.npz` con descriptores 3D de 405 dimensiones precalculados).  
**Nodo de Ejecución**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 hilos lógicos, afinidad CPU aislada, CUDA oculta, límite 8 GiB memoria).  
**Límites de Fila**: Estrictamente $234{,}000 < \text{filas} < 242{,}000$ (sin cabecera).  

---

## 1. Identidad y Punto de Partida

- **Campaña**: 017 — Poda de Bifurcaciones Mitóticas Espurias mediante Invariantes Biológicos 3D y Supresión Condicional de Picos de Jitter en Z.
- **Antecedente Directo**: Campaña 016 (`b0970-temporal-016-rel3`). Demostró que el suavizado axial produce ganancias masivas en datasets con jitter ($+0.0111$ en `44b6_ddf577ad`), pero el suavizado incondicional corta esquinas en trayectorias suaves. Asimismo, el análisis del score oficial reveló que la referencia base V19 acumula 9 falsos positivos de división y solo 2 verdaderos positivos ($J_{\text{div}} = 0.10526$), penalizando fuertemente el score oficial ($\text{Score} = J_{\text{adj\_edge}} + 0.1 \times J_{\text{div}}$).
- **Control**: Grafo de referencia congelado de `b0970-selector-corpus-011` (score base $0.90127$ en desarrollo).
- **Independencia**: Los 8 datasets de desarrollo se evalúan de forma estrictamente ciega sin abrir anotaciones durante la poda ni la inferencia. Las anotaciones GT ingresan única y exclusivamente a través de `tracking_cellmot.metrics.evaluate()` en la fase final de diagnóstico pareado.

---

## 2. Hipótesis Falsable

1. **Cuello de Botella Medido**:
   - En el panel de desarrollo (8 datasets), la referencia genera 9 divisiones FP y solo 2 divisiones TP.
   - 3 datasets sin divisiones biológicas en GT (`44b6_ddf577ad`, `44b6_e57ff5c6`, `6bba_48816121`) acumulan 5 divisiones FP y 5 aristas FP.
   - La inspección de las bifurcaciones de la referencia demostró que las bifurcaciones espurias presentan una hija real (alta correlación de parche $\ge 0.70$, baja distancia) y una hija impuesta por el enlace heurístico que es ruido o célula lejana (correlación $\le 0.40$, distancia $> 5.0\,\mu\text{m}$, correlación entre hermanas $\le 0.20$).
2. **Mecanismo Propuesto**:
   - Para toda bifurcación predicha $(m, d_1, d_2)$, evaluar los 5 invariantes biológicos de mitosis en tejido de pez cebra:
     1. Similitud madre-hija: $\min(\text{dot}(m, d_1), \text{dot}(m, d_2)) \ge \theta_{\text{mother}} = 0.45$.
     2. Similitud entre hermanas: $\text{dot}(d_1, d_2) \ge \theta_{\text{sister}} = 0.35$.
     3. Distancia máxima al plano de escisión: $\max(\text{dist}(m, d_1), \text{dist}(m, d_2)) \le 4.8\,\mu\text{m}$.
     4. Separación entre hermanas: $\text{dist}(d_1, d_2) \le 7.0\,\mu\text{m}$.
     5. Simetría de distancia: $|\text{dist}(m, d_1) - \text{dist}(m, d_2)| \le 2.5\,\mu\text{m}$.
   - Si la bifurcación no satisface los invariantes, se poda la arista impostora (aquella con menor correlación y mayor distancia), preservando la continuación física real.
   - Para las coordenadas de nodos lineales, suprimir condicionalmente únicamente picos de jitter axial en Z: solo si $|z_0 - \frac{z_{-1} + z_1}{2}| \times \text{scale}_z > 2.5\,\mu\text{m}$, aplicar regularización con desplazamiento máximo acotado a $\le 1.5\,\mu\text{m}$, manteniendo $w_{xy}=0.0$.
3. **Criterio de Rechazo (Falsabilidad)**:
   - Si la poda de bifurcaciones espurias reduce el número de verdaderos positivos ($D_{\text{TP}} < 2$) o no mejora el score micro-promedio en desarrollo en al menos $+0.003$, la hipótesis se rechaza.
   - Si la supresión condicional de jitter en Z induce regresión en `44b6_c50204e0` o `6bba_09961292`, la variante combinada se descarta.

---

## 3. Variantes Predeclaradas

- **Control**: Referencia intacta de `b0970-selector-corpus-011` ($0.90127$).
- **Variante 1**: Poda de bifurcaciones espurias mediante invariantes biológicas 3D exclusivamente (sin modificación de coordenadas).
- **Variante 2**: Poda de bifurcaciones espurias + Supresión condicional de picos de jitter axial en Z ($|a_z| > 2.5\,\mu\text{m}$, corrección máxima $1.5\,\mu\text{m}$, $w_{xy}=0.0$).

---

## 4. Gates de Decisión Predeclarados

- **Gate de Desarrollo**:
  - $\Delta \text{score}_{\text{dev}} \ge +0.003$ sobre el control pareado.
  - $\Delta \text{edge\_jaccard} \ge 0.0$.
  - $\Delta \text{division\_jaccard} \ge 0.0$.
  - No regresión por grupo: $\Delta \text{group} \ge 0.0$ tanto en `44b6` como en `6bba`.
- **Gate de Confirmación**:
  - Si la mejor variante supera el gate de desarrollo, se evalúa en el panel de confirmación (5 datasets reservados).
  - Requiere $\Delta \text{score}_{\text{conf}} \ge +0.002$ y no regresión.
- **Límites de Fila y Formato**:
  - Estrictamente $234{,}000 < \text{filas} < 242{,}000$.
  - Inferencia sin GT, determinista y testeada.
