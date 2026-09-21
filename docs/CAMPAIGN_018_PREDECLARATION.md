# Predeclaración — Campaña 018: Regularización Axial Pura en Z y Reconexión Conservadora de Huecos de 1 Frame

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Continuidad Autónoma)  
**Hipótesis**: Rama E + Rama D — Regularización Axial Pura (Z-only) y Cierre de Huecos de 1 Frame con Validación de Parches 3D  
**Run Fuente**: `b0970-selector-corpus-011` en A1555 (53 tensores `.evidence.npz` con descriptores 3D de 405 dimensiones precalculados).  
**Nodo de Ejecución**: `A1555`  
**Perfil de Recursos**: `cpu_evaluation_v1` (2 hilos lógicos, afinidad CPU aislada, CUDA oculta, límite 8 GiB memoria).  
**Límites de Fila**: Estrictamente $234{,}000 < \text{filas} < 242{,}000$ (sin cabecera; referencia V19 = 235,674 filas).  

---

## 1. Identidad y Punto de Partida

- **Campaña**: 018 — Regularización Axial Pura en Z y Reconexión Conservadora de Huecos de 1 Frame.
- **Antecedentes Directos**:
  - **Campaña 016 (`b0970-temporal-016-rel3`)**: Demostró que el suavizado axial produce ganancias notables en datasets con jitter (`+0.0111` en `44b6_ddf577ad`, `+0.0074` en `44b6_d5e7d891`), pero el suavizado incondicional en el plano lateral ($XY$, peso 0.65) cortó esquinas en curvas físicas celulares, induciendo regresiones locales en `44b6_c50204e0` y `6bba_09961292`.
  - **Campaña 017 (`b0970-temporal-017-rel2`)**: Demostró que fijar $w_{xy} = 0.0$ elimina al 100% las regresiones en trayectorias curvas. Sin embargo, la poda heurística de bifurcaciones (`prune_spurious_forks`) podó 434 forks y eliminó las 3 únicas divisiones verdaderas en `6bba` ($TP_{\text{div}}: 3 \to 0$), lo que desplomó el $J_{\text{div}}$ a cero y causó una pérdida de $-0.0133$ en el score combinado.
- **Invariante Topológico de Mitosis**:
  - Se fija como regla absoluta **NO podar ninguna bifurcación ni arista de la referencia base** ($\Delta \text{forks} = 0$, $\text{prune\_edges} = \emptyset$). Esto garantiza matemáticamente que $TP_{\text{div}}$ no disminuye y que $J_{\text{div}}$ se mantiene al 100% blindado contra degradaciones.
- **Control**: Grafo de referencia congelado de `b0970-selector-corpus-011` (score base $0.901274$ en desarrollo).
- **Independencia**: Los datasets se procesan sin etiquetas ni grafos oráculo. Las anotaciones GT ingresan exclusivamente a través del evaluador oficial de métricas `tracking_cellmot.metrics.evaluate()` en la fase pareada final.

---

## 2. Hipótesis Falsable

1. **Premisas Físicas y Biológicas**:
   - **Anisotropía de la PSF**: La microscopía confocal y de hoja de luz presenta una elongación axial de 3× a 4× a lo largo de $Z$ ($s_z = 1.625-2.0\,\mu\text{m}$ frente a $s_{xy} = 0.40625\,\mu\text{m}$). El ruido de centroide está fuertemente concentrado en $Z$.
   - **Preservación Lateral**: Las coordenadas $XY$ corresponden a migración celular real en el plano del tejido; cualquier suavizado en $XY$ reduce la curvatura natural y arriesga superar el umbral de matching de $7.0\,\mu\text{m}$. Fijar $w_{xy} = 0.0$ preserva exactamente la fidelidad de las curvas celulares.
   - **Huecos de 1 Frame ($t \to t+2$)**: La fluctuación temporal de intensidad y la segmentación ocasionan que el detector omita ocasionalmente una célula en un frame intermedio $t+1$. Dos tracks maduros que terminan en $t$ y comienzan en $t+2$ con distancia $\le 4.5\,\mu\text{m}$ y similitud de parche 3D $\ge 0.60$ corresponden inequívocamente a la misma célula. Insertar el centroide interpolado en $t+1$ añade 2 aristas verdaderas ($u \to w$ y $w \to v$), eliminando falsos negativos de arista.
2. **Criterio de Falsabilidad**:
   - Si la regularización axial pura ($w_{xy} = 0.0$) no incrementa el Jaccard de aristas ajustado sobre la referencia ($\Delta J_{\text{adj\_edge}} \le 0.0$), la hipótesis de anisotropía queda refutada.
   - Si el cierre de huecos de 1 frame incrementa las aristas falsas positivas por encima de las verdaderas positivas, o si desborda la cota de filas ($> 242{,}000$), la variante queda rechazada.

---

## 3. Variantes Predeclaradas

- **Control**: Referencia intacta de `b0970-selector-corpus-011` ($0.901274$).
- **Variante 1 (Regularización Axial Moderada - Rama E)**:
  - Suavizado lineal interior en $Z$: $w_z = 0.70$, $w_{xy} = 0.0$, ventana = 2.
  - Topología intacta: $\Delta \text{edges} = 0$, $\Delta \text{forks} = 0$, $\Delta \text{nodes} = 0$.
- **Variante 2 (Regularización Axial Fuerte - Rama E)**:
  - Suavizado lineal interior en $Z$: $w_z = 0.85$, $w_{xy} = 0.0$, ventana = 2.
  - Topología intacta: $\Delta \text{edges} = 0$, $\Delta \text{forks} = 0$, $\Delta \text{nodes} = 0$.
- **Variante 3 (Regularización Axial + Cierre Conservador de Huecos - Rama E + Rama D)**:
  - Suavizado en $Z$: $w_z = 0.75$, $w_{xy} = 0.0$, ventana = 2.
  - Cierre de huecos de 1 frame ($t \to t+2$):
    - Extremos aislados de track simple ($out\_deg=0, in\_deg=1$ en $t$; $in\_deg=0, out\_deg=1$ en $t+2$).
    - Vecinos más cercanos mutuos con $\text{dist} \le 4.5\,\mu\text{m}$.
    - Similitud de parches 3D $\ge 0.60$.
    - Inserción de nodo interpolado en $t+1$: $\mathbf{x}_{\text{mid}} = (\mathbf{x}_u + \mathbf{x}_v)/2$.
    - Inserción de aristas $(u \to w)$ y $(w \to v)$.
    - Guardia de no colisión con nodos existentes en $t+1$ a distancia $< 2.0\,\mu\text{m}$.

---

## 4. Gates de Decisión Predeclarados

- **Gate de Desarrollo**:
  - $\Delta \text{score}_{\text{dev}} \ge +0.003$ sobre el control pareado.
  - $\Delta \text{edge\_jaccard} \ge 0.0$.
  - $\Delta \text{division\_jaccard} \ge 0.0$ (estrictamente no regresión de divisiones).
  - No regresión por grupo: $\Delta \text{group} \ge 0.0$ tanto en `44b6` como en `6bba`.
- **Gate de Confirmación**:
  - Si la mejor variante supera el gate de desarrollo, se evalúa en el panel de confirmación (5 datasets reservados).
  - Requiere $\Delta \text{score}_{\text{conf}} \ge +0.002$ y no regresión.
- **Límites de Fila y Formato**:
  - Estrictamente $234{,}000 < \text{filas} < 242{,}000$ (sin cabecera).
  - Cada rescate de hueco añade exactamente 1 nodo y 2 aristas (+3 filas en el CSV). El impacto máximo proyectado es $< 100$ filas, perfectamente dentro del rango admisible.
  - Inferencia sin GT, determinista y completamente verificada por suite de tests.
