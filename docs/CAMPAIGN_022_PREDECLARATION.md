# Predeclaración — Campaña 022: Regularización Axial con Guardias de Choque de Cuadro y Densidad Espacial (Rama E)

Documento normativo de predeclaración para la Campaña 022 per `docs/AUTONOMOUS_CONTINUATION.md` y `docs/NEXT_EXPERIMENT_TEMPLATE.md`.
Congelado antes de la ejecución científica. No debe modificarse a posteriori; cualquier ajuste exige una nueva release/run.

---

## 1. Identidad y punto de partida

- **Campaña/fecha/responsable**: `b0970-temporal-022` / 2026-09-14 / Antigravity.
- **Hipótesis**: Rama E (Asociación Temporal en Tejido Denso y Regularización Axial Cinemática).
- **Run fuente**: `b0970-selector-corpus-011` en `A1555`.
  - `receipt.json`: `f7c9dccea0ce5b23aebfa6a05d892aee7fe57d161229a7d526e14c6ab2254e1f`
  - `corpus.json`: `584d3ffa640906816577c30a41566a020bd79952b0e88e8bed0fa2aa0c422239`
  - 53 archivos `*.evidence.npz` verificados con SHA256.
- **Nodo de ejecución**: `A1555` (RTX 4090 libre, CPU libre, 168.1 GiB libres, perfil `cpu_evaluation_v1`).
- **Release/control archivados**:
  - Control de referencia: grafo de referencia V19 congelado de `b0970-selector-corpus-011` (score = 0.901274).
  - Resultados precedentes:
    - 021 rel5: Regularización Simétrica Pura demostró $+0.001788$ de ganancia neta ($0.901274 \to 0.903063$) con EXACTAMENTE 0.00000 regresiones en los 8 volúmenes de desarrollo ($+0.01524$ en `44b6_ddf577ad`).
    - 018 rel3: La regularización de reversiones de dirección generó ganancias sustanciales en 4 datasets: `44b6_c50204e0` ($+0.00730$), `44b6_ddf577ad` ($+0.00351$), `6bba_07e24132` ($+0.00271$), `6bba_48816121` ($+0.00199$), totalizando $+0.01551$ en mejoras positivas.
- **Exposiciones previas**:
  - Los 8 datasets de desarrollo se utilizan estrictamente para selección de receta bajo protocolo pareado.
  - Ningún GT es accesible durante la inferencia o generación del grafo candidato.

---

## 2. Hipótesis falsable

### Cuello de botella medido
En la microscopía confocal y de hoja de luz, la resolución axial Z es de 4 a 5 veces menor que la resolución lateral XY ($\text{scale}_z = 1.625\,\mu\text{m}$ vs $\text{scale}_{xy} = 0.40625\,\mu\text{m}$). Esto genera oscilaciones y fluctuaciones de un solo cuadro (jitter) entre cortes ópticos adyacentes.

Cuando una célula oscila artificialmente en Z, su centroide predicho en el cuadro $t$ se desplaza más allá del radio de matching del Ground Truth ($7.0\,\mu\text{m}$ en el espacio escalado o penalización euclídea de emparejamiento bipartito), provocando la caída simultánea del enlace anterior ($t-1 \to t$) y posterior ($t \to t+1$), generando $2 \times \text{FP}_{\text{edge}}$ y $2 \times \text{FN}_{\text{edge}}$.

En la Campaña 018, la regularización de reversiones asimétricas reveló dos mecanismos de fallo específicos:
1. **Choque Colectivo de Cuadro (Frame Shock)**: En `6bba_09961292`, 18 de las 22 reversiones ocurrieron en el mismo cuadro $t=39$. Este fenómeno responde a un desplazamiento colectivo del espécimen o de la platina del microscopio (focal shift), donde el Ground Truth se desplazó conjuntamente con la muestra. Suavizar individualmente las células en ese cuadro las aleja del GT.
2. **Colisión en Tejido Denso (Density Collision)**: En cúmulos densos ($\text{nn\_dist} < 6.0\,\mu\text{m}$), desplazar Z puede inducir un intercambio de identidad en el matching bipartito con una célula vecina. Cuando $\text{nn\_dist} \ge 6.0\,\mu\text{m}$, el vecino más cercano dista más de 10 veces la corrección máxima ($0.6\,\mu\text{m}$), haciendo matemáticamente imposible el conflicto de identidad.

### Mecanismo propuesto
1. **Guardia de Choque de Cuadro (`max_frame_reversals <= 4`)**: Contabilizar las reversiones asimétricas por cuadro. Si un cuadro $t$ excede el umbral, se clasifica como evento de platina/enfoque global y se inhibe toda corrección asimétrica en dicho cuadro.
2. **Guardia de Densidad Espacial (`nn_dist >= isolated_nn_thresh_um`)**: Para reversiones asimétricas, exigir que la distancia euclídea 3D al vecino más cercano en el cuadro $t$ sea superior a $6.5\,\mu\text{m}$ (Variante 2) o $5.5\,\mu\text{m}$ (Variante 3).
3. **Regularización Simétrica Incondicional**: Las reversiones simétricas puras ($z_{t-1} = z_{t+1}$) operan de forma incondicional, manteniendo intacta la base demostrada de $+0.001788$ sin regresiones.
4. **Coordenadas Laterales y Topología Intactas**: Las coordenadas XY permanecen 100% inalteradas ($w_{xy} = 0.0$). El número de filas y aristas es idéntico al control ($N_{\text{rows}} = 235{,}674$).

### Qué observación concreta lo rechazaría
- Si cualquiera de los 8 datasets de desarrollo presenta una regresión respecto al control V19 ($\Delta \text{score}_i < -0.0005$).
- Si el score delta promedio no supera el umbral mínimo de desarrollo de $+0.00300$.

### Qué incertidumbre NO resuelve el ensayo
- No resuelve la cobertura de bifurcaciones mitóticas que no están conectadas en el grafo base (Rama B).
- No altera el número total de nodos predichos.

---

## 3. Cambio acotado

### Módulos y configuraciones
- Nuevo módulo: `core/temporal_association_022.py`.
- Nueva configuración: `configs/temporal_association_022.json`.
- Nueva auditoría: `ops/audit_temporal_inputs_022.py`.
- Nueva suite de tests: `tests/test_temporal_association_022.py`.
- Drivers actualizados: `ops/biohub.py` y `ops/verify.py`.

### Variantes predeclaradas
- **Control**: Grafo de referencia V19 de `b0970-selector-corpus-011` ($N_{\text{rows}} = 235{,}674$, score base = 0.901274).
- **Variante 1 (Control Simétrico Puro)**:
  - `reversal_only`: true
  - `symmetric_only`: true
  - `max_correction_um`: 0.6
  - `jitter_threshold_um`: 0.8
  - `weight_z`: 0.80
  - `weight_xy`: 0.0
- **Variante 2 (Regularización Axial con Guardias Conservadoras)**:
  - `reversal_only`: true
  - `symmetric_only`: false
  - `max_frame_reversals`: 4
  - `isolated_nn_thresh_um`: 6.5
  - `max_correction_um`: 0.6
  - `jitter_threshold_um`: 0.8
  - `weight_z`: 0.80
  - `weight_xy`: 0.0
- **Variante 3 (Regularización Axial con Guardias Calibradas)**:
  - `reversal_only`: true
  - `symmetric_only`: false
  - `max_frame_reversals`: 4
  - `isolated_nn_thresh_um`: 5.5
  - `max_correction_um`: 0.6
  - `jitter_threshold_um`: 0.75
  - `weight_z`: 0.85
  - `weight_xy`: 0.0

---

## 4. Datos y ejecución

- **Datasets de desarrollo (8)**:
  - `44b6`: `44b6_d5e7d891`, `44b6_c50204e0`, `44b6_e57ff5c6`, `44b6_ddf577ad`
  - `6bba`: `6bba_48816121`, `6bba_09961292`, `6bba_5a4d9360`, `6bba_07e24132`
- **Inferencia**: 100% label-free. GT no se abre durante la generación de variantes ni filtrado.
- **Recursos**: CPU de A1555 bajo perfil `cpu_evaluation_v1`. Límite de RAM 24 GiB. Tiempo estimado: < 120s.

---

## 5. Verificación y compuertas de decisión

### Compuertas formales
1. **Compuerta de Desarrollo**: $\Delta \text{score}_{\text{dev}} \ge +0.00300$.
2. **Compuerta de Confirmación**: $\Delta \text{score}_{\text{conf}} \ge +0.00200$.
3. **Compuerta de Grupo**: $\Delta \text{score}_{44\text{b}6} \ge 0.0000$ y $\Delta \text{score}_{6\text{bba}} \ge 0.0000$.
4. **Compuerta de Presupuesto de Filas**: $234{,}000 < N_{\text{rows}} < 242{,}000$ (garantizado estrictamente por diseño con 235,674 filas).

### Decisión predeclarada
- Si la Variante 2 o Variante 3 supera $+0.00300$ con cero regresiones de grupo:
  - Se selecciona la variante óptima y se programa la evaluación pareada de confirmación.
  - Al superar confirmación ($\ge +0.00200$), se genera el bundle de producción y se lanza el submission a Kaggle via `ops/kaggle.py`.
- Si ninguna variante con guardias asimétricas supera $+0.00300$ pero la Variante 1 mantiene $+0.001788$ sin regresiones:
  - Se conserva la Variante 1 como base consolidada para combinación sinérgica con recuperación de detecciones de 1 cuadro (Rama D / SNR >= 4.0).
