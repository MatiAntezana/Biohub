# Predeclaración de Campaña 031 — Track-Length Persistence Guarded Cytokinesis Regularization (TL-PG-CR)

**Fecha**: 2026-09-14 07:45 UTC  
**Responsable**: Antigravity (Modo Autónomo Nocturno)  
**Protocolo Base**: `docs/AUTONOMOUS_CONTINUATION.md` (Sección 5)  
**Plantilla Oficial**: `docs/NEXT_EXPERIMENT_TEMPLATE.md`  
**Run Fuente**: `b0970-selector-corpus-011` en `A1555`  
**Nodo de Ejecución**: `A1555` (`cpu_evaluation_v1`, 2 vCPUs, 8 GiB RAM, CUDA oculta)

---

## 1. Identidad y Punto de Partida

- **Campaña / Fecha / Responsable**: Campaña 031, 2026-09-14, Antigravity.
- **Hipótesis**: Track-Length Persistence Guarded Cytokinesis Regularization (TL-PG-CR).
- **Evidencia Nueva Observada**:
  - En Campaña 029 (Desarrollo), la regularización cinemática y fotométrica (KP-CR-HGP) logró un récord histórico de $+0.004312$ con **cero regresiones** en las 8 muestras. Los 6 forks podados en Desarrollo eran todos prolongaciones transitorias de ruido ($L \le 2$ o $L \le 3$).
  - En Campaña 030 (Confirmación), la regularización logró una ganancia neta positiva ($\Delta = +0.000592$), eliminando 1 FP de división y conservando el $100\%$ de TPs de división ($4/4$). Sin embargo, no superó el umbral de $+0.00200$ debido a que en `44b6_2a2eff9f`, la poda sin límite de longitud de trayectoria podó ramas con $L = 5, 10, 15$ frames, seccionando 1 arista biológica continua ($TP_{\text{edge}}: 200 \to 199$, $\Delta_{\text{adj}} = -0.00452$).
- **Run Fuente**: `b0970-selector-corpus-011` en `A1555`.
- **Nodo de Ejecución**: `A1555` (`cpu_evaluation_v1`).
- **Control**: Referencia congelada dual-seed / DeepCenter de `b0970-selector-corpus-011` (score dev: $0.901274$).
- **Campeona Vigente de Desarrollo**: Variante 1 de Campaña 029 (score dev: $0.905586$, $\Delta = +0.004312$).

---

## 2. Hipótesis Falsable

### Cuello de Botella Medido
1. En 3D confocal time-lapse, las divisiones falsas (FPs) se originan principalmente por dos causas distintas:
   - **Causa 1 (Espolón de Ruido Transitorio)**: Una célula real en el tiempo $t$ se asocia con dos detecciones en $t+1$: una es la célula real (que continúa por decenas de fotogramas, $L \ge 10$) y la otra es una detección falsa efímera de segmentación o debris (que muere en 1–2 fotogramas, $L \le 2$).
   - **Causa 2 (Interacción de Células Persistentes)**: Dos células biológicas independientes y persistentes se aproximan en el espacio, haciendo que el solver conecte erróneamente ambas al mismo progenitor previo. En este caso, **ambas ramas hijas continúan por trayectorias largas y persistentes** ($L \ge 4$).
2. Podar una bifurcación de Causa 2 sin transferir o reconectar la rama secundaria secciona una célula biológica real y destruye Edge True Positives.
3. Por tanto, restringir la poda de bifurcaciones no mitóticas exclusivamente a **espolones de ruido transitorio** ($L(d_{\text{pruned}}) \le 2$, o $L \le 3$ con marcada asimetría de longitud $\Delta L \ge 3$) eliminará los falsos positivos de división sin seccionar jamás trayectorias celulares persistentes.

### Mecanismo Propuesto: Track-Length Persistence Guard (TL-PG)
Para toda bifurcación identificada cinemáticamente como no mitótica ($\cos\theta > 0.0$ o asimetría angular):
1. Calcular la persistencia de trayectoria aguas abajo $L(d_1)$ y $L(d_2)$, donde $L(v)$ es la longitud del camino simple dirigido más largo que parte del nodo $v$.
2. Clasificar la hija con mayor puntuación como $d_{\text{keeper}}$ y la otra como $d_{\text{candidate\_prune}}$.
3. **Condición de Poda Segura por Longitud**:
   La rama $(m, d_{\text{candidate\_prune}})$ **solo se poda si**:
   - $L(d_{\text{candidate\_prune}}) \le L_{\max\_prune}$ (donde $L_{\max\_prune} \in \{2, 3\}$); O
   - $L(d_{\text{candidate\_prune}}) \le 3$ Y $L(d_{\text{keeper}}) - L(d_{\text{candidate\_prune}}) \ge \Delta L_{\min}$ (donde $\Delta L_{\min} = 3$).
4. Si **ambas ramas son persistentes** ($L(d_1) \ge 4$ y $L(d_2) \ge 4$), **no podar ninguna rama** (la bifurcación involucra células biológicas reales persistentes).

### Qué Observación Concreta Rechazaría la Hipótesis
- Que en el panel de desarrollo, el score agregado no alcance $\Delta \ge +0.00300$, o que se pierda alguna división verdadera ($TP < 3$).

---

## 3. Variantes Predeclaradas (Máximo 3)

- **Variante 1 (TL-PG-V1 — Filtro Estricto de Espolón Transitorio)**:
  - $L_{\max\_prune} = 2$.
  - Solo se podan ramas con persistencia aguas abajo $\le 2$ frames.
  - Poda cero ramas si ambas tienen $L \ge 3$.
  - Suavizado axial z: $w_z = 0.40$, max displacement $0.6$ µm.
  - Guardias de heading y selección fotométrica activas.
- **Variante 2 (TL-PG-V2 — Poda con Asimetría de Persistencia)**:
  - $L_{\max\_prune} = 2$, o bien ($L(d_{\text{prune}}) \le 3$ y $L(d_{\text{keeper}}) - L(d_{\text{prune}}) \ge 3$).
  - Permite podar ramas de longitud 3 solo si la rama conservada es marcadamente más larga ($\ge 6$ frames).
- **Variante 3 (TL-PG-V3 — Poda Adaptativa por Grupo)**:
  - Grupo `44b6`: $L_{\max\_prune} = 2$ (alta sensibilidad a pérdida de aristas).
  - Grupo `6bba`: $L_{\max\_prune} = 3$ con asimetría $\Delta L \ge 3$ (mayor densidad de divisiones).

---

## 4. Gates y Criterios de Decisión

- **Gate de Desarrollo (protocol.json)**:
  - $\Delta \text{score}_{\text{dev}} \ge +0.00300$.
  - Cero regresiones de grupo: $\Delta \text{score}_{44\text{b}6} \ge 0$, $\Delta \text{score}_{6\text{bba}} \ge 0$.
  - Peor delta individual $\ge -0.01500$.
- **Presupuesto de Filas**: Estrictamente $234{,}000 < \text{filas} < 242{,}000$.
- **Invariantes**: Inferencia estrictamente ciega a GT. GT se abre solo durante evaluación pareada.
- **Si una variante pasa**: Se seleccionará la mejor variante para someterla a Confirmación en Campaña 032.
- **Si falla**: Se preservarán todos los artefactos y receipts y se investigará la causa raíz.
