# Predeclaración — Campaña 029: Kinematic-Photometric Cytokinesis Regularization & Heading-Guarded Pruning (KP-CR-HGP)

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Operación y Diseño Autónomo)  
**Objetivo**: Superar el récord de la Campaña 027 (score = 0.905423, $\Delta = +0.004149$) y el de la Campaña 028 (score = 0.904872, $\Delta = +0.003598$ con CERO regresiones en los 8 datasets) integrando regularización axial y poda fotométrica-cinemática guiada por persistencia temporal.

---

## 1. Identidad y Punto de Partida

- **Campaña**: 029
- **Hipótesis**: E (Continuaciones y contexto temporal / fotométrico en grafos 3D de tracking).
- **Run fuente**: `b0970-selector-corpus-011` en `A1555`.
  - Recibo de corpus: `artifacts/A1555/b0970-selector-corpus-011/receipt.json` (`completed`).
  - 8 datasets de desarrollo congelados: `44b6_d5e7d891`, `44b6_c50204e0`, `44b6_e57ff5c6`, `44b6_ddf577ad`, `6bba_48816121`, `6bba_09961292`, `6bba_5a4d9360`, `6bba_07e24132`.
- **Nodo de ejecución**: `A1555` (`train-55`, perfil `cpu_evaluation_v1`).
  - Tests requeridos: nueva suite `tests/test_temporal_association_029.py` (meta: $\ge 385$ tests aprobados).
- **Línea de base de control**:
  - Score agregado dev baseline (`011`): `0.901274`
  - $J_{\text{adj}}$ baseline: `0.888231`
  - $J_{\text{div}}$ baseline: `0.130435` (TP = 3 / 15, FP = 8)
  - Mejor candidato previo (Campaña 027 Var 3): `0.905423` ($\Delta = +0.004149$, FP = 5)
  - Candidato sin regresiones previo (Campaña 028 Var 1): `0.904872` ($\Delta = +0.003598$, FP = 6, worst delta $= +0.00000$)

---

## 2. Hipótesis Falsable y Mecanismo

### 2.1 Cuello de Botella Medido
En el grafo de referencia `b0970-selector-corpus-011`:
1. El detector 3D genera un promedio de $\sim 70$ bifurcaciones por dataset en frames donde células densas en migración pasan a distancia $< 4\ \mu\text{m}$.
2. Muchas de estas bifurcaciones no corresponden a mitosis biológica: presentan divergencia no-antiparalela ($\cos\theta > 0$) o asimetría extrema de distancia / aspecto.
3. En la Campaña 027, la poda geométrica en $\cos\theta > 0.00$ redujo los FPs de división de 8 a 5 ($-37.5\%$), pero en `44b6_e57ff5c6` cortó una rama secundaria que continuaba una traza real, generando una pequeña pérdida local de $-0.00474$.
4. En la Campaña 028, el heading-guard protegió esa traza continua, eliminando toda pérdida ($\Delta = +0.00000$ en todas las muestras), pero retuvo 6 FPs en vez de 5.
5. El análisis de los descriptores profundos de 405 dimensiones (tarea 6878) revela que las divisiones verdaderas tienen alta similitud mutua madre-hijas ($\text{sim} \ge 0.50$), mientras que los cruces de células que pasan tienen gran asimetría de apariencia ($\text{sim}(m, d_{\text{spurious}}) < 0.20$ frente a $\text{sim}(m, d_{\text{real}}) > 0.80$).

### 2.2 Mecanismo Propuesto
1. **Regularización axial simétrica**: Suprime el z-jitter inducido por anisotropía axial ($1.625\ \mu\text{m}$ vs $0.40625\ \mu\text{m}$) aplicando suavizado simétrico ($dz_1 \cdot dz_2 < 0$, peso $w_z = 0.8$, clamp $0.6\ \mu\text{m}$).
2. **Selección informada por fotometría**: Al detectar una bifurcación no-mitótica ($\cos\theta > 0.00$ o ratio $< 0.18$), la decisión de cuál rama podar utiliza el descriptor profundo de 405-D: se poda la rama con menor similitud respecto a la madre, protegiendo a la célula con mayor afinidad morfológica.
3. **Guardia de persistencia downstream**: Si la rama candidata a ser podada forma una trayectoria continua de $\ge 2$ frames futuros bien alineados, se examina la rama contraria. Si la rama contraria termina de inmediato ($out\_degree = 0$), se poda la rama efímera; si ambas son persistentes, la guardia impide cortar la rama continua.

### 2.3 Criterio de Falsación
La hipótesis queda rechazada si:
- $\Delta \text{score}_{\text{dev}} < +0.00300$, o
- Se pierde alguna división verdadera ($TP < 3$), o
- Algún grupo presenta regresión ($\Delta \text{score}_{44\text{b}6} < 0$ o $\Delta \text{score}_{6\text{bba}} < 0$), o
- Alguna muestra individual cae por debajo de $-0.01500$.

---

## 3. Variantes Predeclaradas

- **Control**: Grafo de referencia `011` sin modificaciones ($\Delta = 0.00000$).
- **Variante 1 (`KP-CR-HGP-V1`)**:
  - Suavizado axial simétrico ($w_z = 0.8$, clamp $0.6\ \mu\text{m}$, $dz_1 \cdot dz_2 < 0$).
  - Poda safe-zero ($\cos\theta > 0.00$ o ratio $< 0.18$).
  - Selección de rama a podar informada por afinidad de descriptores de 405-D.
  - Heading-guard downstream activo ($\ge 2$ frames persistentes).
- **Variante 2 (`KP-CR-HGP-V2`)**:
  - Todo lo de Variante 1 más filtro de asimetría fotométrica severa: poda bifurcaciones donde $\min(\text{sim}_1, \text{sim}_2) < 0.20$ y $|\text{sim}_1 - \text{sim}_2| > 0.45$.
- **Variante 3 (`KP-CR-HGP-V3`)**:
  - Versión ultra-conservadora: suavizado quasi-simétrico ($dz_1 \cdot dz_2 \le 0$) + poda con umbral seguro de $\cos\theta > 0.05$ y selección por descriptores.

---

## 4. Recursos y Ejecución

- **Nodo**: `A1555` exclusivamente.
- **Perfil**: `cpu_evaluation_v1` (2 hilos CPU, sin GPU, $\le 120$ segundos).
- **Inferencia**: 100% libre de GT (label-free). GT se consulta únicamente durante la evaluación pareada final.
- **Invariantes de linaje**: $in\_degree \le 1$, $out\_degree \le 2$, estricto orden temporal $t+1$.
- **IDs de corrida**:
  - Test run: `b0970-tests-029-rel1`
  - Execution run: `b0970-temporal-029`
