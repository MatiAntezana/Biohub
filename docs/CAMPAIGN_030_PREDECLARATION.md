# Predeclaración — Campaña 030: Evaluación Oficial de Confirmación del Campeón KP-CR-HGP

**Fecha**: 2026-09-14  
**Responsable**: Antigravity (Operación y Diseño Autónomo)  
**Objetivo**: Evaluar la receta campeona única de la Campaña 029 (Variante 1: KP-CR-HGP-V1) sobre el panel de confirmación no expuesto de 8 datasets, superando el gate oficial de confirmación ($\Delta \text{score}_{\text{conf}} \ge +0.00200$, cero regresiones grupales, peor muestra $\ge -0.01500$) para habilitar el empaquetado de producción y submission a Kaggle.

---

## 1. Identidad y Punto de Partida

- **Campaña**: 030
- **Fase**: Confirmación Oficial (Holdout Protocolar).
- **Receta Campeona Evaluada**: Variante 1 de la Campaña 029 (KP-CR-HGP-V1).
  - Regularización axial simétrica ($w_z = 0.8$, clamp $0.6\ \mu\text{m}$, supresión de inversión $dz_1 \cdot dz_2 < 0$).
  - Poda safe-zero fotométricamente informada ($\cos\theta > 0.00$ o ratio $< 0.18$).
  - Selección de rama a podar ponderada por afinidad de descriptores de 405-D ($+1.5 \times \text{sim}(m, d_i)$).
  - Guardia de persistencia downstream activa ($\ge 2$ frames bien alineados).
- **Run fuente**: `b0970-prepare-005` en `A1555` (`cache-confirmation`).
- **Nodo de ejecución**: `A1555` (`train-55`, perfil `cpu_evaluation_v1`).
- **Datasets de confirmación (8 muestras no expuestas)**:
  - Grupo `44b6`: `44b6_2a2eff9f`, `44b6_587a1e22`, `44b6_c15fded2`, `44b6_d2f34f90`.
  - Grupo `6bba`: `6bba_062c8d37`, `6bba_afb141ff`, `6bba_bb9f20c3`, `6bba_df673a83`.
- **Antecedente de desarrollo**:
  - Score Agregado en desarrollo: `0.905586` ($\Delta = \mathbf{+0.004312}$).
  - CERO regresiones en las 8 muestras de desarrollo (peor delta $= +0.00000$).

---

## 2. Hipótesis Falsable y Mecanismo

### 2.1 Generalización del Mecanismo en el Panel de Confirmación
El suavizado de jitter axial z y la poda de bifurcaciones espurias mediante afinidad fotométrica y persistencia de trayectoria responden a invariantes biológicas y de adquisición óptica (anisotropía de muestreo $1.625\ \mu\text{m}$ vs $0.40625\ \mu\text{m}$ y exclusión estérica celular).
Por tanto, esta regularización debe transferirse íntegramente a las 8 muestras de confirmación no expuestas, eliminando falsos positivos de división y mejorando la precisión de aristas sin degradar divisiones verdaderas.

### 2.2 Criterio de Aprobación de Confirmación
Conforme a `configs/protocol.json` y `AUTONOMOUS_CONTINUATION.md` (Sección 7):
- $\Delta \text{score}_{\text{conf}} \ge +0.00200$
- $\Delta \text{score}_{44\text{b}6} \ge 0.000$
- $\Delta \text{score}_{6\text{bba}} \ge 0.000$
- Delta de Jaccard de aristas ajustadas en cualquier muestra individual $\ge -0.01500$.
- Delta agregado de Jaccard de aristas $\ge 0.000$.
- Delta agregado de Jaccard de división $\ge 0.000$.

### 2.3 Criterio de Rechazo
La candidata queda rechazada para publicación si:
- $\Delta \text{score}_{\text{conf}} < +0.00200$, o
- Se pierde alguna división verdadera en confirmación, o
- Ocurre alguna regresión en los promedios grupales, o
- Alguna muestra individual cae por debajo de $-0.01500$.
Si se rechaza, NO se probará una segunda variante sobre confirmación ni se ajustarán hiperparámetros reactivamente.

---

## 3. Variantes Predeclaradas

- **Receta Única (Confirmación prohíbe barridos)**:
  - **`KP-CR-HGP-V1`**:
    - Suavizado axial simétrico ($w_z = 0.8$, clamp $0.6\ \mu\text{m}$).
    - Poda safe-zero ($\cos\theta > 0.00$ o ratio $< 0.18$).
    - Selección por descriptores y guardia de persistencia downstream.

---

## 4. Recursos y Ejecución

- **Nodo**: `A1555` exclusivamente.
- **Perfil**: `cpu_evaluation_v1` (2 hilos CPU, sin GPU, $\le 120$ segundos).
- **Inferencia**: 100% libre de GT (label-free). GT de confirmación se consulta únicamente durante la evaluación pareada final.
- **IDs de corrida**:
  - Test run: `b0970-tests-030-rel1`
  - Execution run: `b0970-confirm-030`
