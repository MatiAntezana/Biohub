# Predeclaración de Campaña 025 — Desanudado de Cruces por Conservación de Momento Cinético y Asociación Temporal Biomecánica

**Fecha**: 2026-09-14  
**Responsable Autónomo**: Antigravity  
**Marco Normativo**: `docs/AUTONOMOUS_CONTINUATION.md` (Sección 5), `docs/NEXT_EXPERIMENT_TEMPLATE.md`  
**Nodo de Ejecución Autorizado**: A1555 (entorno `cpu_evaluation_v1`, 2 hilos CPU, 8 GiB memoria, lock exclusivo de proceso)  
**Run Fuente**: `b0970-selector-corpus-011` en A1555 (corpus congelado con 53 NPZ de evidencia label-free)  
**Run de Tests Previos**: `b0970-tests-025-rel1` (suite `full_discovery`, 0 skips, 0 errores, >= 356 tests)  
**Run ID de Campaña**: `b0970-temporal-025`  

---

## 1. Identidad y Punto de Partida

- **Antecedente Inmediato**: La Campaña 024 (`b0970-temporal-024-rel2`) superó exitosamente el gate oficial de desarrollo con $\Delta \text{score} = \mathbf{+0.003598}$, cero regresiones en los 8 datasets de desarrollo y mejora en $J_{\text{div}}$ de $+0.012422$.
- **Evidencia Nueva Observada**:
  - La auditoría cinemática de los 8 datasets de desarrollo reveló puntos críticos de cruce erróneo (crossover track swaps) en marcos densos (e.g. $t=9, 18$ en `44b6_ddf577ad`, $t=37, 48$ en `44b6_d5e7d891`, $t=70$ en `44b6_c50204e0`, $t=2, 8$ en `6bba_48816121`).
  - En cada cruce, dos células que se mueven en trayectorias inerciales lineales aproximan sus centros a distancia $< 6.0\ \mu\text{m}$. La asignación estática basada en mínima distancia genera giros en horquilla (hairpin turns con $\cos\theta \le 0.00$).
  - Si se desanudan las aristas respetando la conservación de momento lineal, la colinealidad angular salta a $\cos_{s_1}, \cos_{s_2} \ge 0.85$, restaurando 2 aristas TP y eliminando 2 aristas FP por cada cruce resuelto.
- **Invariantes Inquebrantables**:
  - Inferencia 100% *label-free*: la propuesta de desanudado opera únicamente sobre coordenadas predichas, escala espacial y morfología de parches 3D. El GT permanece cerrado hasta la evaluación pareada final.
  - Presupuesto de filas: exactamente 235,674 nodos (dentro de la cota estricta $234{,}000 < \text{filas} < 242{,}000$).
  - Grados: $\text{in\_degree} \le 1, \text{out\_degree} \le 2$.

---

## 2. Hipótesis Falsable

1. **Cuello de Botella Medido**:
   - En el panel de desarrollo, existen 373 aristas FP y 357 aristas FN.
   - En `44b6_e57ff5c6` ($J_{\text{edge}} = 0.713$) y `44b6_c50204e0` ($J_{\text{edge}} = 0.795$), los errores de enlace entre células anotadas adyacentes reducen la métrica en $> 0.05$.
   - Los cruces erróneos violan flagrantemente la inercia biofísica celular al forzar cambios angulares abruptos de $90^\circ$ a $180^\circ$.
2. **Mecanismo Biofísico Propuesto**:
   - Las células eucariotas en migración tienen masa e interacciones citoesqueléticas que limitan las aceleraciones transversales repentinas en intervalos de $\Delta t = 1$ marco.
   - Al evaluar dos células cercanas $u_1, u_2$ en $t$ y sus enlaces actuales $v_1, v_2$ en $t+1$:
     $$\Delta_{\text{momentum}} = \big(\cos(\vec{v}_{u_1 \to v_2}, \vec{v}_{\text{in}, 1}) + \cos(\vec{v}_{u_2 \to v_1}, \vec{v}_{\text{in}, 2})\big) - \big(\cos(\vec{v}_{u_1 \to v_1}, \vec{v}_{\text{in}, 1}) + \cos(\vec{v}_{u_2 \to v_2}, \vec{v}_{\text{in}, 2})\big)$$
     Si $\Delta_{\text{momentum}} \ge 1.00$ y la asignación actual exhibe desvío agudo ($\cos \le 0.20$) mientras la permutada es fuertemente colineal ($\cos \ge 0.50$), el par $(u_1, u_2)$ representa un cruce espurio que debe desanudarse.
3. **Criterio de Falsación**:
   - Si el desanudado cinemático reduce el Adjusted Edge Jaccard global o genera regresión $> 0.00000$ en cualquier dataset frente a la Variante 1 de la Campaña 024, la hipótesis queda refutada y se preserva 024-V1.

---

## 3. Variantes Evaluadas (Máximo 2)

### Variante 1: Desanudado Cinemático Conservador + Receta 024-V1
- **Reglas de Desanudado**:
  - Proximidad espacial: $\text{dist}(u_1, u_2) \le 5.5\ \mu\text{m}$.
  - Ganancia angular: $\Delta_{\text{momentum}} \ge 1.20$.
  - Ángulo previo deficiente: $\max(\cos_1, \cos_2) \le 0.10$.
  - Ángulo permutado excelente: $\min(\cos_{s_1}, \cos_{s_2}) \ge 0.60$.
  - Máximo de intercambios por dataset: 50.
- **Módulos Previos Integrados**:
  - Regularizador axial simétrico puro ($Z$-jitter suppression).
  - Poda de falsas citocinesis ($\cos > 0.20$, ratio $< 0.15$).

### Variante 2: Desanudado Cinemático Moderado con Guardia Morfológica
- **Reglas de Desanudado**:
  - Proximidad espacial: $\text{dist}(u_1, u_2) \le 6.0\ \mu\text{m}$.
  - Ganancia angular: $\Delta_{\text{momentum}} \ge 1.00$.
  - Ángulo previo deficiente: $\max(\cos_1, \cos_2) \le 0.20$.
  - Ángulo permutado: $\min(\cos_{s_1}, \cos_{s_2}) \ge 0.50$.
  - Guardia de descriptores: $\text{sim}(u_1, v_2) + \text{sim}(u_2, v_1) \ge \text{sim}(u_1, v_1) + \text{sim}(u_2, v_2) - 0.10$.

---

## 4. Gates de Decisión Predeclarados

1. **Gate Principal de Desarrollo**:
   - $\Delta \text{score}_{\text{dev}} \ge +0.00300$ respecto al control original (y $\ge +0.00360$ respecto a 024).
   - $\Delta \text{score}_{44\text{b}6} \ge 0.00000$, $\Delta \text{score}_{6\text{bba}} \ge 0.00000$.
   - Ninguna regresión individual inferior a $-0.015$.
2. **Si Supera los Gates**:
   - Declarar `eligible_for_confirmation_and_package`.
   - Proceder al empaquetado del bundle offline para Kaggle.
3. **Si Falla**:
   - Conservar artefactos y receipts completos; registrar reporte y mantener 024-V1 como mejor candidata vigente.
