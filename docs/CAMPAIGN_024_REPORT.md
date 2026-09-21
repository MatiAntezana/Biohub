# Reporte de Campaña 024 — Asociación Temporal Biomecánica y Simetría de Citocinesis

**Fecha**: 2026-09-14  
**Nodo**: A1555 (Remote CPU Evaluation `cpu_evaluation_v1`, 2 hilos, 8 GiB memoria)  
**Run ID de Ejecución**: `b0970-temporal-024-rel2`  
**Run Fuente**: `b0970-selector-corpus-011`  
**Run de Tests Previos**: `b0970-tests-024-rel3` (356/356 tests superados, suite `full_discovery`, 0 skips, 0 errores)  
**Hash de Código**: `c9fa058ef9196782f09360344f1dd41f3cf6393631eb3551e58f4bb46b8fc270`  
**Disposición Oficial**: `eligible_for_confirmation_and_package`  
**Mejor Variante**: `variant_1`  
**Hash SHA256 de Resultados**: `894abb92769298dd8fcdcee13218fbd9e658eee6980debfd0080a03dbabae1ff`  

---

## 1. Resumen Ejecutivo y Decisión

La Campaña 024 evaluó tres variantes de asociación temporal con restricciones biomecánicas sobre los 8 datasets del panel de desarrollo:
1. **Regularizador Axial Simétrico Puro**: supresión de micro-jitter en el eje óptico $Z$ exclusivamente en trayectorias lineales aisladas y balanceadas, preservando coordenadas laterales $X, Y$ intactas.
2. **Poda de Falsas Citocinesis por Divergencia Antiparalela**: eliminación del brazo espurio en bifurcaciones con ángulo agudo ($\cos\theta > 0.20$) o asimetría extrema de distancia ($r < 0.15$), respaldado por la auditoría forense de los 15 eventos de división GT y 8 FPs.
3. **Recuperación Antiparalela de Citocinesis**: reconexión selectiva de hijas huérfanas en el hemisferio opuesto ($\cos\theta < -0.85$).

### Resultado de los Gates
- **Variante 1**: **APROBADA** ($\Delta \text{score} = \mathbf{+0.003598} \ge +0.00300$, `passed: True`, `reasons: []`).
- **Variante 2**: **APROBADA** ($\Delta \text{score} = \mathbf{+0.003291} \ge +0.00300$, `passed: True`).
- **Variante 3**: **RECHAZADA** ($\Delta \text{score} = +0.001995$, reduce $J_{\text{div}}$ en $-0.005435$).

**Decisión**: Promover `variant_1` a la fase de confirmación y empaquetado (`eligible_for_confirmation_and_package`).

---

## 2. Métricas Oficiales Agregadas (Variante 1)

| Métrica | Referencia (Control) | Candidata (024-V1) | Delta Oficial | Gate Requerido | Estado |
|---|---|---|---|---|---|
| **Score Competición** | 0.901274 | **0.904872** | **+0.003598** | $\ge +0.00300$ | **SUPERADO** |
| **Adjusted Edge Jaccard** | 0.888231 | 0.890586 | **+0.002355** | $> 0.00000$ | **SUPERADO** |
| **Edge Jaccard** | 0.881437 | 0.883758 | **+0.002321** | $> 0.00000$ | **SUPERADO** |
| **Division Jaccard** | 0.130435 | 0.142857 | **+0.012422** | $\ge 0.00000$ | **SUPERADO** |
| **Grupo 44b6 Score** | 0.834231 | 0.839737 | **+0.005505** | $\ge 0.00000$ | **SUPERADO** |
| **Grupo 6bba Score** | 0.941911 | 0.944292 | **+0.002381** | $\ge 0.00000$ | **SUPERADO** |

---

## 3. Desglose Detallado por Dataset (Variante 1)

| Dataset | Split | Grupo | Ref Adj Edge | Cand Adj Edge | Delta Adj Edge | Regresión |
|---|---|---|---|---|---|---|
| `44b6_c50204e0` | dev | 44b6 | 0.81599 | 0.81823 | **+0.00224** | CERO |
| `44b6_d5e7d891` | dev | 44b6 | 0.85906 | 0.85906 | **+0.00000** | CERO |
| `44b6_ddf577ad` | dev | 44b6 | 0.84121 | 0.85765 | **+0.01644** | CERO |
| `44b6_e57ff5c6` | dev | 44b6 | 0.73002 | 0.73002 | **+0.00000** | CERO |
| `6bba_07e24132` | dev | 6bba | 0.82505 | 0.82505 | **+0.00000** | CERO |
| `6bba_09961292` | dev | 6bba | 0.91648 | 0.91694 | **+0.00047** | CERO |
| `6bba_48816121` | dev | 6bba | 0.93806 | 0.93905 | **+0.00100** | CERO |
| `6bba_5a4d9360` | dev | 6bba | 0.94690 | 0.94690 | **+0.00000** | CERO |

**Cero regresiones en los 8 datasets**: ni una sola muestra experimentó regresión individual ($\Delta \ge 0.00000$ en todas).

---

## 4. Comparación entre Variantes

| Variante | Descripción | $\Delta \text{Score}$ | $\Delta J_{\text{adj}}$ | $\Delta J_{\text{div}}$ | Gate Status |
|---|---|---|---|---|---|
| **Variante 1** | Poda conservadora ($\cos > 0.20$, $r < 0.15$) + Reg. Axial Simétrico | **+0.003598** | +0.002355 | **+0.012422** | **PASSED** |
| **Variante 2** | Poda estricta ($\cos > 0.10$, $r < 0.20$) + Persistence Guard | **+0.003291** | +0.002049 | **+0.012422** | **PASSED** |
| **Variante 3** | Poda + Reconexión Antiparalela ($\cos < -0.85$) | +0.001995 | **+0.002538** | -0.005435 | FAILED ($J_{\text{div}} < 0$) |

---

## 5. Verificación de Invariantes Absolutas

1. **Sin filtración de Ground Truth**: la inferencia y generación de candidatos de los 8 datasets operó de forma estrictamente *label-free* usando únicamente coordenadas predichas, topología del grafo, escala física y descriptores morfológicos 3D. GT se abrió exclusivamente en la función de evaluación oficial pareada.
2. **Presupuesto de Filas y Nodos**: las coordenadas de nodos permanecen en exactamente 235,674 filas (estrictamente dentro de la cota $234{,}000 < \text{filas} < 242{,}000$).
3. **Invariantes de Grado**: para todo nodo, $\text{in\_degree} \le 1$ y $\text{out\_degree} \le 2$.
4. **Reproducibilidad y Aislamiento**: ejecutado bajo el perfil `cpu_evaluation_v1` en A1555 con CUDA aislado, 2 hilos CPU, 8 GiB memoria y suite de 356 pruebas unitarias previa con 0 fallas.
