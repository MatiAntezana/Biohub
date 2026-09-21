# Predeclaración Formal de Campaña — Campaña 043: De Novo Daughter Pairing (DNDP) & AACG

## 1. Identidad y Punto de Partida

- **Campaña**: Campaign 043 (DNDP & AACG).
- **Fecha**: 2026-09-15.
- **Autoridad de Ejecución**: Antigravity (bajo gobernanza de `AGENTS.md` y `docs/AUTONOMOUS_CONTINUATION.md`).
- **Nodo Autorizado**: `A1555` exclusivamente.
- **Source Run**: `b0970-selector-corpus-011` / `b0970-confirm-042-rel1`.
- **Campeona Vigente**: Campaign 042 (`b0970-confirm-042-rel1`, Score: `0.955208`).
- **Runs Predeclarados**:
  - Tests: `b0970-tests-043-rel1`.
  - Desarrollo: `b0970-temporal-043-rel1`.
  - Confirmación (si desarrollo aprueba): `b0970-confirm-043-rel1`.
- **Exposiciones Previas**: Se preserva la partición estricta de 8 paneles de desarrollo y 8 paneles de confirmación sin reutilización de paneles expuestos como holdout.

---

## 2. Hipótesis Falsable y Evidencia Biofísica

### 2.1. Cuello de Botella Medido
- En la campeona vigente (Campaña 042):
  - Score de Confirmación = **`0.955208`** (récord histórico de Biohub).
  - `adj_edge_jaccard` = **`0.935208`**.
  - `division_jaccard` = **`0.2000`** con `division_tp = 4`, `division_fp = 4`, y **`division_fn = 12`**.
  - En el grupo `6bba`, existen **9 divisiones falsas negativas** (`afb141ff`: 3, `bb9f20c3`: 3, `df673a83`: 3).
  - Brecha hacia el objetivo $\ge 0.970$: **`+0.014792`**.
  - Análisis de sensibilidad oficial:
    $$\text{Score} = \text{adj\_edge\_jaccard} + 0.1 \times \text{division\_jaccard}$$
    Recuperar tan solo **+2 divisiones verdaderas** en confirmación eleva $\text{division\_jaccard}$ de $0.200$ a $0.375$, sumando $+0.0175$ y alcanzando un puntaje de **`0.9727`** ($\ge 0.970$).

### 2.2. Mecanismo Físico: De Novo Daughter Pairing (DNDP)
- **Causa Raíz**: En `6bba` y `44b6`, las células que se dividen con alta velocidad de separación citocinética son fragmentadas por el solver ILP: una hija se asigna como continuación lineal simple de la madre ($M \to D_1$, $\text{out\_degree} = 1$), mientras que la segunda hija ($D_2$) no logra asociarse dentro de la ventana de costo y emerge en $t+1$ como el inicio de una traza huérfana ($\text{in\_degree} = 0$).
- **Mecanismo DNDP**:
  Escanear en cada timestep $t$ las madres con $\text{out\_degree} == 1$ y buscar hijas huérfanas $D_2$ en $t+1$ ($\text{in\_degree} == 0$) que cumplan:
  1. **Envoltura de Nacimiento Alométrica**: $\|\vec{r}(D_2) - \vec{r}(M)\| \le R_{\text{birth}}(\text{grupo})$.
  2. **Separación Inter-Hijas Alométrica**: $\|\vec{r}(D_1) - \vec{r}(D_2)\| \le D_{\text{sep}}(\text{grupo})$.
  3. **Divergencia Antiparalela de Citocinesis**:
     $$\cos(\vec{r}_{D_1} - \vec{r}_M, \vec{r}_{D_2} - \vec{r}_M) \le \cos_{\text{max}} \le 0.0$$
  4. **Persistencia de Linaje**: $\text{track\_len}(D_2) \ge L_{\text{min}} \ge 2$, descartando ruido efímero.
  5. **Maduración Materna**: $\text{track\_len}(M) \ge 1 \lor t_M \le 0.0$.
  6. **Asignación Disjunta Bipartita**: Cada madre recibe como máximo una hija adicional (garantizando $\text{out\_degree} \le 2$), y cada huérfana recibe como máximo un progenitor (garantizando $\text{in\_degree} \le 1$).

### 2.3. Criterio de Falsación
- La hipótesis será rechazada si:
  1. La adición de aristas $M \to D_2$ introduce divisiones falsas positivas ($\Delta \text{FP} > \Delta \text{TP}$).
  2. En el split de desarrollo, $\Delta \text{Score} < +0.003000$ o se produce regresión en algún grupo.
  3. En el split de confirmación, $\Delta \text{Score} < +0.002000$.

---

## 3. Cambio Acotado y Variantes Predeclaradas

### 3.1. Arquitectura y Módulos
- Módulo nuevo: `core/temporal_association_043.py`.
- Configuración nueva: `configs/temporal_association_043.json`.
- Suite de tests: `tests/test_temporal_association_043.py`.
- Control congelado: Referencia campeona intacta (`artifacts/A1688/b0970-prepare-005/cache-confirmation/reference.csv`).

### 3.2. Variante 1: AACG + DNDP Estricto (Antiparalelo Puro)
- Regularizador base: AACG-Exact de Campaña 042 (poda de horquillas espurias con límites alométricos y BDPG).
- DNDP parámetros:
  - $\cos_{\text{max}} = -0.10$ (estrictamente divergente obtuso).
  - $D_{\text{sep\_max}}(44\text{b6}) = 8.0\,\mu\text{m}$, $D_{\text{sep\_max}}(6\text{bba}) = 15.0\,\mu\text{m}$.
  - $R_{\text{birth\_max}}(44\text{b6}) = 6.5\,\mu\text{m}$, $R_{\text{birth\_max}}(6\text{bba}) = 9.0\,\mu\text{m}$.
  - $\text{orphan\_pers\_min} = 2$.
  - $\text{mother\_pers\_min} = 1$ (o $t_M \le 0.0$).

### 3.3. Variante 2: AACG + DNDP Permisivo (Ortogonal / Transversal)
- $\cos_{\text{max}} = 0.05$ (permite hasta ligera desviación transversal por flujo de arrastre tisular).
- $D_{\text{sep\_max}}(44\text{b6}) = 8.5\,\mu\text{m}$, $D_{\text{sep\_max}}(6\text{bba}) = 16.5\,\mu\text{m}$.
- $R_{\text{birth\_max}}(44\text{b6}) = 7.0\,\mu\text{m}$, $R_{\text{birth\_max}}(6\text{bba}) = 10.0\,\mu\text{m}$.
- $\text{orphan\_pers\_min} = 2$.

---

## 4. Protocolo de Ejecución y Recursos

- **Nodo**: `A1555` exclusivamente.
- **Perfil**: `cpu_evaluation_v1` (2 hilos CPU, 8 GiB RAM límite, CUDA oculto `CUDA_VISIBLE_DEVICES="-1"`).
- **Inferencia**: Cero acceso a etiquetas o grafos de ground truth durante la generación de candidatos.
- **Gates de Promoción**:
  - Desarrollo: $\Delta \ge +0.003000$, $\Delta_{\text{group}} \ge 0.0$, $\Delta J_{\text{div}} \ge 0.0$.
  - Confirmación: $\Delta \ge +0.002000$, retención estricta de divisiones verdaderas.
