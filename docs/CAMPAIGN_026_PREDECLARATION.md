# Campaign 026 Predeclaration: Cytokinesis Collinearity Gated Pruning & Antiparallel Mitosis Recovery

- **Campaign**: 026
- **Date**: 2026-09-14
- **Author**: Antigravity (Autonomous Execution)
- **Target Node**: `A1555` (with test suite parallel verification on `A1557`)
- **Resource Profile**: `cpu_evaluation_v1`
- **Source Run**: `b0970-selector-corpus-011`
- **Target Task**: `associate-temporal-026`
- **Anticipated Run ID**: `b0970-temporal-026`

---

## 1. Scientific Background & Hypothesis

### Forensic Foundation (Subagent 1 Findings)
Across the 8 official development datasets:
- **Baseline Division Metrics**: $TP = 3, FP = 8, FN = 12$ ($J_{\text{div}} = 3/23 = 0.130435$).
- **True Mitotic Divisions ($N = 15$)**: 14 of 15 (93.3%) exhibit strict biological cytokinesis collinearity ($\cos\theta < -0.60$, median $-0.7677$, distance ratio $\ge 0.60$). In eukaryotic anaphase/telophase, the mitotic spindle pushes daughter chromatids in diametrically opposing directions along the division axis ($\cos \theta \to -1$).
- **False Positive Divisions ($N = 8$)**: 6 of the 8 evaluated FPs violate this biological constraint ($\cos\theta > -0.40$ or distance ratio $< 0.40$), resulting from spurious linking of converging migrating tracks.
- **Missed Divisions ($N = 12$)**: Include textbook additive second daughters (e.g. Div 12 in `48816121`, $t=11$, with $\cos\theta = -0.9984$, ratio $= 0.9917$, displacements $7.05\ \mu\text{m}$ and $6.99\ \mu\text{m}$) where the second daughter was detected and unassigned, but never linked by the linear assignment tracker.

### Hypotheses
1. **Hypothesis 1 (Collinearity Cytokinesis Pruning)**:
   Pruning bifurcations where $\cos\theta > -0.10$ or ratio $< 0.25$ will eliminate spurious acute/orthogonal divisions while strictly preserving all 3 reference True Positives, boosting Division Jaccard and overall score without edge regressions.
2. **Hypothesis 2 (Deep Cytokinesis Filter)**:
   Extending the pruning threshold to $\cos\theta > -0.30$ or ratio $< 0.35$ eliminates up to 6 of the 8 reference False Positives.
3. **Hypothesis 3 (Ultra-Conservative Antiparallel Recovery)**:
   Targeted recovery of near-perfect antiparallel pairs ($\cos\theta \le -0.92$, ratio $\ge 0.70$, $\text{sister dist} \le 15.0\ \mu\text{m}$, mother track $\ge 3$) will recover true cytokinesis divisions (such as Div 12) without creating false positive bifurcation noise.

---

## 2. Experimental Variants

### Variant 1: Pure Axial Regularizer + Collinearity Cytokinesis Pruning
- **Axial Regularizer**: Reversal only ($dz_1 \cdot dz_2 < 0$), symmetric only ($z_p == z_s$), clamp $\le 0.6\ \mu\text{m}$, jitter threshold $0.8\ \mu\text{m}$, $w_z = 0.80$.
- **Fork Pruning**:
  - `cos_thresh = -0.10` (prunes acute and orthogonal forks).
  - `min_ratio = 0.25` with `min_max_dist_um = 3.0`.
  - Keeper selection: kinetic mother alignment + patch descriptor correlation.
- **Cytokinesis Recovery**: Disabled.

### Variant 2: Pure Axial Regularizer + Deep Cytokinesis Filter
- **Axial Regularizer**: Same as Variant 1.
- **Fork Pruning**:
  - `cos_thresh = -0.30` (prunes non-antiparallel forks).
  - `min_ratio = 0.35` with `min_max_dist_um = 3.0`.
  - Keeper selection: kinetic mother alignment + patch descriptor correlation.
- **Cytokinesis Recovery**: Disabled.

### Variant 3: Variant 1 Pruning + Ultra-Conservative Antiparallel Recovery
- **Axial Regularizer**: Same as Variant 1.
- **Fork Pruning**: Same as Variant 1 (`cos_thresh = -0.10`, `min_ratio = 0.25`).
- **Cytokinesis Recovery**:
  - `cos_max = -0.92` (strictly antiparallel).
  - `min_ratio = 0.70` (highly symmetric daughter displacement).
  - `max_dist_um = 8.0`.
  - `max_sister_dist_um = 15.0`.
  - `min_mother_track_len = 3`.
  - `min_desc_corr = 0.40`.

---

## 3. Gates & Admissibility Criteria

| Metric | Target | Mandatory Gate |
| :--- | :--- | :--- |
| **Development Score Delta** | $\Delta \text{score}_{\text{dev}} \ge +0.00300$ | Yes |
| **Confirmation Score Delta** | $\Delta \text{score}_{\text{conf}} \ge +0.00200$ | Yes (when confirmed) |
| **Group Score Deltas** | $\Delta_{\text{44b6}} \ge 0.00000$, $\Delta_{\text{6bba}} \ge 0.00000$ | Yes (zero group regressions) |
| **Individual Sample Delta** | $\Delta_{\text{adj\_edge}} \ge -0.01500$ across all 8 datasets | Yes |
| **Edge Jaccard Delta** | $\Delta_{\text{edge}} \ge 0.00000$ | Yes |
| **Division Jaccard Delta** | $\Delta_{\text{div}} \ge 0.00000$ | Yes |
| **Resource Isolation** | `cpu_evaluation_v1`, isolated Python, no MLflow | Yes |
