# Campaign 027 Predeclaration: Harmonized Quasi-Symmetric Axial Regularization & Safe-Zero Cytokinesis Pruning

- **Campaign**: 027
- **Date**: 2026-09-14
- **Author**: Antigravity (Autonomous Execution)
- **Target Node**: `A1555` (with unit tests verified on `A1555` / `A1557`)
- **Resource Profile**: `cpu_evaluation_v1`
- **Source Run**: `b0970-selector-corpus-011`
- **Target Task**: `associate-temporal-027`
- **Anticipated Run ID**: `b0970-temporal-027`

---

## 1. Scientific Background & Hypothesis

### Empirical Foundation from Campaigns 023, 024, 025, and 026
1. **Mitosis Boundary Discovery (026)**:
   - Campaign 026 revealed that reference True Positive division **TP 2** (`6bba_09961292`, mother node 15339) possesses an empirical angle of $\cos\theta = -0.0205$ ($\approx 91^\circ$) in the predicted centroid graph due to $z$-axis optical anisotropy ($1.625\ \mu\text{m}$ slice spacing).
   - Setting $\cos_{\text{thresh}} \le -0.10$ pruned TP 2, causing a regression in Division Jaccard.
   - However, setting $\cos_{\text{thresh}} = +0.05$ or $0.00$ mathematically guarantees that TP 2 (at $-0.0205$), TP 1 (at $-0.4828$), and TP 3 (at $-0.7889$) are **$100\%$ preserved** ($TP = 3$).
   - Concurrently, all acute and strictly orthogonal false positive forks ($\cos > 0.05$) are safely eliminated, reducing reference false positives from 8 down to 5 or 4.
2. **Axial Regularization Gains (024)**:
   - In Campaign 024, pure symmetric axial regularizer ($w_z = 0.80$, clamp $\le 0.6\ \mu\text{m}$, jitter threshold $0.8\ \mu\text{m}$) achieved $+0.01644$ in `44b6_ddf577ad` and $+0.00224$ in `44b6_c50204e0` with **zero regressions across all 8 development datasets**.
   - Expanding the symmetry condition from exact integer equality ($z_p == z_s$) to quasi-symmetry ($|z_p - z_s| \times \text{scale}_z \le 0.8\ \mu\text{m}$, or $\le 0.5$ voxel) allows regularizing single-frame spikes where the cell returns to a near-identical focal plane.

### Core Hypotheses
- **Hypothesis 1 (Safe-Zero Cytokinesis Pruning)**:
  Pruning bifurcations with $\cos\theta > +0.05$ (or $> 0.00$) and distance ratio $< 0.18$ will eliminate at least 3 false positive divisions without touching any true positive divisions ($TP = 3$, $FP \le 5$).
- **Hypothesis 2 (Quasi-Symmetric Axial Regularization)**:
  Permitting small focal drift ($|z_p - z_s| \times \text{scale}_z \le 0.8\ \mu\text{m}$) while enforcing direction reversal ($dz_1 \cdot dz_2 < 0$) will capture additional unphysical optical spikes without inducing trajectory drift.
- **Combined Expectation**:
  Synergistic development score delta $\Delta \text{score}_{\text{dev}} \ge +0.00400$ (well above the $+0.00300$ gate threshold), zero sample regressions, and strict preservation of the row budget ($234{,}000 < \text{rows} < 242{,}000$).

---

## 2. Experimental Variants

### Variant 1: Pure Symmetric Axial Regularizer + Safe-Zero Cytokinesis Pruning ($\cos > +0.05$)
- **Axial Regularizer**:
  - `reversal_only = true` ($dz_1 \cdot dz_2 < 0$).
  - `symmetric_only = true` ($z_p == z_s$).
  - `jitter_threshold_um = 0.8`.
  - `max_correction_um = 0.6`.
  - `weight_z = 0.80`.
- **Cytokinesis Fork Pruning**:
  - `cos_thresh = 0.05`.
  - `min_ratio = 0.15`.
  - `min_max_dist_um = 3.0`.
  - Keeper selection: kinetic alignment with mother trajectory + patch descriptor correlation.
- **Cytokinesis Recovery**: Disabled.

### Variant 2: Quasi-Symmetric Axial Regularizer + Safe-Zero Cytokinesis Pruning
- **Axial Regularizer**:
  - `reversal_only = true` ($dz_1 \cdot dz_2 < 0$).
  - `symmetric_only = false`.
  - `max_asymmetry_um = 0.80` ($|z_p - z_s| \times \text{scale}_z \le 0.8\ \mu\text{m}$).
  - `jitter_threshold_um = 0.8`.
  - `max_correction_um = 0.6`.
  - `weight_z = 0.80`.
- **Cytokinesis Fork Pruning**: Same as Variant 1 (`cos_thresh = 0.05`, `min_ratio = 0.15`).
- **Cytokinesis Recovery**: Disabled.

### Variant 3: Pure Symmetric Axial Regularizer + Strict-Zero Cytokinesis Pruning ($\cos > 0.00$)
- **Axial Regularizer**: Pure symmetric.
- **Cytokinesis Fork Pruning**:
  - `cos_thresh = 0.00`.
  - `min_ratio = 0.18`.
  - `min_max_dist_um = 3.0`.
- **Cytokinesis Recovery**: Disabled.

---

## 3. Gates & Admissibility Criteria

| Metric | Target | Mandatory Gate |
| :--- | :--- | :--- |
| **Development Score Delta** | $\Delta \text{score}_{\text{dev}} \ge +0.00300$ | Yes |
| **Group Score Deltas** | $\Delta_{\text{44b6}} \ge 0.00000$, $\Delta_{\text{6bba}} \ge 0.00000$ | Yes (zero group regressions) |
| **Individual Sample Delta** | $\Delta_{\text{adj\_edge}} \ge -0.01500$ across all 8 datasets | Yes |
| **Edge Jaccard Delta** | $\Delta_{\text{edge}} \ge 0.00000$ | Yes |
| **Division Jaccard Delta** | $\Delta_{\text{div}} \ge 0.00000$ | Yes |
| **Resource Profile** | `cpu_evaluation_v1`, isolated Python, no MLflow | Yes |
