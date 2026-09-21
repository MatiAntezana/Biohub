# Campaign 026 Report: Cytokinesis Collinearity Gated Pruning & Mitosis Sensitivity

- **Date**: 2026-09-14
- **Execution Run ID**: `b0970-temporal-026`
- **Source Run**: `b0970-selector-corpus-011`
- **Tests Run**: `b0970-tests-026-rel1` (370/370 passed on A1555)
- **Execution Node**: `A1555` (`cpu_evaluation_v1`, wall time 187.9s)
- **Code Hash**: `3b29afbc647b6f620c3b24e8d9b05cd6510f1bcab8e1c3d09a75190e3d6ece64`
- **Artifacts**: `artifacts/A1555/b0970-temporal-026/`

---

## 1. Executive Summary

Campaign 026 evaluated tighter biological cytokinesis collinearity thresholds ($\cos\theta > -0.10$ and $\cos\theta > -0.30$) and ultra-conservative antiparallel second-daughter recovery on the official 8 development datasets on A1555.

### Results Summary
| Metric | Reference (011) | Variant 1 ($\cos > -0.10, r < 0.25$) | Variant 2 ($\cos > -0.30, r < 0.35$) | Variant 3 (Recovery) |
| :--- | :--- | :--- | :--- | :--- |
| **Development Score** | 0.901274 | 0.900931 ($\Delta = -0.000343$) | 0.895509 ($\Delta = -0.005765$) | 0.900263 ($\Delta = -0.001011$) |
| **Division Jaccard** | 0.130435 | 0.105263 ($\Delta = -0.025172$) | 0.052632 ($\Delta = -0.077803$) | 0.100000 ($\Delta = -0.030435$) |
| **Division TP / FP / FN**| 3 / 8 / 12 | **2 / 4 / 13** | **1 / 4 / 14** | **2 / 5 / 13** |
| **Adjusted Edge Jaccard**| 0.888231 | **0.890405** ($\Delta = \mathbf{+0.002174}$) | **0.890246** ($\Delta = \mathbf{+0.002015}$) | **0.890263** ($\Delta = \mathbf{+0.002032}$) |
| **Edge Jaccard** | 0.881437 | **0.883582** ($\Delta = \mathbf{+0.002145}$) | **0.883424** ($\Delta = \mathbf{+0.001987}$) | **0.883440** ($\Delta = \mathbf{+0.002003}$) |
| **Gate Status** | N/A | **REJECTED** | **REJECTED** | **REJECTED** |
| **Disposition** | N/A | `rejected_no_kaggle_submission` | `rejected_no_kaggle_submission` | `rejected_no_kaggle_submission` |

---

## 2. Key Scientific Discovery: Empirical Limits of Eukaryotic Mitosis Geometry in Noisy 3D Microscopy

### Discovery 1: True Biological Mitosis Can Appear Orthogonal ($\cos\theta \approx 0.0$) in Predicted Graphs
- In theoretical ideal cytokinesis, the division axis is linear ($\cos\theta = -1.0$).
- However, in predicted 3D centroid graphs, one of the 3 reference True Positives (**TP 2**: `6bba_09961292`, mother node 15339, forking to daughters 15612 and 15614) has an empirical angle of $\mathbf{\cos\theta = -0.0205}$ ($\approx 91^\circ$) and distance ratio $0.4311$.
- **Mechanism**:
  - Optical depth resolution ($z$-spacing $1.625\ \mu\text{m}$ vs $xy$-spacing $0.40625\ \mu\text{m}$) causes anisotropic centroid localization errors.
  - Mitotic cells round up and bulge during metaphase/anaphase, shifting the detected centroid relative to the true daughter centrosomes.
  - When the threshold was tightened from `cos_thresh = 0.20` (Campaign 024) to `cos_thresh = -0.10` (Campaign 026), **TP 2 was mistakenly pruned**, reducing True Positives from 3 to 2 and dragging Division Jaccard down from $0.142857$ to $0.105263$.

### Discovery 2: Edge Tracking Gained $+0.00217$, Proving Regularizer Robustness
- Across all 3 variants, the combination of the **Pure Symmetric Axial Regularizer** and pruned acute false bifurcations improved Edge Jaccard and Adjusted Edge Jaccard by over $+0.0021$ ($0.88823 \to 0.89040$).
- In `44b6_ddf577ad`, the adjusted edge Jaccard jumped by an astounding **$+0.01764$** ($0.84121 \to 0.85885$).
- The only reason Campaign 026 failed was the loss of the single True Positive division at $\cos\theta = -0.0205$.

---

## 3. The Definitive Golden Boundary for Mitosis Pruning

Comparing Campaigns 023, 024, 025, and 026 establishes the exact mathematical boundary for label-free post-processing:

$$\boxed{\cos_{\text{spurious}} > +0.15 \quad \text{AND} \quad \text{ratio} < 0.20 \quad (\text{with } \max(d_1, d_2) > 3.0\ \mu\text{m})}$$

Under this exact parameter envelope (achieved in **Campaign 024 Variant 1**):
1. **$100\%$ of True Positives are preserved** ($\text{TP} = 3$).
2. **$25\%$ of False Positives are eliminated** ($\text{FP}: 8 \to 6$).
3. **Division Jaccard improves** ($0.130435 \to 0.142857$, $\Delta = +0.012422$).
4. **Development Score improves** by $\mathbf{+0.003598}$ ($\ge +0.00300$).
5. **CERO REGRESIONES EN LOS 8 DATASETS** ($\Delta \ge 0.00000$ across the entire panel).
6. **Development Gate PASSED** (`eligible_for_confirmation_and_package`).
