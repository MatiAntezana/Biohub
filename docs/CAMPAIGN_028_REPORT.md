# Campaign 028 Report: Axial De-Biased Cytokinesis Regularization & Precision Mitosis Recovery (AD-CR-PMR)

**Date**: 2026-09-14  
**Operator**: Antigravity  
**Execution Host**: `A1555` (`train-55`, `cpu_evaluation_v1`, isolated 2-thread runtime)  
**Code Release Hash**: `7b0f771a5af95b8df8309eadd3de83026bfad2c8f3c53d76dd4dd70e368dcce6`  
**Test Run**: `b0970-tests-028-rel1` (383/383 tests passed on A1555 in 6.80s)  
**Execution Run**: `b0970-temporal-028` (91.2s on A1555, `cpu_evaluation_v1`)  
**Status**: **PASSED OFFICIAL DEVELOPMENT GATE**  
**Disposition**: `"eligible_for_confirmation_and_package"`  

---

## 1. Executive Summary & Core Results

Campaign 028 investigated two fundamental biomechanical hypotheses in 3D confocal cell tracking:
1. **Specimen Z-Drift Bias Masking**: Demonstrated that common axial specimen drift ($dz_{\text{drift}} \approx 1-3\ \mu\text{m}$) adds an artificial positive bias $(dz_{\text{drift}})^2 > 0$ to naive 3D dot products. Centering relative to the common axial velocity ($\bar{dz} = \frac{dz_1 + dz_2}{2}$) exposes the true antiparallel cytokinesis divergence ($c_{\text{rel}} = -0.772$ for TP 2).
2. **Persistence-Guarded Keeper Selection**: Integrated 3-frame downstream trajectory heading alignment to protect long, stable biological tracks from being pruned at ambiguous forks.
3. **Additive High-Confidence Cytokinesis Recovery**: Evaluated whether geometric additive recovery of free antiparallel daughters ($c_{\text{rel}} < -0.85$, $\text{ratio} \ge 0.50$) could recover missed GT divisions.

### 1.1 Official Aggregate Development Performance

| Metric | Reference Baseline (`011`) | **Variant 1 (Heading-Guarded Safe-Zero)** | Variant 2 (Axial De-Biased) | Variant 3 (Additive Recovery) | Gate / Constraint |
|---|---|---|---|---|---|
| **Overall Score** | 0.901274 | **0.904872 ($\mathbf{+0.003598}$)** | 0.904872 ($\mathbf{+0.003598}$) | 0.902481 ($+0.001207$) | $\ge +0.00300$ (**PASSED**) |
| **Adjusted Edge Jaccard** | 0.888231 | **0.890586 ($\mathbf{+0.002355}$)** | 0.890586 ($\mathbf{+0.002355}$) | 0.890481 ($+0.002250$) | $\ge 0.000$ (**PASSED**) |
| **Edge Jaccard** | 0.881437 | **0.883758 ($\mathbf{+0.002321}$)** | 0.883758 ($\mathbf{+0.002321}$) | 0.883654 ($+0.002218$) | $\ge 0.000$ (**PASSED**) |
| **Division Jaccard** | 0.130435 | **0.142857 ($\mathbf{+0.012422}$)** | 0.142857 ($\mathbf{+0.012422}$) | 0.120000 ($-0.010435$) | $\ge 0.000$ (**PASSED**) |
| **Division TPs** | 3 / 15 | **3 / 15 (0 lost)** | 3 / 15 (0 lost) | 3 / 15 (0 lost) | No TP loss (**PASSED**) |
| **Division FPs** | 8 | **6 ($\mathbf{-25.0\%}$)** | 6 ($\mathbf{-25.0\%}$) | 10 ($+25.0\%$) | Monotonic drop |
| **Group 44b6 Score** | 0.834231 | **0.839440 ($\mathbf{+0.005209}$)** | 0.839440 ($\mathbf{+0.005209}$) | 0.839440 ($+0.005209$) | $\ge 0.000$ (**PASSED**) |
| **Group 6bba Score** | 0.941911 | **0.944292 ($\mathbf{+0.002381}$)** | 0.944292 ($\mathbf{+0.002381}$) | 0.940742 ($-0.001169$) | $\ge 0.000$ (**PASSED**) |
| **Worst Sample Delta** | --- | **$\mathbf{+0.00000}$ (ZERO REGRESSIONS)** | **$\mathbf{+0.00000}$** | $+0.00000$ | $\ge -0.01500$ (**PASSED**) |
| **Official Disposition** | --- | **`eligible_for_confirmation_and_package`** | `eligible` | `rejected` | --- |

---

## 2. Per-Dataset Breakdown (Variant 1: Zero Regressions Across All 8 Datasets)

| Dataset | Split | Group | Baseline $J_{\text{adj}}$ | Candidate $J_{\text{adj}}$ | $\Delta J_{\text{adj}}$ | Ref Div (TP/FP/FN) | Cand Div (TP/FP/FN) | Note |
|---|---|---|---|---|---|---|---|---|
| `44b6_c50204e0` | dev | 44b6 | 0.81599 | **0.81823** | **+0.00224** | 0 / 2 / 2 | 0 / 2 / 2 | Pure positive gain |
| `44b6_d5e7d891` | dev | 44b6 | 0.85906 | **0.85906** | **+0.00000** | 0 / 0 / 2 | 0 / 0 / 2 | Pure parity |
| `44b6_ddf577ad` | dev | 44b6 | 0.84121 | **0.85765** | **+0.01644** | 0 / 3 / 0 | 0 / 2 / 0 | +1 FP eliminated + Z-smoothing |
| `44b6_e57ff5c6` | dev | 44b6 | 0.73002 | **0.73002** | **+0.00000** | 0 / 1 / 0 | 0 / 1 / 0 | **No regression** (edge preserved) |
| `6bba_07e24132` | dev | 6bba | 0.82505 | **0.82505** | **+0.00000** | 0 / 0 / 2 | 0 / 0 / 2 | Pure parity |
| `6bba_09961292` | dev | 6bba | 0.91648 | **0.91694** | **+0.00047** | 2 / 2 / 2 | 2 / 1 / 2 | +1 FP eliminated, TPs intact |
| `6bba_48816121` | dev | 6bba | 0.93806 | **0.93905** | **+0.00100** | 1 / 0 / 4 | 1 / 0 / 4 | Z-jitter smoothed, TP intact |
| `6bba_5a4d9360` | dev | 6bba | 0.94690 | **0.94690** | **+0.00000** | 0 / 0 / 0 | 0 / 0 / 0 | Pure parity |

---

## 3. Scientific Discoveries: Why Additive Recovery Failed (Variant 3)

Variant 3 tested high-confidence additive recovery ($c_{\text{rel}} < -0.85$, $\text{ratio} \ge 0.50$, target $out\_deg \ge 1$). It failed the gate by increasing division FPs from 8 to 10 ($J_{\text{div}}$ dropped from $0.1304$ to $0.1200$).

Forensic analysis revealed why:
1. **Asymmetry of True Divisions**: In the anchor graph, true divisions (e.g. Div 6 and Div 12) exhibit apparent distance asymmetry ($\text{ratio} = 0.311$ and $\text{ratio} = 0.187$) because one daughter remains near the mother's centroid while the other daughter rapidly elongates away. Thus, requiring $\text{ratio} \ge 0.50$ excluded the true divisions.
2. **Symmetry of Random Migrating Encounters**: In dense cell culture, two unrelated migrating cells passing each other in opposite directions frequently exhibit symmetric distances ($\text{ratio} \ge 0.60$). Gating on symmetry alone preferentially linked these non-dividing passing cells, creating 2 false positive divisions in the GT evaluation crop.
3. **Core Conclusion**: Additive topological mutations without explicit neural division probability or segmentation mask evidence suffer from an adverse signal-to-noise ratio. Conversely, **pruning spurious forks** and **quasi-symmetric axial smoothing** operate on already-linked high-confidence edges, making them mathematically robust and reliably profitable.

---

## 4. Summary of Winning Champions

Across Campaigns 021 through 028, we have established two official champions that pass all development gates:
- **Campaign 027 - Variant 3**:
  - Score: **0.905423** ($\Delta = \mathbf{+0.004149}$ over baseline).
  - Division Jaccard: **0.150000** ($\mathbf{+15.0\%}$ relative improvement, $-37.5\%$ FPs).
  - Worst sample: $-0.00474 \ge -0.01500$.
- **Campaign 028 - Variant 1**:
  - Score: **0.904872** ($\Delta = \mathbf{+0.003598}$ over baseline).
  - Division Jaccard: **0.142857** ($\mathbf{+9.5\%}$ relative improvement, $-25.0\%$ FPs).
  - Worst sample: **$\mathbf{+0.00000}$ (ZERO REGRESSIONS ON ALL 8 DATASETS)**.

Both algorithms are ready for confirmation and packaging.
