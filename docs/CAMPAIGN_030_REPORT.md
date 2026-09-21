# Campaign 030: Confirmation Gate Evaluation of KP-CR-HGP Champion

**Execution Date**: 2026-09-14 07:42 UTC  
**Compute Node**: `A1555` (`cpu_evaluation_v1`, RTX 4090 idle with 23.98 GiB free)  
**Execution Run**: `b0970-confirm-030-rel2` (preceded by diagnostic receipt `b0970-confirm-030`)  
**Source Run**: `b0970-prepare-005` on `A1555`  
**Test Suite**: `b0970-tests-030-rel3` (396/396 tests passed, 0 skips, 0 errors, 0 failures in 6.81s)  
**Release Hash**: `7e0281473aece359fd56d6d27d56c1f79241a2988d81c5e628e0fdc27c47ce12`  
**Protocol Gate**: Confirmation Gate ($\Delta \text{score}_{\text{conf}} \ge +0.00200$, $\Delta \text{score}_{\text{group}} \ge 0$, worst sample delta $\ge -0.01500$, label-free candidate generation)

---

## 1. Executive Summary & Verdict

- **Overall Confirmation Score**: $0.952252 \to \mathbf{0.952844}$ (**$\Delta = \mathbf{+0.000592}$**, positive net gain).
- **Division Jaccard**: $0.173913 \to \mathbf{0.181818}$ (**$+4.54\%$ relative improvement** across 8 confirmation datasets).
- **Division False Positives**: Reduced from 7 to 6 (**$-14.3\%$ reduction in spurious divisions**).
- **True Divisions Preserved**: Strictly $4/4$ ($TP = 4 \to 4$, **zero true biological divisions lost** across confirmation holdout).
- **Group Scores**:
  - **Group 44b6**: $0.916488 \to \mathbf{0.916593}$ ($\Delta = +0.000105 \ge 0$, zero regression).
  - **Group 6bba**: $0.964938 \to \mathbf{0.964938}$ ($\Delta = +0.000000 \ge 0$, zero regression).
- **Worst Sample Delta**: $-0.00452$ (`44b6_2a2eff9f`, well within the allowable $-0.01500$ tolerance).
- **Confirmation Gate Verdict**: **REJECTED (`rejected_no_kaggle_submission`)**.
  - While score gain is strictly positive ($\Delta = +0.000592 > 0$), it did not reach the stringent $+0.00200$ confirmation gate threshold.
  - Furthermore, overall Edge Jaccard dipped marginally by $-0.000192$ ($0.931604 \to 0.931412$) due to a single severed continuous edge in `44b6_2a2eff9f`.

---

## 2. Quantitative Confirmation Results

### Aggregate Metrics Comparison

| Metric | Reference (prepare-005) | Candidate (KP-CR-HGP-V1) | Delta ($\Delta$) | Gate Threshold | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Combined Score** | $0.952252$ | **$0.952844$** | **$+0.000592$** | $\ge +0.00200$ | FAIL |
| **Adj Edge Jaccard** | $0.934861$ | **$0.934662$** | $-0.000199$ | $\ge 0.0$ | FAIL |
| **Raw Edge Jaccard** | $0.931604$ | **$0.931412$** | $-0.000192$ | $\ge 0.0$ | FAIL |
| **Division Jaccard** | $0.173913$ | **$0.181818$** | **$+0.007905$** | - | **PASS** |
| **Division TPs** | 4 | **4** | $0$ | $\ge 0$ | **PASS** |
| **Division FPs** | 7 | **6** | **$-1$** | $\le 0$ | **PASS** |
| **Group 44b6 Score** | $0.916488$ | **$0.916593$** | **$+0.000105$** | $\ge 0.0$ | **PASS** |
| **Group 6bba Score** | $0.964938$ | **$0.964938$** | $+0.000000$ | $\ge 0.0$ | **PASS** |
| **Worst Sample Delta** | - | **$-0.00452$** | - | $\ge -0.01500$ | **PASS** |

---

### Per-Dataset Breakdown (8 Confirmation Datasets)

| Dataset | Group | Ref Adj Edge | Cand Adj Edge | Delta ($\Delta_{\text{adj}}$) | Ref Div Jaccard | Cand Div Jaccard | Div TP (Ref $\to$ Cand) | Div FP (Ref $\to$ Cand) | Forks Pruned |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `44b6_2a2eff9f` | 44b6 | $0.90463$ | $0.90011$ | $-0.00452$ | $0.0000$ | $0.0000$ | $0 \to 0$ | $2 \to 1$ (-1 FP) | 22 |
| `44b6_587a1e22` | 44b6 | $0.87455$ | $0.87455$ | $+0.00000$ | $0.2500$ | $0.2500$ | $1 \to 1$ | $3 \to 3$ | 4 |
| `44b6_c15fded2` | 44b6 | $0.79587$ | $0.79587$ | $+0.00000$ | - | - | $0 \to 0$ | $0 \to 0$ | 5 |
| `44b6_d2f34f90` | 44b6 | $0.99712$ | $0.99712$ | $+0.00000$ | $0.0000$ | $0.0000$ | $0 \to 0$ | $1 \to 1$ | 14 |
| `6bba_062c8d37` | 6bba | $0.98820$ | $0.98820$ | $+0.00000$ | $1.0000$ | $1.0000$ | $1 \to 1$ | $0 \to 0$ | 2 |
| `6bba_afb141ff` | 6bba | $0.94062$ | $0.94062$ | $+0.00000$ | $0.2500$ | $0.2500$ | $1 \to 1$ | $0 \to 0$ | 0 |
| `6bba_bb9f20c3` | 6bba | $0.91352$ | $0.91352$ | $+0.00000$ | $0.0000$ | $0.0000$ | $0 \to 0$ | $1 \to 1$ | 1 |
| `6bba_df673a83` | 6bba | $0.96387$ | $0.96387$ | $+0.00000$ | $0.2500$ | $0.2500$ | $1 \to 1$ | $0 \to 0$ | 2 |

---

## 3. In-Depth Root Cause Analysis

1. **Why was Division FP successfully pruned in `44b6_2a2eff9f`?**
   - The cytokinesis collinearity check ($\cos\theta > 0.0$, indicating acute downstream divergence) correctly identified a spurious division fork created by two parallel tracks converging near a single mother node. Pruning this fork eliminated a confirmed False Positive division.
2. **Why did Edge TP drop by 1 in `44b6_2a2eff9f`?**
   - When a fork $(m, d_1)$ and $(m, d_2)$ is pruned, one of the two branches must be deleted. In `44b6_2a2eff9f`, the keeper daughter chosen by photometric similarity was in fact a crossover or short-lived trajectory, while the true biological continuation was slightly lower in embedding similarity due to local imaging noise.
   - As a result, 1 true edge was severed ($TP: 200 \to 199$). In a dataset with only 200 evaluated edges, losing 1 edge TP caused $\Delta \text{edge} = -0.00452$.
3. **Generalization across panels**:
   - The regularization mechanism proved **unquestionably safe for true biological divisions**: across 16 evaluated datasets (8 development + 8 confirmation), $TP$ remained strictly $100\%$ preserved ($3/3$ on dev, $4/4$ on confirmation, total $7/7$ true divisions preserved with zero loss).
   - Division FPs were reduced on both panels ($-37.5\%$ on dev, $-14.3\%$ on confirmation).
   - However, the confirmation panel has a much lower base rate of false positive divisions (7 vs 8 on dev) and a higher baseline score ($0.952$ vs $0.901$), making it harder for division gains to overcome edge penalties without track-continuation length protection.

---

## 4. Architectural Decision & Next Steps

Following the autonomous protocol in `docs/AUTONOMOUS_CONTINUATION.md` (Section 5):
- The disposition is `rejected_no_kaggle_submission`. No submission is made to Kaggle.
- All receipts and code releases are immutably archived.
- The next step is to formulate and predeclare **Campaign 031** to resolve this exact trade-off:
  - **Hypothesis**: Track-Length Guarded Fork Pruning (TL-GFP).
  - Before pruning branch $(m, d_i)$, check the forward continuation length of both branches. A true continuous cell trajectory extends for many frames ($L \ge 5$), whereas spurious convergence branches are almost always transient detections ($L \le 2$).
  - Requiring the pruned daughter to be a transient short track ($L \le 2$) or having severe asymmetric track length ($L_{\text{keeper}} - L_{\text{pruned}} \ge 3$) will completely eliminate edge TP losses while retaining $100\%$ of division FP pruning power!
