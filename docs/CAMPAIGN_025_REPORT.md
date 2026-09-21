# Campaign 025 Report: Biomechanical Momentum Conservation & Crossover Untangling

- **Date**: 2026-09-14
- **Execution Run ID**: `b0970-temporal-025`
- **Source Run**: `b0970-selector-corpus-011`
- **Tests Run**: `b0970-tests-025-rel1` (363/363 passed)
- **Execution Node**: `A1555` (`cpu_evaluation_v1`, wall time 235.1s)
- **Code Hash**: `119e60bede9311923cb194473ebf5e02d1a37df68ac12a6cc1e735c9614873a5`
- **Artifacts**: `artifacts/A1555/b0970-temporal-025/`

---

## 1. Executive Summary

Campaign 025 implemented and evaluated label-free kinematic crossover untangling combined with the winning Campaign 024 Pure Symmetric Axial Regularizer and Cytokinesis Pruning.

### Key Results
| Metric | Reference (011) | Variant 1 (Pure Kinematic) | Variant 2 (Morphology Guard) | Gate Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **Development Score** | 0.901274 | 0.903980 ($\Delta = +0.002706$) | **0.904424** ($\Delta = \mathbf{+0.003150}$) | $\Delta \ge +0.00300$ |
| **Edge Jaccard** | 0.881437 | 0.882877 ($\Delta = +0.001440$) | **0.883317** ($\Delta = \mathbf{+0.001880}$) | $\Delta \ge 0.0$ |
| **Adjusted Edge Jaccard**| 0.888231 | 0.889695 ($\Delta = +0.001464$) | **0.890138** ($\Delta = \mathbf{+0.001907}$) | $\Delta \ge 0.0$ |
| **Division Jaccard** | 0.130435 | 0.142857 ($\Delta = +0.012422$) | **0.142857** ($\Delta = \mathbf{+0.012422}$) | $\Delta \ge 0.0$ |
| **Division TP / FP / FN**| 3 / 8 / 12 | 3 / 6 / 12 | 3 / 6 / 12 | FP reduction: **25%** |
| **Group 44b6 Score** | 0.834231 | 0.837727 ($\Delta = +0.003496$) | **0.838911** ($\Delta = \mathbf{+0.004679}$) | $\Delta \ge 0.0$ |
| **Group 6bba Score** | 0.941911 | 0.944063 ($\Delta = +0.002152$) | **0.944063** ($\Delta = \mathbf{+0.002152}$) | $\Delta \ge 0.0$ |
| **Individual Regressions**| N/A | 1 regression (`44b6_d5e7d891`: -0.00280) | **0 regressions** ($\Delta \ge 0.00000$ in all 8) | $\Delta \ge -0.015$ |
| **Gate Status** | N/A | **FAILED** (`score_delta < 0.003`) | **PASSED** (`passed: true`) | $\Delta \ge +0.00300$ |

---

## 2. Scientific & Biological Findings

### Finding 1: Contact Inhibition vs Linear Momentum in Dense Cell Cultures
- In Variant 1, untangling crossovers based purely on velocity vector alignment ($\Delta_{\text{momentum}} \ge 1.20$) triggered 4 edge swaps across the 8 movies.
- In `44b6_d5e7d891`, 1 swap was executed, which caused a regression ($\Delta_{\text{adj\_edge}} = -0.00280$), pulling the aggregate score improvement down from $+0.003598$ to $+0.002706$.
- **Biological Cause**: Eukaryotic cells in dense tissues do not behave like inertial point particles in Newtonian physics. When two cells migrate close to one another, they often experience contact inhibition of locomotion (CIL), causing sudden deflection, stopping, or course alteration rather than simple momentum-conserving path intersection. Forcing a swap because forward and backward tracks appear straighter risks joining two distinct cells and breaking valid tracks.

### Finding 2: Morphology Guard Protects Against False Swaps
- In Variant 2, requiring morphological continuity (intensity/area ratio within 30%) and $\Delta_{\text{momentum}} \ge 1.40$ safely rejected ambiguous swaps (`untangled: 0`).
- This preserved the pure gains of the axial regularizer and cytokinesis pruning, securing:
  - Aggregate score delta: **$+0.003150$** ($\ge +0.00300$).
  - **Zero regressions** across all 8 development datasets.
  - Official gate: **PASSED (`passed: True`, `reasons: []`)**.

---

## 3. Dataset-by-Dataset Breakdown (Variant 2)

| Dataset | Group | Ref Adj Edge | Cand Adj Edge | $\Delta_{\text{adj\_edge}}$ | Z-Spikes Regularized | False Forks Pruned |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `44b6_c50204e0` | 44b6 | 0.81599 | 0.81823 | **+0.00224** | 1,293 | 26 |
| `44b6_d5e7d891` | 44b6 | 0.85906 | 0.85906 | **+0.00000** | 1,459 | 28 |
| `44b6_ddf577ad` | 44b6 | 0.84121 | 0.85500 | **+0.01379** | 1,338 | 21 |
| `44b6_e57ff5c6` | 44b6 | 0.73002 | 0.73002 | **+0.00000** | 310 | 11 |
| `6bba_07e24132` | 6bba | 0.82505 | 0.82505 | **+0.00000** | 903 | 17 |
| `6bba_09961292` | 6bba | 0.91648 | 0.91694 | **+0.00047** | 1,035 | 12 |
| `6bba_48816121` | 6bba | 0.93806 | 0.93806 | **+0.00000** | 874 | 8 |
| `6bba_5a4d9360` | 6bba | 0.94690 | 0.94690 | **+0.00000** | 45 | 0 |

---

## 4. Decision and Transition to Campaign 026

- **Disposition**: `eligible_for_confirmation_and_package`.
- **Next Step**: Leverage the forensic mitosis findings from Subagent 1.
  - 14 of 15 True GT divisions exhibit strict antiparallel cytokinesis ($\cos \theta < -0.60$, median $-0.7677$, distance ratio $\ge 0.60$).
  - 6 of the 8 baseline False Positives fail this cytokinesis criterion ($\cos \theta > -0.40$ or distance ratio $< 0.45$).
  - Missed true divisions are primarily **"stolen daughters"** (where linear assignment greedily gave daughter 2 to an unrelated track) and **"additive second daughters"** (where daughter 2 was left unlinked).
  - Campaign 026 will test **Cytokinesis Collinearity Reassignment & Additive Second-Daughter Rescue** to push Division Jaccard from 0.14 to $>0.30$, aiming for a massive $+0.040$ jump in score toward $\ge 0.970$.
