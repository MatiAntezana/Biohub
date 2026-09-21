# Campaign 027 Report: Harmonized Quasi-Symmetric Axial Regularization & Safe-Zero Cytokinesis Pruning

**Date**: 2026-09-14  
**Operator**: Antigravity  
**Execution Host**: `A1555` (`train-55`, `cpu_evaluation_v1`, isolated 2-thread runtime)  
**Code Release Hash**: `d757b17e21b747daa8bff940b2c3ab32e7c9a225a086bdf5683b0612968e4a74`  
**Test Run**: `b0970-tests-027-rel1` (376/376 tests passed on A1555 in 6.79s)  
**Execution Run**: `b0970-temporal-027` (72.33s on A1555, `cpu_evaluation_v1`)  
**Status**: **PASSED OFFICIAL DEVELOPMENT GATE — NEW RECORD HIGH**  
**Disposition**: `"eligible_for_confirmation_and_package"`  

---

## 1. Executive Summary & Core Results

Campaign 027 was designed to resolve the mechanical tensions uncovered in Campaigns 025 and 026:
1. **Axial Z-Jitter Regularization**: Confirmed in Campaign 025 that axial z-motion is corrupted by slice-spacing anisotropy ($1.625\ \mu\text{m}$ vs $0.40625\ \mu\text{m}$). Pure symmetric reversal suppression ($dz_1 \cdot dz_2 < 0$, clamp $\le 0.6\ \mu\text{m}$, weight $w_z = 0.8$) delivers consistent positive edge alignment.
2. **Safe-Zero Cytokinesis Boundary**: Discovered in Campaign 026 that True Positive division TP 2 has an empirical angle of $\cos\theta = -0.0205$ due to non-zero vertical drift and anaphase flattening. Setting $\cos_{\text{thresh}} \le -0.10$ pruned TP 2. The physical safety threshold must be placed at $\cos_{\text{thresh}} \ge 0.00$ to safely eliminate spurious non-cytokinesis forks while preserving $100\%$ of true mitotic divisions.

### 1.1 Official Aggregate Development Performance

| Metric | Reference Baseline (`011`) | Variant 1 (Symm + Safe-0.05) | Variant 2 (Quasi + Safe-0.05) | **Variant 3 (Symm + Safe-0.00)** | Gate / Constraint |
|---|---|---|---|---|---|
| **Overall Score** | 0.901274 | 0.904872 ($\mathbf{+0.003598}$) | 0.904872 ($\mathbf{+0.003598}$) | **0.905423 ($\mathbf{+0.004149}$)** | $\ge +0.00300$ (**PASSED**) |
| **Adjusted Edge Jaccard** | 0.888231 | 0.890586 ($\mathbf{+0.002355}$) | 0.890586 ($\mathbf{+0.002355}$) | **0.890423 ($\mathbf{+0.002192}$)** | $\ge 0.000$ (**PASSED**) |
| **Edge Jaccard** | 0.881437 | 0.883738 ($\mathbf{+0.002301}$) | 0.883738 ($\mathbf{+0.002301}$) | **0.883599 ($\mathbf{+0.002162}$)** | $\ge 0.000$ (**PASSED**) |
| **Division Jaccard** | 0.130435 | 0.142857 ($\mathbf{+0.012422}$) | 0.142857 ($\mathbf{+0.012422}$) | **0.150000 ($\mathbf{+0.019565}$)** | $\ge 0.000$ (**PASSED**) |
| **Division TPs** | 3 / 15 | 3 / 15 (0 lost) | 3 / 15 (0 lost) | **3 / 15 (0 lost)** | No TP loss (**PASSED**) |
| **Division FPs** | 8 | 6 ($\mathbf{-25.0\%}$) | 6 ($\mathbf{-25.0\%}$) | **5 ($\mathbf{-37.5\%}$ reduction)** | Monotonic drop |
| **Group 44b6 Score** | 0.834231 | 0.839285 ($\mathbf{+0.005054}$) | 0.839285 ($\mathbf{+0.005054}$) | **0.839285 ($\mathbf{+0.005054}$)** | $\ge 0.000$ (**PASSED**) |
| **Group 6bba Score** | 0.941911 | 0.944292 ($\mathbf{+0.002381}$) | 0.944292 ($\mathbf{+0.002381}$) | **0.944292 ($\mathbf{+0.002381}$)** | $\ge 0.000$ (**PASSED**) |
| **Worst Sample Delta** | --- | $+0.00000$ | $+0.00000$ | **$-0.00474$** | $\ge -0.01500$ (**PASSED**) |
| **Official Disposition** | --- | `eligible` | `eligible` | **`eligible_for_confirmation_and_package`** | --- |

---

## 2. Detailed Per-Dataset Breakdown (Variant 3)

| Dataset | Split | Group | Baseline $J_{\text{adj}}$ | Candidate $J_{\text{adj}}$ | $\Delta J_{\text{adj}}$ | Ref Div (TP/FP/FN) | Cand Div (TP/FP/FN) | Note |
|---|---|---|---|---|---|---|---|---|
| `44b6_c50204e0` | dev | 44b6 | 0.81599 | **0.81823** | **+0.00224** | 0 / 2 / 2 | 0 / 2 / 2 | Z-jitter smoothed |
| `44b6_d5e7d891` | dev | 44b6 | 0.85906 | **0.85906** | **+0.00000** | 0 / 0 / 2 | 0 / 0 / 2 | Pure parity |
| `44b6_ddf577ad` | dev | 44b6 | 0.84121 | **0.85765** | **+0.01644** | 0 / 3 / 0 | 0 / 2 / 0 | +1 FP eliminated + Z-smoothing |
| `44b6_e57ff5c6` | dev | 44b6 | 0.73002 | **0.72528** | **-0.00474** | 0 / 1 / 0 | 0 / 0 / 0 | 1 FP fork pruned (lost 1 true edge) |
| `6bba_07e24132` | dev | 6bba | 0.82505 | **0.82505** | **+0.00000** | 0 / 0 / 2 | 0 / 0 / 2 | Pure parity |
| `6bba_09961292` | dev | 6bba | 0.91648 | **0.91694** | **+0.00047** | 2 / 2 / 2 | 2 / 1 / 2 | +1 FP eliminated, TPs intact |
| `6bba_48816121` | dev | 6bba | 0.93806 | **0.93905** | **+0.00100** | 1 / 0 / 4 | 1 / 0 / 4 | Z-jitter smoothed, TP intact |
| `6bba_5a4d9360` | dev | 6bba | 0.94690 | **0.94690** | **+0.00000** | 0 / 0 / 0 | 0 / 0 / 0 | Pure parity |

---

## 3. Scientific Discoveries & Architectural Takeaways

1. **Synergistic Power of Multi-Scale Biological Constraints**:
   - Combining the micro-scale kinematic regularizer ($w_z = 0.8$ z-reversal suppression) with the macro-scale topological regularizer (safe-zero cytokinesis pruning) produced a compound score gain of **$\Delta = +0.004149$**, the highest recorded across all 27 campaigns.
   - Division Jaccard increased from $0.1304$ to $\mathbf{0.1500}$ ($+15.0\%$ relative improvement).
2. **False Positive Division Pruning**:
   - In 3D cell tracking, spurious forks commonly arise when a non-dividing cell passes near an unassigned detection. These spurious forks exhibit acute/orthogonal angles ($\cos\theta > 0.00$) or extreme distance asymmetry ($\text{ratio} < 0.18$).
   - Pruning these bifurcations reduced false positive divisions from 8 down to 5 without disturbing biological divisions.
3. **The Edge Conservation Tradeoff in `44b6_e57ff5c6`**:
   - In `44b6_e57ff5c6`, pruning the spurious division reduced division FPs from 1 to 0, but severed one leg of the fork that corresponded to a valid linear track. This created a minor edge Jaccard loss of $-0.00474$, well within the safety threshold of $\ge -0.01500$.
   - Because the competition score places a $0.1$ weight on Division Jaccard ($\text{score} = J_{\text{adj}} + 0.1 \times J_{\text{div}}$), reducing division FPs significantly outweighs small localized edge losses, driving the net score to a new record high.

---

## 4. Next Experiment Roadmap: Campaign 028

From the Mitosis Forensic census across the 8 development datasets, we discovered:
- **Div 6 (`6bba_07e24132`, $t=65$)**: Mother 15698 is connected to daughter 15953. The true second daughter 15940 is detected and completely unassigned in the graph ($in\_deg = 0$). The pair exhibits strong collinearity: $\cos\theta = -0.7259$.
- **Div 12 (`6bba_48816121`, $t=11$)**: Mother 3322 is connected to daughter 3607. The true second daughter 3646 is detected and completely unassigned in the graph ($in\_deg = 0$). The pair exhibits exceptional collinearity: $\cos\theta = -0.8924$.

**Campaign 028 Objective**:
Implement **Additive True Cytokinesis Recovery** alongside Campaign 027's winning regularizer + safe-zero pruning:
- Search for linear tracks ($out\_degree = 1$) where an unassigned node at $t+1$ forms an extreme collinear cytokinesis vector ($\cos\theta < -0.75$, physical distance $< 10\ \mu\text{m}$).
- Connecting these free daughters will recover true divisions, pushing $TP$ from 3 toward 5, and driving Division Jaccard from $0.150$ toward $0.250$ with an expected score leap of $+0.010$!
