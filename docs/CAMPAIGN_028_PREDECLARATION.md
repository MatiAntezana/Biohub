# Campaign 028 Predeclaration: Axial De-Biased Cytokinesis Regularization & Precision Mitosis Recovery (AD-CR-PMR)

**Date**: 2026-09-14  
**Author / Operator**: Antigravity  
**Execution Node**: `A1555` (`train-55`, `cpu_evaluation_v1`, isolated 2-thread runtime)  
**Baseline Source Run**: `b0970-selector-corpus-011` on A1555 (Official Development Baseline: Score = 0.901274)  
**Parent Campaign**: Campaign 027 (Score = 0.905423, $\Delta = +0.004149$, Gate Passed)  
**Task Identifier**: `associate-temporal-028`  

---

## 1. Scientific Motivation & Physical Bottlenecks

1. **Specimen Z-Drift Bias Masking in 3D Microscopy**:
   - In 3D confocal/light-sheet microscopy, slice spacing in Z ($1.625\ \mu\text{m}$) is $4\times$ coarser than in XY ($0.40625\ \mu\text{m}$).
   - Imaging volumes experience subtle specimen/stage drift along the optical axis between frames $t$ and $t+1$.
   - For anaphase cell division occurring in the epithelial plane, both daughter cells share this vertical drift component: $dz_1 \approx dz_2 \approx dz_{\text{drift}}$.
   - Calculating the naive 3D dot product $\vec{v}_1 \cdot \vec{v}_2 = (v_{1,xy} \cdot v_{2,xy}) + dz_1 dz_2$ adds an artificial positive term $dz_{\text{drift}}^2 \ge 0$, which can push $\cos\theta$ from a strongly negative separation angle (e.g. $-0.772$ in TP 2) up into the near-zero or slightly positive domain ($-0.021$ in TP 2, $+0.040$ in FP 6).
   - **Solution**: Centering each candidate bifurcation relative to the local vertical centroid motion $\bar{dz} = \frac{dz_1 + dz_2}{2}$ removes common axial drift, yielding the true de-biased biological divergence angle $c_{\text{rel}}$.

2. **Downstream Multi-Frame Trajectory Heading Alignment**:
   - In Campaign 027-V3, pruning spurious forks with naive $\cos > 0.00$ removed 3 false positive divisions, but severed one leg of a valid linear track in `44b6_e57ff5c6` because instantaneous 1-frame kinematic alignment preferred the other daughter.
   - Using a 3-frame downstream heading vector ($\vec{u}_{\text{down}} = \text{coords}(d, t+2) - \text{coords}(d, t+1)$) accurately distinguishes continuing biological tracks from transient detection noise.

3. **Additive High-Confidence Cytokinesis Recovery**:
   - Across the 8 development datasets, 12 out of 15 true ground truth divisions were omitted in the baseline graph.
   - For linear mothers ($out\_degree = 1$), searching for unassigned candidate daughters at $t+1$ with antiparallel relative divergence ($c_{\text{rel}} < -0.85$), balanced distance ratio ($\text{ratio} \ge 0.50$), and proven downstream persistence ($out\_degree \ge 1$) safely recovers true divisions without adding spurious noise.

---

## 2. Predeclared Variants

- **Variant 1: Harmonized Axial Regularizer + Downstream Persistence-Guarded Safe-Zero Pruning**:
  - Symmetric z-reversal suppression: $w_z = 0.8$, clamp $\le 0.6\ \mu\text{m}$, jitter threshold $0.8\ \mu\text{m}$.
  - Spurious fork pruning: $c_{\text{3D}} > 0.05$ or ($\text{ratio} < 0.15$ with $\max(d_1, d_2) > 3.0\ \mu\text{m}$).
  - Keeper edge selection: 3-frame heading alignment and track length weighting.
  - Cytokinesis recovery: disabled.

- **Variant 2: Harmonized Axial Regularizer + Axial De-Biased Cytokinesis Pruning (AD-CP)**:
  - Symmetric z-reversal suppression: $w_z = 0.8$, clamp $\le 0.6\ \mu\text{m}$.
  - Spurious fork pruning: $c_{\text{rel}} > -0.10$ or ($\text{ratio} < 0.18$ with $\max(d_1, d_2) > 3.0\ \mu\text{m}$).
  - Keeper edge selection: 3-frame heading alignment.
  - Cytokinesis recovery: disabled.

- **Variant 3: Compound Model (AD-CR-PMR: Axial Regularizer + AD-CP + Additive Recovery)**:
  - Symmetric z-reversal suppression: $w_z = 0.8$, clamp $\le 0.6\ \mu\text{m}$.
  - Axial de-biased cytokinesis pruning: $c_{\text{rel}} > -0.10$ or ($\text{ratio} < 0.18$).
  - Additive cytokinesis recovery: For linear mothers with $in\_degree \ge 1$, link unassigned target at $t+1$ if $c_{\text{rel}} < -0.85$, $\text{ratio} \ge 0.50$, $1.0 \le \|v_1\|, \|v_2\| \le 7.0\ \mu\text{m}$, and target $out\_degree \ge 1$.

---

## 3. Protocol Gates & Resource Constraints

1. **Development Gate**:
   - $\Delta \text{score}_{\text{dev}} \ge +0.00300$ over baseline ($0.901274 \to \ge 0.904274$).
   - $\Delta \text{score}_{44\text{b}6} \ge 0.00000$ (no group regression).
   - $\Delta \text{score}_{6\text{bba}} \ge 0.00000$ (no group regression).
   - $\Delta J_{\text{adj}} \ge -0.01500$ on all 8 individual datasets.
2. **Row Budget**: Strictly $234{,}000 < \text{rows} < 242{,}000$ (excluding header).
3. **Execution Resource Profile**: `cpu_evaluation_v1` on `A1555` (isolated 2 threads, RLIMIT_AS 8 GiB, CUDA hidden).
4. **Label-Free Inference Invariant**: Candidate graph generation must not access ground truth; GT is opened solely by the official evaluator.
