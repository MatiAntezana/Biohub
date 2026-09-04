# 📡 LATEST COMMUNITY DISCOVERIES & EXTERNAL INTEL

> Last updated: `2026-09-04 18:55:36 UTC`
> This file provides actionable domain insights and hypotheses for the Autoresearch Agent.

---

## 💡 Top Actionable Competitive Insights

### 🎯 [INSIGHT-001] Division Jaccard: The +0.050 direct score opportunity
- **Category**: Mitosis & Topology
- **Source**: Top Kaggle Solutions & biohub-base-0937
- **Executive Summary**: Standard bipartite Hungarian matching enforces 1:1 edges, completely failing on mitotic divisions (1 cell -> 2 daughters). Division Jaccard is scaled by 0.1 in the competition metric. Implementing a safe mitotic branching stage ('add_safe_divisions_postlink') that links high-confidence dividing mothers to two sister candidates with geometric gating (daughter dist <= 8um, sister dist <= 12um) boosts Division Jaccard from ~0.25 to >0.70, adding +0.045 to +0.050 directly to the leaderboard.
- **Actionable Levers**:
  - `MITOSIS_CONF_THRESHOLD: tune in [0.45, 0.65]`
  - `SAFE_DIV_MAX_UM: tune in [7.0, 9.5]`
  - `SAFE_DIV_SISTER_MAX_UM: tune in [10.0, 14.0]`
  - `SAFE_DIV_FRAME_FRAC_CAP: keep <= 0.02 to avoid false positive forks`

---

### 🎯 [INSIGHT-002] Eliminating the Node Overprediction Penalty Multiplier
- **Category**: Metric Optimization
- **Source**: RoyerLab Official Metric Specification (metrics.md)
- **Executive Summary**: Adjusted Edge Jaccard scales by max(0, 1 - 0.1 * (T_pred - T_true) / T_true). Neural networks often output extra low-confidence background centroids. If T_pred is 20% higher than T_true, you immediately lose 2% of your entire edge score! By taking the coarse 'estimated_number_of_nodes' metadata from the Zarr volume and sorting peak candidates by confidence, keeping only top K = int(estimated_nodes * 0.98), the penalty multiplier stays strictly at 1.000.
- **Actionable Levers**:
  - `NODE_BUDGET_FACTOR: set to [0.95 - 1.00]`
  - `CONF_THRESHOLD: prune low-confidence peaks (< 0.40)`

---

### 🎯 [INSIGHT-003] Optical Anisotropy Weighting in Z (3.25um vs 0.4um)
- **Category**: Physics & Geometry
- **Source**: Zarr v3 Dataset Metadata & starter kit
- **Executive Summary**: Light-sheet microscopy has lower resolution along the optical axis (Z). In physical coordinates, 1 voxel in Z is 1.625 um (or 3.25 um depending on downsampling), while 1 voxel in XY is 0.40625 um (a 4:1 to 8:1 ratio). Euclidean distance in voxel space severely underestimates Z displacement. Scaling coordinates by [1.625, 0.40625, 0.40625] before computing assignment costs prevents erratic Z-plane jumps.
- **Actionable Levers**:
  - `SCALE_ZYX: [1.625, 0.40625, 0.40625]`
  - `MAX_LINK_DISTANCE_UM: calibrate around 10.0 - 14.0 um`

---

### 🎯 [INSIGHT-004] Gap Closing (t -> t+2) & Adaptive Short Track Rescue
- **Category**: Temporal Continuity
- **Source**: Trackastra & Ultrack Papers
- **Executive Summary**: Cells occasionally dim or undergo temporary focal drift for 1 frame. A second pass linking dead-ends at t to new-starts at t+2 recovers broken trajectories. Furthermore, orphan trajectories with length <= 2 frames are predominantly false positives and should be pruned, UNLESS they participate in a mitotic division.
- **Actionable Levers**:
  - `MAX_GAP2_DISTANCE_UM: set to [14.0 - 18.0 um]`
  - `OUTPUT_MIN_TRACK_LEN: set to 3 or 4`
  - `KEEP_DIVISION_COMPONENTS: set to True`

---
