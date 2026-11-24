# Biohub — 3D Cell Tracking During Development

End-to-end 3D spatiotemporal cell detection and lineage reconstruction pipeline for the [Chan Zuckerberg Biohub Kaggle Competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

**Competition Result:** **Top 100 out of 3,502 teams (Top 3% overall, 0.932 public score)**.

---

## The Problem

Tracking cells across 3D time-lapse microscopy of developing zebrafish embryos is harder than standard multi-object tracking for three reasons:

1. **Anisotropic resolution:** Voxel spacing is $1.625\,\mu\text{m}$ along $Z$ versus $0.40625\,\mu\text{m}$ in $X$ and $Y$ (a 4:1 ratio). Nearest-neighbor heuristics in voxel space break down immediately; all candidate radii, velocities, and divergence angles have to be computed in physical microns.
2. **Dense, drifting tissue:** Thousands of nuclei move together under convective tissue drift. During mitosis, two daughter cells can appear to angle slightly forward along the drift vector rather than splitting cleanly at $180^\circ$.
3. **Strict lineage constraints:** Every valid tracking graph must respect biological invariants (`in_degree <= 1`, `out_degree <= 2`, no self-loops, consecutive frame transitions) while recovering true mitotic forks without exploding false-positive divisions.

---

## Pipeline Overview

```text
3D+T Microscopy Volume (Zarr)
  │
  ├──► Dual-Seed TemporalUNet3D + DeepCenter Prior (8-view TTA)
  │      └─► Cell-center detection & dense 32-dim spatiotemporal features
  │
  ├──► SimpleNodeTransformer (4-layer Cross-Attention)
  │      └─► Pairwise edge logits between consecutive frames (t -> t+1)
  │
  └──► Lineage Graph Solver & Post-Processing
         ├─► SCIP ILP + Bidirectional Hungarian Matching
         ├─► Harmonized Axial Regularizer (suppresses Z-reversal jitter)
         └─► Bounded Divergence Persistence Gating (BDPG + DDKS-SPI)
               └─► Final Submission Graph (submission.csv)
```

### 1. Detection & Spatiotemporal Features (`src/models/temporal_unet.py`)
* **Backbone:** `TemporalUNet3D` operating on consecutive frame windows (`(B, T, C, Z, Y, X)`) with encoder widths `(32, 64, 128)`, trilinear upsampling, and gradient checkpointing.
* **Temporal Attention:** Per-voxel multi-head self-attention across time at deeper encoder stages (skipping full-resolution stage 0 for ~3x faster runtime and ~30% lower VRAM).
* **Ensemble & TTA:** Combines two independent seeds (`primary` and `secondary`) with a `deepcenter` 3D U-Net center prior (`assets/champion/`) and 8-view Test-Time Augmentation.

### 2. Edge Scoring (`src/models/simple_node_transformer.py`)
* **Architecture:** `SimpleNodeTransformer` projects per-node U-Net features + sinusoidal positional encodings into 4 cross-attention blocks (`hidden_dim=128`, `n_heads=4`), followed by a pairwise MLP scoring `(feat_t, feat_t1, delta_zyx)`.
* **Physical Candidate Filtering:** Candidate edges are restricted in physical micron space (`um`) rather than raw voxel coordinates to account for the 4:1 axial anisotropy.

### 3. Lineage Optimization & Cytokinesis Gating (`src/tracking/solver.py`, `src/tracking/bdpg.py`)
* **ILP Formulation (SCIP):** Solves global track linking with binary edge and division-triplet variables, penalizing unconfirmed splits and enforcing `in_degree <= 1` and `out_degree <= 2`.
* **Axial Regularization:** Suppresses symmetric single-frame Z-reversal spikes along linear tracks.
* **Bounded Divergence Persistence Gating (BDPG):** Stratifies candidate mitotic forks into three physical regimes:
  * *Obtuse divergent forks* ($\cos_{3\text{D}} \le 0$): Retained when both daughters persist ($\ge 3$ frames) or show consistent forward heading.
  * *Drift-tilted cytokinesis* ($0 < \cos_{3\text{D}} \le 0.20$): Retained only when both daughters persist ($\ge 5$ frames) within a bounded birth distance ($\le 6.0\,\mu\text{m}$).
  * *Acute crossovers* ($\cos_{3\text{D}} > 0.20$): Pruned via Distance-Dominant Keeper Selection & Spatial Proximity Invariant (DDKS-SPI).

---

## Repository Structure

```text
Biohub/
├── assets/
│   ├── champion/          # Canonical ensemble checkpoints (primary, secondary, deepcenter)
│   └── nextgen.json       # Candidate generation & solver configuration
├── notebooks/
│   └── submission.ipynb   # Kaggle offline inference & submission notebook
├── src/
│   ├── models/            # TemporalUNet3D and SimpleNodeTransformer
│   ├── tracking/          # Lineage solver, BDPG fork pruning, Zarr I/O, official metrics
│   ├── train.py           # End-to-end U-Net + Transformer training script
│   ├── predict.py         # Inference & GEFF graph export
│   └── evaluate.py        # Edge and division Jaccard evaluation
├── tests/                 # Unit tests for models, tracking contracts, and BDPG
├── requirements.txt
└── LICENSE
```

---

## Quickstart

### Installation

```bash
git clone https://github.com/MatiAntezana/Biohub.git
cd Biohub
pip install -r requirements.txt
```

### Run Unit Tests

```bash
PYTHONPATH=. pytest tests -v
```

### Training & Inference

Training and evaluation expect the competition Zarr/GEFF dataset layout configured in `src/dataspec.py`:

```bash
# Train TemporalUNet3D + SimpleNodeTransformer on split 0
python -m src.train --split 0 --epochs 50

# Run inference and export .geff tracking graphs
python -m src.predict --split 0

# Evaluate edge and division Jaccard against ground truth
python -m src.evaluate --split 0
```

For Kaggle submission, `notebooks/submission.ipynb` loads the verified checkpoints from `assets/champion/` and writes `/kaggle/working/submission.csv`.
