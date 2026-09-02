# Cell Tracking: Getting Started w/ Nearest Neighbor
**Author:** inversion (@inversion)
**URL:** https://www.kaggle.com/code/inversion/cell-tracking-getting-started-w-nearest-neighbor
**Kernel ID:** 124558834
**Best Public Score:** 0.143

---

## Overview
This notebook provides a minimal working end-to-end baseline for the Biohub Cell Tracking competition.
It demonstrates how to:
1. Read and decompress timepoint chunks from Zarr v3 volume arrays using `blosc2`
2. Downsample and smooth 3D volumes with uniform filtering (`scipy.ndimage.uniform_filter`)
3. Segment cell candidates via intensity percentiles and connected components labeling (`scipy.ndimage.label`)
4. Compute 3D centroid coordinates and scale them to physical dimensions ($Z=1.625, Y=0.40625, X=0.40625\,\mu\text{m}$)
5. Link centroids across adjacent timepoints using Hungarian matching (`scipy.optimize.linear_sum_assignment`) bounded by maximum Euclidean distance
6. Format and export the final `submission.csv` containing both `node` and `edge` records

---

## Notebook Code Cells

### Cell 1 [Code]
```python
import json
import os

import blosc2
import numpy as np
import pandas as pd
from scipy.ndimage import label, uniform_filter
from scipy.optimize import linear_sum_assignment
```

### Cell 2 [Code]
```python
TEST_DIR = '/kaggle/input/competitions/biohub-cell-tracking-during-development/test'
SCALE = np.array([1.625, 0.40625, 0.40625])  # Z, Y, X µm/voxel
DOWNSAMPLE = 4
PERCENTILE = 90
MAX_LINK_DISTANCE = 15.0  # µm
```

### Cell 3 [Code]
```python
test_folder_names = sorted(
    d.replace('.zarr', '') for d in os.listdir(TEST_DIR) if d.endswith('.zarr')
)

all_rows = []
for folder_name in test_folder_names:
    zarr_path = os.path.join(TEST_DIR, folder_name + '.zarr')

    with open(os.path.join(zarr_path, '0', 'zarr.json')) as f:
        arr_meta = json.load(f)
    shape = tuple(arr_meta['shape'])  # (T, Z, Y, X)
    dtype = np.dtype(arr_meta['data_type'])
    n_t = shape[0]

    prev_centroids = {}
    node_id_counter = 1

    for t in range(n_t):
        chunk_path = os.path.join(zarr_path, '0', 'c', str(t), '0', '0', '0')
        with open(chunk_path, 'rb') as f:
            compressed = f.read()
        decompressed = blosc2.decompress(compressed)
        vol = np.frombuffer(decompressed, dtype=dtype).reshape(shape[1:])  # (Z, Y, X)

        ds = vol[::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE]

        smoothed = uniform_filter(ds.astype(np.float32), size=3)
        threshold = np.percentile(smoothed, PERCENTILE)
        binary = smoothed > threshold

        labeled, n_features = label(binary)

        centroids = {}
        for comp_id in range(1, n_features + 1):
            coords = np.argwhere(labeled == comp_id)
            centroid = coords.mean(axis=0) * DOWNSAMPLE
            nid = node_id_counter
            node_id_counter += 1
            centroids[nid] = (int(round(centroid[0])), int(round(centroid[1])), int(round(centroid[2])))
            all_rows.append({
                'dataset': folder_name,
                'row_type': 'node',
                'node_id': nid,
                't': t,
                'z': centroids[nid][0],
                'y': centroids[nid][1],
                'x': centroids[nid][2],
                'source_id': -1,
                'target_id': -1,
            })

        if prev_centroids:
            prev_ids = list(prev_centroids.keys())
            curr_ids = list(centroids.keys())
            if prev_ids and curr_ids:
                prev_coords = np.array([prev_centroids[pid] for pid in prev_ids], dtype=np.float64)
                curr_coords = np.array([centroids[cid] for cid in curr_ids], dtype=np.float64)
                prev_phys = prev_coords * SCALE
                curr_phys = curr_coords * SCALE
                dist = np.sqrt(((prev_phys[:, None] - curr_phys[None, :]) ** 2).sum(axis=2))
                row_ind, col_ind = linear_sum_assignment(dist)
                for ri, ci in zip(row_ind, col_ind):
                    if dist[ri, ci] <= MAX_LINK_DISTANCE:
                        all_rows.append({
                            'dataset': folder_name,
                            'row_type': 'edge',
                            'node_id': -1,
                            't': -1,
                            'z': -1,
                            'y': -1,
                            'x': -1,
                            'source_id': prev_ids[ri],
                            'target_id': curr_ids[ci],
                        })

        prev_centroids = centroids

    print(f'{folder_name}: {node_id_counter - 1} nodes')
```

### Cell 4 [Code]
```python
submission = pd.DataFrame(all_rows)
submission.index.name = 'id'
submission.to_csv('submission.csv')
print(f'Done. {len(submission)} rows written to submission.csv')
```
