import os, glob, subprocess, torch, json
import numpy as np, pandas as pd
from skimage.feature import peak_local_max
from src.models.unet3d_anisotropic import AnisotropicUNet3D
from src.dataset import load_zarr_chunk, SCALE
from src.tracking.mitosis_detector import link_and_detect_mitosis

def run_evaluation(peak_thresh=0.15, max_link_dist=14.0, enable_mitosis=True):
    device = torch.device("cuda")
    model = AnisotropicUNet3D().to(device)
    ckpt_path = os.path.expanduser("~/Documents/Biohub/checkpoints/exp001a_unet3d_best.pt")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    train_dir = os.path.expanduser("~/Documents/Biohub/data/train")
    val_samples = sorted([os.path.basename(p).replace(".zarr", "") for p in glob.glob(f"{train_dir}/44b6_*.zarr")])[:8]
    print(f"\n--- Testing peak_thresh={peak_thresh}, max_link_dist={max_link_dist}, mitosis={enable_mitosis} on {len(val_samples)} samples ---")

    val_rows = []
    node_id_counter = 1

    for s in val_samples:
        z_path = os.path.join(train_dir, f"{s}.zarr")
        with open(os.path.join(z_path, '0', 'zarr.json')) as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        dtype = np.dtype(meta['data_type'])
        n_t = shape[0]

        prev_centroids = {}
        for t in range(n_t):
            vol = load_zarr_chunk(z_path, t, shape, dtype).astype(np.float32)
            p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
            vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

            vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad(), torch.amp.autocast('cuda'):
                heat = model(vol_t).squeeze().detach().cpu().numpy().astype(np.float32)

            peaks = peak_local_max(heat, min_distance=2, threshold_abs=peak_thresh, exclude_border=False)
            curr_centroids = {}
            for p in peaks:
                nid = node_id_counter
                node_id_counter += 1
                curr_centroids[nid] = (int(p[0]), int(p[1]), int(p[2]))
                val_rows.append({
                    'dataset': s,
                    'row_type': 'node',
                    'node_id': nid,
                    't': t,
                    'z': p[0],
                    'y': p[1],
                    'x': p[2],
                    'source_id': -1,
                    'target_id': -1,
                })

            if prev_centroids and len(curr_centroids) > 0:
                edges = link_and_detect_mitosis(
                    prev_centroids,
                    curr_centroids,
                    scale=SCALE,
                    max_link_dist=max_link_dist,
                    enable_mitosis=enable_mitosis
                )
                for pid, cid in edges:
                    val_rows.append({
                        'dataset': s,
                        'row_type': 'edge',
                        'node_id': -1,
                        't': -1,
                        'z': -1,
                        'y': -1,
                        'x': -1,
                        'source_id': pid,
                        'target_id': cid,
                    })
            prev_centroids = curr_centroids

    sub_df = pd.DataFrame(val_rows)
    sub_df.index.name = "id"
    sub_csv = os.path.expanduser(f"~/Documents/Biohub/eval_tmp/val_thresh_{peak_thresh}_mit_{enable_mitosis}.csv")
    sub_df.to_csv(sub_csv)

    pred_geff_dir = os.path.expanduser(f"~/Documents/Biohub/eval_tmp/geffs_thresh_{peak_thresh}_mit_{enable_mitosis}")
    os.makedirs(pred_geff_dir, exist_ok=True)
    subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/csv_to_geffs.py"),
        "--csv", sub_csv,
        "--out-dir", pred_geff_dir
    ], check=True, stdout=subprocess.DEVNULL)

    eval_out = subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/evaluate.py"),
        "--pred-dir", pred_geff_dir,
        "--gt-dir", train_dir
    ], capture_output=True, text=True)

    print(eval_out.stdout)
    return eval_out.stdout

if __name__ == "__main__":
    for th in [0.15, 0.10, 0.08]:
        run_evaluation(peak_thresh=th, max_link_dist=14.0, enable_mitosis=True)
