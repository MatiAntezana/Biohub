import os, glob, subprocess, torch, json
import numpy as np, pandas as pd
from skimage.feature import peak_local_max

from src.models.unet3d_anisotropic import AnisotropicUNet3D
from src.dataset import load_zarr_chunk, SCALE
from src.tracking.gap_closing_tracker import track_with_gap_closing

@torch.no_grad()
def predict_tta(model, vol, device):
    vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
    with torch.amp.autocast('cuda'):
        h0 = model(vol_t).squeeze().detach().cpu().numpy().astype(np.float32)
        vx = torch.flip(vol_t, dims=[-1])
        hx = torch.flip(model(vx).squeeze(), dims=[-1]).detach().cpu().numpy().astype(np.float32)
        vy = torch.flip(vol_t, dims=[-2])
        hy = torch.flip(model(vy).squeeze(), dims=[-2]).detach().cpu().numpy().astype(np.float32)
        vz = torch.flip(vol_t, dims=[-3])
        hz = torch.flip(model(vz).squeeze(), dims=[-3]).detach().cpu().numpy().astype(np.float32)
    return (h0 + hx + hy + hz) / 4.0

def evaluate_exp005(ckpt_name="exp005_fullfit_best.pt", peak_thresh=0.010, max_link_dist=14.0):
    device = torch.device("cuda")
    model = AnisotropicUNet3D().to(device)
    ckpt_path = os.path.expanduser(f"~/Documents/Biohub/checkpoints/{ckpt_name}")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    train_dir = os.path.expanduser("~/Documents/Biohub/data/train")
    val_samples = sorted([os.path.basename(p).replace(".zarr", "") for p in glob.glob(f"{train_dir}/44b6_*.zarr")])[:8]
    print(f"\n=== Evaluating {ckpt_name} with 3D TTA on {len(val_samples)} samples (thresh={peak_thresh}) ===")

    all_submission_rows = []
    node_id_counter = 1

    for s in val_samples:
        z_path = os.path.join(train_dir, f"{s}.zarr")
        with open(os.path.join(z_path, '0', 'zarr.json')) as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        dtype = np.dtype(meta['data_type'])
        n_t = shape[0]

        nodes_by_t = {}
        for t in range(n_t):
            vol = load_zarr_chunk(z_path, t, shape, dtype).astype(np.float32)
            p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
            vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

            heat = predict_tta(model, vol, device)
            peaks = peak_local_max(heat, min_distance=2, threshold_abs=peak_thresh, exclude_border=False)
            nodes_by_t[t] = []
            for p in peaks:
                nid = node_id_counter
                node_id_counter += 1
                nodes_by_t[t].append((nid, (int(p[0]), int(p[1]), int(p[2]))))

        filtered_nodes, edges = track_with_gap_closing(
            nodes_by_t,
            scale=SCALE,
            max_link_dist=max_link_dist,
            enable_gap_closing=True,
            min_track_length=2,
            enable_mitosis=False
        )

        for nid, t, z, y, x in filtered_nodes:
            all_submission_rows.append({
                'dataset': s,
                'row_type': 'node',
                'node_id': nid,
                't': t,
                'z': z,
                'y': y,
                'x': x,
                'source_id': -1,
                'target_id': -1,
            })

        for pid, cid in edges:
            all_submission_rows.append({
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

    sub_df = pd.DataFrame(all_submission_rows)
    sub_df.index.name = "id"
    sub_csv = os.path.expanduser(f"~/Documents/Biohub/eval_tmp/val_exp005.csv")
    sub_df.to_csv(sub_csv)

    pred_geff_dir = os.path.expanduser(f"~/Documents/Biohub/eval_tmp/pred_geffs_exp005")
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

if __name__ == "__main__":
    evaluate_exp005("exp005_fullfit_best.pt", peak_thresh=0.010)
