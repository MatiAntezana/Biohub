import os, glob, json, time, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.feature import peak_local_max
from scipy.optimize import linear_sum_assignment

from src.dataset_mpm import BiohubMPMDataset, load_zarr_chunk, SCALE
from src.models.mpm_unet3d import MPMUNet3D

def mpm_loss(outputs, heat_gt, flow_gt, flow_mask, div_gt):
    heat_pred = outputs['heatmap']
    flow_pred = outputs['flow_fwd']
    div_pred = outputs['mitosis']

    # 1. Heatmap loss
    w_heat = 1.0 + 10.0 * heat_gt
    loss_heat = torch.mean(w_heat * (heat_pred - heat_gt) ** 2)

    # 2. Flow loss (masked on cell regions)
    flow_err = torch.abs(flow_pred - flow_gt) # (B, 3, Z, Y, X)
    denom = flow_mask.sum() * 3.0 + 1e-5
    loss_flow = (flow_err * flow_mask).sum() / denom

    # 3. Mitosis loss
    w_div = 1.0 + 20.0 * div_gt
    loss_div = torch.mean(w_div * (div_pred - div_gt) ** 2)

    return loss_heat + 0.3 * loss_flow + 2.0 * loss_div, loss_heat, loss_flow, loss_div

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    tot_loss = 0.0
    for vols, heat_gt, flow_gt, fmask, div_gt in loader:
        vols = vols.to(device)
        heat_gt = heat_gt.to(device)
        flow_gt = flow_gt.to(device)
        fmask = fmask.to(device)
        div_gt = div_gt.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(vols)
            loss, lh, lf, ld = mpm_loss(outputs, heat_gt, flow_gt, fmask, div_gt)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tot_loss += loss.item()
    return tot_loss / len(loader)

@torch.no_grad()
def evaluate_loss(model, loader, device):
    model.eval()
    tot_loss = 0.0
    for vols, heat_gt, flow_gt, fmask, div_gt in loader:
        vols = vols.to(device)
        heat_gt = heat_gt.to(device)
        flow_gt = flow_gt.to(device)
        fmask = fmask.to(device)
        div_gt = div_gt.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(vols)
            loss, lh, lf, ld = mpm_loss(outputs, heat_gt, flow_gt, fmask, div_gt)
        tot_loss += loss.item()
    return tot_loss / len(loader)

@torch.no_grad()
def predict_mpm_sample(model, zarr_path, device, peak_thresh=0.015, max_link_dist=14.0, div_thresh=0.35):
    model.eval()
    with open(os.path.join(zarr_path, '0', 'zarr.json')) as f:
        meta = json.load(f)
    shape = tuple(meta['shape'])
    dtype = np.dtype(meta['data_type'])
    n_t, Z, Y, X = shape
    folder_name = os.path.basename(zarr_path).replace('.zarr', '')

    all_rows = []
    prev_nodes = {} # {nid: {'pos': (z, y, x), 'flow': (dz, dy, dx), 'div': prob}}
    node_id_counter = 1

    for t in range(n_t):
        vol = load_zarr_chunk(zarr_path, t, shape, dtype).astype(np.float32)
        p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99.8)
        vol = np.clip((vol - p1) / max(1e-5, p99 - p1), 0.0, 1.0)

        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            out = model(vol_t)
            heat = out['heatmap'].squeeze().detach().cpu().numpy().astype(np.float32)
            flow = out['flow_fwd'].squeeze().detach().cpu().numpy().astype(np.float32) # (3, Z, Y, X)
            div_map = out['mitosis'].squeeze().detach().cpu().numpy().astype(np.float32)

        peaks = peak_local_max(heat, min_distance=2, threshold_abs=peak_thresh, exclude_border=False)
        curr_nodes = {}
        for p in peaks:
            nid = node_id_counter
            node_id_counter += 1
            pz, py, px = int(p[0]), int(p[1]), int(p[2])
            node_flow = (float(flow[0, pz, py, px]), float(flow[1, pz, py, px]), float(flow[2, pz, py, px]))
            node_div = float(div_map[pz, py, px])

            curr_nodes[nid] = {
                'pos': (pz, py, px),
                'flow': node_flow,
                'div': node_div
            }
            all_rows.append({
                'dataset': folder_name,
                'row_type': 'node',
                'node_id': nid,
                't': t,
                'z': pz,
                'y': py,
                'x': px,
                'source_id': -1,
                'target_id': -1,
            })

        # Flow-guided matching with previous frame
        if prev_nodes and len(curr_nodes) > 0:
            prev_ids = list(prev_nodes.keys())
            curr_ids = list(curr_nodes.keys())

            # Projected physical positions of previous nodes at t: P_t * scale + flow
            prev_projected = []
            for pid in prev_ids:
                p_pos = np.array(prev_nodes[pid]['pos'], dtype=np.float64) * SCALE
                p_flow = np.array(prev_nodes[pid]['flow'], dtype=np.float64)
                prev_projected.append(p_pos + p_flow)
            prev_projected = np.array(prev_projected, dtype=np.float64) # (N_prev, 3)

            curr_phys = np.array([np.array(curr_nodes[cid]['pos'], dtype=np.float64) * SCALE for cid in curr_ids]) # (N_curr, 3)

            dist_mat = np.sqrt(((prev_projected[:, None] - curr_phys[None, :]) ** 2).sum(axis=2))
            row_ind, col_ind = linear_sum_assignment(dist_mat)

            matched_prev = set()
            matched_curr = set()
            primary_matches = {}

            for ri, ci in zip(row_ind, col_ind):
                if dist_mat[ri, ci] <= max_link_dist:
                    pid = prev_ids[ri]
                    cid = curr_ids[ci]
                    matched_prev.add(ri)
                    matched_curr.add(ci)
                    primary_matches[ri] = ci
                    all_rows.append({
                        'dataset': folder_name,
                        'row_type': 'edge',
                        'node_id': -1,
                        't': -1,
                        'z': -1,
                        'y': -1,
                        'x': -1,
                        'source_id': pid,
                        'target_id': cid,
                    })

            # Mitosis Branching for high division probability nodes
            unmatched_curr = [ci for ci in range(len(curr_ids)) if ci not in matched_curr]
            for ri, c1 in primary_matches.items():
                pid = prev_ids[ri]
                if prev_nodes[pid]['div'] >= div_thresh and len(unmatched_curr) > 0:
                    # Find closest unmatched daughter to projected parent position
                    dists_to_parent = dist_mat[ri, unmatched_curr]
                    best_idx = np.argmin(dists_to_parent)
                    if dists_to_parent[best_idx] <= max_link_dist:
                        best_c2 = unmatched_curr[best_idx]
                        cid2 = curr_ids[best_c2]
                        all_rows.append({
                            'dataset': folder_name,
                            'row_type': 'edge',
                            'node_id': -1,
                            't': -1,
                            'z': -1,
                            'y': -1,
                            'x': -1,
                            'source_id': pid,
                            'target_id': cid2,
                        })
                        unmatched_curr.pop(best_idx)

        prev_nodes = curr_nodes

    return all_rows

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dir = os.path.expanduser("~/Documents/Biohub/data/train")
    os.makedirs(os.path.expanduser("~/Documents/Biohub/checkpoints"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Documents/Biohub/results"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/Documents/Biohub/eval_tmp"), exist_ok=True)

    all_zarrs = sorted(glob.glob(f"{train_dir}/*.zarr"))
    all_samples = [os.path.basename(z).replace(".zarr", "") for z in all_zarrs]

    train_samples = [s for s in all_samples if s.startswith("6bba")]
    val_samples = [s for s in all_samples if s.startswith("44b6")]
    print(f"MPM Training -> Train ({len(train_samples)} samples from 6bba), Val ({len(val_samples)} samples from 44b6)")

    train_ds = BiohubMPMDataset(train_dir, train_samples, crop_size=(32, 128, 128), is_train=True)
    val_ds = BiohubMPMDataset(train_dir, val_samples[:10], crop_size=(32, 128, 128), is_train=False)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    print(f"MPM Dataset ready: {len(train_ds)} train crops, {len(val_ds)} val crops")

    model = MPMUNet3D(in_channels=1, base_channels=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')
    epochs = 15
    ckpt_path = os.path.expanduser("~/Documents/Biohub/checkpoints/exp002a_mpm_best.pt")

    print(f"Starting MPM training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = evaluate_loss(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {elapsed:.1f}s", flush=True)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Saved best checkpoint: {ckpt_path} (val_loss={val_loss:.6f})", flush=True)

    print("\nTraining complete! Running end-to-end evaluation on validation set...")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))

    eval_val_samples = val_samples[:8]
    val_rows = []
    for s in eval_val_samples:
        z_path = os.path.join(train_dir, f"{s}.zarr")
        val_rows.extend(predict_mpm_sample(model, z_path, device, peak_thresh=0.015, max_link_dist=14.0, div_thresh=0.35))

    sub_df = pd.DataFrame(val_rows)
    sub_df.index.name = 'id'
    sub_csv_path = os.path.expanduser("~/Documents/Biohub/eval_tmp/val_exp002a_mpm.csv")
    sub_df.to_csv(sub_csv_path)

    pred_geff_dir = os.path.expanduser("~/Documents/Biohub/eval_tmp/pred_geffs_exp002a")
    os.makedirs(pred_geff_dir, exist_ok=True)
    subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/csv_to_geffs.py"),
        "--csv", sub_csv_path,
        "--out-dir", pred_geff_dir
    ], check=True)

    eval_out = subprocess.run([
        "python", os.path.expanduser("~/Documents/Biohub/starter_kit/scripts/evaluate.py"),
        "--pred-dir", pred_geff_dir,
        "--gt-dir", train_dir
    ], capture_output=True, text=True)

    print("\n=== Official Metric Results for EXP-002A (MPM) ===")
    print(eval_out.stdout)

    with open(os.path.expanduser("~/Documents/Biohub/results/exp002a_mpm_eval.txt"), "w") as f:
        f.write(eval_out.stdout)

if __name__ == "__main__":
    main()
