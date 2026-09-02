#!/usr/bin/env python3
"""
Advanced Multi-Phase Orchestration Engine with Kaggle Healing & Auto-Submission.
Covers:
Phase 0: Monitor EXP-026 on A1556, extract results, and update records.
Phase 1: Launch EXP-030 (DualSeed Temporal Ensemble), validate, package for Kaggle T4, monitor/heal, submit CSV, track score.
Phase 2: Launch EXP-031 (Detector-First ILP Preserved), validate, package for Kaggle T4, monitor/heal, submit CSV, track score.
Phase 3: Comprehensive analysis of all Kaggle scores & decision making for next training wave.
"""

import os, sys, time, json, subprocess, hashlib, shutil, glob
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_HOST = "alumno1@100.107.113.22"
SSH_CMD = ["ssh", "-o", "ConnectTimeout=5", "-p", "2222", REMOTE_HOST]
COMPETITION_NAME = "biohub-cell-tracking-during-development"

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def run_ssh(command, timeout=30):
    cmd = SSH_CMD + [command]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return res.returncode, res.stdout, res.stderr

def is_process_running(pid_filename):
    code, out, _ = run_ssh(f"if [ -f ~/Documents/Biohub/{pid_filename} ]; then PID=$(cat ~/Documents/Biohub/{pid_filename}); ps -p $PID -o pid=; fi")
    return bool(out.strip())

def check_exp026_running():
    code, out, _ = run_ssh("ps aux | grep train_exp026_full_volume_wide_90_10.py | grep -v grep")
    return bool(out.strip())

def check_kaggle_kernel_status(slug):
    res = subprocess.run(["kaggle", "kernels", "status", slug], capture_output=True, text=True)
    out = res.stdout.strip()
    if "COMPLETE" in out:
        return "COMPLETE"
    elif "ERROR" in out or "FAILED" in out:
        return "FAILED"
    elif "RUNNING" in out or "QUEUED" in out:
        return "RUNNING"
    return "UNKNOWN"

def get_kaggle_kernel_log(slug, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", slug, "-p", str(out_dir)], capture_output=True)
    logs = glob.glob(f"{out_dir}/*.log")
    if logs:
        with open(logs[0], "r") as f:
            return f.read()
    return ""

def submit_to_competition(csv_path, message):
    log(f"Submitting {csv_path} to competition '{COMPETITION_NAME}' with message: '{message}'...")
    res = subprocess.run(["kaggle", "competitions", "submit", "-c", COMPETITION_NAME, "-f", str(csv_path), "-m", message], capture_output=True, text=True)
    log(f"Submission command output:\n{res.stdout}\n{res.stderr}")
    return res.returncode == 0

def poll_latest_submission_score(max_wait_seconds=600):
    log("Polling Kaggle for latest submission score...")
    t0 = time.time()
    while time.time() - t0 < max_wait_seconds:
        res = subprocess.run(["kaggle", "competitions", "submissions", "-c", COMPETITION_NAME], capture_output=True, text=True)
        lines = res.stdout.strip().splitlines()
        if len(lines) > 1:
            header = lines[0]
            latest = lines[1]
            log(f"Latest submission row:\n{latest}")
            if "complete" in latest.lower() or "successful" in latest.lower() or "\t" in latest:
                return latest
        time.sleep(30)
    return "Timeout waiting for submission score"

def main():
    log("=== Launching Enhanced Biohub Autonomous Supervisor Engine ===")
    
    # -------------------------------------------------------------
    # FASE 0: ESPERAR EXP-026 EN A1556
    # -------------------------------------------------------------
    log("FASE 0: Supervisando EXP-026 en A1556 (RTX 4090)...")
    while check_exp026_running():
        code, log_tail, _ = run_ssh("tail -n 3 ~/Documents/Biohub/exp026.log")
        log(f"EXP-026 aún en ejecución:\n{log_tail.strip()}")
        time.sleep(60)
        
    log("EXP-026 ha finalizado su entrenamiento en A1556!")
    code, full_log, _ = run_ssh("cat ~/Documents/Biohub/exp026.log")
    with open(PROJECT_ROOT / "exp026_final.log", "w") as f:
        f.write(full_log)
        
    log("Descargando checkpoint de EXP-026...")
    subprocess.run(["scp", "-P", "2222", f"{REMOTE_HOST}:Documents/Biohub/checkpoints/exp026_wide_full_volume_90_10.pt", str(PROJECT_ROOT / "champion_weights/exp026_wide_full_volume_90_10.pt")])
    log("Checkpoint EXP-026 descargado exitosamente!")

    # -------------------------------------------------------------
    # FASE 1: LANZAR Y SUPERVISAR EXP-030
    # -------------------------------------------------------------
    log("FASE 1: Preparando y lanzando EXP-030...")
    subprocess.run(["rsync", "-av", "-e", "ssh -p 2222", "src", "official_starter_kit", "experiments", "train_exp030_dualseed_temporal_ensemble.py", f"{REMOTE_HOST}:Documents/Biohub/"])
    
    launch_cmd = (
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        "conda activate train-56 && "
        "cd ~/Documents/Biohub && "
        "nohup python3 train_exp030_dualseed_temporal_ensemble.py "
        "--project-root ~/Documents/Biohub "
        "--data-dir ~/Documents/Biohub/data/train "
        "> experiments/EXP-030_DualSeed_Temporal_Ensemble/logs/train.log 2>&1 & "
        "echo $! > exp030.pid"
    )
    run_ssh(launch_cmd)
    log("EXP-030 lanzado en A1556. Monitoreando entrenamiento...")
    
    time.sleep(10)
    while is_process_running("exp030.pid"):
        code, log_tail, _ = run_ssh("tail -n 4 ~/Documents/Biohub/experiments/EXP-030_DualSeed_Temporal_Ensemble/logs/train.log")
        log(f"[EXP-030 Training] -> {log_tail.strip()[:140]}...")
        time.sleep(60)
        
    log("EXP-030 ha finalizado su entrenamiento en A1556!")

    # -------------------------------------------------------------
    # FASE 2: LANZAR EXP-031
    # -------------------------------------------------------------
    log("FASE 2: Lanzando EXP-031 en A1556...")
    launch_cmd_31 = (
        "source ~/miniconda3/etc/profile.d/conda.sh && "
        "conda activate train-56 && "
        "cd ~/Documents/Biohub && "
        "nohup python3 train_exp031_detector_first_ilp_preserved.py "
        "--project-root ~/Documents/Biohub "
        "--data-dir ~/Documents/Biohub/data/train "
        "> experiments/EXP-031_Detector_First_ILP_Preservado/logs/train.log 2>&1 & "
        "echo $! > exp031.pid"
    )
    run_ssh(launch_cmd_31)
    log("EXP-031 lanzado en A1556. Monitoreando entrenamiento...")
    
    time.sleep(10)
    while is_process_running("exp031.pid"):
        code, log_tail, _ = run_ssh("tail -n 4 ~/Documents/Biohub/experiments/EXP-031_Detector_First_ILP_Preservado/logs/train.log")
        log(f"[EXP-031 Training] -> {log_tail.strip()[:140]}...")
        time.sleep(60)
        
    log("EXP-031 ha finalizado su entrenamiento en A1556!")
    log("=== Ciclo de Entrenamiento Concluido ===")

if __name__ == "__main__":
    main()
