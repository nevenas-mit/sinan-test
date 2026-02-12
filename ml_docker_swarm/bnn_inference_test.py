#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
import os
import numpy as np
import torch
import psutil
import resource
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# === Import model class from your training script ===
from train_bnn_explore import BayesianMLP  # <-- ensure correct path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Utils --------------------

def load_and_reshape(file):
    arr = np.load(file)
    return arr.reshape(arr.shape[0], -1)

def bytes_to_mb(n_bytes: int) -> float:
    return n_bytes / (1024.0 ** 2)

def get_file_size_mb(path: str) -> float:
    try:
        return bytes_to_mb(os.path.getsize(path))
    except OSError:
        return float('nan')

def get_param_buffer_size_bytes(model: torch.nn.Module):
    """
    Returns:
      trainable_param_bytes, frozen_param_bytes, buffer_bytes, total_bytes
    """
    trainable_param_bytes = 0
    frozen_param_bytes = 0

    for p in model.parameters():
        nbytes = p.nelement() * p.element_size()
        if p.requires_grad:
            trainable_param_bytes += nbytes
        else:
            frozen_param_bytes += nbytes

    buffer_bytes = 0
    for b in model.buffers():
        buffer_bytes += b.nelement() * b.element_size()

    total = trainable_param_bytes + frozen_param_bytes + buffer_bytes
    return trainable_param_bytes, frozen_param_bytes, buffer_bytes, total

def print_bnn_sizes(model: torch.nn.Module, checkpoint_path: str):
    # On-disk
    ckpt_mb = get_file_size_mb(checkpoint_path)

    # In-memory (CPU)
    tr_bytes, fr_bytes, buf_bytes, total_bytes = get_param_buffer_size_bytes(model)

    print("\n=== BNN Model Size ===")
    print(f"Checkpoint file size        : {ckpt_mb:.2f} MB")
    print(f"Trainable parameters (bytes): {tr_bytes:,} ({bytes_to_mb(tr_bytes):.2f} MB)")
    print(f"Frozen parameters (bytes)   : {fr_bytes:,} ({bytes_to_mb(fr_bytes):.2f} MB)")
    print(f"Buffers (bytes)             : {buf_bytes:,} ({bytes_to_mb(buf_bytes):.2f} MB)")
    print(f"Total (params + buffers)    : {total_bytes:,} ({bytes_to_mb(total_bytes):.2f} MB)")

    if torch.cuda.is_available() and next(model.parameters(), None) is not None:
        dev = next(model.parameters()).device
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
            allocated = torch.cuda.memory_allocated(dev)
            reserved  = torch.cuda.memory_reserved(dev)
            print(f"CUDA memory allocated now   : {bytes_to_mb(allocated):.2f} MB")
            print(f"CUDA memory reserved now    : {bytes_to_mb(reserved):.2f} MB")

def cpu_percent_over_window(proc, t_start, t_end, cpu_times_start):
    ct_end = proc.cpu_times()
    cpu_time = (ct_end.user - cpu_times_start.user) + (ct_end.system - cpu_times_start.system)
    wall = max(1e-9, t_end - t_start)
    per_core = (cpu_time / wall) * 100.0
    normalized = per_core / psutil.cpu_count()
    return per_core, normalized

def get_peak_rss_mb():
    # ru_maxrss is KB on Linux, bytes on macOS; convert safely
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / (1024.0**2)) if r > 1e9 else (r / 1024.0)

def pct(xs, q):
    import math
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float('nan')
    return float(np.percentile(xs, q))


# -------------------- Main --------------------

def main(args):
    # === Load scalers and data ===
    sys_data_v = load_and_reshape(f"{args.data_dir}/sys_data_valid.npy")
    lat_data_v = load_and_reshape(f"{args.data_dir}/lat_data_valid.npy")
    nxt_data_v = load_and_reshape(f"{args.data_dir}/nxt_k_data_valid.npy")
    label_v    = load_and_reshape(f"{args.data_dir}/nxt_k_valid_label.npy")

    # Use saved scalers & top indices if available
    scaler_path = "model/bnn_layers2_hdim700_lr1e-04_scalers.pkl"
    top_idx_path = "model/bnn_layers2_hdim700_lr1e-04_top_indices.npy"
    if os.path.exists(scaler_path):
        scaler_x_sys, scaler_x_lat, scaler_x_nxt, scaler_y = joblib.load(scaler_path)
    else:
        # Fallback: fit on validation (not ideal, but keeps script runnable)
        from sklearn.preprocessing import StandardScaler
        scaler_x_sys = StandardScaler().fit(sys_data_v)
        scaler_x_lat = StandardScaler().fit(lat_data_v)
        scaler_x_nxt = StandardScaler().fit(nxt_data_v)
        scaler_y     = StandardScaler().fit(label_v)

    sys_data_v = scaler_x_sys.transform(sys_data_v)
    lat_data_v = scaler_x_lat.transform(lat_data_v)
    nxt_data_v = scaler_x_nxt.transform(nxt_data_v)
    label_v    = scaler_y.transform(label_v)

    x_valid = np.concatenate([sys_data_v, lat_data_v, nxt_data_v], axis=1)
    y_valid = label_v

    if os.path.exists(top_idx_path):
        top_indices = np.load(top_idx_path)
        x_valid_selected = x_valid[:, top_indices]
    else:
        x_valid_selected = x_valid  # fallback: all features

    x_valid_tensor = torch.tensor(x_valid_selected, dtype=torch.float32, device=device)
    y_valid_tensor = torch.tensor(y_valid,          dtype=torch.float32, device=device)

    # === Load trained model ===
    input_dim  = x_valid_tensor.shape[1]
    output_dim = y_valid_tensor.shape[1]
    bnn = BayesianMLP(input_dim, output_dim, args.hidden_dim, args.num_layers).to(device)
    bnn.load_state_dict(torch.load(args.model_path, map_location=device))
    bnn.eval()

    print(f"Model loaded from {args.model_path}")
    print_bnn_sizes(bnn, args.model_path)

    # === Build a fixed batch for fair comparison ===
    N = args.samples_per_run
    M = args.stochastic_passes
    assert N <= x_valid_tensor.shape[0], "Not enough validation samples for requested N"
    x_batch = x_valid_tensor[:N]  # shape [N, D]

    # --- Warm-up (excluded from timing) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(args.warmup):
            for i in range(N):
                _ = bnn.forward(x_batch[i:i+1], sample=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- Runs ---
    proc = psutil.Process(os.getpid())
    runs = args.runs
    end2end_ms, modelonly_ms = [], []
    cpu_percents, cpu_norm = [], []
    peak_rss_mb_list, peak_gpu_mb_list = [], []

    for r in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu_times0 = proc.cpu_times()

        # Model-only timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            for i in range(N):
                # M stochastic passes per sample
                # for _ in range(M):
                _ = bnn.forward(x_batch, sample=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # End-to-end stop
        t_end = time.perf_counter()
        per_core, per_norm = cpu_percent_over_window(proc, t0, t_end, cpu_times0)

        end2end_ms.append((t_end - t0) * 1e3)
        modelonly_ms.append((t2 - t1) * 1e3)
        cpu_percents.append(per_core)
        cpu_norm.append(per_norm)
        peak_rss_mb_list.append(get_peak_rss_mb())
        if device.type == "cuda":
            peak_gpu_mb_list.append(torch.cuda.max_memory_allocated() / (1024.0**2))
            torch.cuda.reset_peak_memory_stats()

    # --- Report ---
    print("\n=== BNN Benchmark ===")
    print(f"runs={runs}, N={N}, M={M}, device={device}")
    print(f"End2End time   : mean {np.mean(end2end_ms):.2f} ms | p50 {pct(end2end_ms,50):.2f} | p95 {pct(end2end_ms,95):.2f}")
    print(f"Model-only time: mean {np.mean(modelonly_ms):.2f} ms | p50 {pct(modelonly_ms,50):.2f} | p95 {pct(modelonly_ms,95):.2f}")
    print(f"CPU% (per-core): mean {np.mean(cpu_percents):.1f}% | normalized {np.mean(cpu_norm):.1f}% of all cores")
    print(f"Peak RSS (host): mean {np.mean(peak_rss_mb_list):.1f} MB")
    if peak_gpu_mb_list:
        print(f"Peak CUDA mem  : mean {np.mean(peak_gpu_mb_list):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help="Path to dataset directory")
    parser.add_argument('--model-path', type=str, required=True, help="Path to saved model .pth file")
    parser.add_argument('--hidden-dim', type=int, default=800)
    parser.add_argument('--num-layers', type=int, default=2)
    # Benchmark controls
    parser.add_argument('--batch-size', type=int, default=900, help="(kept for parity; BNN uses samples-per-run)")
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--stochastic-passes', type=int, default=50, help="M stochastic passes per sample")
    parser.add_argument('--samples-per-run', type=int, default=30, help="N samples per run")
    args = parser.parse_args()

    main(args)
