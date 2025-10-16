# ============================ bnn_cp_benchmark.py ============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import time
import math
import numpy as np
import torch
import psutil
import resource
import joblib

from train_bnn_explore import BayesianMLP  # ensure path/name matches your training file

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
    ckpt_mb = get_file_size_mb(checkpoint_path)
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
    normalized = per_core / max(1, psutil.cpu_count())
    return per_core, normalized

def get_peak_rss_mb():
    # ru_maxrss is KB on Linux, bytes on macOS; convert safely
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / (1024.0**2)) if r > 1e9 else (r / 1024.0)

def pct(xs, q):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float('nan')
    return float(np.percentile(xs, q))


# -------------------- CP helpers --------------------

@torch.no_grad()
def bnn_mean_std(bnn, x, M=50):
    bnn.eval()
    preds = []
    for _ in range(M):
        preds.append(bnn.forward(x, sample=True))    # (B, D)
    P = torch.stack(preds, dim=0)                    # (M, B, D)
    return P.mean(dim=0), P.std(dim=0, unbiased=True)

@torch.no_grad()
def cp_predict_intervals(bnn, x, scaler_y, cp_stats, M=50):
    t1 = time.perf_counter()
    mu, sd = bnn_mean_std(bnn, x, M=M)
    t2 = time.perf_counter()
    print(f"  [Debug] BNN MC mean/std time: {(t2 - t1)*1e3:.2f} ms for M={M}")
    t3 = time.perf_counter()
    mu_np  = mu.detach().cpu().numpy()
    sd_np  = sd.detach().cpu().numpy()

    mu_real = scaler_y.inverse_transform(mu_np)
    scale   = np.asarray(scaler_y.scale_)
    sd_real = sd_np * scale.reshape(1, -1)

    mode = cp_stats["mode"]
    per_output = cp_stats["per_output"]
    q = cp_stats["q"]

    if mode == "stdnorm":
        if per_output:
            lower = mu_real - q.reshape(1, -1) * sd_real
            upper = mu_real + q.reshape(1, -1) * sd_real
        else:
            lower = mu_real - q * sd_real
            upper = mu_real + q * sd_real
    else:  # "absolute"
        if per_output:
            lower = mu_real - q.reshape(1, -1)
            upper = mu_real + q.reshape(1, -1)
        else:
            lower = mu_real - q
            upper = mu_real + q

    t4 = time.perf_counter()
    print(f"  [Debug] CP interval construction time: {(t4 - t3)*1e3:.2f} ms")

    return mu_real, lower, upper


@torch.no_grad()
def cp_predict_intervals_euclidean(bnn, x, scaler_y, cp_stats=None, M=50, alpha=0.1):
    """
    Compute prediction intervals using Euclidean distance instead of CP stats.
    alpha: significance level, e.g., 0.1 for ~90% confidence
    """
    t1 = time.perf_counter()
    mu, sd = bnn_mean_std(bnn, x, M=M)
    t2 = time.perf_counter()
    print(f"  [Debug] BNN MC mean/std time: {(t2 - t1)*1e3:.2f} ms for M={M}")

    mu_np  = mu.detach().cpu().numpy()
    sd_np  = sd.detach().cpu().numpy()

    # inverse transform to real scale
    mu_real = scaler_y.inverse_transform(mu_np)
    scale   = np.asarray(scaler_y.scale_)
    sd_real = sd_np * scale.reshape(1, -1)

    # --- New: Euclidean distance based confidence ---
    # radius = quantile of Euclidean norm of std across outputs
    dists = np.linalg.norm(sd_real, axis=1)   # (B,)
    q = np.quantile(dists, 1 - alpha)

    lower = mu_real - q
    upper = mu_real + q

    t3 = time.perf_counter()
    print(f"  [Debug] Euclidean interval construction time: {(t3 - t2)*1e3:.2f} ms "
          f"| radius={q:.4f}")

    return mu_real, lower, upper


# -------------------- Main --------------------

def main(args):
    # --- Load artifacts ---
    scaler_x_sys, scaler_x_lat, scaler_x_nxt, scaler_y = joblib.load(args.scalers_path)
    top_indices = np.load(args.top_indices_path)
    cp_bundle   = joblib.load(args.cp_stats_path)
    cp_stats    = cp_bundle["cp_stats"]

    # --- Load validation data (as a stand-in for inference inputs) ---
    sys_v = load_and_reshape(f"{args.data_dir}/sys_data_valid.npy")
    lat_v = load_and_reshape(f"{args.data_dir}/lat_data_valid.npy")
    nxt_v = load_and_reshape(f"{args.data_dir}/nxt_k_data_valid.npy")
    y_v   = load_and_reshape(f"{args.data_dir}/nxt_k_valid_label.npy")

    sys_v = scaler_x_sys.transform(sys_v)
    lat_v = scaler_x_lat.transform(lat_v)
    nxt_v = scaler_x_nxt.transform(nxt_v)
    y_v   = scaler_y.transform(y_v)

    Xv = np.concatenate([sys_v, lat_v, nxt_v], axis=1)
    Xv = Xv[:, top_indices]

    x_tensor = torch.tensor(Xv, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_v, dtype=torch.float32, device=device)

    # --- Rebuild & load model ---
    input_dim  = x_tensor.shape[1]
    output_dim = y_tensor.shape[1]
    bnn = BayesianMLP(input_dim, output_dim, args.hidden_dim, args.num_layers).to(device)
    bnn.load_state_dict(torch.load(args.model_path, map_location=device))
    bnn.eval()

    print(f"Loaded model: {args.model_path}")
    print(f"Loaded scalers: {args.scalers_path}")
    print(f"Loaded feature indices: {args.top_indices_path}")
    print(f"Loaded CP stats: {args.cp_stats_path} | mode={cp_stats['mode']} per_output={cp_stats['per_output']}")
    print_bnn_sizes(bnn, args.model_path)

    # --- Build batch for benchmarking ---
    N = min(args.samples_per_run, x_tensor.shape[0])
    xb = x_tensor[:N]
    yb = y_tensor[:N]
    y_real = scaler_y.inverse_transform(yb.cpu().numpy())

    # --- Warm-up (excluded from timing) ---
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        _ = cp_predict_intervals(bnn, xb, scaler_y, cp_stats, M=args.mc_samples)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # --- Runs with timing, CPU, memory ---
    proc = psutil.Process(os.getpid())
    runs = args.runs
    end2end_ms, modelonly_ms = [], []
    cpu_percents, cpu_norm = [], []
    peak_rss_mb_list, peak_gpu_mb_list = [], []
    avg_widths, batch_coverages = [], []

    for r in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        cpu_times0 = proc.cpu_times()

        # Model-only section: CP intervals (includes MC passes + interval construction)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            mu_real, lo, hi = cp_predict_intervals(bnn, xb, scaler_y, cp_stats, M=args.mc_samples)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

        # End-to-end timer stop
        t_end = time.perf_counter()
        per_core, per_norm = cpu_percent_over_window(proc, t0, t_end, cpu_times0)

        # Metrics
        end2end_ms.append((t_end - t0) * 1e3)
        modelonly_ms.append((t2 - t1) * 1e3)
        cpu_percents.append(per_core)
        cpu_norm.append(per_norm)
        peak_rss_mb_list.append(get_peak_rss_mb())
        if device.type == "cuda":
            peak_gpu_mb_list.append(torch.cuda.max_memory_allocated() / (1024.0**2))
            torch.cuda.reset_peak_memory_stats()

        # Interval diagnostics (optional but useful)
        covered = ((y_real >= lo) & (y_real <= hi)).all(axis=1)
        batch_coverages.append(covered.mean())
        avg_widths.append((hi - lo).mean())

    # --- Report ---
    print("\n=== BNN + CP Benchmark ===")
    print(f"device={device}, runs={runs}, N(batch)={N}, MC(samples)={args.mc_samples}")
    print(f"End2End time   : mean {np.mean(end2end_ms):.2f} ms | p50 {pct(end2end_ms,50):.2f} | p95 {pct(end2end_ms,95):.2f}")
    print(f"Model-only time: mean {np.mean(modelonly_ms):.2f} ms | p50 {pct(modelonly_ms,50):.2f} | p95 {pct(modelonly_ms,95):.2f}")
    print(f"CPU% (per-core): mean {np.mean(cpu_percents):.1f}% | normalized {np.mean(cpu_norm):.1f}% of all cores")
    print(f"Peak RSS (host): mean {np.mean(peak_rss_mb_list):.1f} MB")
    if peak_gpu_mb_list:
        print(f"Peak CUDA mem  : mean {np.mean(peak_gpu_mb_list):.2f} MB")
    print(f"Avg interval width (mean over runs): {np.mean(avg_widths):.4f}")
    print(f"Empirical coverage on batch (mean over runs, joint): {np.mean(batch_coverages)*100:.1f}%")

    # Print a couple of example predictions
    mu_real, lo, hi = cp_predict_intervals(bnn, xb[:min(3, N)], scaler_y, cp_stats, M=args.mc_samples)
    for i in range(min(3, N)):
        print(f"\nSample {i+1}:")
        print("  mu:", np.array2string(mu_real[i], precision=4))
        print("  lo:", np.array2string(lo[i], precision=4))
        print("  hi:", np.array2string(hi[i], precision=4))
        print("  y :", np.array2string(y_real[i], precision=4))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, required=True, help="Path to dataset directory")
    p.add_argument('--model-path', type=str, required=True, help="Path to saved model .pth")
    p.add_argument('--scalers-path', type=str, required=True, help="Path to saved scalers .pkl")
    p.add_argument('--top-indices-path', type=str, required=True, help="Path to saved top_indices .npy")
    p.add_argument('--cp-stats-path', type=str, required=True, help="Path to saved CP stats .pkl")
    p.add_argument('--hidden-dim', type=int, default=800)
    p.add_argument('--num-layers', type=int, default=2)
    p.add_argument('--mc-samples', type=int, default=50, help="MC passes per inference")
    p.add_argument('--samples-per-run', type=int, default=1, help="Batch size evaluated per run")
    p.add_argument('--runs', type=int, default=20)
    p.add_argument('--warmup', type=int, default=1)
    args = p.parse_args()
    main(args)
