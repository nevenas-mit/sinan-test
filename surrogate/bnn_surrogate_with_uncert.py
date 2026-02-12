#!/usr/bin/env python3
"""
Generate surrogate training data by running the BNN (teacher) on the dataset.
Saves (X, Y_bnn) where X is scaled full-dim input and Y_bnn contains:
  - Column 0: 99th percentile latency (BNN output index 15)
  - Column 1: Violation probability (1.0 if any 99th pct in future steps >= QoS, else 0.0)

This matches the output format expected by the master controller (same as CNN+XGBoost).

Fits scalers from train data (893 features: 840 sys + 25 lat + 28 nxt), then applies
top_indices feature selection to match the BNN checkpoint (100 features).

NEW: Computes distance-based uncertainty in DT input space (scaled + top_indices applied) and saves:
  - dt_uncertainty_ref.npy: reference feature bank for kNN distance uncertainty
  - dt_uncertainty_thresholds.json: suggested tau values from train-distance percentiles
  - dt_uncertainty_{split}.npy: optional per-sample uncertainty for train/valid/test
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler

# Add ml_docker_swarm to path for BayesianMLP import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ml_docker_swarm'))
from train_bnn_explore import BayesianMLP


def load_and_reshape(filepath):
    arr = np.load(filepath)
    return arr.reshape(arr.shape[0], -1)


def knn_distance_uncertainty(x, ref, k=1, batch_size=4096):
    """
    Distance-based uncertainty: distance to nearest (or mean of k nearest) points in ref.

    x:   (N, D) float
    ref: (M, D) float
    returns: (N,) uncertainty
    """
    if x.ndim != 2 or ref.ndim != 2:
        raise ValueError(f"x and ref must be 2D; got x={x.shape}, ref={ref.shape}")
    if x.shape[1] != ref.shape[1]:
        raise ValueError(f"Dim mismatch: x={x.shape}, ref={ref.shape}")
    k = int(max(1, k))
    k = min(k, ref.shape[0])

    x = x.astype(np.float32, copy=False)
    ref = ref.astype(np.float32, copy=False)

    # Precompute ref norms once
    ref_norm2 = np.sum(ref * ref, axis=1, keepdims=True).T  # (1, M)

    out = np.empty((x.shape[0],), dtype=np.float32)

    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        xb = x[start:end]  # (B, D)
        x_norm2 = np.sum(xb * xb, axis=1, keepdims=True)  # (B, 1)

        # squared distances (B, M)
        d2 = x_norm2 + ref_norm2 - 2.0 * (xb @ ref.T)
        d2 = np.maximum(d2, 0.0)

        if k == 1:
            out[start:end] = np.sqrt(np.min(d2, axis=1))
        else:
            idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]  # (B, k)
            dk = np.take_along_axis(d2, idx, axis=1)              # (B, k)
            out[start:end] = np.sqrt(np.mean(dk, axis=1))

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with sys_data_train.npy, lat_data_*.npy, nxt_k_*.npy, nxt_k_*_label.npy")
    parser.add_argument("--bnn-model", type=str, required=True,
                        help="Path to BNN .pth checkpoint")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Directory to save X_surrogate_*.npy, Y_bnn_*.npy (and scalers/uncertainty artifacts)")
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="Number of MC samples for BNN mean prediction")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size for BNN inference")
    parser.add_argument("--save-scalers", action="store_true",
                        help="Save fitted scalers to out-dir for use by predictor with this surrogate")
    parser.add_argument("--qos", type=float, default=500.0,
                        help="QoS threshold (ms) for violation probability derivation")

    # NEW uncertainty args
    parser.add_argument("--save-uncertainty", action="store_true",
                        help="Save per-sample dt_uncertainty_{split}.npy for train/valid/test (can be big).")
    parser.add_argument("--uncertainty-ref-size", type=int, default=5000,
                        help="How many TRAIN points to store as dt_uncertainty_ref.npy.")
    parser.add_argument("--uncertainty-k", type=int, default=1,
                        help="k for kNN distance uncertainty (1 = nearest neighbor).")
    parser.add_argument("--uncertainty-batch", type=int, default=4096,
                        help="Batch size for distance uncertainty computation.")
    parser.add_argument("--uncertainty-seed", type=int, default=0,
                        help="Random seed for subsampling uncertainty ref bank.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir

    # Infer BNN arch from checkpoint
    sd = torch.load(args.bnn_model, map_location="cpu")
    input_dim = sd["weight_mus.0"].shape[1]
    output_dim = sd["bias_mus.2"].shape[0]
    hidden_dim = sd["weight_mus.0"].shape[0]
    num_layers = 2
    print(f"BNN from checkpoint: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}")

    # Load top_indices for feature selection (must match BNN training)
    bnn_model_dir = os.path.dirname(args.bnn_model)
    top_indices_path = os.path.join(bnn_model_dir, "top_feature_indices.npy")
    if os.path.isfile(top_indices_path):
        top_indices = np.load(top_indices_path)
        print(f"Loaded top_indices ({len(top_indices)} features) from {top_indices_path}")
    else:
        raise FileNotFoundError(f"top_feature_indices.npy not found at {top_indices_path}. "
                                "BNN must be trained with feature selection enabled.")

    # Load train data and fit scalers
    sys_train = load_and_reshape(os.path.join(data_dir, "sys_data_train.npy"))
    lat_train = load_and_reshape(os.path.join(data_dir, "lat_data_train.npy"))
    nxt_raw_train = np.load(os.path.join(data_dir, "nxt_k_data_train.npy"))
    nxt_train = nxt_raw_train.reshape(nxt_raw_train.shape[0], -1)  # (N, 140)
    label_train = load_and_reshape(os.path.join(data_dir, "nxt_k_train_label.npy"))

    scaler_sys = StandardScaler().fit(sys_train)
    scaler_lat = StandardScaler().fit(lat_train)
    scaler_nxt = StandardScaler().fit(nxt_train)
    scaler_y = StandardScaler().fit(label_train)

    sys_s = scaler_sys.transform(sys_train)
    lat_s = scaler_lat.transform(lat_train)
    nxt_s = scaler_nxt.transform(nxt_train)
    x_train_full = np.concatenate([sys_s, lat_s, nxt_s], axis=1)
    print(f"Full feature dimension (before selection): {x_train_full.shape[1]}")

    # Apply feature selection to match BNN input
    x_train = x_train_full[:, top_indices]
    assert x_train.shape[1] == input_dim, f"x_train {x_train.shape[1]} vs input_dim {input_dim}"

    # Load BNN
    bnn = BayesianMLP(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    bnn.load_state_dict(sd, strict=True)
    bnn.eval()

    def process_split(suffix):
        sys_path = os.path.join(data_dir, f"sys_data_{suffix}.npy")
        if not os.path.isfile(sys_path):
            return None
        sys_data = load_and_reshape(sys_path)
        lat_data = load_and_reshape(os.path.join(data_dir, f"lat_data_{suffix}.npy"))
        nxt_raw = np.load(os.path.join(data_dir, f"nxt_k_data_{suffix}.npy"))
        nxt_data = nxt_raw.reshape(nxt_raw.shape[0], -1)  # (N, 140)

        sys_s_ = scaler_sys.transform(sys_data)
        lat_s_ = scaler_lat.transform(lat_data)
        nxt_s_ = scaler_nxt.transform(nxt_data)
        x_full = np.concatenate([sys_s_, lat_s_, nxt_s_], axis=1)

        return x_full[:, top_indices]

    def run_bnn_mean(x_np, mc_samples, batch_size):
        n = x_np.shape[0]
        all_preds = []
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x_b = torch.tensor(x_np[start:end], dtype=torch.float32).to(device)
                batch_preds = []
                for _ in range(mc_samples):
                    pred = bnn(x_b, sample=True).cpu().numpy()
                    batch_preds.append(pred)
                batch_preds = np.stack(batch_preds, axis=0).mean(axis=0)
                all_preds.append(batch_preds)
        return np.concatenate(all_preds, axis=0)

    def extract_targets(y_bnn_scaled, qos_threshold):
        """
        Extract CNN+XGBoost compatible outputs from BNN predictions.

        Args:
            y_bnn_scaled: BNN predictions in scaled space, shape (N, 25)
            qos_threshold: QoS threshold in ms for violation detection

        Returns:
            y_target: shape (N, 2) with [99th_pct_latency, violation_prob]
        """
        y_bnn_real = scaler_y.inverse_transform(y_bnn_scaled)

        pred_lat_99 = y_bnn_real[:, 15]
        future_99_pcts = y_bnn_real[:, 15:20]  # (N, 5)
        max_future_lat = np.max(future_99_pcts, axis=1)
        viol_prob = (max_future_lat >= qos_threshold).astype(np.float64)

        return np.column_stack([pred_lat_99, viol_prob])

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------
    # 1) Run BNN teacher and save surrogate train targets
    # -------------------------
    print(f"Train X shape: {x_train.shape}")
    print(f"Running BNN with {args.mc_samples} MC samples...")
    print(f"QoS threshold for violation: {args.qos} ms")

    y_bnn_train_scaled = run_bnn_mean(x_train, args.mc_samples, args.batch_size)
    y_bnn_train_real = extract_targets(y_bnn_train_scaled, args.qos)

    np.save(os.path.join(args.out_dir, "X_surrogate_train.npy"), x_train)
    np.save(os.path.join(args.out_dir, "Y_bnn_train.npy"), y_bnn_train_real)
    print(f"Saved X_surrogate_train.npy, Y_bnn_train.npy to {args.out_dir}")
    print(f"  Violation rate (train): {y_bnn_train_real[:, 1].mean():.2%}")

    # -------------------------
    # 2) NEW: Save uncertainty reference bank + recommended taus
    # -------------------------
    rng = np.random.default_rng(args.uncertainty_seed)
    ref_size = min(int(args.uncertainty_ref_size), x_train.shape[0])
    ref_idx = rng.choice(x_train.shape[0], size=ref_size, replace=False)

    dt_ref = x_train[ref_idx].astype(np.float32, copy=False)
    ref_path = os.path.join(args.out_dir, "dt_uncertainty_ref.npy")
    np.save(ref_path, dt_ref)
    print(f"Saved dt_uncertainty_ref.npy (shape {dt_ref.shape}) to {args.out_dir}")

    print(f"Computing DT distance uncertainty on train (k={args.uncertainty_k})...")
    unc_train = knn_distance_uncertainty(
        x_train, dt_ref, k=args.uncertainty_k, batch_size=args.uncertainty_batch
    )

    percentiles = [90, 95, 97, 99]
    tau_suggestions = {f"p{p}": float(np.percentile(unc_train, p)) for p in percentiles}
    thr_path = os.path.join(args.out_dir, "dt_uncertainty_thresholds.json")
    with open(thr_path, "w") as f:
        json.dump({
            "k": int(args.uncertainty_k),
            "ref_size": int(ref_size),
            "seed": int(args.uncertainty_seed),
            "percentile_taus": tau_suggestions
        }, f, indent=2)
    print(f"Saved dt_uncertainty_thresholds.json with taus: {tau_suggestions}")

    if args.save_uncertainty:
        np.save(os.path.join(args.out_dir, "dt_uncertainty_train.npy"), unc_train)
        print("Saved dt_uncertainty_train.npy")

    # -------------------------
    # 3) Valid split (if present): save X/Y and optional uncertainty
    # -------------------------
    x_valid = process_split("valid")
    if x_valid is not None:
        y_bnn_valid_scaled = run_bnn_mean(x_valid, args.mc_samples, args.batch_size)
        y_bnn_valid_real = extract_targets(y_bnn_valid_scaled, args.qos)

        np.save(os.path.join(args.out_dir, "X_surrogate_valid.npy"), x_valid)
        np.save(os.path.join(args.out_dir, "Y_bnn_valid.npy"), y_bnn_valid_real)
        print(f"Saved X_surrogate_valid.npy, Y_bnn_valid.npy (shape {y_bnn_valid_real.shape})")
        print(f"  Violation rate (valid): {y_bnn_valid_real[:, 1].mean():.2%}")

        if args.save_uncertainty:
            print("Computing DT distance uncertainty on valid...")
            unc_valid = knn_distance_uncertainty(
                x_valid, dt_ref, k=args.uncertainty_k, batch_size=args.uncertainty_batch
            )
            np.save(os.path.join(args.out_dir, "dt_uncertainty_valid.npy"), unc_valid)
            print("Saved dt_uncertainty_valid.npy")
    else:
        print("No valid split found; skipping.")

    # -------------------------
    # 4) Test split (optional): save X/Y and optional uncertainty
    # -------------------------
    x_test = process_split("test")
    if x_test is not None:
        y_bnn_test_scaled = run_bnn_mean(x_test, args.mc_samples, args.batch_size)
        y_bnn_test_real = extract_targets(y_bnn_test_scaled, args.qos)

        np.save(os.path.join(args.out_dir, "X_surrogate_test.npy"), x_test)
        np.save(os.path.join(args.out_dir, "Y_bnn_test.npy"), y_bnn_test_real)
        print(f"Saved X_surrogate_test.npy, Y_bnn_test.npy (shape {y_bnn_test_real.shape})")
        print(f"  Violation rate (test): {y_bnn_test_real[:, 1].mean():.2%}")

        if args.save_uncertainty:
            print("Computing DT distance uncertainty on test...")
            unc_test = knn_distance_uncertainty(
                x_test, dt_ref, k=args.uncertainty_k, batch_size=args.uncertainty_batch
            )
            np.save(os.path.join(args.out_dir, "dt_uncertainty_test.npy"), unc_test)
            print("Saved dt_uncertainty_test.npy")

    # -------------------------
    # 5) Save scalers (optional)
    # -------------------------
    if args.save_scalers:
        joblib.dump(scaler_sys, os.path.join(args.out_dir, "scaler_sys.pkl"))
        joblib.dump(scaler_lat, os.path.join(args.out_dir, "scaler_lat.pkl"))
        joblib.dump(scaler_nxt, os.path.join(args.out_dir, "scaler_nxt.pkl"))
        joblib.dump(scaler_y, os.path.join(args.out_dir, "scaler_y.pkl"))
        np.save(os.path.join(args.out_dir, "top_feature_indices.npy"), top_indices)
        print(f"Saved scalers and top_feature_indices ({len(top_indices)} features) to out-dir.")

    print("Done.")


if __name__ == "__main__":
    main()