#!/usr/bin/env python3
"""
Generate surrogate training data by running the BNN (teacher) on the dataset.
Saves (X, Y_bnn) where X is scaled full-dim input and Y_bnn contains:
  - Column 0: 99th percentile latency (BNN output index 15)
  - Column 1: Violation probability (1.0 if any 99th pct in future steps >= QoS, else 0.0)

This matches the output format expected by the master controller (same as CNN+XGBoost).

Fits scalers from train data (893 features: 840 sys + 25 lat + 28 nxt), then applies
top_indices feature selection to match the BNN checkpoint (100 features).
"""

import argparse
import os
import sys
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with sys_data_train.npy, lat_data_*.npy, nxt_k_*.npy, nxt_k_*_label.npy")
    parser.add_argument("--bnn-model", type=str, required=True,
                        help="Path to BNN .pth checkpoint")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Directory to save X_surrogate_*.npy, Y_bnn_*.npy (and optional scalers)")
    parser.add_argument("--mc-samples", type=int, default=50,
                        help="Number of MC samples for BNN mean prediction")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size for BNN inference")
    parser.add_argument("--save-scalers", action="store_true",
                        help="Save fitted scalers to out-dir for use by predictor with this surrogate")
    parser.add_argument("--qos", type=float, default=500.0,
                        help="QoS threshold (ms) for violation probability derivation")
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

    # Load train data and fit scalers (893 = 840+25+28, then select top 100)
    sys_train = load_and_reshape(os.path.join(data_dir, "sys_data_train.npy"))
    lat_train = load_and_reshape(os.path.join(data_dir, "lat_data_train.npy"))
    # Use only immediate step (index 0) for nxt features: (N, 28, 5) -> (N, 28)
    nxt_raw_train = np.load(os.path.join(data_dir, "nxt_k_data_train.npy"))
    nxt_train = nxt_raw_train.reshape(nxt_raw_train.shape[0], -1)  # (N, 140)
    label_train = load_and_reshape(os.path.join(data_dir, "nxt_k_train_label.npy"))
    scaler_sys = StandardScaler().fit(sys_train)
    scaler_lat = StandardScaler().fit(lat_train)
    scaler_nxt = StandardScaler().fit(nxt_train)
    # Fit scaler on all 25 output columns (same as BNN training)
    # Label layout (flattened from 5 percentiles x 5 steps):
    #   [0:5]   = 90th percentile, steps 0-4
    #   [5:10]  = 95th percentile, steps 0-4
    #   [10:15] = 98th percentile, steps 0-4
    #   [15:20] = 99th percentile, steps 0-4
    #   [20:25] = 99.9th percentile, steps 0-4
    scaler_y = StandardScaler().fit(label_train)

    sys_s = scaler_sys.transform(sys_train)
    lat_s = scaler_lat.transform(lat_train)
    nxt_s = scaler_nxt.transform(nxt_train)
    x_train_full = np.concatenate([sys_s, lat_s, nxt_s], axis=1)
    print(f"Full feature dimension (before selection): {x_train_full.shape[1]}")
    # Apply feature selection to match BNN input
    x_train = x_train_full[:, top_indices]
    assert x_train.shape[1] == input_dim, f"x_train {x_train.shape[1]} vs input_dim {input_dim}"

    bnn = BayesianMLP(input_dim, output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    bnn.load_state_dict(sd, strict=True)
    bnn.eval()

    def process_split(suffix):
        sys_path = os.path.join(data_dir, f"sys_data_{suffix}.npy")
        if not os.path.isfile(sys_path):
            return None
        sys_data = load_and_reshape(sys_path)
        lat_data = load_and_reshape(os.path.join(data_dir, f"lat_data_{suffix}.npy"))
        # Use only immediate step (index 0) for nxt features: (N, 28, 5) -> (N, 28)
        # nxt_raw = np.load(os.path.join(data_dir, f"nxt_k_data_{suffix}.npy"))
        # nxt_data = nxt_raw[:, :, 0]  # Shape: (N, 28)
        nxt_raw = np.load(os.path.join(data_dir, f"nxt_k_data_{suffix}.npy"))
        nxt_data = nxt_raw.reshape(nxt_raw.shape[0], -1)  # (N, 140)
        sys_s = scaler_sys.transform(sys_data)
        lat_s = scaler_lat.transform(lat_data)
        nxt_s = scaler_nxt.transform(nxt_data)
        x_full = np.concatenate([sys_s, lat_s, nxt_s], axis=1)
        # Apply feature selection to match BNN input
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
        # Inverse transform to real scale
        y_bnn_real = scaler_y.inverse_transform(y_bnn_scaled)
        
        # Extract 99th percentile at step 0 (index 15)
        pred_lat_99 = y_bnn_real[:, 15]
        
        # Derive violation probability from 99th percentile at all future steps (indices 15:20)
        future_99_pcts = y_bnn_real[:, 15:20]  # shape (N, 5)
        max_future_lat = np.max(future_99_pcts, axis=1)  # shape (N,)
        viol_prob = (max_future_lat >= qos_threshold).astype(np.float64)
        
        return np.column_stack([pred_lat_99, viol_prob])

    # Train (x_train already built above)
    print(f"Train X shape: {x_train.shape}")
    print(f"Running BNN with {args.mc_samples} MC samples...")
    print(f"QoS threshold for violation: {args.qos} ms")
    y_bnn_train_scaled = run_bnn_mean(x_train, args.mc_samples, args.batch_size)
    y_bnn_train_real = extract_targets(y_bnn_train_scaled, args.qos)
    print(f"Y_bnn_train shape: {y_bnn_train_real.shape}")
    print(f"  Violation rate (train): {y_bnn_train_real[:, 1].mean():.2%}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X_surrogate_train.npy"), x_train)
    np.save(os.path.join(args.out_dir, "Y_bnn_train.npy"), y_bnn_train_real)
    print(f"Saved X_surrogate_train.npy, Y_bnn_train.npy to {args.out_dir}")

    # Valid
    x_valid = process_split("valid")
    if x_valid is not None:
        y_bnn_valid_scaled = run_bnn_mean(x_valid, args.mc_samples, args.batch_size)
        y_bnn_valid_real = extract_targets(y_bnn_valid_scaled, args.qos)
        np.save(os.path.join(args.out_dir, "X_surrogate_valid.npy"), x_valid)
        np.save(os.path.join(args.out_dir, "Y_bnn_valid.npy"), y_bnn_valid_real)
        print(f"Saved X_surrogate_valid.npy, Y_bnn_valid.npy (shape {y_bnn_valid_real.shape})")
        print(f"  Violation rate (valid): {y_bnn_valid_real[:, 1].mean():.2%}")
    else:
        print("No valid split found; skipping.")
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
