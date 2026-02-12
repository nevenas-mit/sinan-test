#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import torch
import joblib
import psutil
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# === Import your BNN class from training ===
from train_bnn_explore import BayesianMLP  # <-- adjust to your file/module name

# Reproducibility (optional)
torch.manual_seed(2333)
np.random.seed(2333)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_reshape(path):
    arr = np.load(path)
    return arr.reshape(arr.shape[0], -1)


def cdf_plot(values, title, xlabel, out_path):
    values = np.asarray(values)
    values_sorted = np.sort(values)
    cdf = np.arange(1, values_sorted.size + 1) / values_sorted.size

    plt.figure(figsize=(7, 5))
    plt.plot(values_sorted, cdf, linewidth=3)
    plt.title(title, fontsize=18)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel("Cumulative Probability", fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")


def main(args):
    # === Load validation arrays ===
    x_sys_v  = load_and_reshape(os.path.join(args.data_dir, "sys_data_valid.npy"))
    x_lat_v  = load_and_reshape(os.path.join(args.data_dir, "lat_data_valid.npy"))
    x_nxt_v  = load_and_reshape(os.path.join(args.data_dir, "nxt_k_data_valid.npy"))
    y_v      = load_and_reshape(os.path.join(args.data_dir, "nxt_k_valid_label.npy"))

    # === Load training-time artifacts (scalers + top feature indices) ===
    print(f"Loading scalers")
    scaler_x_sys, scaler_x_lat, scaler_x_nxt, scaler_y = joblib.load("model/bnn_layers2_hdim700_lr1e-04_scalers.pkl")

    print(f"Loading top feature indices")
    top_indices = np.load("model/bnn_layers2_hdim700_lr1e-04_top_indices.npy")

    # === Apply the same preprocessing as training ===
    x_sys_v  = scaler_x_sys.transform(x_sys_v)
    x_lat_v  = scaler_x_lat.transform(x_lat_v)
    x_nxt_v  = scaler_x_nxt.transform(x_nxt_v)
    y_v_scaled = scaler_y.transform(y_v)  # not strictly required; we’ll evaluate in real space

    X_valid = np.concatenate([x_sys_v, x_lat_v, x_nxt_v], axis=1)
    X_valid_sel = X_valid[:, top_indices]

    # Tensors
    Xv_t = torch.tensor(X_valid_sel, dtype=torch.float32, device=device)
    Yv_t = torch.tensor(y_v_scaled, dtype=torch.float32, device=device)

    # === Load model ===
    input_dim = Xv_t.shape[1]
    output_dim = Yv_t.shape[1]
    bnn = BayesianMLP(input_dim, output_dim, args.hidden_dim, args.num_layers).to(device)
    bnn.load_state_dict(torch.load(args.model_path, map_location=device))
    bnn.eval()
    print(f"Model loaded from {args.model_path}")

    # === Inference settings ===
    TARGET_INFERENCES = args.num_inferences
    T = args.stochastic_passes       # stochastic forward passes per inference for mean prediction
    EPS = 1e-8                       # small epsilon for stability in relative error

    # Metrics storage
    reg_accuracies = []              # 1 - mean relative error (per sample), in real units
    rmse_accuracies = []             # 1 / (1 + RMSE), in real units
    rmses = []                       # raw RMSE values (per sample), in real units
    times_ms = []                    # elapsed time for T passes

    def regression_accuracy(pred_vec, true_vec, eps=EPS):
        """1 - mean relative error, clipped to [0,1]."""
        rel_err = np.abs(pred_vec - true_vec) / (np.abs(true_vec) + eps)
        acc = 1.0 - np.mean(rel_err)
        return float(np.clip(acc, 0.0, 1.0))

    def rmse_value(pred_vec, true_vec):
        return float(np.sqrt(np.mean((pred_vec - true_vec) ** 2)))

    n_valid = Xv_t.shape[0]
    idx = 0
    process = psutil.Process(os.getpid())

    with torch.no_grad():
        while len(reg_accuracies) < TARGET_INFERENCES:
            # Round-robin selection from validation set
            x_sample = Xv_t[idx % n_valid].unsqueeze(0)             # (1, input_dim)
            y_true_scaled = Yv_t[idx % n_valid].squeeze(0).cpu().numpy()  # (output_dim,)
            y_true_real = scaler_y.inverse_transform(y_true_scaled.reshape(1, -1)).flatten()
            idx += 1

            t0 = time.time()
            preds_scaled = []
            for _ in range(T):
                y_hat_scaled = bnn.forward(x_sample, sample=True).squeeze(0)  # (output_dim,)
                preds_scaled.append(y_hat_scaled.cpu().numpy())
            t1 = time.time()

            preds_scaled = np.stack(preds_scaled, axis=0)  # (T, output_dim)
            mean_pred_scaled = preds_scaled.mean(axis=0)

            # Convert prediction to original units
            mean_pred_real = scaler_y.inverse_transform(mean_pred_scaled.reshape(1, -1)).flatten()

            # Compute metrics
            reg_acc = regression_accuracy(mean_pred_real, y_true_real, eps=EPS)
            rmse = rmse_value(mean_pred_real, y_true_real)
            rmse_acc = 1.0 / (1.0 + rmse)  # bounded in (0,1]; higher is better

            reg_accuracies.append(reg_acc)
            rmses.append(rmse)
            rmse_accuracies.append(rmse_acc)
            times_ms.append((t1 - t0) * 1000.0)

            if len(reg_accuracies) % 1000 == 0:
                print(f"Inferences: {len(reg_accuracies)} / {TARGET_INFERENCES} | "
                      f"reg-acc={reg_acc:.3f}, rmse={rmse:.4f}, rmse-acc={rmse_acc:.3f}")

    # === Summary ===
    print("\n=== Inference Summary ===")
    print(f"Samples: {len(reg_accuracies)}  |  T={T} passes each")
    print(f"Mean Regression-Accuracy: {np.mean(reg_accuracies):.3f}  "
          f"(median={np.median(reg_accuracies):.3f})")
    print(f"Mean RMSE: {np.mean(rmses):.4f}  (median={np.median(rmses):.4f})")
    print(f"Mean RMSE-Accuracy: {np.mean(rmse_accuracies):.3f}  "
          f"(median={np.median(rmse_accuracies):.3f})")
    print(f"Avg time per sample: {np.mean(times_ms):.2f} ms")

    # === Output directory ===
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save raw arrays for later analysis
    np.save(os.path.join(out_dir, "regression_accuracies.npy"), np.array(reg_accuracies))
    np.save(os.path.join(out_dir, "rmse_values.npy"), np.array(rmses))
    np.save(os.path.join(out_dir, "rmse_accuracies.npy"), np.array(rmse_accuracies))
    np.save(os.path.join(out_dir, "inference_times_ms.npy"), np.array(times_ms))
    print(f"Saved metrics to: {out_dir}")

    # === CDF plots ===
    cdf_plot(reg_accuracies,
             title=f"CDF of Regression Accuracy (N={len(reg_accuracies)}, T={T})",
             xlabel="Regression Accuracy (1 - mean relative error)",
             out_path=os.path.join(out_dir, "cdf_regression_accuracy.png"))

    cdf_plot(rmse_accuracies,
             title=f"CDF of RMSE-Accuracy (N={len(rmse_accuracies)}, T={T})",
             xlabel="RMSE-Accuracy = 1 / (1 + RMSE)",
             out_path=os.path.join(out_dir, "cdf_rmse_accuracy.png"))

    # (Optional) Also plot CDF of raw RMSE (lower is better)
    cdf_plot(rmses,
             title=f"CDF of RMSE (N={len(rmses)}, T={T})",
             xlabel="RMSE (original units)",
             out_path=os.path.join(out_dir, "cdf_rmse.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run many BNN inferences and plot CDFs of accuracies.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with *_valid.npy files")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to saved .pth model (must match hidden-dim / num-layers)")
    parser.add_argument("--hidden-dim", type=int, default=800)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-inferences", type=int, default=10_000,
                        help="Total inferences to run (will loop over validation set as needed)")
    parser.add_argument("--stochastic-passes", type=int, default=50,
                        help="Stochastic forward passes per inference (Monte Carlo)")
    parser.add_argument("--out-dir", type=str, default="model_eval",
                        help="Directory to write metrics and plots")
    args = parser.parse_args()

    main(args)
