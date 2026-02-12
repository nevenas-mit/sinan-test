import argparse
import logging
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

torch.manual_seed(2333)
np.random.seed(2333)
pyro.set_rng_seed(2333)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flexible multi-layer Bayesian MLP
class BayesianMLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.weight_mus = torch.nn.ParameterList()
        self.bias_mus = torch.nn.ParameterList()
        self.weight_logstds = torch.nn.ParameterList()
        self.bias_logstds = torch.nn.ParameterList()

        layer_dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]

        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            self.weight_mus.append(torch.nn.Parameter(torch.randn(out_dim, in_dim) * 0.01))
            self.bias_mus.append(torch.nn.Parameter(torch.zeros(out_dim)))
            self.weight_logstds.append(torch.nn.Parameter(torch.ones(out_dim, in_dim) * -5))
            self.bias_logstds.append(torch.nn.Parameter(torch.ones(out_dim) * -5))

        self.log_noise = torch.nn.Parameter(torch.tensor(-3.0))

    def sample_weights(self):
        weights = []
        for w_mu, w_logstd, b_mu, b_logstd in zip(
                self.weight_mus, self.weight_logstds, self.bias_mus, self.bias_logstds):
            w_std = torch.nn.functional.softplus(w_logstd)
            b_std = torch.nn.functional.softplus(b_logstd)
            w = dist.Normal(w_mu, w_std).rsample()
            b = dist.Normal(b_mu, b_std).rsample()
            weights.append((w, b))
        return weights

    def forward(self, x, weights=None, sample=True):
        if sample or weights is None:
            weights = self.sample_weights()
        else:
            weights = [(w_mu, b_mu) for w_mu, b_mu in zip(self.weight_mus, self.bias_mus)]

        for i, (w, b) in enumerate(weights):
            x = torch.nn.functional.linear(x, w, b)
            if i < len(weights) - 1:
                x = torch.relu(x)
        return x

def model(x, y=None):
    net = pyro.module("bnn", bnn)
    weights = []

    for i in range(len(bnn.weight_mus)):
        w_prior = dist.Normal(torch.zeros_like(bnn.weight_mus[i]), torch.ones_like(bnn.weight_mus[i])).to_event(2)
        b_prior = dist.Normal(torch.zeros_like(bnn.bias_mus[i]), torch.ones_like(bnn.bias_mus[i])).to_event(1)

        w = pyro.sample(f"w_{i}", w_prior)
        b = pyro.sample(f"b_{i}", b_prior)
        weights.append((w, b))

    mean = bnn.forward(x, weights, sample=False)
    sigma = torch.exp(bnn.log_noise)

    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)
    return mean

def guide(x, y=None):
    for i in range(len(bnn.weight_mus)):
        w_mu = pyro.param(f"w_{i}_mu", bnn.weight_mus[i])
        b_mu = pyro.param(f"b_{i}_mu", bnn.bias_mus[i])
        w_logstd = pyro.param(f"w_{i}_logstd", bnn.weight_logstds[i])
        b_logstd = pyro.param(f"b_{i}_logstd", bnn.bias_logstds[i])

        w_std = torch.nn.functional.softplus(w_logstd)
        b_std = torch.nn.functional.softplus(b_logstd)

        pyro.sample(f"w_{i}", dist.Normal(w_mu, w_std).to_event(2))
        pyro.sample(f"b_{i}", dist.Normal(b_mu, b_std).to_event(1))

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def evaluate_rmse(model, data_loader):
    model.eval()
    total_rmse = 0
    n_samples = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            preds = model.forward(xb, sample=False)
            batch_rmse = rmse(preds, yb) * xb.shape[0]
            total_rmse += batch_rmse.item()
            n_samples += xb.shape[0]
    return total_rmse / n_samples

def main(args):
    # === Load and normalize data ===
    data_dir = args.data_dir
    def load_and_reshape(file): return np.load(file).reshape(np.load(file).shape[0], -1)

    sys_data_t = load_and_reshape(f"{data_dir}/sys_data_train.npy")
    lat_data_t = load_and_reshape(f"{data_dir}/lat_data_train.npy")
    nxt_data_t = load_and_reshape(f"{data_dir}/nxt_k_data_train.npy")
    label_t = load_and_reshape(f"{data_dir}/nxt_k_train_label.npy")

    scaler_x_sys = StandardScaler().fit(sys_data_t)
    scaler_x_lat = StandardScaler().fit(lat_data_t)
    scaler_x_nxt = StandardScaler().fit(nxt_data_t)
    scaler_y = StandardScaler().fit(label_t)

    sys_data_t = scaler_x_sys.transform(sys_data_t)
    lat_data_t = scaler_x_lat.transform(lat_data_t)
    nxt_data_t = scaler_x_nxt.transform(nxt_data_t)
    label_t = scaler_y.transform(label_t)

    x_train = np.concatenate([sys_data_t, lat_data_t, nxt_data_t], axis=1)
    y_train = label_t

    # === Select top 100 most important features using RandomForest ===
    print("Fitting RandomForest to compute feature importances...")
    rf = RandomForestRegressor(n_estimators=50, random_state=2333, n_jobs=-1)
    rf.fit(x_train, y_train)

    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-100:]
    print(f"Top 100 feature indices: {top_indices}")

    # Keep only the selected top features
    x_train_selected = x_train[:, top_indices]

    x_train_tensor = torch.tensor(x_train_selected, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=args.batch_size, shuffle=True)

    # === Validation set ===
    sys_data_v = load_and_reshape(f"{data_dir}/sys_data_valid.npy")
    lat_data_v = load_and_reshape(f"{data_dir}/lat_data_valid.npy")
    nxt_data_v = load_and_reshape(f"{data_dir}/nxt_k_data_valid.npy")
    label_v = load_and_reshape(f"{data_dir}/nxt_k_valid_label.npy")

    sys_data_v = scaler_x_sys.transform(sys_data_v)
    lat_data_v = scaler_x_lat.transform(lat_data_v)
    nxt_data_v = scaler_x_nxt.transform(nxt_data_v)
    label_v = scaler_y.transform(label_v)

    x_valid = np.concatenate([sys_data_v, lat_data_v, nxt_data_v], axis=1)
    y_valid = label_v

    x_valid_selected = x_valid[:, top_indices]

    x_valid_tensor = torch.tensor(x_valid_selected, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

    valid_loader = DataLoader(TensorDataset(x_valid_tensor, y_valid_tensor), batch_size=args.batch_size)

    input_dim = x_train_tensor.shape[1]
    output_dim = y_train_tensor.shape[1]

    # === Create and train model ===
    global bnn
    bnn = BayesianMLP(input_dim, output_dim, args.hidden_dim, args.num_layers).to(device)
    pyro.clear_param_store()
    optimizer = ClippedAdam({"lr": args.lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    val_rmses = []

    for epoch in range(args.epochs):
        bnn.train()
        epoch_loss = sum(svi.step(xb, yb) for xb, yb in train_loader) / len(train_loader.dataset)
        losses.append(epoch_loss)

        val_rmse = evaluate_rmse(bnn, valid_loader)
        val_rmses.append(val_rmse)

        logging.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Validation RMSE={val_rmse:.4f}")

    print(f"\nFinal Train RMSE: {evaluate_rmse(bnn, train_loader):.4f}")
    print(f"Final Valid RMSE: {evaluate_rmse(bnn, valid_loader):.4f}")

    TOP_K_BAR = 20
    CUTOFF_K  = 100

    # ---------- after training loop, before saving plots/models ----------
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"bnn_layers{args.num_layers}_hdim{args.hidden_dim}_lr{args.lr:.0e}"

    # === (1) ELBO Loss plot ===
    loss_plot_path = os.path.join(model_dir, f"{model_name}_elbo_loss.png")
    plt.figure(figsize=(7, 5))
    plt.plot(losses, linewidth=4)
    plt.title("ELBO Loss", fontsize=22)
    plt.xlabel("Training Step", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"Plot saved to {loss_plot_path}")

    # === (2) Validation RMSE plot ===
    rmse_plot_path = os.path.join(model_dir, f"{model_name}_val_rmse.png")
    plt.figure(figsize=(7, 5))
    plt.plot(val_rmses, linewidth=4)
    plt.title("Validation RMSE", fontsize=22)
    plt.xlabel("Training Step", fontsize=20)
    plt.ylabel("RMSE", fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(rmse_plot_path)
    print(f"Plot saved to {rmse_plot_path}")

    # === Prepare importance orderings for the next two plots ===
    n_features = importances.shape[0]
    order_desc = np.argsort(importances)[::-1]
    sorted_imps = importances[order_desc]

    # === (3) Top-K Feature Importances (bar chart) ===
    top_k = min(TOP_K_BAR, n_features)
    bar_plot_path = os.path.join(model_dir, f"{model_name}_feature_importance_top{top_k}.png")

    plt.figure(figsize=(8, 6))
    ypos = np.arange(top_k)
    top_idx_desc = order_desc[:top_k]
    top_imps = importances[top_idx_desc]
    plt.barh(ypos, top_imps, height=0.7)
    plt.yticks(ypos, [f"f{idx}" for idx in top_idx_desc], fontsize=16)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance", fontsize=20)
    plt.title(f"Top {top_k} Feature Importances", fontsize=22)
    plt.xticks(fontsize=20)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(bar_plot_path)
    print(f"Plot saved to {bar_plot_path}")

    # === (4) Cumulative Importance curve with cutoff line at K=100 ===
    cum_plot_path = os.path.join(model_dir, f"{model_name}_cumulative_importance.png")
    cum = np.cumsum(sorted_imps)

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, n_features + 1), cum, linewidth=4)
    plt.title("Cumulative Feature Importance", fontsize=22)
    plt.xlabel("Number of Features (sorted by importance)", fontsize=20)
    plt.ylabel("Cumulative Importance", fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if n_features >= CUTOFF_K:
        cutoff_val = cum[CUTOFF_K - 1]
        plt.axvline(CUTOFF_K, linestyle="--")
        plt.axhline(cutoff_val, linestyle="--")
    else:
        cutoff_val = cum[-1]
        plt.axvline(n_features, linestyle="--")
    plt.tight_layout()
    plt.savefig(cum_plot_path)
    print(f"Plot saved to {cum_plot_path}")

    # === Save combined loss/RMSE panel (you already had this) ===
    plot_path = os.path.join(model_dir, f"{model_name}_loss_rmse_plot.png")
    model_path = os.path.join(model_dir, f"{model_name}_model.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses, linewidth=4)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title("ELBO Loss", fontsize=18)
    plt.xlabel("Training Step", fontsize=18)
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(val_rmses, linewidth=4)
    plt.title("Validation RMSE", fontsize=18)
    plt.xlabel("Training Step", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    torch.save(bnn.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    np.save(os.path.join(model_dir, f"{model_name}_top_indices.npy"), top_indices)
    import joblib
    joblib.dump((scaler_x_sys, scaler_x_lat, scaler_x_nxt, scaler_y),
                os.path.join(model_dir, f"{model_name}_scalers.pkl"))

    # === Inspect predictions vs. ground truth + collect uncertainty/accuracy ===
    print("\nSample Predictions on Validation Set:")
    N = min(500, x_valid_tensor.shape[0])  # choose how many val samples to analyze
    bnn.eval()

    all_ps = []
    all_corr = []
    all_u = []       # mean relative uncertainty (%) per sample
    all_acc = []     # regression accuracy (0..1) per sample
    all_rmse = []    # RMSE per sample (in original units)

    with torch.no_grad():
        for i in range(N):
            input_sample = x_valid_tensor[i].unsqueeze(0)  # (1, input_dim)
            true_output = y_valid_tensor[i]                # (output_dim,)

            true_output_np = scaler_y.inverse_transform(true_output.cpu().numpy().reshape(1, -1)).flatten()

            # Number of forward passes to estimate uncertainty
            M = 100
            predictions = []

            t1 = time.time()
            for _ in range(M):
                pred = bnn.forward(input_sample, sample=True).squeeze(0)
                predictions.append(pred.cpu().numpy())

            predictions = np.stack(predictions)  # (M, output_dim)

            # Mean & std over stochastic passes (in scaled space)
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)

            # Back to original units
            mean_pred_real = scaler_y.inverse_transform(mean_pred.reshape(1, -1)).flatten()
            std_pred_real = std_pred * scaler_y.scale_

            # Relative uncertainty (%) per-dimension, then mean over dims
            relative_uncertainty = (std_pred_real / (np.abs(mean_pred_real) + 1e-8)) * 100.0
            u_i = float(np.mean(relative_uncertainty))  # %
            all_u.append(u_i)

            # Errors
            abs_error = np.abs(mean_pred_real - true_output_np)
            epsilon = 1e-8
            rel_error = abs_error / (np.abs(true_output_np) + epsilon) * 100.0  # %
            rmse_i = float(np.sqrt(np.mean((mean_pred_real - true_output_np) ** 2)))
            all_rmse.append(rmse_i)

            # Regression accuracy = 1 - mean relative error (in fraction, not %)
            acc_i = float(1.0 - np.mean(rel_error) / 100.0)
            all_acc.append(np.clip(acc_i, 0.0, 1.0))

            t2 = time.time()

            # Correlations you had per-sample (kept)
            corr_abs, p_val_abs = pearsonr(relative_uncertainty, abs_error)
            corr_rel, p_val_rel = pearsonr(relative_uncertainty, rel_error)
            all_ps.append(p_val_rel)
            all_corr.append(corr_rel)

            # Optional per-sample prints can be noisy; keep or remove
            # print(f"Sample {i+1}: acc={acc_i:.3f}, rmse={rmse_i:.4f}, u%={u_i:.1f}, time={(t2-t1)*1000:.2f} ms")

    print(f"Average Pearson Correlation (per-dim rel-uncertainty vs rel-error): "
          f"{(sum(all_corr)/len(all_corr)):.2f} (p={(sum(all_ps)/len(all_ps)):.2f})")

    # === Global correlations (uncertainty vs accuracy) ===
    U = np.array(all_u)
    ACC = np.array(all_acc)
    RMSE = np.array(all_rmse)
    r_p, p_p = pearsonr(U, ACC)
    r_s, p_s = spearmanr(U, ACC)
    print(f"\nUncertainty vs Accuracy: Pearson r={r_p:.3f} (p={p_p:.2g}), "
          f"Spearman ρ={r_s:.3f} (p={p_s:.2g})")
    r_p_rmse, p_p_rmse = pearsonr(U, -RMSE)  # negative so 'higher is better' aligns with accuracy
    print(f"Uncertainty vs -RMSE: Pearson r={r_p_rmse:.3f} (p={p_p_rmse:.2g})")

    # === Plots for Uncertainty & Correlation with Accuracy ===
    def save_cdf(arr, title, xlabel, path):
        arr = np.sort(np.asarray(arr))
        cdf = np.arange(1, arr.size + 1) / arr.size
        plt.figure(figsize=(7,5))
        plt.plot(arr, cdf, linewidth=3)
        plt.title(title, fontsize=18)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel("Cumulative Probability", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.xticks(fontsize=14); plt.yticks(fontsize=14)
        plt.tight_layout(); plt.savefig(path)
        print(f"Saved: {path}")

    # (a) Histogram of mean relative uncertainty
    unc_hist_path = os.path.join(model_dir, f"{model_name}_uncertainty_hist.png")
    plt.figure(figsize=(7,5))
    plt.hist(U, bins=30, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Mean Relative Uncertainty", fontsize=18)
    plt.xlabel("Mean Relative Uncertainty (%)", fontsize=16)
    plt.ylabel("Count", fontsize=16)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout(); plt.savefig(unc_hist_path)
    print(f"Saved: {unc_hist_path}")

    # (b) Scatter of Uncertainty vs Accuracy
    ua_scatter_path = os.path.join(model_dir, f"{model_name}_uncertainty_vs_accuracy.png")
    plt.figure(figsize=(7,5))
    plt.scatter(U, ACC, s=25, alpha=0.5)
    plt.title(f"Uncertainty vs Accuracy (Pearson r={r_p:.2f})", fontsize=18)
    plt.xlabel("Mean Relative Uncertainty (%)", fontsize=16)
    plt.ylabel("Regression Accuracy (1 - mean rel err)", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout(); plt.savefig(ua_scatter_path)
    print(f"Saved: {ua_scatter_path}")

    # (c) Binned “reliability” curve: accuracy vs uncertainty deciles
    rel_curve_path = os.path.join(model_dir, f"{model_name}_accuracy_by_uncertainty_bin.png")
    q = np.quantile(U, np.linspace(0, 1, 11))
    bin_centers, bin_mean_acc = [], []
    for b in range(10):
        lo, hi = q[b], q[b+1]
        mask = (U >= lo) & (U <= hi if b == 9 else U < hi)
        if np.any(mask):
            bin_centers.append(0.5*(lo+hi))
            bin_mean_acc.append(ACC[mask].mean())

    plt.figure(figsize=(7,5))
    plt.plot(bin_centers, bin_mean_acc, marker="o", linewidth=3)
    plt.title("Accuracy vs Uncertainty (Binned by Deciles)", fontsize=18)
    plt.xlabel("Mean Relative Uncertainty (%)", fontsize=16)
    plt.ylabel("Mean Regression Accuracy", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.tight_layout(); plt.savefig(rel_curve_path)
    print(f"Saved: {rel_curve_path}")

    # (d) CDFs: Uncertainty and Accuracy
    save_cdf(U,
             "CDF of Mean Relative Uncertainty",
             "Mean Relative Uncertainty (%)",
             os.path.join(model_dir, f"{model_name}_uncertainty_cdf.png"))
    save_cdf(ACC,
             "CDF of Regression Accuracy",
             "Regression Accuracy",
             os.path.join(model_dir, f"{model_name}_accuracy_cdf.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default="/home/jovans2/test_llms/sinan-curr/docker_swarm/logs/collected_data/dataset")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--hidden-dim', type=int, default=800)
    parser.add_argument('--num-layers', type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
