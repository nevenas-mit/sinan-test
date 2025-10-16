# ============================ train_bnn_explore.py ============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import numpy as np
import torch
import os
import time
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

t_start = time.time()

torch.manual_seed(2333)
np.random.seed(2333)
pyro.set_rng_seed(2333)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Bayesian MLP --------------------
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

# will be set in main()
bnn = None

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
    return total_rmse / max(1, n_samples)

# -------------------- Conformal Helpers (training-time calibration) --------------------
@torch.no_grad()
def bnn_mean_std(bnn, x, M=100):
    bnn.eval()
    preds = []
    for _ in range(M):
        preds.append(bnn.forward(x, sample=True))  # (B, D)
    P = torch.stack(preds, dim=0)                  # (M, B, D)
    mean = P.mean(dim=0)                           # (B, D)
    std = P.std(dim=0, unbiased=True)              # (B, D)
    return mean, std

def _conservative_quantile(scores_1d, alpha):
    s = np.sort(np.asarray(scores_1d, dtype=float))
    n = s.shape[0]
    if n == 0:
        return 0.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])

def compute_cp_stats(bnn, x_cal, y_cal, scaler_y, M=100, alpha=0.1, mode="stdnorm", per_output=False):
    mu_cal, sd_cal = bnn_mean_std(bnn, x_cal, M=M)        # (Nc, D)
    mu_cal_np = mu_cal.cpu().numpy()
    sd_cal_np = sd_cal.cpu().numpy()
    y_cal_np  = y_cal.cpu().numpy()

    mu_real = scaler_y.inverse_transform(mu_cal_np)       # (Nc, D)
    y_real  = scaler_y.inverse_transform(y_cal_np)        # (Nc, D)
    scale   = np.asarray(scaler_y.scale_)
    sd_real = sd_cal_np * scale.reshape(1, -1)

    resid = np.abs(y_real - mu_real)                      # (Nc, D)
    eps = 1e-12

    if mode == "stdnorm":
        sn = resid / (sd_real + eps)                      # (Nc, D)
        if per_output:
            qs = np.array([_conservative_quantile(sn[:, j], alpha) for j in range(sn.shape[1])], dtype=float)
            return {"mode": mode, "per_output": True, "q": qs}
        else:
            sn_max = sn.max(axis=1)                       # (Nc,)
            q = _conservative_quantile(sn_max, alpha)
            return {"mode": mode, "per_output": False, "q": q}
    elif mode == "absolute":
        if per_output:
            qs = np.array([_conservative_quantile(resid[:, j], alpha) for j in range(resid.shape[1])], dtype=float)
            return {"mode": mode, "per_output": True, "q": qs}
        else:
            resid_max = resid.max(axis=1)
            q = _conservative_quantile(resid_max, alpha)
            return {"mode": mode, "per_output": False, "q": q}
    else:
        raise ValueError("mode must be 'stdnorm' or 'absolute'")

# -------------------- Main --------------------
def main(args):
    # === Load and normalize data ===
    data_dir = args.data_dir
    def load_and_reshape(file):
        arr = np.load(file)
        return arr.reshape(arr.shape[0], -1)

    sys_data_t = load_and_reshape(f"{data_dir}/sys_data_train.npy")
    lat_data_t = load_and_reshape(f"{data_dir}/lat_data_train.npy")
    nxt_data_t = load_and_reshape(f"{data_dir}/nxt_k_data_train.npy")
    label_t    = load_and_reshape(f"{data_dir}/nxt_k_train_label.npy")

    scaler_x_sys = StandardScaler().fit(sys_data_t)
    scaler_x_lat = StandardScaler().fit(lat_data_t)
    scaler_x_nxt = StandardScaler().fit(nxt_data_t)
    scaler_y     = StandardScaler().fit(label_t)

    sys_data_t = scaler_x_sys.transform(sys_data_t)
    lat_data_t = scaler_x_lat.transform(lat_data_t)
    nxt_data_t = scaler_x_nxt.transform(nxt_data_t)
    label_t    = scaler_y.transform(label_t)

    x_train = np.concatenate([sys_data_t, lat_data_t, nxt_data_t], axis=1)
    y_train = label_t

    # === RandomForest feature selection ===
    print("Fitting RandomForest to compute feature importances...")
    rf = RandomForestRegressor(n_estimators=50, random_state=2333, n_jobs=-1)
    rf.fit(x_train, y_train)

    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-args.num_features:]
    print(f"Top {args.num_features} feature indices: {top_indices}")

    x_train_selected = x_train[:, top_indices]

    x_train_tensor = torch.tensor(x_train_selected, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train,          dtype=torch.float32).to(device)

    train_loader = DataLoader(TensorDataset(x_train_tensor, y_train_tensor),
                              batch_size=args.batch_size, shuffle=True)

    # === Validation set (for RMSE reporting & CP calibration/test split) ===
    sys_data_v = load_and_reshape(f"{data_dir}/sys_data_valid.npy")
    lat_data_v = load_and_reshape(f"{data_dir}/lat_data_valid.npy")
    nxt_data_v = load_and_reshape(f"{data_dir}/nxt_k_data_valid.npy")
    label_v    = load_and_reshape(f"{data_dir}/nxt_k_valid_label.npy")

    sys_data_v = scaler_x_sys.transform(sys_data_v)
    lat_data_v = scaler_x_lat.transform(lat_data_v)
    nxt_data_v = scaler_x_nxt.transform(nxt_data_v)
    label_v    = scaler_y.transform(label_v)

    x_valid = np.concatenate([sys_data_v, lat_data_v, nxt_data_v], axis=1)
    y_valid = label_v

    x_valid_selected = x_valid[:, top_indices]
    x_valid_tensor = torch.tensor(x_valid_selected, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid,          dtype=torch.float32).to(device)

    valid_loader = DataLoader(TensorDataset(x_valid_tensor, y_valid_tensor),
                              batch_size=args.batch_size)

    input_dim  = x_train_tensor.shape[1]
    output_dim = y_train_tensor.shape[1]

    # === Create and train model ===
    global bnn
    bnn = BayesianMLP(input_dim, output_dim, args.hidden_dim, args.num_layers).to(device)
    pyro.clear_param_store()
    optimizer = ClippedAdam({"lr": args.lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    for epoch in range(args.epochs):
        bnn.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            epoch_loss += svi.step(xb, yb)
        epoch_loss /= max(len(train_loader.dataset), 1)
        val_rmse = evaluate_rmse(bnn, valid_loader)
        logging.info(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val RMSE={val_rmse:.4f}")

    print(f"\nFinal Train RMSE: {evaluate_rmse(bnn, train_loader):.4f}")
    print(f"Final Valid RMSE: {evaluate_rmse(bnn, valid_loader):.4f}")

    # === Save model & artifacts ===
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)
    model_name = f"bnn_layers{args.num_layers}_hdim{args.hidden_dim}_lr{args.lr:.0e}"

    model_path = os.path.join(model_dir, f"{model_name}_model.pth")
    torch.save(bnn.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    np.save(os.path.join(model_dir, f"{model_name}_top_indices.npy"), top_indices)
    joblib.dump(
        (scaler_x_sys, scaler_x_lat, scaler_x_nxt, scaler_y),
        os.path.join(model_dir, f"{model_name}_scalers.pkl")
    )
    print("Scalers and top_indices saved.")

    # === Conformal Calibration Split from validation ===
    cal_frac = args.cal_frac
    n_valid = x_valid_tensor.shape[0]
    n_cal = int(max(1, round(n_valid * cal_frac)))
    if n_cal >= n_valid:
        n_cal = max(1, n_valid - 1)

    x_cal = x_valid_tensor[:n_cal]
    y_cal = y_valid_tensor[:n_cal]

    # === Compute and save CP stats ===
    cp_stats = compute_cp_stats(
        bnn, x_cal, y_cal, scaler_y,
        M=args.mc_cal, alpha=args.cp_alpha,
        mode=args.cp_mode, per_output=bool(args.cp_per_output)
    )
    cp_path = os.path.join(model_dir, f"{model_name}_cp_stats.pkl")
    joblib.dump(
        {
            "cp_stats": cp_stats,
            "alpha": args.cp_alpha,
            "cal_frac": args.cal_frac,
            "mc_cal": args.mc_cal,
            "mode": args.cp_mode,
            "per_output": bool(args.cp_per_output),
        },
        cp_path
    )
    print(f"CP stats saved to {cp_path}")

    print("\nDone.")

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--model-dir', type=str, default="model")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--hidden-dim', type=int, default=800)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-features', type=int, default=100)

    # CP args
    parser.add_argument('--cal-frac', type=float, default=0.30)
    parser.add_argument('--cp-alpha', type=float, default=0.10)
    parser.add_argument('--cp-mode', type=str, default="stdnorm", choices=["stdnorm", "absolute"])
    parser.add_argument('--cp-per-output', type=int, default=0)  # 1 per-dim, 0 joint via max
    parser.add_argument('--mc-cal', type=int, default=100)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    t1 = time.time()
    main(args)
    t2 = time.time()
    print(f"\nTotal time: {(t2 - t1):.2f} seconds")
