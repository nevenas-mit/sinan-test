import argparse
import logging
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.preprocessing import StandardScaler

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

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
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

    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32).to(device)
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

    '''
    # === Save plot and model with hyperparameter info ===
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)

    model_name = f"bnn_layers{args.num_layers}_hdim{args.hidden_dim}_lr{args.lr:.0e}"
    plot_path = os.path.join(model_dir, f"{model_name}_loss_rmse_plot.png")
    model_path = os.path.join(model_dir, f"{model_name}_model.pth")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title("ELBO Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_rmses)
    plt.title("Validation RMSE")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    torch.save(bnn.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    '''

    # === Inspect predictions vs. ground truth ===
    print("\nSample Predictions on Validation Set:")

    # Use first N examples from validation set
    N = 5  # or 10 if you want more
    bnn.eval()
    with torch.no_grad():
        for i in range(N):
            input_sample = x_valid_tensor[i].unsqueeze(0)  # shape: (1, input_dim)
            true_output = y_valid_tensor[i]                # shape: (output_dim,)

            pred_output = bnn.forward(input_sample, sample=False).squeeze(0)  # shape: (output_dim,)

            # Optional: inverse transform to get real-world scale
            true_output_np = scaler_y.inverse_transform(true_output.cpu().numpy().reshape(1, -1)).flatten()
            pred_output_np = scaler_y.inverse_transform(pred_output.cpu().numpy().reshape(1, -1)).flatten()

            print(f"Sample {i+1}:")
            print(f"  Prediction : {pred_output_np}")
            print(f"  Ground Truth: {true_output_np}")
            print()

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

