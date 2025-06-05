import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
from sklearn.preprocessing import StandardScaler

# Fix random seeds
torch.manual_seed(2333)
np.random.seed(2333)
pyro.set_rng_seed(2333)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BayesianRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=100):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Initialize variational parameters for weights and biases
        # Using normal initialization with small std
        # Means
        self.fc1_weight_mu = torch.nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.01)
        self.fc1_bias_mu = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.fc2_weight_mu = torch.nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.01)
        self.fc2_bias_mu = torch.nn.Parameter(torch.zeros(output_dim))
        # Log stddevs (initialized small)
        self.fc1_weight_logstd = torch.nn.Parameter(torch.ones(hidden_dim, input_dim) * -5)
        self.fc1_bias_logstd = torch.nn.Parameter(torch.ones(hidden_dim) * -5)
        self.fc2_weight_logstd = torch.nn.Parameter(torch.ones(output_dim, hidden_dim) * -5)
        self.fc2_bias_logstd = torch.nn.Parameter(torch.ones(output_dim) * -5)

        # Observation noise log sigma
        self.log_noise = torch.nn.Parameter(torch.tensor(-3.0))

    def sample_weights(self):
        # Softplus for positive std
        fc1_weight_std = torch.nn.functional.softplus(self.fc1_weight_logstd)
        fc1_bias_std = torch.nn.functional.softplus(self.fc1_bias_logstd)
        fc2_weight_std = torch.nn.functional.softplus(self.fc2_weight_logstd)
        fc2_bias_std = torch.nn.functional.softplus(self.fc2_bias_logstd)

        fc1_weight = dist.Normal(self.fc1_weight_mu, fc1_weight_std).rsample()
        fc1_bias = dist.Normal(self.fc1_bias_mu, fc1_bias_std).rsample()
        fc2_weight = dist.Normal(self.fc2_weight_mu, fc2_weight_std).rsample()
        fc2_bias = dist.Normal(self.fc2_bias_mu, fc2_bias_std).rsample()

        return fc1_weight, fc1_bias, fc2_weight, fc2_bias

    def forward(self, x, weights=None, sample=True):
        if sample or weights is None:
            fc1_weight, fc1_bias, fc2_weight, fc2_bias = self.sample_weights()
        else:
            # Use means for evaluation
            fc1_weight = self.fc1_weight_mu
            fc1_bias = self.fc1_bias_mu
            fc2_weight = self.fc2_weight_mu
            fc2_bias = self.fc2_bias_mu

        x = torch.nn.functional.linear(x, fc1_weight, fc1_bias)
        x = torch.relu(x)
        x = torch.nn.functional.linear(x, fc2_weight, fc2_bias)
        return x

def model(x, y=None):
    net = pyro.module("bnn", bnn)

    fc1_weight_prior = dist.Normal(torch.zeros_like(bnn.fc1_weight_mu), torch.ones_like(bnn.fc1_weight_mu))
    fc1_bias_prior = dist.Normal(torch.zeros_like(bnn.fc1_bias_mu), torch.ones_like(bnn.fc1_bias_mu))
    fc2_weight_prior = dist.Normal(torch.zeros_like(bnn.fc2_weight_mu), torch.ones_like(bnn.fc2_weight_mu))
    fc2_bias_prior = dist.Normal(torch.zeros_like(bnn.fc2_bias_mu), torch.ones_like(bnn.fc2_bias_mu))

    fc1_weight = pyro.sample("fc1_weight", fc1_weight_prior.to_event(2))
    fc1_bias = pyro.sample("fc1_bias", fc1_bias_prior.to_event(1))
    fc2_weight = pyro.sample("fc2_weight", fc2_weight_prior.to_event(2))
    fc2_bias = pyro.sample("fc2_bias", fc2_bias_prior.to_event(1))

    weights = (fc1_weight, fc1_bias, fc2_weight, fc2_bias)

    mean = bnn.forward(x, weights, sample=False)  # Use mean for likelihood mean

    sigma = torch.exp(bnn.log_noise)
    with pyro.plate("data", x.shape[0]):
        pyro.sample("obs", dist.Normal(mean, sigma).to_event(1), obs=y)

    return mean

def guide(x, y=None):
    fc1_weight_std = torch.nn.functional.softplus(pyro.param("fc1_weight_logstd", bnn.fc1_weight_logstd))
    fc1_bias_std = torch.nn.functional.softplus(pyro.param("fc1_bias_logstd", bnn.fc1_bias_logstd))
    fc2_weight_std = torch.nn.functional.softplus(pyro.param("fc2_weight_logstd", bnn.fc2_weight_logstd))
    fc2_bias_std = torch.nn.functional.softplus(pyro.param("fc2_bias_logstd", bnn.fc2_bias_logstd))

    fc1_weight_mu = pyro.param("fc1_weight_mu", bnn.fc1_weight_mu)
    fc1_bias_mu = pyro.param("fc1_bias_mu", bnn.fc1_bias_mu)
    fc2_weight_mu = pyro.param("fc2_weight_mu", bnn.fc2_weight_mu)
    fc2_bias_mu = pyro.param("fc2_bias_mu", bnn.fc2_bias_mu)

    pyro.sample("fc1_weight", dist.Normal(fc1_weight_mu, fc1_weight_std).to_event(2))
    pyro.sample("fc1_bias", dist.Normal(fc1_bias_mu, fc1_bias_std).to_event(1))
    pyro.sample("fc2_weight", dist.Normal(fc2_weight_mu, fc2_weight_std).to_event(2))
    pyro.sample("fc2_bias", dist.Normal(fc2_bias_mu, fc2_bias_std).to_event(1))

def rmse(pred, true):
    return torch.sqrt(torch.mean((pred - true) ** 2))

def evaluate_rmse(model, data_loader):
    model.eval()
    total_rmse = 0
    n_samples = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            preds = model.forward(xb)  # deterministic forward pass
            batch_rmse = rmse(preds, yb) * xb.shape[0]
            total_rmse += batch_rmse.item()
            n_samples += xb.shape[0]
    return total_rmse / n_samples

def main(args):
    # Load data
    data_dir = args.data_dir

    sys_data_t = np.load(f"{data_dir}/sys_data_train.npy")
    lat_data_t = np.load(f"{data_dir}/lat_data_train.npy")
    nxt_data_t = np.load(f"{data_dir}/nxt_k_data_train.npy")
    label_t = np.load(f"{data_dir}/nxt_k_train_label.npy")

    # Normalize inputs and outputs
    scaler_x_sys = StandardScaler()
    scaler_x_lat = StandardScaler()
    scaler_x_nxt = StandardScaler()
    scaler_y = StandardScaler()

    sys_data_t = sys_data_t.reshape(sys_data_t.shape[0], -1)
    lat_data_t = lat_data_t.reshape(lat_data_t.shape[0], -1)
    nxt_data_t = nxt_data_t.reshape(nxt_data_t.shape[0], -1)
    label_t = label_t.reshape(label_t.shape[0], -1)

    sys_data_t = scaler_x_sys.fit_transform(sys_data_t)
    lat_data_t = scaler_x_lat.fit_transform(lat_data_t)
    nxt_data_t = scaler_x_nxt.fit_transform(nxt_data_t)
    label_t = scaler_y.fit_transform(label_t)

    x_train = np.concatenate([sys_data_t, lat_data_t, nxt_data_t], axis=1)
    y_train = label_t

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Validation data
    sys_data_v = np.load(f"{data_dir}/sys_data_valid.npy")
    lat_data_v = np.load(f"{data_dir}/lat_data_valid.npy")
    nxt_data_v = np.load(f"{data_dir}/nxt_k_data_valid.npy")
    label_v = np.load(f"{data_dir}/nxt_k_valid_label.npy")

    sys_data_v = sys_data_v.reshape(sys_data_v.shape[0], -1)
    lat_data_v = lat_data_v.reshape(lat_data_v.shape[0], -1)
    nxt_data_v = nxt_data_v.reshape(nxt_data_v.shape[0], -1)
    label_v = label_v.reshape(label_v.shape[0], -1)

    sys_data_v = scaler_x_sys.transform(sys_data_v)
    lat_data_v = scaler_x_lat.transform(lat_data_v)
    nxt_data_v = scaler_x_nxt.transform(nxt_data_v)
    label_v = scaler_y.transform(label_v)

    x_valid = np.concatenate([sys_data_v, lat_data_v, nxt_data_v], axis=1)
    y_valid = label_v

    x_valid_tensor = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).to(device)

    valid_dataset = TensorDataset(x_valid_tensor, y_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = x_train_tensor.shape[1]
    output_dim = y_train_tensor.shape[1]

    global bnn
    bnn = BayesianRegression(input_dim=input_dim, output_dim=output_dim, hidden_dim=100).to(device)

    pyro.clear_param_store()
    optimizer = ClippedAdam({"lr": args.lr})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    losses = []
    val_rmses = []

    for epoch in range(args.epochs):
        bnn.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            loss = svi.step(xb, yb)
            epoch_loss += loss

        epoch_loss /= len(train_loader.dataset)
        losses.append(epoch_loss)
        logging.info(f"Epoch {epoch + 1} - ELBO Loss per sample: {epoch_loss:.6f}")

        # Evaluation
        bnn.eval()
        total_rmse = 0
        n_samples = 0
        with torch.no_grad():
            for xb, yb in valid_loader:
                mean_pred = bnn.forward(xb, sample=False)
                batch_rmse = rmse(mean_pred, yb) * xb.shape[0]
                total_rmse += batch_rmse.item()
                n_samples += xb.shape[0]
        avg_rmse = total_rmse / n_samples
        val_rmses.append(avg_rmse)
        logging.info(f"Validation RMSE: {avg_rmse:.6f}")
        
    final_train_rmse = evaluate_rmse(bnn, train_loader)
    final_valid_rmse = evaluate_rmse(bnn, valid_loader)

    print(f"Final Training RMSE: {final_train_rmse:.6f}")
    print(f"Final Validation RMSE: {final_valid_rmse:.6f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(losses, label="ELBO Loss per sample")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training ELBO Loss")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_rmses, label="Validation RMSE", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Validation RMSE")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("model/loss_rmse_plot.png")

    torch.save(bnn.state_dict(), "model/bnn_model.pth")
    print("Model saved to model/bnn_model.pth")

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
    parser.add_argument('--data-dir', type=str, required=True, help="Directory with .npy data files")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--batch-size', type=int, default=256, help="Batch size")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
