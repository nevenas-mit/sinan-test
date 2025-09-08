# predictor_bnn.py
# Note: must run from microservices directory!

import sys
import os
import socket
import time
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import pyro
from sklearn.preprocessing import StandardScaler
import joblib  # for saving/loading scalers and feature indices

# ------------------------
# Args definition
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--server-port', dest='server_port', type=int, default=40010)
parser.add_argument('--model-path', type=str, default='./model/bnn_layers2_hdim800_lr1e-04_model.pth')
parser.add_argument('--scaler-sys', type=str, default='./model/scaler_sys.pkl')
parser.add_argument('--scaler-lat', type=str, default='./model/scaler_lat.pkl')
parser.add_argument('--scaler-nxt', type=str, default='./model/scaler_nxt.pkl')
parser.add_argument('--scaler-y', type=str, default='./model/scaler_y.pkl')
parser.add_argument('--top-features', type=str, default='./model/top_feature_indices.npy')
parser.add_argument('--mc-samples', type=int, default=50, help='number of MC forward passes for uncertainty')
parser.add_argument('--uncertainty-threshold', type=float, default=0.1)
args = parser.parse_args()

ServerPort = args.server_port
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# BNN Definition (must match training)
# ------------------------
class BayesianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.weight_mus = nn.ParameterList()
        self.bias_mus = nn.ParameterList()
        self.weight_logstds = nn.ParameterList()
        self.bias_logstds = nn.ParameterList()

        layer_dims = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            self.weight_mus.append(nn.Parameter(torch.randn(out_dim, in_dim) * 0.01))
            self.bias_mus.append(nn.Parameter(torch.zeros(out_dim)))
            self.weight_logstds.append(nn.Parameter(torch.ones(out_dim, in_dim) * -5))
            self.bias_logstds.append(nn.Parameter(torch.ones(out_dim) * -5))
        self.log_noise = nn.Parameter(torch.tensor(-3.0))

    def sample_weights(self):
        import pyro.distributions as dist
        weights = []
        for w_mu, w_logstd, b_mu, b_logstd in zip(
            self.weight_mus, self.weight_logstds, self.bias_mus, self.bias_logstds
        ):
            w_std = torch.nn.functional.softplus(w_logstd)
            b_std = torch.nn.functional.softplus(b_logstd)
            w = dist.Normal(w_mu, w_std).rsample()
            b = dist.Normal(b_mu, b_std).rsample()
            weights.append((w, b))
        return weights

    def forward(self, x, sample=True):
        if sample:
            weights = self.sample_weights()
        else:
            weights = [(w_mu, b_mu) for w_mu, b_mu in zip(self.weight_mus, self.bias_mus)]
        for i, (w, b) in enumerate(weights):
            x = torch.nn.functional.linear(x, w, b)
            if i < len(weights) - 1:
                x = torch.relu(x)
        return x

# ------------------------
# Load model + preprocessing
# ------------------------
logging.info("Loading BNN model and scalers...")

# Load scalers
scaler_sys = joblib.load(args.scaler_sys)
scaler_lat = joblib.load(args.scaler_lat)
scaler_nxt = joblib.load(args.scaler_nxt)
scaler_y = joblib.load(args.scaler_y)

# Load top feature indices
top_indices = np.load(args.top_features)

# Load BNN model
# (input_dim must match training, here it equals len(top_indices))
input_dim = len(top_indices)
# During training you had output_dim = y_train.shape[1]
# We reload scalery to know that dimension:
output_dim = len(scaler_y.mean_)
bnn = BayesianMLP(input_dim, output_dim, hidden_dim=800, num_layers=2).to(device)
bnn.load_state_dict(torch.load(args.model_path, map_location=device))
bnn.eval()

logging.info("BNN model loaded successfully.")

# ------------------------
# Data preprocessing helper
# ------------------------
Services = [
    'compose-post-redis', 'compose-post-service', 'home-timeline-redis', 'home-timeline-service',
    'nginx-thrift', 'post-storage-memcached', 'post-storage-mongodb', 'post-storage-service',
    'social-graph-mongodb', 'social-graph-redis', 'social-graph-service',
    'text-service', 'text-filter-service', 'unique-id-service',
    'url-shorten-service', 'media-service', 'media-filter-service',
    'user-mention-service', 'user-memcached', 'user-mongodb', 'user-service',
    'user-timeline-mongodb', 'user-timeline-redis', 'user-timeline-service',
    'write-home-timeline-service', 'write-home-timeline-rabbitmq',
    'write-user-timeline-service', 'write-user-timeline-rabbitmq'
]

CnnTimeSteps = 5

def _compose_sys_data_channel(sys_data, field, batch_size):
    for i, service in enumerate(Services):
        assert len(sys_data[service][field]) == CnnTimeSteps
        if i == 0:
            data = np.array(sys_data[service][field])
        else:
            data = np.vstack((data, np.array(sys_data[service][field])))
    data = data.reshape([1, data.shape[0], data.shape[1]])
    for i in range(0, batch_size):
        if i == 0:
            channel_data = np.array(data)
        else:
            channel_data = np.vstack((channel_data, data))
    channel_data = channel_data.reshape([channel_data.shape[0], channel_data.shape[1] * channel_data.shape[2]])
    return channel_data

# ------------------------
# Prediction logic with BNN
# ------------------------
def _predict(info):
    raw_sys_data = info['sys_data']
    raw_next_info = info['next_info']
    batch_size = len(raw_next_info)

    # Prepare features (flatten)
    rps_data = _compose_sys_data_channel(raw_sys_data, 'rps', batch_size)
    replica_data = _compose_sys_data_channel(raw_sys_data, 'replica', batch_size)
    cpu_limit_data = _compose_sys_data_channel(raw_sys_data, 'cpu_limit', batch_size)
    cpu_usage_mean_data = _compose_sys_data_channel(raw_sys_data, 'cpu_usage_mean', batch_size)
    rss_mean_data = _compose_sys_data_channel(raw_sys_data, 'rss_mean', batch_size)
    cache_mem_mean_data = _compose_sys_data_channel(raw_sys_data, 'cache_mem_mean', batch_size)

    sys_data = np.concatenate(
        (rps_data, replica_data, cpu_limit_data, cpu_usage_mean_data, rss_mean_data, cache_mem_mean_data), axis=1
    )

    # e2e_lat
    for key in ['90.0', '95.0', '98.0', '99.0', '99.9']:
        assert len(raw_sys_data['e2e_lat'][key]) == CnnTimeSteps
        if key == '90.0':
            e2e_lat = np.array(raw_sys_data['e2e_lat'][key])
        else:
            e2e_lat = np.vstack((e2e_lat, np.array(raw_sys_data['e2e_lat'][key])))
    e2e_lat = e2e_lat.reshape([1, e2e_lat.shape[0], e2e_lat.shape[1]])
    lat_data = np.repeat(e2e_lat, batch_size, axis=0).reshape(batch_size, -1)

    # next_info (proposal of CPUs)
    nxt_data = []
    for proposal in raw_next_info:
        ncore_proposal = []
        for service in Services:
            ncore_proposal.append(proposal[service]['cpus'])
        nxt_data.append(ncore_proposal)
    nxt_data = np.array(nxt_data)

    # Apply scalers and feature selection
    sys_scaled = scaler_sys.transform(sys_data)
    lat_scaled = scaler_lat.transform(lat_data)
    nxt_scaled = scaler_nxt.transform(nxt_data)

    x = np.concatenate([sys_scaled, lat_scaled, nxt_scaled], axis=1)
    x = x[:, top_indices]
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    # BNN Monte Carlo prediction
    predictions = []
    with torch.no_grad():
        for _ in range(args.mc_samples):
            pred = bnn(x_tensor, sample=True).cpu().numpy()
            predictions.append(pred)
    predictions = np.stack(predictions)  # (M, batch, output_dim)
    mean_pred = predictions.mean(axis=0)  # (batch, output_dim)
    std_pred = predictions.std(axis=0)    # (batch, output_dim)

    # Inverse transform to real scale
    mean_pred_real = scaler_y.inverse_transform(mean_pred)
    std_pred_real = std_pred * scaler_y.scale_

    # Decide predictions with uncertainty threshold
    predict = []
    for i in range(batch_size):
        unc = np.mean(std_pred_real[i])  # aggregate uncertainty
        if unc > args.uncertainty_threshold:
            predict.append([-1, -1])
        else:
            # just take first two outputs for compatibility with old interface
            predict.append([
                round(mean_pred_real[i][0], 2),
                round(mean_pred_real[i][1] if mean_pred_real.shape[1] > 1 else mean_pred_real[i][0], 3)
            ])
    return predict

# ------------------------
# Socket server loop (unchanged)
# ------------------------
def main():
    logging.info('Starting BNN predictor server...')
    local_serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    local_serv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    local_serv_sock.bind(('0.0.0.0', ServerPort))
    local_serv_sock.listen(1024)
    host_sock, addr = local_serv_sock.accept()
    host_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    logging.info('master connected')
    MsgBuffer = ''
    terminate = False

    while True:
        data = host_sock.recv(2048).decode('utf-8')
        if len(data) == 0:
            logging.warning('connection reset by host, exiting...')
            break
        MsgBuffer += data
        while '\n' in MsgBuffer:
            (cmd, rest) = MsgBuffer.split('\n', 1)
            MsgBuffer = rest
            if cmd.startswith('pred----'):
                info = json.loads(cmd.split('----')[-1])
                pred_lat = _predict(info)
                ret_msg = 'pred----' + json.dumps(pred_lat) + '\n'
                host_sock.sendall(ret_msg.encode('utf-8'))
            elif cmd.startswith('terminate'):
                ret_msg = 'experiment_done\n'
                host_sock.sendall(ret_msg.encode('utf-8'))
                terminate = True
                break
            else:
                logging.error('Unknown cmd format')
                logging.error(cmd)
                terminate = True
                break
        if terminate:
            break

    host_sock.close()
    local_serv_sock.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    main()
