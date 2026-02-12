# Surrogate predictor: decision tree / forest mimicking BNN. Same socket protocol as social_media_predictor_bnn.py.
# Run from the surrogate directory.

import socket
import json
import argparse
import logging
import numpy as np
import joblib

# ------------------------
# Args definition
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--server-port', dest='server_port', type=int, default=40010)
parser.add_argument('--model-path', type=str, default='./model/bnn_surrogate_tree.joblib',
                    help='Path to surrogate tree/forest joblib from train_bnn_surrogate_tree.py')
parser.add_argument('--scaler-sys', type=str, default='./model/scaler_sys.pkl')
parser.add_argument('--scaler-lat', type=str, default='./model/scaler_lat.pkl')
parser.add_argument('--scaler-nxt', type=str, default='./model/scaler_nxt.pkl')
parser.add_argument('--scaler-y', type=str, default='./model/scaler_y.pkl')
parser.add_argument('--top-features', type=str, default='./model/top_feature_indices.npy')
args = parser.parse_args()

ServerPort = args.server_port

# ------------------------
# Load model + preprocessing
# ------------------------
logging.info("Loading surrogate model and scalers...")

scaler_sys = joblib.load(args.scaler_sys)
scaler_lat = joblib.load(args.scaler_lat)
scaler_nxt = joblib.load(args.scaler_nxt)
scaler_y = joblib.load(args.scaler_y)
top_indices = np.load(args.top_features)

model = joblib.load(args.model_path)
logging.info("Surrogate model loaded successfully.")

# ------------------------
# Data preprocessing helper (same as BNN predictor)
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
# Prediction logic with surrogate tree
# ------------------------
def _predict(info):
    raw_sys_data = info['sys_data']
    raw_next_info = info['next_info']
    batch_size = len(raw_next_info)

    rps_data = _compose_sys_data_channel(raw_sys_data, 'rps', batch_size)
    replica_data = _compose_sys_data_channel(raw_sys_data, 'replica', batch_size)
    cpu_limit_data = _compose_sys_data_channel(raw_sys_data, 'cpu_limit', batch_size)
    cpu_usage_mean_data = _compose_sys_data_channel(raw_sys_data, 'cpu_usage_mean', batch_size)
    rss_mean_data = _compose_sys_data_channel(raw_sys_data, 'rss_mean', batch_size)
    cache_mem_mean_data = _compose_sys_data_channel(raw_sys_data, 'cache_mem_mean', batch_size)

    sys_data = np.concatenate(
        (rps_data, replica_data, cpu_limit_data, cpu_usage_mean_data, rss_mean_data, cache_mem_mean_data), axis=1
    )

    for key in ['90.0', '95.0', '98.0', '99.0', '99.9']:
        assert len(raw_sys_data['e2e_lat'][key]) == CnnTimeSteps
        if key == '90.0':
            e2e_lat = np.array(raw_sys_data['e2e_lat'][key])
        else:
            e2e_lat = np.vstack((e2e_lat, np.array(raw_sys_data['e2e_lat'][key])))
    e2e_lat = e2e_lat.reshape([1, e2e_lat.shape[0], e2e_lat.shape[1]])
    lat_data = np.repeat(e2e_lat, batch_size, axis=0).reshape(batch_size, -1)

    nxt_data = []
    for proposal in raw_next_info:
        ncore_proposal = []
        for service in Services:
            ncore_proposal.append(proposal[service]['cpus'])
        nxt_data.append(ncore_proposal)
    nxt_data = np.array(nxt_data)

    sys_scaled = scaler_sys.transform(sys_data)
    lat_scaled = scaler_lat.transform(lat_data)
    nxt_scaled = scaler_nxt.transform(nxt_data)

    x = np.concatenate([sys_scaled, lat_scaled, nxt_scaled], axis=1)
    x = x[:, top_indices]

    # Surrogate predicts in original scale (Y_bnn was inverse-transformed during training)
    pred = model.predict(x)  # (batch_size, output_dim)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    pred_lat_col = pred[:, 0]
    pred_viol_col = pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]

    predict = []
    for i in range(batch_size):
        predict.append([
            round(float(pred_lat_col[i]), 2),
            round(float(pred_viol_col[i]), 3)
        ])
    return predict


# ------------------------
# Socket server loop (same as BNN predictor)
# ------------------------
def main():
    logging.info('Starting BNN surrogate predictor server...')
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
