# ============================ social_media_predictor.py ============================
# NOTE: Must run from microservices directory!
# This version adds Conformal Prediction (CP) as an uncertainty estimator for the
# XGBoost binary classifier output (objective=binary:logistic).
#
# Behavior:
#  - Loads XGB model as before
#  - (Optional) Loads CP stats JSON and uses conformal prediction sets:
#       include1 iff p >= 1-qhat
#       include0 iff p <= qhat
#       uncertain iff both included  => return [-1, -1]
#  - Auto-discovers CP file for the chosen look-forward (no exact alpha match required)

import sys
import os
import socket
import time
import json
import argparse
import logging
import glob
import re

import mxnet as mx
import xgboost as xgb
import numpy as np

# ml parameters
Model = None
InternalSysState = None
BoostTree = None

# Conformal stats (loaded once)
CP_QHAT = None
CP_ALPHA = None

Services = ['compose-post-redis',
            'compose-post-service',
            'home-timeline-redis',
            'home-timeline-service',
            # 'jaeger',
            'nginx-thrift',
            'post-storage-memcached',
            'post-storage-mongodb',
            'post-storage-service',
            'social-graph-mongodb',
            'social-graph-redis',
            'social-graph-service',
            'text-service',
            'text-filter-service',
            'unique-id-service',
            'url-shorten-service',
            'media-service',
            'media-filter-service',
            'user-mention-service',
            'user-memcached',
            'user-mongodb',
            'user-service',
            'user-timeline-mongodb',
            'user-timeline-redis',
            'user-timeline-service',
            'write-home-timeline-service',
            'write-home-timeline-rabbitmq',
            'write-user-timeline-service',
            'write-user-timeline-rabbitmq']

# -----------------------------------------------------------------------
# parser args definition
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--cnn-time-steps', dest='cnn_time_steps', type=int, default=5)
parser.add_argument('--xgb-look-forward', dest='xgb_look_forward', type=int, default=4)
parser.add_argument('--server-port', dest='server_port', type=int, default=40010)
parser.add_argument('--model-prefix', dest='model_prefix', type=str, default='./model/cnv')
parser.add_argument('--load-epoch', dest='load_epoch', type=int, default=200)
parser.add_argument('--xgb-prefix', dest='xgb_prefix', type=str,
                    default='./xgb_model/social_nn_sys_state_look_forward_')
parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')

# CP args
parser.add_argument('--cp-enabled', action='store_true',
                    help='Enable conformal prediction gating for XGB output')
parser.add_argument('--cp-dir', type=str, default='./xgb_model',
                    help='Directory containing cp_xgb_binary_lf{lf}_alpha*.json')
parser.add_argument('--cp-alpha', type=float, default=None,
                    help='Optional. Prefer CP file with this alpha; otherwise auto-pick.')

args = parser.parse_args()

CnnTimeSteps = args.cnn_time_steps
XgbLookForward = args.xgb_look_forward
ServerPort = args.server_port


def _load_model(args, rank=0):
    if (not hasattr(args, "load_epoch")) or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)


def _load_cp():
    """
    Auto-discover CP stats for the chosen look-forward.
    Expected files:
      cp_xgb_binary_lf{look_forward}_alpha{alpha}.json
    """
    global CP_QHAT, CP_ALPHA

    if not args.cp_enabled:
        CP_QHAT = None
        CP_ALPHA = None
        return

    pattern = os.path.join(args.cp_dir, f'cp_xgb_binary_lf{XgbLookForward}_alpha*.json')
    candidates = sorted(glob.glob(pattern))

    if not candidates:
        logging.warning(f'CP enabled but no CP files found matching: {pattern}. Running without CP.')
        CP_QHAT = None
        CP_ALPHA = None
        return

    alpha_re = re.compile(r'_alpha([0-9]+\.[0-9]+)\.json$')

    def extract_alpha(path):
        m = alpha_re.search(path)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    parsed = [(p, extract_alpha(p)) for p in candidates]

    chosen = None
    if args.cp_alpha is not None:
        want = float(args.cp_alpha)
        parsable = [(p, a) for (p, a) in parsed if a is not None]
        if parsable:
            parsable.sort(key=lambda pa: abs(pa[1] - want))
            chosen = parsable[0][0]
        else:
            chosen = max(candidates, key=lambda p: os.path.getmtime(p))
    else:
        # Default: pick the smallest alpha available (most conservative coverage).
        parsable = [(p, a) for (p, a) in parsed if a is not None]
        if parsable:
            parsable.sort(key=lambda pa: pa[1])  # smallest alpha first
            chosen = parsable[0][0]
        else:
            chosen = max(candidates, key=lambda p: os.path.getmtime(p))

    try:
        with open(chosen, 'r') as f:
            d = json.load(f)

        CP_QHAT = float(d.get("qhat"))
        # Prefer alpha inside JSON; fallback to filename alpha
        CP_ALPHA = float(d.get("alpha")) if d.get("alpha") is not None else extract_alpha(chosen)

        if CP_QHAT is None:
            raise ValueError("Missing qhat in CP JSON.")

        logging.info(f'Loaded CP stats from {chosen}: alpha={CP_ALPHA}, qhat={CP_QHAT}')
        logging.info(f'CP uncertain region is p in [{CP_QHAT}, {1.0 - CP_QHAT}] (approx)')

    except Exception as e:
        logging.warning(f'Failed to load CP file {chosen}: {e}. Running without CP.')
        CP_QHAT = None
        CP_ALPHA = None


def _compose_sys_data_channel(sys_data, field, batch_size):
    global Services
    global CnnTimeSteps

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
    channel_data = channel_data.reshape([channel_data.shape[0], 1, channel_data.shape[1], channel_data.shape[2]])
    return channel_data


def _predict(info):
    global Services
    global Model
    global InternalSysState
    global BoostTree
    global CnnTimeSteps
    global XgbLookForward
    global CP_QHAT

    raw_sys_data = info['sys_data']
    raw_next_info = info['next_info']
    batch_size = len(raw_next_info)

    # rps
    rps_data = _compose_sys_data_channel(raw_sys_data, 'rps', batch_size)
    # replica
    replica_data = _compose_sys_data_channel(raw_sys_data, 'replica', batch_size)
    # cpu limit
    cpu_limit_data = _compose_sys_data_channel(raw_sys_data, 'cpu_limit', batch_size)
    # cpu usage
    cpu_usage_mean_data = _compose_sys_data_channel(raw_sys_data, 'cpu_usage_mean', batch_size)
    # memory
    rss_mean_data = _compose_sys_data_channel(raw_sys_data, 'rss_mean', batch_size)
    cache_mem_mean_data = _compose_sys_data_channel(raw_sys_data, 'cache_mem_mean', batch_size)

    sys_data = np.concatenate(
        (rps_data,
         replica_data,
         cpu_limit_data,
         cpu_usage_mean_data,
         rss_mean_data,
         cache_mem_mean_data),
        axis=1)

    # -------------------------- e2e_lat -------------------------- #
    for key in ['90.0', '95.0', '98.0', '99.0', '99.9']:
        assert len(raw_sys_data['e2e_lat'][key]) == CnnTimeSteps
        if key == '90.0':
            e2e_lat = np.array(raw_sys_data['e2e_lat'][key])
        else:
            e2e_lat = np.vstack((e2e_lat, np.array(raw_sys_data['e2e_lat'][key])))

    e2e_lat = e2e_lat.reshape([1, e2e_lat.shape[0], e2e_lat.shape[1]])

    for i in range(0, batch_size):
        if i == 0:
            lat_data = np.array(e2e_lat)
        else:
            lat_data = np.vstack((lat_data, e2e_lat))

    # -------------------------- next_info -------------------------- #
    ncore_next = None
    ncore_next_k = None
    for i, proposal in enumerate(raw_next_info):
        for j, service in enumerate(Services):
            if j == 0:
                ncore_proposal = np.array(proposal[service]['cpus'])
            else:
                ncore_proposal = np.vstack((ncore_proposal, np.array(proposal[service]['cpus'])))

        for k in range(0, XgbLookForward):
            if k == 0:
                ncore_proposal_next_k = np.array(ncore_proposal).reshape([-1, 1])
            else:
                ncore_proposal_next_k = np.hstack((ncore_proposal_next_k,
                                                   np.array(ncore_proposal).reshape([-1, 1])))

        if i == 0:
            ncore_next = ncore_proposal.reshape([1, ncore_proposal.shape[0]])
            ncore_next_k = ncore_proposal_next_k.reshape(
                [1, ncore_proposal_next_k.shape[0], ncore_proposal_next_k.shape[1]])
        else:
            ncore_next = np.vstack((ncore_next, ncore_proposal.reshape([1, ncore_proposal.shape[0]])))
            ncore_next_k = np.vstack((ncore_next_k,
                                      ncore_proposal_next_k.reshape(
                                          [1, ncore_proposal_next_k.shape[0], ncore_proposal_next_k.shape[1]])))

    next_data = ncore_next
    next_k_data = ncore_next_k

    pred_data = {'data1': sys_data, 'data2': lat_data, 'data3': next_data}
    pred_iter = mx.io.NDArrayIter(pred_data, batch_size=batch_size)
    cnn_pred = Model.predict(pred_iter).asnumpy()

    # -------------------- predicting next_k cycle with xgb -------------------- #
    internal_sys_state = InternalSysState.predict(pred_iter).asnumpy()
    next_k_info = next_k_data.reshape(next_k_data.shape[0], -1)
    xgb_input = np.concatenate((internal_sys_state, next_k_info), axis=1)
    dpred = xgb.DMatrix(xgb_input)
    xgb_predict = BoostTree.predict(dpred)  # probabilities in [0,1]

    predict = []
    use_cp = (CP_QHAT is not None)

    for i in range(batch_size):
        cnn_point = float(cnn_pred[i, -2])
        p = float(xgb_predict[i])

        if use_cp:
            q = float(CP_QHAT)
            include1 = (p >= 1.0 - q)
            include0 = (p <= q)
            uncertain = include0 and include1  # prediction set {0,1}

            if uncertain:
                predict.append([-1, -1])
            else:
                predict.append([round(cnn_point, 2), round(p, 3)])
        else:
            predict.append([round(cnn_point, 2), round(p, 3)])

    return predict


def main():
    global Model
    global InternalSysState
    global BoostTree
    global ServerPort
    global CnnTimeSteps
    global XgbLookForward

    # load model for prediction
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    load_params = _load_model(args, kv.rank)
    sym = load_params[0]
    if sym is None:
        raise RuntimeError("Failed to load MXNet checkpoint. Check --model-prefix and --load-epoch.")

    all_layers = sym.get_internals()

    # ---------------- cnn ---------------- #
    Model_sym = all_layers['latency_output']
    Model = mx.mod.Module(
        context=devs,
        symbol=Model_sym,
        data_names=('data1', 'data2', 'data3'),
    )

    default_batch_size = 2048
    Model.bind(for_training=False,
               data_shapes=[('data1', (default_batch_size, 6, len(Services), CnnTimeSteps)),
                            ('data2', (default_batch_size, 5, CnnTimeSteps)),
                            ('data3', (default_batch_size, len(Services)))])
    Model.set_params(load_params[1], load_params[2], allow_missing=True, allow_extra=True)

    # ---------------- xgb internal rep ---------------- #
    InternalSysState_sym = all_layers['full_feature_output']
    InternalSysState = mx.mod.Module(
        context=devs,
        symbol=InternalSysState_sym,
        data_names=('data1', 'data2', 'data3')
    )
    InternalSysState.bind(for_training=False,
                          data_shapes=[('data1', (default_batch_size, 6, len(Services), CnnTimeSteps)),
                                       ('data2', (default_batch_size, 5, CnnTimeSteps)),
                                       ('data3', (default_batch_size, len(Services)))])
    InternalSysState.set_params(load_params[1], load_params[2], allow_missing=True, allow_extra=True)

    BoostTree = xgb.Booster()
    logging.info('load ' + args.xgb_prefix + str(XgbLookForward) + '.model')
    BoostTree.load_model(args.xgb_prefix + str(XgbLookForward) + '.model')

    _load_cp()
    logging.info('models loaded...')

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
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    main()
