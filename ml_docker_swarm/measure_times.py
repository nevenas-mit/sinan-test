#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Note: must run from microservices directory!

import sys
import os
import time
import json
import argparse, logging
import resource
import psutil

import mxnet as mx
import xgboost as xgb
import numpy as np

# -------------------- Globals --------------------

Model = None
InternalSysState = None
BoostTree = None

Services = [
    'compose-post-redis', 'compose-post-service',
    'home-timeline-redis', 'home-timeline-service',
    'nginx-thrift',
    'post-storage-memcached', 'post-storage-mongodb', 'post-storage-service',
    'social-graph-mongodb', 'social-graph-redis', 'social-graph-service',
    'text-service', 'text-filter-service',
    'unique-id-service', 'url-shorten-service',
    'media-service', 'media-filter-service',
    'user-mention-service', 'user-memcached', 'user-mongodb', 'user-service',
    'user-timeline-mongodb', 'user-timeline-redis', 'user-timeline-service',
    'write-home-timeline-service', 'write-home-timeline-rabbitmq',
    'write-user-timeline-service', 'write-user-timeline-rabbitmq'
]

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

# Benchmark controls
parser.add_argument('--batch-size', type=int, default=900)
parser.add_argument('--runs', type=int, default=30)
parser.add_argument('--warmup', type=int, default=1)

args = parser.parse_args()

CnnTimeSteps   = args.cnn_time_steps
XgbLookForward = args.xgb_look_forward


# -------------------- Helpers --------------------

def cpu_percent_over_window(proc, t_start, t_end, cpu_times_start):
    ct_end = proc.cpu_times()
    cpu_time = (ct_end.user - cpu_times_start.user) + (ct_end.system - cpu_times_start.system)
    wall = max(1e-9, t_end - t_start)
    per_core = (cpu_time / wall) * 100.0
    normalized = per_core / psutil.cpu_count()
    return per_core, normalized

def get_peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return (r / (1024.0**2)) if r > 1e9 else (r / 1024.0)

def pct(xs, q):
    import math
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if not xs:
        return float('nan')
    return float(np.percentile(xs, q))

def get_mxnet_model_size(arg_params, aux_params):
    """Return total size of MXNet model params in MB (host bytes)."""
    total_bytes = 0
    for d in (arg_params, aux_params):
        if not d:
            continue
        for _, v in d.items():
            try:
                total_bytes += v.asnumpy().nbytes  # robust across mxnet versions
            except Exception:
                total_bytes += int(v.size) * 4  # assume float32
    return total_bytes / (1024.0 ** 2)

def get_xgb_model_size(xgb_model):
    """Return XGBoost model size in MB by saving to a temp file."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".xgb")
    os.close(fd)
    try:
        xgb_model.save_model(path)
        size_mb = os.path.getsize(path) / (1024.0 ** 2)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return size_mb


# -------------------- Model Loading --------------------

def _load_model(rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _compose_sys_data_channel(sys_data, field, batch_size):
    data = None
    for i, service in enumerate(Services):
        assert len(sys_data[service][field]) == CnnTimeSteps
        v = np.array(sys_data[service][field], dtype=np.float32)
        data = v if data is None else np.vstack((data, v))
    data = data.reshape([1, data.shape[0], data.shape[1]])
    # tile to batch
    channel_data = np.repeat(data, repeats=batch_size, axis=0)
    channel_data = channel_data.reshape([channel_data.shape[0], 1, channel_data.shape[1], channel_data.shape[2]])
    return channel_data

def _predict(info):
    """Returns (pred_list, model_only_ms) where model_only_ms = CNN forward + InternalSysState+XGB forward (ms)."""
    global Model, InternalSysState, BoostTree

    raw_sys_data  = info['sys_data']
    raw_next_info = info['next_info']
    batch_size    = len(raw_next_info)

    # --- Build inputs (prep work; not counted in model_only_ms) ---
    rps_data          = _compose_sys_data_channel(raw_sys_data, 'rps', batch_size)
    replica_data      = _compose_sys_data_channel(raw_sys_data, 'replica', batch_size)
    cpu_limit_data    = _compose_sys_data_channel(raw_sys_data, 'cpu_limit', batch_size)
    cpu_usage_mean    = _compose_sys_data_channel(raw_sys_data, 'cpu_usage_mean', batch_size)
    rss_mean_data     = _compose_sys_data_channel(raw_sys_data, 'rss_mean', batch_size)
    cache_mem_mean    = _compose_sys_data_channel(raw_sys_data, 'cache_mem_mean', batch_size)

    sys_data = np.concatenate(
        (rps_data, replica_data, cpu_limit_data, cpu_usage_mean, rss_mean_data, cache_mem_mean),
        axis=1
    ).astype(np.float32)

    # e2e_lat (5 percentiles)
    e2e_lat = None
    for key in ['90.0', '95.0', '98.0', '99.0', '99.9']:
        assert len(raw_sys_data['e2e_lat'][key]) == CnnTimeSteps
        v = np.array(raw_sys_data['e2e_lat'][key], dtype=np.float32)
        e2e_lat = v if e2e_lat is None else np.vstack((e2e_lat, v))
    e2e_lat = e2e_lat.reshape([1, e2e_lat.shape[0], e2e_lat.shape[1]])
    lat_data = np.repeat(e2e_lat, repeats=batch_size, axis=0).astype(np.float32)

    # next_info tensors
    ncore_next = None
    ncore_next_k = None
    for i, proposal in enumerate(raw_next_info):
        ncore_proposal = None
        for j, service in enumerate(Services):
            val = np.array(proposal[service]['cpus'], dtype=np.float32)
            ncore_proposal = val if j == 0 else np.vstack((ncore_proposal, val))
        # build look-forward stack
        ncore_proposal_next_k = np.hstack([ncore_proposal.reshape([-1, 1])] * XgbLookForward)
        if i == 0:
            ncore_next   = ncore_proposal.reshape([1, ncore_proposal.shape[0]])
            ncore_next_k = ncore_proposal_next_k.reshape([1, ncore_proposal_next_k.shape[0], ncore_proposal_next_k.shape[1]])
        else:
            ncore_next   = np.vstack((ncore_next, ncore_proposal.reshape([1, ncore_proposal.shape[0]])))
            ncore_next_k = np.vstack((ncore_next_k, ncore_proposal_next_k.reshape([1, ncore_proposal_next_k.shape[0], ncore_proposal_next_k.shape[1]])))

    next_data  = ncore_next.astype(np.float32)
    next_k_data = ncore_next_k.astype(np.float32)

    pred_data = {'data1': sys_data, 'data2': lat_data, 'data3': next_data}
    pred_iter = mx.io.NDArrayIter(pred_data, batch_size=batch_size)

    # --- Model-only timing ---
    t7 = time.perf_counter()
    cnn_pred = Model.predict(pred_iter).asnumpy()
    t8 = time.perf_counter()

    # internal state + xgb
    internal_sys_state = InternalSysState.predict(pred_iter).asnumpy()
    next_k_info = next_k_data.reshape(next_k_data.shape[0], -1)
    xgb_input = np.concatenate((internal_sys_state, next_k_info), axis=1)
    dpred = xgb.DMatrix(xgb_input)
    xgb_predict = BoostTree.predict(dpred)
    t10 = time.perf_counter()

    model_only_ms = ((t8 - t7) + (t10 - t8)) * 1e3

    # pack prediction
    predict = []
    for i in range(0, batch_size):
        # keep 99% percentile from cnn_pred and xgb next-k prob
        t = [round(float(cnn_pred[i, -2]), 2), round(float(xgb_predict[i]), 3)]
        predict.append(t)

    return predict, model_only_ms


# -------------------- Benchmark --------------------

def run_benchmark():
    global Model, InternalSysState, BoostTree

    logging.info("Loading models...")
    kv   = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    sym, arg_params, aux_params = _load_model(kv.rank)
    all_layers = sym.get_internals()

    # Sizes
    cnn_model_size_mb = get_mxnet_model_size(arg_params, aux_params)
    print(f"CNN + InternalSysState model size: {cnn_model_size_mb:.2f} MB")

    # CNN head
    Model = all_layers['latency_output']
    Model = mx.mod.Module(
        context=devs,
        symbol=Model,
        data_names=('data1', 'data2', 'data3'),
        label_names=None
    )
    default_batch_size = args.batch_size
    Model.bind(for_training=False,
        data_shapes=[('data1', (default_batch_size, 6, 28, CnnTimeSteps)),
                     ('data2', (default_batch_size, 5, CnnTimeSteps)),
                     ('data3', (default_batch_size, 28))])
    Model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    # Internal features
    InternalSysState = all_layers['full_feature_output']
    InternalSysState = mx.mod.Module(
        context=devs,
        symbol=InternalSysState,
        data_names=('data1', 'data2', 'data3'),
        label_names=None
    )
    InternalSysState.bind(for_training=False,
        data_shapes=[('data1', (default_batch_size, 6, 28, CnnTimeSteps)),
                     ('data2', (default_batch_size, 5, CnnTimeSteps)),
                     ('data3', (default_batch_size, 28))])
    InternalSysState.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    # XGB
    BoostTree = xgb.Booster()
    print("load ", args.xgb_prefix + str(XgbLookForward) + '.model')
    BoostTree.load_model(args.xgb_prefix + str(XgbLookForward) + '.model')
    xgb_size_mb = get_xgb_model_size(BoostTree)
    print(f"XGBoost model size: {xgb_size_mb:.2f} MB")
    print(f"Total memory footprint of loaded models: {cnn_model_size_mb + xgb_size_mb:.2f} MB")

    # --- Build fixed synthetic batch (same as before) ---
    info = {}
    sys_data = {}
    sys_data['e2e_lat'] = {}
    for i, key in enumerate(['90.0', '95.0', '98.0', '99.0', '99.9']):
        sys_data['e2e_lat'][key] = [1.0 + i/10.0] * CnnTimeSteps

    for service in Services:
        sys_data[service] = {
            'rps': [50] * CnnTimeSteps,
            'cpu_limit': [12] * CnnTimeSteps,
            'replica': [10] * CnnTimeSteps,
            'cpu_usage_mean': [5.0] * CnnTimeSteps,
            'rss_mean': [1.0] * CnnTimeSteps,
            'cache_mem_mean': [0.0] * CnnTimeSteps
        }

    info['sys_data'] = sys_data
    next_info = []
    for _ in range(args.batch_size):
        proposal = {}
        for service in Services:
            proposal[service] = {'cpus': 12, 'rps': 50}
        next_info.append(proposal)
    info['next_info'] = next_info

    # --- Warm-up (excluded; triggers cuDNN autotune) ---
    _ = _predict(info)
    mx.nd.waitall()

    # --- Runs ---
    proc = psutil.Process(os.getpid())
    end2end_ms, modelonly_ms = [], []
    cpu_percents, cpu_norm = [], []
    peak_rss_mb_list = []

    for r in range(args.runs):
        t0 = time.perf_counter()
        cpu_times0 = proc.cpu_times()

        preds, mdl_ms = _predict(info)
        mx.nd.waitall()
        t_end = time.perf_counter()

        per_core, per_norm = cpu_percent_over_window(proc, t0, t_end, cpu_times0)
        end2end_ms.append((t_end - t0) * 1e3)
        modelonly_ms.append(mdl_ms)
        cpu_percents.append(per_core)
        cpu_norm.append(per_norm)
        peak_rss_mb_list.append(get_peak_rss_mb())

    # --- Report ---
    device_name = "gpu" if args.gpus and args.gpus.strip() not in ("", "None") else "cpu"
    print("\n=== CNN+XGB Benchmark ===")
    print(f"runs={args.runs}, batch={args.batch_size}, device={device_name}")
    print(f"End2End time   : mean {np.mean(end2end_ms):.2f} ms | p50 {pct(end2end_ms,50):.2f} | p95 {pct(end2end_ms,95):.2f}")
    print(f"Model-only time: mean {np.mean(modelonly_ms):.2f} ms | p50 {pct(modelonly_ms,50):.2f} | p95 {pct(modelonly_ms,95):.2f}")
    print(f"CPU% (per-core): mean {np.mean(cpu_percents):.1f}% | normalized {np.mean(cpu_norm):.1f}% of all cores")
    print(f"Peak RSS (host): mean {np.mean(peak_rss_mb_list):.1f} MB")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    run_benchmark()
