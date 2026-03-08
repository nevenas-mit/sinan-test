# ============================ xgb_train_latent.py ============================
# python xgb_train_latent.py --gpus 1 --data-dir ./swarm_data_next_5s_upsample/
# multiclass classification (actually binary:logistic in this script)
import os
import time
import math
import json
import argparse

import mxnet as mx
import xgboost as xgb
import numpy as np

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


def _load_model(args, rank=0):
    if (not hasattr(args, "load_epoch")) or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)


TimeSteps = 5
QoS = 500.0


def _conservative_quantile(scores_1d, alpha):
    """
    Same 'ceil((n+1)(1-alpha))' conservative quantile you used in BNN CP.
    """
    s = np.sort(np.asarray(scores_1d, dtype=float))
    n = s.shape[0]
    if n == 0:
        return 1.0
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(s[k - 1])


def cp_calibrate_binary_prob(p_cal, y_cal, alpha):
    """
    Split conformal for binary probabilistic classifier.

    score_i = 1 - p_true_label
      where p_true_label = p if y=1 else (1-p)

    qhat = Quantile(score, ceil((n+1)(1-alpha)))

    At test, include label y in prediction set if p_y >= 1 - qhat.
    For binary:
      include1 iff p >= 1-qhat
      include0 iff p <= qhat
      uncertain iff both included (qhat <= p <= 1-qhat)
    """
    p_cal = np.asarray(p_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=int)
    p_true = np.where(y_cal == 1, p_cal, 1.0 - p_cal)
    scores = 1.0 - p_true
    qhat = _conservative_quantile(scores, alpha)
    return qhat


def main(args):
    global QoS
    look_forward = args.look_forward

    mx.random.seed(2333)
    np.random.seed(2333)

    data_dir = args.data_dir

    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]

    load_params = _load_model(args, kv.rank)
    sym = load_params[0]
    if sym is None:
        raise RuntimeError("Failed to load MXNet checkpoint. Check --model-prefix and --load-epoch.")

    # model's internal representation
    all_layers = sym.get_internals()
    sym_sys_state = all_layers['full_feature_output']

    model_sys_state = mx.mod.Module(
        context=devs,
        symbol=sym_sys_state,
        data_names=('data1', 'data2', 'data3')
    )

    batch_size = args.batch_size
    model_sys_state.bind(
        for_training=False,
        data_shapes=[('data1', (batch_size, 6, len(Services), TimeSteps)),
                     ('data2', (batch_size, 5, TimeSteps)),
                     ('data3', (batch_size, len(Services)))]
    )
    model_sys_state.set_params(load_params[1], load_params[2], allow_missing=True, allow_extra=True)

    # ---------------------- training data ---------------------- #
    sys_data_t = np.load(os.path.join(data_dir, 'sys_data_train.npy'))
    lat_data_t = np.load(os.path.join(data_dir, 'lat_data_train.npy'))

    nxt_data_t = np.squeeze(np.load(os.path.join(data_dir, 'nxt_k_data_train.npy'))[:, :, 0])
    nxt_k_data_t = np.load(os.path.join(data_dir, 'nxt_k_data_train.npy'))[:, :, 1:]
    nxt_k_data_t = nxt_k_data_t.reshape(nxt_k_data_t.shape[0], -1)

    label_t = np.load(os.path.join(data_dir, 'nxt_k_train_label.npy'))
    label_t = np.squeeze(label_t[:, :, 0])  # only keep immediate future

    train_data = {'data1': sys_data_t, 'data2': lat_data_t, 'data3': nxt_data_t}
    train_label = {'label': label_t}

    train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size)
    internal_rep_train = model_sys_state.predict(train_iter).asnumpy()

    # only keep 99% percentile of far future (excluding immediate future)
    label_nxt_t = np.load(os.path.join(data_dir, 'nxt_k_train_label.npy'))[:, -2, 1:]
    label_nxt_t = np.squeeze(label_nxt_t)
    label_nxt_t = np.greater_equal(label_nxt_t, QoS)

    if look_forward > 1:
        label_nxt_t = np.sum(label_nxt_t, axis=1)
    final_label_t = np.greater_equal(label_nxt_t, 1)

    X_train = np.concatenate((internal_rep_train, nxt_k_data_t), axis=1)
    y_train = final_label_t.astype(int)

    print('X_train.shape = ', X_train.shape)
    print('y_train.shape = ', y_train.shape, 'pos_rate=', np.mean(y_train))

    # ---------------------- validation data ---------------------- #
    sys_data_v = np.load(os.path.join(data_dir, 'sys_data_valid.npy'))
    lat_data_v = np.load(os.path.join(data_dir, 'lat_data_valid.npy'))

    nxt_data_v = np.squeeze(np.load(os.path.join(data_dir, 'nxt_k_data_valid.npy'))[:, :, 0])
    nxt_k_data_v = np.load(os.path.join(data_dir, 'nxt_k_data_valid.npy'))[:, :, 1:]
    nxt_k_data_v = nxt_k_data_v.reshape(nxt_k_data_v.shape[0], -1)

    label_v = np.load(os.path.join(data_dir, 'nxt_k_valid_label.npy'))
    label_v = np.squeeze(label_v[:, :, 0])  # only keep immediate future

    valid_data = {'data1': sys_data_v, 'data2': lat_data_v, 'data3': nxt_data_v}
    valid_label = {'label': label_v}

    valid_iter = mx.io.NDArrayIter(valid_data, valid_label, batch_size=batch_size)
    internal_rep_valid = model_sys_state.predict(valid_iter).asnumpy()

    label_nxt_v = np.load(os.path.join(data_dir, 'nxt_k_valid_label.npy'))[:, -2, 1:]
    label_nxt_v = np.squeeze(label_nxt_v)
    label_nxt_v = np.greater_equal(label_nxt_v, QoS)

    if look_forward > 1:
        label_nxt_v = np.sum(label_nxt_v, axis=1)
    final_label_v = np.greater_equal(label_nxt_v, 1)

    X_valid = np.concatenate((internal_rep_valid, nxt_k_data_v), axis=1)
    y_valid = final_label_v.astype(int)

    print('X_valid.shape = ', X_valid.shape)
    print('y_valid.shape = ', y_valid.shape, 'pos_rate=', np.mean(y_valid))

    # ---------------------- CP split from validation ---------------------- #
    n = X_valid.shape[0]
    n_cal = int(max(1, round(n * args.cp_cal_frac)))
    if n_cal >= n:
        n_cal = max(1, n - 1)

    X_cal, y_cal = X_valid[:n_cal], y_valid[:n_cal]
    X_eval, y_eval = X_valid[n_cal:], y_valid[n_cal:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dcal = xgb.DMatrix(X_cal, label=y_cal)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    # ---------------------- Train model ---------------------- #
    progress = dict()
    watchlist = [(dtrain, 'train-err'), (deval, 'eval-err')]

    params = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eval_metric': 'error',
        'feature_selector': 'greedy',
        'eta': 0.01,
        'max_depth': 6,
        'tree_method': 'gpu_exact',  # keep your original
        'gamma': 0.0,
        'grow_policy': 'lossguide'
    }

    tmp = time.time()
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=watchlist,
        evals_result=progress,
        early_stopping_rounds=50
    )
    print("GPU Training Time: %s seconds" % (str(time.time() - tmp)))

    # ---------------------- Simple threshold reports on eval split ---------------------- #
    ypred = bst.predict(deval)

    for thr in [0.10, 0.25]:
        binary_ypred = np.greater(ypred, thr * np.ones_like(ypred))
        print(f'\n{thr:.2f} threshold (eval split)')
        print('pred_pos_rate=', np.mean(binary_ypred), 'true_pos_rate=', np.mean(y_eval))
        print('false positive = ', np.sum((1 - y_eval) * binary_ypred) * 1.0 / y_eval.shape[0])
        print('false negative = ', np.sum((1 - binary_ypred) * y_eval) * 1.0 / y_eval.shape[0])

    # ---------------------- Conformal calibration ---------------------- #
    p_cal = bst.predict(dcal)
    qhat = cp_calibrate_binary_prob(p_cal, y_cal, alpha=args.cp_alpha)
    print("\nCP calibration:")
    print("alpha =", args.cp_alpha, "qhat =", qhat, "n_cal =", len(y_cal))
    print("Uncertain region is p in [qhat, 1-qhat] =", (qhat, 1.0 - qhat))

    # ---------------------- Save model + CP stats ---------------------- #
    if not os.path.isdir(args.xgb_out_dir):
        os.makedirs(args.xgb_out_dir, exist_ok=True)

    model_path = os.path.join(args.xgb_out_dir, f'social_nn_sys_state_look_forward_{look_forward}.model')
    bst.save_model(model_path)
    print("Saved XGB model to:", model_path)

    cp_path = os.path.join(args.xgb_out_dir, f'cp_xgb_binary_lf{look_forward}_alpha{args.cp_alpha:.3f}.json')
    with open(cp_path, 'w') as f:
        json.dump({
            "type": "split_conformal_classification",
            "objective": "binary:logistic",
            "look_forward": int(look_forward),
            "alpha": float(args.cp_alpha),
            "cal_frac": float(args.cp_cal_frac),
            "qhat": float(qhat),
            "n_cal": int(len(y_cal))
        }, f)
    print("Saved CP stats to:", cp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--look-forward', type=int, default=4)
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--kv-store', type=str, default='local', help='the kvstore type')
    parser.add_argument('--batch-size', type=int, default=2048, help='the batch size')
    parser.add_argument('--network', type=str, default='cnvnet')
    parser.add_argument('--model-prefix', type=str, default='./model/cnv')
    parser.add_argument('--load-epoch', type=int, default=200)

    # CP args
    parser.add_argument('--cp-alpha', type=float, default=0.10)
    parser.add_argument('--cp-cal-frac', type=float, default=0.30)

    # output dir
    parser.add_argument('--xgb-out-dir', type=str, default='./xgb_model')

    args = parser.parse_args()
    main(args)
