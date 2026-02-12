#!/usr/bin/env python3
import os
import argparse
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
import time
import psutil

# ---------------------------
# Helpers
# ---------------------------

def load_split(dirpath):
    sys_x = np.load(os.path.join(dirpath, "sys_data_valid.npy"))
    lat_x = np.load(os.path.join(dirpath, "lat_data_valid.npy"))
    nxt_x = np.squeeze(np.load(os.path.join(dirpath, "nxt_k_data_valid.npy"))[:, :, 0])
    y     = np.squeeze(np.load(os.path.join(dirpath, "nxt_k_valid_label.npy"))[:, :, 0])
    d, k = 505, 0.01
    y = np.where(y < d, y, d + (y - d) / (1.0 + k * (y - d)))
    return sys_x, lat_x, nxt_x, y

def build_iter(sys_x, lat_x, nxt_x, y=None, batch_size=2048, shuffle=False):
    data = {'data1': sys_x, 'data2': lat_x, 'data3': nxt_x}
    label = None if y is None else {'label': y}
    return mx.io.NDArrayIter(data=data, label=label, batch_size=batch_size, shuffle=shuffle)

def forward_get(mod, data_iter, profile=False):
    preds, feats, labels = [], [], []
    data_iter.reset()

    if profile:
        process = psutil.Process(os.getpid())
        cpu_samples, mem_samples = [], []
        start_time = time.time()


    for batch in data_iter:

        if profile:
            # Sample CPU + memory before each forward
            cpu_samples.append(process.cpu_percent(interval=None))
            mem_samples.append(process.memory_info().rss / (1024**2))  # MB

        mod.forward(batch, is_train=False)
        outs = mod.get_outputs()
        yhat = outs[0].asnumpy()
        pen  = outs[1].asnumpy()
        preds.append(yhat)
        feats.append(pen)
        if batch.label:
            labels.append(batch.label[0].asnumpy())
    
    if profile:
        end_time = time.time()
        inference_time = end_time - start_time
        avg_cpu = np.mean(cpu_samples)
        peak_mem = np.max(mem_samples)
        print(f"\n--- Inference Profiling ---")
        print(f"Total inference time: {inference_time:.4f} sec")
        print(f"Average CPU utilization: {avg_cpu:.2f}%")
        print(f"Peak memory consumption: {peak_mem:.2f} MB")
        print("----------------------------\n")

    preds = np.concatenate(preds, axis=0).reshape(-1)
    feats = np.concatenate(feats, axis=0)
    labels = None if not labels else np.concatenate(labels, axis=0).reshape(-1)
    return preds, feats, labels

def fit_mahalanobis(feats_train):
    mu = feats_train.mean(axis=0)
    cov = np.cov(feats_train, rowvar=False)
    eps = 1e-6
    cov_inv = np.linalg.inv(cov + eps * np.eye(cov.shape[0]))
    return mu, cov, cov_inv

def mahalanobis(dist_x, mu, cov_inv):
    diff = dist_x - mu
    return np.sqrt(np.einsum('ij,jk,ik->i', diff, cov_inv, diff))

def reliability_curve(x, y, nbins=10, metric="rmse"):
    x = np.asarray(x); y = np.asarray(y)
    assert len(x) == len(y), f"Length mismatch: x={len(x)}, y={len(y)}"
    q = np.quantile(x, np.linspace(0, 1, nbins + 1))
    centers, vals, counts = [], [], []
    for b in range(nbins):
        lo, hi = q[b], q[b + 1]
        mask = (x >= lo) & (x <= hi if b == nbins - 1 else x < hi)
        if np.any(mask):
            centers.append(0.5 * (lo + hi))
            if metric == "rmse":
                vals.append(np.sqrt(np.mean(y[mask] ** 2)))
            else:
                vals.append(y[mask].mean())
            counts.append(mask.sum())
    return np.array(centers), np.array(vals), np.array(counts)

def plot_reliability(centers, mean_err, out_path):
    plt.figure(figsize=(9, 5))
    plt.plot(centers, mean_err, marker='o', linewidth=3)
    plt.xlabel("Mahalanobis Distance (to training feature distribution)", fontsize=18)
    plt.ylabel("RMSE", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved: {out_path}")

# ---------------------------
# Synthetic OOD generator
# ---------------------------

def chol_psd(cov):
    eps = 1e-8
    try:
        return np.linalg.cholesky(cov + eps * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, a_min=1e-12, a_max=None)
        return V @ np.diag(np.sqrt(w)) @ V.T

def synth_feats_from_train(feats_id, mu, cov, target_radii, per_radius=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(1337)
    L = chol_psd(cov)
    D = feats_id.shape[1]
    synth, tags = [], []
    for r in target_radii:
        Z = rng.normal(size=(per_radius, D))
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        delta = (Z * r) @ L.T
        Xr = mu + delta
        synth.append(Xr)
        tags.extend([r] * per_radius)
    synth = np.vstack(synth)
    tags  = np.array(tags)
    return synth, tags

def simulate_ground_truth(yhat, radii, drift_alpha=0.15, rng=None):
    if rng is None:
        rng = np.random.default_rng(4242)
    signs = rng.choice([-1.0, 1.0], size=yhat.shape[0])
    return yhat + drift_alpha * radii * signs

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--id-dir", required=True)
    parser.add_argument("--ood-dir", default=None)
    parser.add_argument("--model-prefix", default="./model/cnv")
    parser.add_argument("--load-epoch", type=int, default=200)
    parser.add_argument("--penult-name", default="fc1_output")
    parser.add_argument("--latency-name", default="latency_output")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--out", default="reliability.png")
    parser.add_argument("--synthetic-radii", default="0.5,1,1.5,2,3")
    parser.add_argument("--synthetic-per-radius", type=int, default=1500)
    parser.add_argument("--drift-alpha", type=float, default=0.15)
    parser.add_argument("--profile", action="store_true", help="Enable profiling (inference time, CPU, memory)")
    args = parser.parse_args()

    sym, arg_params, aux_params = mx.model.load_checkpoint(args.model_prefix, args.load_epoch)
    internals = sym.get_internals()
    group_sym = mx.sym.Group([internals[args.latency_name], internals[args.penult_name]])

    mod = mx.mod.Module(symbol=group_sym, context=[mx.cpu()],
                        data_names=('data1','data2','data3'), label_names=None)

    # ID set
    sys_id, lat_id, nxt_id, y_id = load_split(args.id_dir)
    it_id = build_iter(sys_id, lat_id, nxt_id, y_id, batch_size=args.batch_size, shuffle=False)
    mod.bind(data_shapes=it_id.provide_data, for_training=False)
    mod.set_params(arg_params, aux_params, allow_missing=False, allow_extra=False)
    preds_id, feats_id, labels_id = forward_get(mod, it_id, profile=args.profile)
    n_id = min(len(preds_id), len(labels_id), feats_id.shape[0])
    preds_id, feats_id, labels_id = preds_id[:n_id], feats_id[:n_id], labels_id[:n_id]

    # Train feats for Mahalanobis
    sys_tr, lat_tr, nxt_tr, y_tr = load_split(args.train_dir)
    it_tr = build_iter(sys_tr, lat_tr, nxt_tr, y_tr, batch_size=args.batch_size, shuffle=False)
    _, feats_tr, _ = forward_get(mod, it_tr)
    mu, cov, cov_inv = fit_mahalanobis(feats_tr)

    dist_id = mahalanobis(feats_id, mu, cov_inv)
    abs_err_id = np.abs(preds_id - labels_id)

    if args.ood_dir:
        sys_ood, lat_ood, nxt_ood, y_ood = load_split(args.ood_dir)
        it_ood = build_iter(sys_ood, lat_ood, nxt_ood, y_ood, batch_size=args.batch_size, shuffle=False)
        preds_ood, feats_ood, labels_ood = forward_get(mod, it_ood)
        n_ood = min(len(preds_ood), len(labels_ood), feats_ood.shape[0])
        preds_ood, feats_ood, labels_ood = preds_ood[:n_ood], feats_ood[:n_ood], labels_ood[:n_ood]
        dist_ood = mahalanobis(feats_ood, mu, cov_inv)
        abs_err_ood = np.abs(preds_ood - labels_ood)
        D = np.concatenate([dist_id, dist_ood])
        E = np.concatenate([abs_err_id, abs_err_ood])
    else:
        radii = np.array([float(x) for x in args.synthetic_radii.split(",")])
        synth_feats, synth_r = synth_feats_from_train(feats_id, mu, cov, radii, per_radius=args.synthetic_per_radius)
        n_fit = min(feats_id.shape[0], preds_id.shape[0])
        F = np.hstack([feats_id[:n_fit], np.ones((n_fit, 1))])
        preds_fit = preds_id[:n_fit]
        w, *_ = np.linalg.lstsq(F, preds_fit, rcond=None)
        Fs = np.hstack([synth_feats, np.ones((synth_feats.shape[0], 1))])
        preds_synth = Fs @ w
        y_synth = simulate_ground_truth(preds_synth, synth_r, drift_alpha=args.drift_alpha)
        abs_err_synth = np.abs(preds_synth - y_synth)
        dist_synth = mahalanobis(synth_feats, mu, cov_inv)
        D = np.concatenate([dist_id, dist_synth])
        E = np.concatenate([abs_err_id, abs_err_synth])

    assert len(D) == len(E), f"Final mismatch: D={len(D)}, E={len(E)}"
    centers, mean_err, counts = reliability_curve(D, E, nbins=args.bins)
    print("Bin centers:", centers)
    print("Mean abs error per bin:", mean_err)
    print("Counts per bin:", counts)
    plot_reliability(centers, mean_err, args.out)

if __name__ == "__main__":
    main()
