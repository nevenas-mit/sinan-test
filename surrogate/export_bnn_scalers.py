#!/usr/bin/env python3
"""
Export BNN scalers from train_bnn_with_cp.py output to the format expected by
social_media_predictor_bnn.py: separate scaler_sys.pkl, scaler_lat.pkl,
scaler_nxt.pkl, scaler_y.pkl and top_feature_indices.npy.

Usage:
  python export_bnn_scalers.py --scalers-pkl model/bnn_layers2_hdim800_lr1e-04_scalers.pkl --top-indices model/bnn_layers2_hdim800_lr1e-04_top_indices.npy --out-dir model
"""

import argparse
import os
import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export BNN scalers to predictor format")
    parser.add_argument("--scalers-pkl", type=str, required=True,
                        help="Path to *_scalers.pkl from train_bnn_with_cp.py")
    parser.add_argument("--top-indices", type=str, required=True,
                        help="Path to *_top_indices.npy from train_bnn_with_cp.py")
    parser.add_argument("--out-dir", type=str, default="model",
                        help="Directory to write scaler_sys.pkl, scaler_lat.pkl, scaler_nxt.pkl, scaler_y.pkl, top_feature_indices.npy")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load tuple (scaler_x_sys, scaler_x_lat, scaler_x_nxt, scaler_y)
    t = joblib.load(args.scalers_pkl)
    scaler_sys, scaler_lat, scaler_nxt, scaler_y = t

    joblib.dump(scaler_sys, os.path.join(args.out_dir, "scaler_sys.pkl"))
    joblib.dump(scaler_lat, os.path.join(args.out_dir, "scaler_lat.pkl"))
    joblib.dump(scaler_nxt, os.path.join(args.out_dir, "scaler_nxt.pkl"))
    joblib.dump(scaler_y, os.path.join(args.out_dir, "scaler_y.pkl"))
    print(f"Saved scaler_sys.pkl, scaler_lat.pkl, scaler_nxt.pkl, scaler_y.pkl to {args.out_dir}")

    top_indices = np.load(args.top_indices)
    out_path = os.path.join(args.out_dir, "top_feature_indices.npy")
    np.save(out_path, top_indices)
    print(f"Saved top_feature_indices.npy (shape {top_indices.shape}) to {args.out_dir}")


if __name__ == "__main__":
    main()
