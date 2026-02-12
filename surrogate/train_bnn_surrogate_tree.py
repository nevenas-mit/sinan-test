#!/usr/bin/env python3
"""
Train a surrogate decision tree (or random forest) on (X, Y_bnn) produced by
generate_bnn_surrogate_data.py. Saves a joblib model for use by the surrogate predictor.

Usage:
  python train_bnn_surrogate_tree.py --data-dir . --out-model model/bnn_surrogate_tree.joblib
  python train_bnn_surrogate_tree.py --data-dir . --model-type forest --out-model model/bnn_surrogate_forest.joblib
"""

import argparse
import os
import numpy as np
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing X_surrogate_train.npy, Y_bnn_train.npy (and valid if present)")
    parser.add_argument("--out-model", type=str, default="model/bnn_surrogate_tree.joblib",
                        help="Path to save the trained model (joblib)")
    parser.add_argument("--model-type", type=str, default="tree", choices=["tree", "forest"],
                        help="tree = DecisionTreeRegressor, forest = RandomForestRegressor")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="Max depth (default: None for tree, 20 for forest)")
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=100,
                        help="Number of trees (forest only)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    x_train = np.load(os.path.join(args.data_dir, "X_surrogate_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "Y_bnn_train.npy"))
    print(f"Train X: {x_train.shape}, Y: {y_train.shape}")

    if args.model_type == "tree":
        model = DecisionTreeRegressor(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth is not None else 20,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1,
        )

    model.fit(x_train, y_train)

    # Validation if present
    x_valid_path = os.path.join(args.data_dir, "X_surrogate_valid.npy")
    y_valid_path = os.path.join(args.data_dir, "Y_bnn_valid.npy")
    if os.path.isfile(x_valid_path) and os.path.isfile(y_valid_path):
        x_valid = np.load(x_valid_path)
        y_valid = np.load(y_valid_path)
        pred_valid = model.predict(x_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, pred_valid))
        print(f"Validation RMSE (all outputs): {rmse:.4f}")
        if y_valid.shape[1] >= 2:
            rmse_0 = np.sqrt(mean_squared_error(y_valid[:, 0], pred_valid[:, 0]))
            rmse_1 = np.sqrt(mean_squared_error(y_valid[:, 1], pred_valid[:, 1]))
            print(f"  Output 0 (pred_lat): RMSE = {rmse_0:.4f}")
            print(f"  Output 1 (pred_viol): RMSE = {rmse_1:.4f}")

    os.makedirs(os.path.dirname(args.out_model) or ".", exist_ok=True)
    joblib.dump(model, args.out_model)
    print(f"Model saved to {args.out_model}")


if __name__ == "__main__":
    main()
