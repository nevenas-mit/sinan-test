#!/usr/bin/env python3
"""
Measure fidelity of the surrogate decision tree vs the BNN.

Uses pre-generated surrogate data where Y_bnn is the BNN mean prediction (teacher).
Loads the surrogate model, predicts on X, and compares to Y_bnn.

Usage:
  python measure_surrogate_fidelity.py --data-dir . --model model/bnn_surrogate_tree.joblib
  python measure_surrogate_fidelity.py --data-dir . --model model/bnn_surrogate_forest.joblib --split valid
"""

import argparse
import os
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    parser = argparse.ArgumentParser(description="Measure surrogate vs BNN fidelity")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory with X_surrogate_*.npy and Y_bnn_*.npy")
    parser.add_argument("--model", type=str, default="model/bnn_surrogate_tree.joblib",
                        help="Path to surrogate model (joblib)")
    parser.add_argument("--split", type=str, default="both", choices=["train", "valid", "both"],
                        help="Which split(s) to evaluate")
    args = parser.parse_args()

    model = joblib.load(args.model)

    def evaluate(x_path, y_path, name):
        if not os.path.isfile(x_path) or not os.path.isfile(y_path):
            print(f"[{name}] Skipping: missing {x_path} or {y_path}")
            return
        X = np.load(x_path)
        Y_bnn = np.load(y_path)
        Y_pred = model.predict(X)
        if Y_pred.ndim == 1:
            Y_pred = Y_pred.reshape(-1, 1)

        n_out = Y_bnn.shape[1]
        assert Y_pred.shape == Y_bnn.shape, f"Shape mismatch pred {Y_pred.shape} vs bnn {Y_bnn.shape}"

        rmse_all = np.sqrt(mean_squared_error(Y_bnn, Y_pred))
        mae_all = mean_absolute_error(Y_bnn, Y_pred)
        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((Y_bnn - Y_pred) ** 2)
        ss_tot = np.sum((Y_bnn - np.mean(Y_bnn, axis=0)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

        print(f"\n--- {name} (n={X.shape[0]}) ---")
        print(f"  RMSE (all outputs): {rmse_all:.4f}")
        print(f"  MAE  (all outputs): {mae_all:.4f}")
        print(f"  R²   (all outputs): {r2:.4f}")

        for j in range(n_out):
            rmse_j = np.sqrt(mean_squared_error(Y_bnn[:, j], Y_pred[:, j]))
            mae_j = mean_absolute_error(Y_bnn[:, j], Y_pred[:, j])
            ss_res_j = np.sum((Y_bnn[:, j] - Y_pred[:, j]) ** 2)
            ss_tot_j = np.sum((Y_bnn[:, j] - np.mean(Y_bnn[:, j])) ** 2)
            r2_j = 1.0 - (ss_res_j / ss_tot_j) if ss_tot_j > 0 else float("nan")
            corr = np.corrcoef(Y_bnn[:, j], Y_pred[:, j])[0, 1] if np.std(Y_bnn[:, j]) > 0 else float("nan")
            label = "pred_lat" if j == 0 else ("pred_viol" if j == 1 else f"out_{j}")
            print(f"  Output {j} ({label}): RMSE={rmse_j:.4f}  MAE={mae_j:.4f}  R²={r2_j:.4f}  corr={corr:.4f}")

    data_dir = args.data_dir
    if args.split in ("train", "both"):
        evaluate(
            os.path.join(data_dir, "X_surrogate_train.npy"),
            os.path.join(data_dir, "Y_bnn_train.npy"),
            "Train",
        )
    if args.split in ("valid", "both"):
        evaluate(
            os.path.join(data_dir, "X_surrogate_valid.npy"),
            os.path.join(data_dir, "Y_bnn_valid.npy"),
            "Valid",
        )

    print()


if __name__ == "__main__":
    main()
