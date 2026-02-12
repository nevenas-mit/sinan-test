# BNN Surrogate Decision Tree — Summary

Summary of work done to train a surrogate decision tree that mimics the original Bayesian neural network (BNN), and how to test it.

---

## 1. What We Did (High Level)

- **Goal:** Replace BNN inference at runtime with a lightweight **decision tree** (or random forest) that reproduces the BNN's predictions, so the system can run without GPU and with lower latency.
- **Approach:** **Knowledge distillation** — use the BNN as a "teacher" to generate targets, then train a tree/forest on `(X, Y_bnn)`.
- **Scope:** Data generation from BNN, training script, surrogate predictor (same socket protocol as BNN), fidelity measurement, tree analysis, deployment config, and small data-parser fix.

The existing plan in `../DECISION_TREE_PLAN.md` describes replacing the older **CNN + XGBoost** pipeline with a single tree; the **actual implementation** is a **BNN → surrogate tree** pipeline (mimic the current BNN, not the legacy CNN+XGBoost).

---

## 2. Directory Structure

All surrogate-related files are now in this standalone `surrogate/` directory:

```
surrogate/
├── README.md                           # This file
├── data/
│   ├── X_surrogate_train.npy           # Scaled input features (train)
│   ├── X_surrogate_valid.npy           # Scaled input features (valid)
│   ├── Y_bnn_train.npy                 # BNN predictions / teacher targets (train)
│   └── Y_bnn_valid.npy                 # BNN predictions / teacher targets (valid)
├── model/
│   ├── bnn_surrogate_tree.joblib       # Trained surrogate tree/forest
│   ├── scaler_sys.pkl                  # Preprocessing scaler (system features)
│   ├── scaler_lat.pkl                  # Preprocessing scaler (latency features)
│   ├── scaler_nxt.pkl                  # Preprocessing scaler (next-k features)
│   ├── scaler_y.pkl                    # Output scaler
│   └── top_feature_indices.npy         # Feature indices used at inference
├── generate_bnn_surrogate_data.py      # Runs BNN on dataset; saves (X, Y_bnn) and scalers
├── train_bnn_surrogate_tree.py         # Trains tree/forest on (X, Y_bnn)
├── social_media_predictor_bnn_surrogate.py  # Socket predictor (same protocol as BNN)
├── measure_surrogate_fidelity.py       # Compares surrogate to Y_bnn (RMSE, R², etc.)
├── analyze_tree_leaf_depths.py         # Analyzes leaf-depth distribution
├── export_bnn_scalers.py               # Exports BNN scalers to model/
└── leaf_depth_distribution.png         # Plot from analysis script
```

---

## 3. Related Files Outside This Directory

| File | Description |
|------|-------------|
| `../DECISION_TREE_PLAN.md` | Original plan (CNN+XGBoost → tree); implementation is BNN→surrogate. |
| `../docker_swarm/config/predictor_bnn_surrogate.json` | Predictor config pointing to this directory. |
| `../docker_swarm/misc/make_gpu_config_bnn_surrogate.py` | Script to regenerate the predictor config. |
| `../ml_docker_swarm/train_bnn_explore.py` | BayesianMLP definition (imported by `generate_bnn_surrogate_data.py`). |

---

## 4. Modified Files (in original codebase)

| File | Change |
|------|--------|
| `../ml_docker_swarm/data_parser_socialml_next_k.py` | Replaced deprecated `dtype=np.float` with `dtype=np.float64` (NumPy compatibility). |
| `../docker_swarm/logs/collected_data/dataset/*.npy` | Regenerated/updated parsed data files. |

---

## 5. How to Test the Tree's Fidelity

Fidelity = how well the surrogate tree's predictions match the BNN's (teacher) predictions.

### Prerequisites

- Python 3 with `numpy`, `scikit-learn`, `joblib` (e.g. use `source ../.venv_surrogate/bin/activate` from this directory).
- Data files in `data/`:
  - `X_surrogate_train.npy`, `Y_bnn_train.npy` (required).
  - `X_surrogate_valid.npy`, `Y_bnn_valid.npy` (optional; for validation metrics).
- Trained model: `model/bnn_surrogate_tree.joblib`.

### Steps

1. **Activate environment and go to surrogate directory**
   ```bash
   cd /home/czarkos/sinan-test
   source .venv_surrogate/bin/activate
   cd surrogate
   ```

2. **Run fidelity script**
   ```bash
   python3 measure_surrogate_fidelity.py --split both
   ```
   - Default `--data-dir` is `data`.
   - Default `--model` is `model/bnn_surrogate_tree.joblib`.
   - `--split both` evaluates on both train and valid; use `train` or `valid` for one only.

3. **Interpret output**
   - **RMSE / MAE:** Lower is better (surrogate closer to BNN).
   - **R²:** Closer to 1.0 is better (surrogate explains BNN variance well).
   - **corr:** Correlation per output; high (>0.97) means surrogate tracks BNN closely.
   - Compare **Train** vs **Valid**: some drop on valid is normal; large drop may indicate overfitting.

### One-line test (from repo root)

```bash
cd /home/czarkos/sinan-test && source .venv_surrogate/bin/activate && cd surrogate && python3 measure_surrogate_fidelity.py --split both
```

### Last run (reference)

On the current artifacts, typical results were:

- **Train (n=24,633):** R² ≈ 0.994, correlation ≈ 0.997 (both outputs).
- **Valid (n=2,737):** R² ≈ 0.963, correlation ≈ 0.98 (both outputs).

So the current tree has high fidelity to the BNN on this data.

---

## 6. Pipeline Overview

```
[Parsed dataset: ../docker_swarm/logs/collected_data/dataset/sys/lat/nxt_*.npy]
         │
         ▼
generate_bnn_surrogate_data.py  (BNN .pth, --save-scalers)
         │
         ├── data/X_surrogate_train.npy, data/Y_bnn_train.npy
         ├── data/X_surrogate_valid.npy, data/Y_bnn_valid.npy
         └── model/scaler_*.pkl (if --save-scalers)
         │
         ▼
train_bnn_surrogate_tree.py  (--data-dir data --out-model model/bnn_surrogate_tree.joblib)
         │
         ▼
model/bnn_surrogate_tree.joblib
         │
         ▼
social_media_predictor_bnn_surrogate.py  (loads tree + scalers + top_feature_indices.npy)
         │
         ▼
Socket server (same protocol as BNN predictor); no GPU.
```

Fidelity is checked by comparing surrogate predictions on `data/X_surrogate_*` to `data/Y_bnn_*` with `measure_surrogate_fidelity.py`.

---

## 7. Important Notes

- **Feature set at inference:** The surrogate predictor uses `model/top_feature_indices.npy` and only passes those columns to the model. The tree in `model/bnn_surrogate_tree.joblib` must have been trained on the **same** feature set (same columns / same dimension). If you retrain from full-dimension `X_surrogate_*.npy`, either train on `X[:, top_indices]` or change the predictor to pass full X; otherwise input dimension will mismatch.
- **Scalers:** Surrogate data generation can save scalers with `--save-scalers`; alternatively they can come from the BNN training pipeline and be exported with `export_bnn_scalers.py`. The predictor expects `scaler_sys.pkl`, `scaler_lat.pkl`, `scaler_nxt.pkl`, `scaler_y.pkl` and `top_feature_indices.npy` in `model/`.
- **Deployment:** Use `../docker_swarm/config/predictor_bnn_surrogate.json` (and `make_gpu_config_bnn_surrogate.py` if regenerating) so the stack runs `social_media_predictor_bnn_surrogate.py` with no GPU.

---

## 8. Optional: Leaf Depth Analysis

To inspect how deep the tree goes on your data:

```bash
cd /home/czarkos/sinan-test/surrogate
python3 analyze_tree_leaf_depths.py --split valid --out-plot leaf_depth_distribution.png
```

This prints min/max/mean/median leaf depth and saves a bar plot of the leaf-depth distribution.

---

## 9. References

- **Plan (historical):** `../DECISION_TREE_PLAN.md` — CNN+XGBoost replacement plan; implementation is BNN surrogate instead.
- **BNN definition:** `../ml_docker_swarm/train_bnn_explore.py` (e.g. `BayesianMLP`) is used by `generate_bnn_surrogate_data.py` to load the teacher checkpoint.
