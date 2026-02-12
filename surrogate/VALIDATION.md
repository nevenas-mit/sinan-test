# Surrogate Model Validation Steps

This document provides steps to independently verify that the surrogate decision tree is correctly trained and has genuine fidelity to the BNN.

---

## Prerequisites

```bash
cd /home/czarkos/sinan-test
source .venv_surrogate/bin/activate
cd surrogate
```

---

## 1. Run the Fidelity Test

Reproduce the fidelity metrics independently:

```bash
python3 measure_surrogate_fidelity.py --split both
```

**Expected output:**
- Train R² ≈ 0.99, Valid R² ≈ 0.96
- Correlations > 0.97 for both outputs

---

## 2. Verify Data Shapes and No Train/Valid Overlap

```bash
python3 -c "
import numpy as np

X_train = np.load('data/X_surrogate_train.npy')
Y_train = np.load('data/Y_bnn_train.npy')
X_valid = np.load('data/X_surrogate_valid.npy')
Y_valid = np.load('data/Y_bnn_valid.npy')

print(f'X_train: {X_train.shape}, Y_train: {Y_train.shape}')
print(f'X_valid: {X_valid.shape}, Y_valid: {Y_valid.shape}')
print(f'Train samples: {len(X_train)}, Valid samples: {len(X_valid)}')
print(f'No overlap check: {len(set(map(tuple, X_train[:100])) & set(map(tuple, X_valid[:100])))} common rows (first 100)')
"
```

**Expected:**
- Train ~24k samples, Valid ~2.7k samples
- Same number of features (columns)
- Y has 2 output columns
- 0 common rows between train and valid

---

## 3. Spot-Check: Compare Individual Predictions

```bash
python3 -c "
import numpy as np
import joblib

model = joblib.load('model/bnn_surrogate_tree.joblib')
X_valid = np.load('data/X_surrogate_valid.npy')
Y_valid = np.load('data/Y_bnn_valid.npy')  # BNN ground truth

# Predict on first 5 validation samples
preds = model.predict(X_valid[:5])

print('Sample | Y_bnn (actual)         | Y_pred (surrogate)     | Diff')
print('-' * 70)
for i in range(5):
    diff = np.abs(Y_valid[i] - preds[i])
    print(f'{i:5d}  | {Y_valid[i]}  | {preds[i]}  | {diff}')
"
```

**Expected:** Small differences between actual and predicted values.

---

## 4. Visual Sanity Check: Scatter Plot

```bash
python3 -c "
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load('model/bnn_surrogate_tree.joblib')
X_valid = np.load('data/X_surrogate_valid.npy')
Y_valid = np.load('data/Y_bnn_valid.npy')

preds = model.predict(X_valid)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, name in enumerate(['pred_lat', 'pred_viol']):
    ax = axes[i]
    ax.scatter(Y_valid[:, i], preds[:, i], alpha=0.3, s=5)
    ax.plot([Y_valid[:, i].min(), Y_valid[:, i].max()],
            [Y_valid[:, i].min(), Y_valid[:, i].max()], 'r--', label='perfect')
    ax.set_xlabel(f'BNN {name}')
    ax.set_ylabel(f'Surrogate {name}')
    ax.set_title(f'{name} (valid set)')
    ax.legend()
plt.tight_layout()
plt.savefig('verification_scatter.png', dpi=150)
print('Saved verification_scatter.png')
"
```

**Expected:** Points cluster tightly around the red diagonal line (y=x). Random scatter would indicate the model isn't learning.

---

## 5. Negative Control: Shuffled Labels Should Fail

```bash
python3 -c "
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

model = joblib.load('model/bnn_surrogate_tree.joblib')
X_valid = np.load('data/X_surrogate_valid.npy')
Y_valid = np.load('data/Y_bnn_valid.npy')

preds = model.predict(X_valid)
rmse_real = np.sqrt(mean_squared_error(Y_valid, preds))

# Shuffle Y_valid - should give much worse RMSE
np.random.seed(42)
Y_shuffled = Y_valid.copy()
np.random.shuffle(Y_shuffled)
rmse_shuffled = np.sqrt(mean_squared_error(Y_shuffled, preds))

print(f'RMSE (real labels):     {rmse_real:.4f}')
print(f'RMSE (shuffled labels): {rmse_shuffled:.4f}')
print(f'Ratio: {rmse_shuffled/rmse_real:.1f}x worse with random labels')
"
```

**Expected:** Shuffled labels give 5-10x worse RMSE. If similar, something is wrong.

---

## 6. Baseline Check: Model Beats Mean Prediction

```bash
python3 -c "
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

model = joblib.load('model/bnn_surrogate_tree.joblib')
X_valid = np.load('data/X_surrogate_valid.npy')
Y_valid = np.load('data/Y_bnn_valid.npy')
Y_train = np.load('data/Y_bnn_train.npy')

preds = model.predict(X_valid)
rmse_model = np.sqrt(mean_squared_error(Y_valid, preds))

# Baseline: just predict mean of training set
mean_pred = np.tile(Y_train.mean(axis=0), (len(X_valid), 1))
rmse_mean = np.sqrt(mean_squared_error(Y_valid, mean_pred))

print(f'RMSE (tree model): {rmse_model:.4f}')
print(f'RMSE (mean only):  {rmse_mean:.4f}')
print(f'Model is {rmse_mean/rmse_model:.1f}x better than mean baseline')
"
```

**Expected:** Tree model significantly outperforms the mean baseline (at least 2-3x better).

---

## 7. Check Tree Structure

```bash
python3 -c "
import joblib

model = joblib.load('model/bnn_surrogate_tree.joblib')

print(f'Model type: {type(model).__name__}')
print(f'Max depth: {model.get_depth()}')
print(f'Number of leaves: {model.get_n_leaves()}')
print(f'Number of features: {model.n_features_in_}')
print(f'Number of outputs: {model.n_outputs_}')
"
```

**Expected:** 
- Model type is DecisionTreeRegressor (or RandomForestRegressor)
- Reasonable depth (e.g., 10-50)
- Multiple leaves (not trivial tree)
- n_outputs_ = 2 (pred_lat, pred_viol)

---

## Summary Checklist

| Check | What It Validates |
|-------|-------------------|
| 1. Fidelity test | Overall accuracy metrics |
| 2. Data shapes | Correct dimensions, no leakage |
| 3. Spot-check | Individual predictions make sense |
| 4. Scatter plot | Visual confirmation of correlation |
| 5. Shuffled labels | Model learned real patterns, not noise |
| 6. Mean baseline | Model is non-trivial |
| 7. Tree structure | Model architecture is reasonable |

If all checks pass, the surrogate model is correctly trained and has genuine fidelity to the BNN.
