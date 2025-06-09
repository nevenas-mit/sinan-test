import subprocess
import itertools
import os
import re

# === Define hyperparameter search space ===
LEARNING_RATES = [1e-4, 1e-3, 1e-2]
HIDDEN_DIMS = [100, 200, 400, 800]
NUM_LAYERS = [1, 2, 4, 8]
BATCH_SIZES = [1024]
EPOCHS = 30

DATA_DIR = "/home/jovans2/test_llms/sinan-curr/docker_swarm/logs/collected_data/dataset"

# === Results tracking ===
results = []

# === Iterate through all combinations ===
for lr, hdim, nlayer, bsz in itertools.product(LEARNING_RATES, HIDDEN_DIMS, NUM_LAYERS, BATCH_SIZES):
    print(f"Running: LR={lr}, HDIM={hdim}, NLAYER={nlayer}, BATCH={bsz}")
    try:
        result = subprocess.run([
            "python3", "train_bnn_explore.py",
            "--data-dir", DATA_DIR,
            "--lr", str(lr),
            "--hidden-dim", str(hdim),
            "--num-layers", str(nlayer),
            "--batch-size", str(bsz),
            "--epochs", str(EPOCHS)
        ], capture_output=True, text=True)

        # Extract final RMSE from stdout
        stdout = result.stdout
        print(stdout)
        rmse_match = re.search(r"Final Valid RMSE:\s*([\d.]+)", stdout)
        if rmse_match:
            val_rmse = float(rmse_match.group(1))
            results.append(((lr, hdim, nlayer, bsz), val_rmse))
            print(f"→ Valid RMSE: {val_rmse}")
        else:
            print("→ Failed to parse RMSE.")
            print(stdout)
            continue

    except Exception as e:
        print(f"Error running training: {e}")
        continue

# === Get best configuration ===
if results:
    best_config, best_rmse = min(results, key=lambda x: x[1])
    print("\n=== Best Configuration ===")
    print(f"LR={best_config[0]}, Hidden Dim={best_config[1]}, Num Layers={best_config[2]}, Batch Size={best_config[3]}")
    print(f"Validation RMSE: {best_rmse:.4f}")
else:
    print("No successful runs.")
