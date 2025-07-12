import subprocess
import itertools
import os
import matplotlib.pyplot as plt
import pandas as pd
import re

# === Define hyperparameter search space ===
LEARNING_RATES = [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
HIDDEN_DIMS = [100, 200, 300, 400, 500, 600, 700, 800]
NUM_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]
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

if results:
    # Unpack results
    configs, rmses = zip(*results)
    labels = [f"lr={c[0]}\nhdim={c[1]}\nnlayer={c[2]}" for c in configs]

    # === Save to CSV ===
    df_results = pd.DataFrame({
        'learning_rate': [c[0] for c in configs],
        'hidden_dim': [c[1] for c in configs],
        'num_layers': [c[2] for c in configs],
        'batch_size': [c[3] for c in configs],
        'validation_rmse': rmses
    })
    df_results.to_csv('grid_search_results_top100.csv', index=False)
    print("Results saved to grid_search_results_top100.csv")

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(rmses)), rmses, color='blue')

    plt.xticks(range(len(rmses)), labels, rotation=45, ha='right', fontsize=8)
    plt.ylabel("Validation RMSE")
    plt.xlabel("Hyperparameter Config")
    plt.title("Hyperparameter Search: Validation RMSE per Config")
    plt.ylim(top=1.0)
    plt.tight_layout()
    plt.grid(True)

    plt.savefig("grid_search_rmse_plot_top100.png")
    plt.show()

    print("Plot saved to grid_search_rmse_plot_top100.png")
else:
    print("No successful runs.")

