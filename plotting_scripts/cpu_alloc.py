import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 9)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

# Data

data = {'AutoScaleOpt': [45.5, 53.375, 60.375, 61.25, 63.875, 65.625, 67.375, 71.75, 84.0], 
        'AutoScaleCons': [61.25, 72.625, 99.75, 108.5, 138.25, 154.0, 165.375, 186.375, 193.375], 
        'CNN+XGBoost': [34.125, 42.0, 51.625, 54.25, 56.0, 56.875, 63.0, 65.625, 69.125], 
        'CNN+XGBoost+EucDist': [35.875, 42.0, 40.25, 50.75, 56.0, 57.75, 64.75, 68.25, 68.25], 
        'CNN+XGBoost+CP': [32.375, 40.25, 49.875, 51.625, 53.375, 53.375, 57.75, 60.375, 63.0], 
        'BNN': [33.25, 42.875, 50.75, 52.5, 54.25, 54.25, 60.375, 63.0, 66.5], 
        # 'BNN+CP': [32.375, 41.125, 49.0, 50.75, 53.375, 53.375, 56.875, 59.5, 63.0]
        }

# Define custom line styles and markers
styles = {
    "AutoScaleOpt": {"linestyle": "--", "marker": "s"},
    "AutoScaleCons": {"linestyle": ":", "marker": "o"},
    "Sinan": {"linestyle": "-", "marker": "D"},
    "CNN+XGBoost+EucDist": {"linestyle": "-.", "marker": "x"},
    "CNN+XGBoost+CP": {"linestyle": "-", "marker": "o"},
    "BNN": {"linestyle": "-", "marker": "^"},
    # "BNN+CP": {"linestyle": ":", "marker": "p"}
}

# Plot
plt.figure(figsize=(13, 6))
for system, values in data.items():
    plt.plot(time_minutes, values,
             label=system,
             **styles[system],
             linewidth=2,
             markersize=8)

plt.xlabel("Time (minutes)", fontsize=20)
plt.ylabel("Mean CPU Allocation", fontsize=20)
# plt.title("Fraction of Requests Violating QoS per Epoch", fontsize=20)
plt.legend(fontsize=20, frameon=False, bbox_to_anchor=(1, 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(top=70)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("sinan_resource_alloc.pdf")
