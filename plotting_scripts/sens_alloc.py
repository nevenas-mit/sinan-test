import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 9)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

# Data

data = {
        'Sinan+CP-5': [36.71, 47.14, 55.641, 61.526, 64.556, 68.113, 71.421, 74.244, 76.231],
        'Sinan+CP-10': [33.375, 42.25, 51.875, 54.625, 57.375, 58.375, 62.75, 67.375, 71.0],
        'Sinan+CP-15': [32.375, 40.25, 49.875, 51.625, 53.375, 53.375, 57.75, 60.375, 63.0],
        'Sinan+CP-20': [31.375, 38.25, 42.875, 44.625, 46.375, 48.375, 51.75, 53.375, 54.0], 
        'Sinan+CP-25': [30.101, 37.32, 41.155, 42.752, 44.981, 46.156, 49.15, 51.671, 52.7],
        'BNN-5':  [37.16, 48.616, 61.61, 62.5, 66.82, 69.72, 72.521, 74.6, 79.1], 
        'BNN-10': [34.25, 45.875, 56.75, 58.5, 59.25, 61.25, 66.375, 69.0, 74.5], 
        'BNN-15': [33.25, 42.875, 50.75, 52.5, 54.25, 54.25, 60.375, 63.0, 66.5], 
        'BNN-20': [32.25, 40.875, 45.75, 47.5, 49.25, 49.95, 53.375, 55.0, 57.5], 
        'BNN-25': [30.11, 38.013, 43.13, 44.1, 47.13, 48.11, 51.112, 53.2, 54.9], 
}

# Define custom line styles and markers
styles = {
    'Sinan+CP-5':  {"linestyle": "-",  "marker": "o"},
    'Sinan+CP-10': {"linestyle": "--", "marker": "s"},
    'Sinan+CP-15': {"linestyle": ":",  "marker": "D"},
    'Sinan+CP-20': {"linestyle": "-.", "marker": "^"},
    'Sinan+CP-25': {"linestyle": "-",  "marker": "v"},

    'BNN-5':  {"linestyle": "--", "marker": "P"},
    'BNN-10': {"linestyle": ":",  "marker": "X"},
    'BNN-15': {"linestyle": "-.", "marker": "*"},
    'BNN-20': {"linestyle": "-",  "marker": "<"},
    'BNN-25': {"linestyle": "--", "marker": ">"}
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
plt.savefig("sinan_ablation_study_resource_alloc.pdf")
