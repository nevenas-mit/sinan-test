import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 9)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

# Data

data = {
    "AutoScaleOpt":        [50, 50, 59, 64, 76, 54, 58, 66, 78],
    "AutoScaleCons":       [68, 68, 82, 119, 167, 74, 86, 122, 168],
    "CNN+XGBoost":         [43, 43, 48, 53, 64, 45, 52, 55, 65],
    "CNN+XGBoost+EucDist": [42, 42, 49, 52, 65, 43, 48, 54, 68],
    "CNN+XGBoost+CP":      [38, 38, 47, 54, 61, 40, 47, 56, 63],
    "BNN":                 [38, 38, 49, 53, 62, 39, 51, 55, 64],
    "BNN+CP":              [37, 37, 46, 53, 61, 38, 48, 56, 60]
}

# Define custom line styles and markers
styles = {
    "AutoScaleOpt": {"linestyle": "--", "marker": "s"},
    "AutoScaleCons": {"linestyle": ":", "marker": "o"},
    "CNN+XGBoost": {"linestyle": "-", "marker": "D"},
    "CNN+XGBoost+EucDist": {"linestyle": "-.", "marker": "x"},
    "CNN+XGBoost+CP": {"linestyle": "-", "marker": "o"},
    "BNN": {"linestyle": "-", "marker": "^"},
    "BNN+CP": {"linestyle": ":", "marker": "p"}
}

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for system, values in data.items():
    print(system)
    print(len(values))
    ax.plot(time_minutes, values,
            label=system,
            **styles[system],
            linewidth=2,
            markersize=8)

# ax.set_xlabel("Time (minutes)", fontsize=20)
ax.set_ylabel("Mean CPU Allocation", fontsize=20)
ax.legend(fontsize=16, frameon=False)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_ylim(top=100)
ax.set_xticks(time_minutes)
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)

# ---- Custom second row of labels ----
labels = []
loads = [100, 150, 200, 250, 100, 150, 200, 250]
for i in range(len(time_minutes)-1):
    users = loads[i]
    workload = "IID" if i < 3 else "OOD"
    labels.append(f"{users} RPS")

# Center labels between ticks
positions = (time_minutes[:-1] + time_minutes[1:]) / 2
for pos, lab in zip(positions, labels):
    ax.text(pos-0.8, 22, lab, fontsize=15)

ax.text(positions[3]-1, 15, "Time (minutes)", fontsize=20)

plt.tight_layout()
plt.savefig("sinan_resource_alloc_loads.pdf", bbox_inches="tight")
plt.show()
