import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 9)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

# Data
data = {
    "AutoScaleOpt":         [0, 0, 0, 0, 21, 0, 0, 0, 24],
    "AutoScaleCons":        [0, 0, 0, 1, 1, 0, 0, 0.8, 1],
    "CNN+XGBoost":          [0, 0, 1, 3, 11, 1, 2, 4, 13],
    "CNN+XGBoost+EucDist":  [0, 0, 0.5, 1, 7, 1, 1, 3, 8],
    "CNN+XGBoost+CP":       [0, 0, 0.4, 0.7, 5.5, 0.8, 0.9, 2.1, 5.6],
    "BNN":                  [0, 0, 0.1, 0.2, 2.4, 0.1, 0.3, 0.4, 2.5],
    # "BNN+CP":               [0, 0, 0.1, 0.1, 2.1, 0.1, 0.1, 0.2, 2.0]
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
ax.set_ylabel("QoS Violations (%)", fontsize=20)
ax.legend(fontsize=16, frameon=False)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_ylim(top=25)
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
    ax.text(pos-0.8, -5, lab, fontsize=15)
#    ax.text(pos, 0, lab, ha="center", va="top", fontsize=0,
#            transform=ax.transAxes)  # use axes coords so it stays below

ax.text(positions[3]-1, -7, "Time (minutes)", fontsize=20)

plt.tight_layout()
plt.savefig("sinan_qos_violations_loads.pdf", bbox_inches="tight")
plt.show()
