import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 9)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

data = {
    "AutoScaleOpt": [0, 10, 15, 11, 28, 27, 21, 22, 15],
    "AutoScaleCons": [0, 1, 2, 1, 1, 1, 2, 1, 1],
    "Sinan": [0, 5, 7, 6, 13, 14, 13, 12, 8],
    "Sinan+Dist": [0, 3, 6, 3, 12, 12, 11, 11, 5],
    "Sinan+CP": [0, 3, 5, 3, 8, 7, 6, 6, 4],
    "BNN": [0, 1, 2, 1, 3, 3, 2, 2, 1],
    "Dec.Tree": [0, 1.4, 2.5, 2.6, 3.4, 3.5, 3.1, 2.9, 2.3]
}

# Data
data = {
    "AutoScaleOpt": [0, 0, 0, 10, 14, 22, 30, 35, 38],
    "AutoScaleCons": [0, 0, 0, 0, 1, 2, 1, 1, 1],
    "Sinan": [0, 0, 0, 5, 7, 13, 14, 19, 22],
    "Sinan+Dist": [0, 0, 0, 3, 5, 10, 13, 16, 15],
    "Sinan+CP": [0, 0, 0, 3, 4, 7, 8, 8, 11],
    "BNN": [0, 0, 0, 1, 2, 3, 3, 5, 6],
    "Dec. Tree": [0, 0, 0.4, 1.7, 2.8, 3.6, 4.7, 6.3, 6.8],
    "Dec. Tree+Thresh.": [0, 0, 0.3, 1.3, 2.3, 3.0, 3.9, 5.7, 5.6],
    "Dec. Tree+BNN":     [0, 0, 0.1, 0.8, 1.4, 2.5, 3.4, 5.6, 5.5]
}

# Define custom line styles and markers
styles = {
    "AutoScaleOpt": {"linestyle": "--", "marker": "s"},
    "AutoScaleCons": {"linestyle": ":", "marker": "o"},
    "Sinan": {"linestyle": "-", "marker": "D"},
    "Sinan+Dist": {"linestyle": "-.", "marker": "x"},
    "Sinan+CP": {"linestyle": "-", "marker": "o"},
    "BNN": {"linestyle": "-", "marker": "^"},
    "Dec. Tree": {"linestyle": ":", "marker": "p"},
    "Dec. Tree+Thresh.": {"linestyle": ":", "marker": "p"},
    "Dec. Tree+BNN": {"linestyle": ":", "marker": "D"}
}

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
for system, values in data.items():
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
for i in range(len(time_minutes)-1):
    users = 50 * (i+1)
    workload = "IID" if i < 3 else "OOD"
    labels.append(f"{users} usr\n{workload}")

# Center labels between ticks
positions = (time_minutes[:-1] + time_minutes[1:]) / 2
for pos, lab in zip(positions, labels):
    ax.text(pos-0.5, -6, lab, fontsize=15)
#    ax.text(pos, 0, lab, ha="center", va="top", fontsize=0,
#            transform=ax.transAxes)  # use axes coords so it stays below

ax.text(positions[3]-1, -8, "Time (minutes)", fontsize=20)

plt.tight_layout()
plt.savefig("sinan_qos_violations_workshop_new.pdf", bbox_inches="tight")
plt.show()
