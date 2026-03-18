import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 18)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

# Data
data = {
    "AutoScaleOpt": [0, 0, 0, 10, 14, 22, 30, 35, 38, 38, 36, 33, 24, 16, 13, 5, 2, 0],
    "AutoScaleCons": [0, 0, 0, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1],
    "Sinan": [0, 0, 0, 5, 7, 13, 14, 19, 22, 22, 20, 16, 15, 10, 6, 3, 1, 0],
    "Sinan+EucDist": [0, 0, 0, 3, 5, 10, 13, 16, 16, 15, 15, 14, 12, 7, 4, 1, 0, 0],
    "Sinan+CP": [0, 0, 0, 3, 4, 7, 8, 8, 11, 11, 10, 9, 8, 6, 3, 2, 2, 0],
    "BNN": [0, 0, 0, 1, 2, 3, 3, 5, 6, 6, 5, 4, 3, 2, 1, 1, 0, 0],
}

# Define custom line styles and markers
styles = {
    "AutoScaleOpt": {"linestyle": "--", "marker": "s"},
    "AutoScaleCons": {"linestyle": ":", "marker": "o"},
    "Sinan": {"linestyle": "-", "marker": "D"},
    "Sinan+EucDist": {"linestyle": "-.", "marker": "x"},
    "Sinan+CP": {"linestyle": "-", "marker": "o"},
    "BNN": {"linestyle": "-", "marker": "^"},
    "BNN+CP": {"linestyle": ":", "marker": "p"}
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
ax.set_xticklabels([])
ax.tick_params(axis="x", labelsize=16)
ax.tick_params(axis="y", labelsize=16)

# ---- Custom second row of labels ----
up = list(range(50, 500, 50))        # 50 → 450
down = list(range(400, 0, -50))      # 400 → 50
user_pattern = up + down

labels = []
for i in range(len(time_minutes)-1):
    users = 50 * (i+1)
    workload = "IID" if i < 3 else "OOD"
    labels.append(f"{users}")

labels = [f"{u}" for u in user_pattern]

# Center labels between ticks
positions = (time_minutes[:-1] + time_minutes[1:]) / 2
for pos, lab in zip(positions, labels):
    ax.text(pos-0.5, -3.5, lab, fontsize=15)
#    ax.text(pos, 0, lab, ha="center", va="top", fontsize=0,
#            transform=ax.transAxes)  # use axes coords so it stays below

ax.text(positions[7]-1, -5.5, "Number of users", fontsize=20)

plt.tight_layout()
plt.savefig("sinan_qos_violations_users.pdf", bbox_inches="tight")
plt.show()
