import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 9)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

# Data
data = {
    "Sinan+CP-5": [0, 0, 0, 2, 2, 4, 5, 5.5, 6],
    "Sinan+CP-10": [0, 0, 0, 2, 3, 6, 6, 7, 8],
    "Sinan+CP-15": [0, 0, 0, 3, 4, 7, 8, 8, 11],
    "Sinan+CP-20": [0, 0, 0, 3.5, 4.5, 8, 10, 11, 14],
    "Sinan+CP-25": [0, 0, 0, 4, 5.5, 9, 12, 13, 15.5],
    "BNN-5": [0, 0, 0, 1, 1, 1, 2, 3, 4],
    "BNN-10": [0, 0, 0, 1, 2, 2, 2, 4, 4],
    "BNN-15": [0, 0, 0, 1, 2, 3, 3, 5, 6],
    "BNN-20": [0, 0, 0, 1, 3, 4, 5, 7, 8],
    "BNN-25": [0, 0, 0, 1, 4, 4, 5, 8, 8.5],
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
ax.set_ylim(top=15)
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
# for pos, lab in zip(positions, labels):
#     ax.text(pos-0.5, -6, lab, fontsize=15)
#    ax.text(pos, 0, lab, ha="center", va="top", fontsize=0,
#            transform=ax.transAxes)  # use axes coords so it stays below

ax.text(positions[3]-1, -3, "Time (minutes)", fontsize=20)

plt.tight_layout()
plt.savefig("sinan_ablation_study_qos_violations.pdf", bbox_inches="tight")
plt.show()
