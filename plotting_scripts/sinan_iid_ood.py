import matplotlib.pyplot as plt
import numpy as np

# Epochs and time (2 minutes each)
epochs = np.arange(0, 9)  # 0 to 8
time_minutes = epochs * 2  # each epoch = 2 minutes

# Data
data = {
    "Sinan": [0, 0, 0, 5, 7, 13, 14, 19, 22]
}

# Define custom line styles and markers
styles = {
    "Sinan": {"linestyle": "-", "marker": "D"}
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

# Vertical line at 4 minutes
ax.axvline(x=6, color="black", linestyle="--", linewidth=2)

# Arrow from the line to the left (IID)
ax.annotate("IID",
            xy=(6, 22), xytext=(2, 22),
            fontsize=20, ha="center", va="center")

# Arrow from the line to the right (OOD)
ax.annotate("OOD",
            xy=(8, 22), xytext=(10, 22),
            fontsize=20, ha="center", va="center")



plt.tight_layout()
plt.savefig("sinan_iid_ood.pdf", bbox_inches="tight")
plt.show()
