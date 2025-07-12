import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load CSV ===
df = pd.read_csv('grid_search_results_top100.csv')

# === Filter RMSE <= 1 ===
df_filtered = df[df['validation_rmse'] <= 1].reset_index(drop=True)
print(f"Filtered down to {len(df_filtered)} configs (RMSE ≤ 1).")

# === Prepare mapping: num_layers → color
unique_layers = sorted(df_filtered['num_layers'].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_layers)))
layer_color_map = {layer: color for layer, color in zip(unique_layers, colors)}

# === Prepare mapping: hidden_dim → marker
unique_dims = sorted(df_filtered['hidden_dim'].unique())
markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']  # as many as you need
dim_marker_map = {dim: markers[i % len(markers)] for i, dim in enumerate(unique_dims)}

# === Find where LR changes (for vertical lines)
lr_changes = df_filtered['learning_rate'] != df_filtered['learning_rate'].shift()
lr_change_indices = df_filtered.index[lr_changes].tolist()

# === Plot ===
plt.figure(figsize=(14,6))

for idx, row in df_filtered.iterrows():
    lr = row['learning_rate']
    nl = row['num_layers']
    hd = row['hidden_dim']
    rmse = row['validation_rmse']
    
    plt.scatter(
        idx, rmse,
        color=layer_color_map[nl],
        marker=dim_marker_map[hd],
        s=60,  # marker size
        edgecolor='black',
        linewidth=0.5,
        label=f'nlayer={nl}, hdim={hd}'  # won't show repeated in legend
    )

# === Add vertical lines where LR changes (skip first index 0)
for i in lr_change_indices[1:]:
    plt.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.5)

# === Add LR text labels centered under each block
# Collect LR blocks: start_idx, end_idx, lr_value
lr_blocks = []
for i, start in enumerate(lr_change_indices):
    lr_value = df_filtered.loc[start, 'learning_rate']
    if i + 1 < len(lr_change_indices):
        end = lr_change_indices[i+1] - 1
    else:
        end = len(df_filtered) - 1
    lr_blocks.append((start, end, lr_value))

# Add text under each block
for start, end, lr_value in lr_blocks:
    center = (start + end) / 2
    plt.text(center, -0.05, f"LR={lr_value:.0e}", ha='center', va='top', fontsize=10)



# === Remove x-ticks
plt.xticks([])

plt.ylabel("Validation RMSE")
plt.xlabel("Configs")
plt.title("Filtered Configs: Validation RMSE ≤ 1")

plt.ylim(0, 1.0)
plt.tight_layout()
plt.grid(axis='y', linestyle=':', alpha=0.7)

# === Custom legend (one entry per unique num_layers and hidden_dim)
# To make a clean legend, build proxy artists
from matplotlib.lines import Line2D

legend_elements = []

# Colors for num_layers
for nl in unique_layers:
    legend_elements.append(Line2D(
        [0], [0], color=layer_color_map[nl], marker='o', linestyle='None',
        markersize=8, label=f'num_layers={nl}'
    ))

# Markers for hidden_dim (using black color for clarity)
for hd in unique_dims:
    legend_elements.append(Line2D(
        [0], [0], color='black', marker=dim_marker_map[hd], linestyle='None',
        markersize=8, label=f'hidden_dim={hd}'
    ))

plt.legend(handles=legend_elements, loc='best', fontsize=9, ncol=2)

plt.savefig("filtered_rmse_plot_top100.png")
plt.show()

print("Plot saved to filtered_rmse_plot_top100.png")
