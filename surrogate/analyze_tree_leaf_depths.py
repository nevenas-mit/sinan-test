#!/usr/bin/env python3
"""
Analyze the distribution of leaf depths accessed when running the surrogate
decision tree on the validation dataset.

Usage:
  python analyze_tree_leaf_depths.py --model model/bnn_surrogate_tree.joblib --data-dir .
"""

import argparse
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from collections import Counter


def compute_node_depths(tree):
    """
    Compute the depth of every node in the tree.
    Returns a dict: node_id -> depth
    """
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right

    node_depth = np.zeros(n_nodes, dtype=int)
    stack = [(0, 0)]  # (node_id, depth)

    while stack:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        left = children_left[node_id]
        right = children_right[node_id]

        if left != -1:  # not a leaf
            stack.append((left, depth + 1))
        if right != -1:
            stack.append((right, depth + 1))

    return node_depth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model/bnn_surrogate_tree.joblib",
                        help="Path to surrogate tree joblib")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory with X_surrogate_valid.npy (or train)")
    parser.add_argument("--split", type=str, default="valid", choices=["train", "valid"],
                        help="Which split to analyze")
    parser.add_argument("--out-plot", type=str, default="leaf_depth_distribution.png",
                        help="Output plot filename")
    args = parser.parse_args()

    model = joblib.load(args.model)

    # Load data
    x_path = os.path.join(args.data_dir, f"X_surrogate_{args.split}.npy")
    X = np.load(x_path)
    print(f"Loaded {args.split} data: {X.shape}")

    # Get leaf node index for each sample
    leaf_ids = model.apply(X)  # shape (n_samples,) or (n_samples, n_outputs)
    if leaf_ids.ndim == 2:
        leaf_ids = leaf_ids[:, 0]  # multi-output trees have same structure

    # Compute depth for every node in tree
    node_depths = compute_node_depths(model.tree_)

    # Get depth of each leaf reached
    leaf_depths = node_depths[leaf_ids]

    # Count distribution
    depth_counts = Counter(leaf_depths)
    depths = sorted(depth_counts.keys())
    counts = [depth_counts[d] for d in depths]

    print(f"\nLeaf depth distribution ({args.split}, n={len(X)}):")
    print(f"  Min depth: {min(depths)}")
    print(f"  Max depth: {max(depths)}")
    print(f"  Mean depth: {np.mean(leaf_depths):.2f}")
    print(f"  Median depth: {np.median(leaf_depths):.1f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(depths, counts, color='steelblue', edgecolor='black', alpha=0.8)
    plt.xlabel('Leaf Depth (number of comparisons)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Distribution of Decision Tree Leaf Depths ({args.split} set, n={len(X)})', fontsize=13)
    plt.xticks(range(min(depths), max(depths) + 1, 2))
    plt.grid(axis='y', alpha=0.3)

    # Add stats annotation
    stats_text = f"Mean: {np.mean(leaf_depths):.1f}\nMedian: {np.median(leaf_depths):.0f}\nMax: {max(depths)}"
    plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    print(f"\nPlot saved to {args.out_plot}")

    # Also print top-10 most common depths
    print("\nTop 10 most common leaf depths:")
    for depth, count in sorted(depth_counts.items(), key=lambda x: -x[1])[:10]:
        pct = 100.0 * count / len(X)
        print(f"  Depth {depth:2d}: {count:5d} samples ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
