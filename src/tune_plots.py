"""
tune_plots.py

Generate threshold tuning visualizations from tune_thresholds.py CSV output.
Reads results/tuning/{method}.csv and produces three plots:

    1. accuracy_vs_threshold   — CV accuracy vs pruning percentile, with
                                 confidence bands and best-point markers
    2. leaf_count_vs_threshold — mean subject leaf count vs threshold, showing
                                 how aggressively each method prunes the tree
    3. accuracy_vs_leaf_count  — accuracy vs tree size, dots coloured by
                                 threshold percentile via plasma colormap

Usage:
    # Default: reads results/tuning/{shannon,compression,variance}.csv
    python3 src/tune_plots.py
    
    # Explicit input files
    python3 src/tune_plots.py --input results/tuning/compression.csv
    
    # Multiple files
    python3 src/tune_plots.py \
        --input results/tuning/compression.csv results/tuning/compression_spatial_map.csv
    
    # Custom output folder (default: results/tuning/)
    python3 src/tune_plots.py --output results/tuning/

Output (three PNGs written to output folder):
    accuracy_vs_thresholds.png
    leaf_count_vs_thresholds.png
    accuracy_vs_leaf_count.png
"""

import argparse
import itertools
import os
import sys
import matplotlib.pyplot as plt
import csv
import numpy as np

PLOT_COLORS = ["steelblue", "tomato", "mediumseagreen", "orchid", "orange", "cyan"]
DEFAULT_METHODS = ["shannon", "compression", "variance"]
DEFAULT_INPUT_DIR = "results/tuning"


def parse_args():
    parser = argparse.ArgumentParser(description="Plot threshold tuning results.")
    parser.add_argument(
        "--input", nargs="+", default=None,
        help="One or more tuning CSVs. If omitted, auto-discovers all CSVs "
             "under results/tuning/ (or the folder given by --folder)."
    )
    parser.add_argument(
        "--folder", default=None, metavar="DIR",
        help="Restrict auto-discovery to a specific subfolder, e.g. "
             "results/tuning/deepfacelab/ — plots only that dataset's methods. "
             "Ignored when --input is provided explicitly."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output folder for plots. Default: same folder as --folder if given, "
             "otherwise results/tuning/"
    )
    return parser.parse_args()


def label_from_path(path: str) -> str:
    """
    Derive a short legend label from a CSV path.

    results/tuning/stylegan_v1/shannon.csv   -> stylegan_v1/shannon
    results/tuning/deepfacelab/compression.csv -> deepfacelab/compression
    results/tuning/shannon.csv               -> shannon  (legacy flat)
    """
    stem   = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    # If parent is the tuning root itself, just use the stem (legacy)
    if parent == DEFAULT_INPUT_DIR.rstrip("/").split("/")[-1] or parent == "tuning":
        return stem
    return f"{parent}/{stem}"


def load_csv(path: str) -> list:
    with open(path, mode="r") as f:
        return list(csv.DictReader(f))


def _styled_ax(fig, ax):
    """Apply dark theme styling to an axes."""
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#333", linewidth=0.5)


def accuracy_vs_threshold(datasets: list, output_dir: str):
    """datasets: list of (label, rows) tuples."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
    _styled_ax(fig, ax)
    
    for (label, rows), color in zip(datasets, itertools.cycle(PLOT_COLORS)):
        thresholds  = [float(r["threshold"]) for r in rows]
        accuracies  = [float(r["cv_accuracy_mean"]) for r in rows]
        std         = [float(r["cv_accuracy_std"]) for r in rows]
    
        ax.plot(thresholds, accuracies, color=color, linewidth=2, marker="o", label=label)
        upper = [m + s for m, s in zip(accuracies, std)]
        lower = [m - s for m, s in zip(accuracies, std)]
        ax.fill_between(thresholds, lower, upper, color=color, alpha=0.15)
        
        best_i = accuracies.index(max(accuracies))
        ax.axvline(thresholds[best_i], color=color, linestyle="--", linewidth=1, alpha = 0.5)
        ax.annotate(
            f"best={thresholds[best_i]}",
            xy=(thresholds[best_i], accuracies[best_i]),
            xytext=(thresholds[best_i] + 0.75, accuracies[best_i] + 0.001),
            color=color, fontsize=9
        )
    
    ax.set_xlim(0, 100)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Accuracy vs Threshold")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    out = os.path.join(output_dir, "accuracy_vs_thresholds.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"Saved: {out}")


def leaf_count_vs_threshold(datasets: list, output_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
    _styled_ax(fig, ax)
    
    all_max_leaves = []
    for(label, rows), color in zip(datasets, itertools.cycle(PLOT_COLORS)):
        thresholds = [float(r["threshold"]) for r in rows]
        mean_leaves = [float(r["mean_leaves"]) for r in rows]
        all_max_leaves.append(max(mean_leaves))
        ax.plot(thresholds, mean_leaves, color=color, linewidth=2, marker="o", label=label)
    
    # Reference line at the maximum unpruned leaf count across all datasets
    full_grid = max(all_max_leaves)
    ax.axhline(full_grid, color="#888", linestyle=":", linewidth=1)
    ax.annotate(
        "full grid (no pruning)",
        xy=(1, 1387),
        xytext=(1, 1387 + 5),
        color="#888",
        fontsize=8
    )
    
    ax.set_xlim(0, 100)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean Subject Leaves")
    ax.set_title("Leaf Count vs Threshold")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    out = os.path.join(output_dir, "leaf_count_vs_thresholds.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"saved: {out}")


def accuracy_vs_leaf_count(datasets: list, output_dir: str):
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
    _styled_ax(fig, ax)
    
    sc = None
    for (label, rows), color in zip(datasets, itertools.cycle(PLOT_COLORS)):
        thresholds  = [float(r["threshold"]) for r in rows]
        mean_leaves = [float(r["mean_leaves"]) for r in rows]
        accuracies  = [float(r["cv_accuracy_mean"]) for r in rows]

        sc = ax.scatter(
            mean_leaves,
            accuracies,
            c=thresholds,
            cmap="plasma",
            s=50,
            edgecolors=color,
            linewidths=1.2,
            label=label,
            zorder=3
        )
        ax.plot(mean_leaves, accuracies, color=color, linewidth=1, alpha=0.4)

        best_i = accuracies.index(max(accuracies))
        ax.annotate(
            f"{label}\nthr={thresholds[best_i]:.0f}",
            xy=(mean_leaves[best_i], accuracies[best_i]),
            xytext=(mean_leaves[best_i] + 5, accuracies[best_i] + 0.001),
            color=color, fontsize=8
        )
    
    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Threshold percentile", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Mean Subject Leaves")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Accuracy vs Leaf Count")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    out = os.path.join(output_dir, "accuracy_vs_leaf_count.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#111")
    plt.close()
    print(f"Saved: {out}")


def main():
    args = parse_args()
    
    # Resolve input file list
    if args.input:
        paths = args.input
    else:
        # If --folder given, restrict discovery to that directory
        search_root = args.folder.rstrip("/") if args.folder else DEFAULT_INPUT_DIR
        import glob
        paths = sorted(glob.glob(
            os.path.join(search_root, "**", "*.csv"), recursive=True
        ))
        # Also check legacy flat location: results/tuning/{method}.csv
        if not args.folder:
            for m in DEFAULT_METHODS:
                flat = os.path.join(DEFAULT_INPUT_DIR, f"{m}.csv")
                if os.path.exists(flat) and flat not in paths:
                    paths.append(flat)
        if not paths:
            print(f"No CSVs found under {search_root}/. Use --input to specify files.")
            return
    
    # Load datasets
    datasets = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: file not found, skipping: {path}")
            continue
        label = label_from_path(path)
        rows = load_csv(path)
        datasets.append((label, rows))
        print(f"Loaded {len(rows)} rows from {os.path.basename(path)} ({label})")

    if not datasets:
        print("No data loaded — nothing to plot.")
        return

    # Resolve output directory — default to --folder if given, else tuning root
    if args.output:
        output_dir = args.output.rstrip("/")
    elif args.folder:
        output_dir = args.folder.rstrip("/")
    else:
        output_dir = DEFAULT_INPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    accuracy_vs_threshold(datasets, output_dir)
    leaf_count_vs_threshold(datasets, output_dir)
    accuracy_vs_leaf_count(datasets, output_dir)
        

if __name__ == "__main__":
    main()