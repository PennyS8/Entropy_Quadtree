"""
tune_plots.py

Generate threshold tuning visualizations from tune_threshold.py CSV output.
Reads results/tune_threshold/tune_results_{method}.csv and produces three plots:

    1. accuracy_vs_threshold   — CV accuracy vs pruning percentile, with
                                 confidence bands and best-point markers
    2. leaf_count_vs_threshold — mean subject leaf count vs threshold, showing
                                 how aggressively each method prunes the tree
    3. accuracy_vs_leaf_count  — accuracy vs tree size, dots coloured by
                                 threshold percentile via plasma colormap

Usage:
    python3 tune_plots.py

Output:
    results/tune_threshold/tune_plots.png
    results/tune_threshold/leaf_count_vs_threshold.png
    results/tune_threshold/accuracy_vs_leaf_count.png
"""

import matplotlib.pyplot as plt
import csv
import numpy as np

def accuracy_vs_threshold():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#333", linewidth=0.5)
    
    # import csv
    METHODS = ["shannon", "compression", "variance"]
    COLORS = ["steelblue", "tomato", "mediumseagreen"]
    
    for method, color in zip(METHODS, COLORS):
        path = "./results/tune_threshold/tune_results_" + method + ".csv"
        with open(path, mode="r") as f:
            rows = list(csv.DictReader(f))
    
        thresholds  = [float(r["threshold"]) for r in rows]
        accuracies  = [float(r["cv_accuracy_mean"]) for r in rows]
        std         = [float(r["cv_accuracy_std"]) for r in rows]
    
        ax.plot(thresholds, accuracies, color=color, linewidth=2, marker="o", label=method)

        upper = [m + s for m, s in zip(accuracies, std)]
        lower = [m - s for m, s in zip(accuracies, std)]
        ax.fill_between(thresholds, lower, upper, color=color, alpha=0.15)
        
        best_i = accuracies.index(max(accuracies))
        ax.axvline(thresholds[best_i], color=color, linestyle="--", linewidth=1, alpha = 0.5)
        ax.annotate(f"best={thresholds[best_i]}",
                    xy=(thresholds[best_i], accuracies[best_i]),
                    xytext=(thresholds[best_i] + 0.75, accuracies[best_i] + 0.001),
                    color=color, fontsize=9)
    
    ax.set_xlim(0, 100)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Accuracy vs Threshold")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    
    plt.savefig("./results/tune_threshold/tune_plots.png", dpi=150, bbox_inches="tight", facecolor="#111")


def leaf_count_vs_threshold():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#333", linewidth=0.5)

    METHODS = ["shannon", "compression", "variance"]
    COLORS  = ["steelblue", "tomato", "mediumseagreen"]

    for method, color in zip(METHODS, COLORS):
        path = "./results/tune_threshold/tune_results_" + method + ".csv"
        with open(path, mode="r") as f:
            rows = list(csv.DictReader(f))

        thresholds  = [float(r["threshold"]) for r in rows]
        mean_leaves = [float(r["mean_leaves"]) for r in rows]

        ax.plot(thresholds, mean_leaves, color=color, linewidth=2, marker="o", label=method)

    # Horizontal reference line at the full unprunded leaf count
    ax.axhline(1387, color="#888", linestyle=":", linewidth=1)
    ax.annotate("full grid (no pruning)", xy=(1, 1387),
                xytext=(1, 1387 + 5),
                color="#888", fontsize=8)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean Subject Leaves")
    ax.set_title("Leaf Count vs Threshold")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")

    plt.savefig("./results/tune_threshold/leaf_count_vs_threshold.png", dpi=150, bbox_inches="tight", facecolor="#111")


def accuracy_vs_leaf_count():
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="#111")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, color="#333", linewidth=0.5)

    METHODS = ["shannon", "compression", "variance"]
    COLORS  = ["steelblue", "tomato", "mediumseagreen"]

    for method, color in zip(METHODS, COLORS):
        path = "./results/tune_threshold/tune_results_" + method + ".csv"
        with open(path, mode="r") as f:
            rows = list(csv.DictReader(f))

        thresholds  = [float(r["threshold"]) for r in rows]
        mean_leaves = [float(r["mean_leaves"]) for r in rows]
        accuracies  = [float(r["cv_accuracy_mean"]) for r in rows]

        # Scatter coloured by threshold value so you can still read the progression
        sc = ax.scatter(mean_leaves, accuracies, c=thresholds, cmap="plasma",
                        s=50, edgecolors=color, linewidths=1.2, label=method, zorder=3)
        ax.plot(mean_leaves, accuracies, color=color, linewidth=1, alpha=0.4)

        best_i = accuracies.index(max(accuracies))
        ax.annotate(f"{method}\nthr={thresholds[best_i]:.0f}",
                    xy=(mean_leaves[best_i], accuracies[best_i]),
                    xytext=(mean_leaves[best_i] + 5, accuracies[best_i] + 0.001),
                    color=color, fontsize=8)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Threshold percentile", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Mean Subject Leaves")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Accuracy vs Leaf Count")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")

    plt.savefig("./results/tune_threshold/accuracy_vs_leaf_count.png", dpi=150, bbox_inches="tight", facecolor="#111")


def main():
    accuracy_vs_threshold()
    leaf_count_vs_threshold()
    accuracy_vs_leaf_count()


if __name__ == "__main__":
    main()