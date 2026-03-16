"""
depth_distribution.py

Analyse the empirical depth distribution of quadtrees across a dataset by
sampling images and building trees, then reporting the actual max depth
reached per image.

This is used to choose a target depth for the tree-grid spatial map — pick
a depth that every image in the dataset reliably reaches so no cells need
filling by parent propagation.

Usage:
    # Sample 500 images from each of two folders
    python3 src/depth_distribution.py \\
        --input /path/to/Real /path/to/Fake \\
        --labels real ai \\
        --method compression \\
        --max_images 500

    # Use a specific threshold and leaf size
    python3 src/depth_distribution.py \\
        --input /path/to/Real /path/to/Fake \\
        --labels real ai \\
        --method compression --threshold 10 --leaf_size 16 \\
        --max_images 1000

    # Save depth distribution plots (default output: results/depth/)
    python3 src/depth_distribution.py \\
        --input /path/to/Real /path/to/Fake \\
        --labels real ai \\
        --output results/depth/

Output:
    Console report: percentile table, recommended target depth
    PNG per method: results/depth/{method}_spatial_map.png
"""

import argparse
import os
import sys
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image

# Allow running from project root as: python3 src/depth_distribution.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from complexity import get_scorer
from quadtree import QuadTree, BG_THRESHOLD

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse quadtree depth distribution across a dataset.")
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more input folders of images")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="One label per folder, same order as --input")
    parser.add_argument("--method", choices=["shannon", "compression", "variance", "all"],
                        default="all",
                        help="Scoring method, or 'all' to run all three (default: all)")
    parser.add_argument("--leaf_size", type=int, default=None,
                        help="Target leaf side length in pixels (default: method default)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Percentile pruning threshold (default: off)")
    parser.add_argument("--max_images", type=int, default=500,
                        help="Max images to sample per folder (default: 500)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers (default: 4)")
    parser.add_argument("--output", default="results/depth",
                        help="Output folder for plots. Each method saves as "
                             "{method}_spatial_map.png. (default: results/depth)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    return parser.parse_args()


def load_image(path: str):
    img = Image.open(path)
    alpha = None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        alpha = np.array(img.split()[3])
        image_array = np.array(img)
    else:
        img = img.convert("RGB")
        image_array = np.array(img)
    return image_array, alpha


def process_image(args_tuple):
    """Build a quadtree for one image and return depth statistics."""
    filename, path, method, leaf_size, threshold, label = args_tuple
    try:
        image_array, alpha = load_image(path)
        scorer = get_scorer(method)

        DEFAULT_LEAF_SIZE = {"shannon": 4, "compression": 16, "variance": 4}
        effective_leaf_size = leaf_size if leaf_size is not None else DEFAULT_LEAF_SIZE[method]

        qt = QuadTree(scorer=scorer, leaf_size=effective_leaf_size, threshold=threshold)
        root = qt.build(image_array, alpha=alpha, normalize=False)

        all_leaves = root.all_leaves()
        subject_leaves = [n for n in all_leaves if n.background_ratio < BG_THRESHOLD]
        if not subject_leaves:
            subject_leaves = all_leaves

        depths = [n.depth for n in subject_leaves]
        max_depth = max(depths)
        mean_depth = float(np.mean(depths))
        leaf_count = len(subject_leaves)
        h, w = image_array.shape[:2]

        return {
            "filename": filename,
            "label": label,
            "max_depth": max_depth,
            "mean_depth": mean_depth,
            "leaf_depths": depths,
            "leaf_count": leaf_count,
            "image_h": h,
            "image_w": w,
            "error": None,
        }
    except Exception as e:
        import traceback
        return {"filename": filename, "label": label, "error": traceback.format_exc()}


def print_percentile_table(values, label="all"):
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.array(values)
    print(f"\n  {label} (n={len(vals)})")
    print(f"  {'Percentile':>12}  {'Max Depth':>10}")
    print(f"  {'-'*25}")
    for p in percentiles:
        print(f"  {p:>11}%  {np.percentile(vals, p):>10.1f}")
    print(f"\n  Mean: {vals.mean():.2f}  Std: {vals.std():.2f}  "
          f"Min: {vals.min():.0f}  Max: {vals.max():.0f}")


def recommend_depth(all_max_depths):
    """
    Recommend a target grid depth based on the empirical distribution.
    Uses p5 of max depth — 95% of images reach this depth without needing
    parent propagation to fill missing cells.
    """
    p5 = int(np.percentile(all_max_depths, 5))
    # Cap at depth 4 (256 cells) — deeper adds noise without spatial benefit
    # since scalar features already summarise fine-grain complexity
    recommended = min(p5, 4)
    cells = 4 ** recommended
    grid = int(cells ** 0.5)
    return recommended, cells, grid


def run_method(args, method):
    """Run the full analysis pipeline for a single method."""
    DEFAULT_LEAF_SIZE = {"shannon": 4, "compression": 16, "variance": 4}
    effective_leaf_size = args.leaf_size or DEFAULT_LEAF_SIZE[method]
    workers = min(args.workers, cpu_count())

    print(f"\nDepth Distribution Analysis")
    print(f"{'─'*50}")
    print(f"Method:     {method}")
    print(f"Leaf size:  {effective_leaf_size}px")
    print(f"Threshold:  {args.threshold or 'off'}")
    print(f"Max images: {args.max_images} per folder")
    print(f"Workers:    {workers}")

    # Build task list — sample up to max_images per folder
    rng = random.Random(args.seed)
    tasks = []
    for folder, label in zip(args.input, args.labels):
        entries = sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        if not entries:
            print(f"Warning: no images found in '{folder}', skipping.")
            continue
        sampled = rng.sample(entries, min(args.max_images, len(entries)))
        print(f"Sampled {len(sampled)} / {len(entries)} images from '{folder}' (label: {label})")
        for filename in sampled:
            tasks.append((
                filename,
                os.path.join(folder, filename),
                method,
                args.leaf_size,
                args.threshold,
                label,
            ))

    if not tasks:
        print("No images found.")
        sys.exit(1)

    print(f"\nBuilding {len(tasks)} trees...")

    # Process images
    results = []
    errors = 0
    if workers > 1:
        with Pool(processes=workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_image, tasks), 1):
                if r["error"]:
                    errors += 1
                else:
                    results.append(r)
                if i % 100 == 0:
                    print(f"  {i}/{len(tasks)} done...")
    else:
        for i, task in enumerate(tasks, 1):
            r = process_image(task)
            if r["error"]:
                errors += 1
            else:
                results.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} done...")

    print(f"\nDone. {len(results)} processed, {errors} errors.")

    if not results:
        print("No results to analyse.")
        sys.exit(1)

    # Aggregate by label
    labels = sorted(set(r["label"] for r in results))
    all_max_depths = [r["max_depth"] for r in results]

    print(f"\n{'='*50}")
    print(f"  Max depth distribution (subject leaves only)")
    print(f"{'='*50}")

    print_percentile_table(all_max_depths, label="all classes")
    for label in labels:
        label_depths = [r["max_depth"] for r in results if r["label"] == label]
        print_percentile_table(label_depths, label=label)

    # Leaf count distribution
    all_leaf_counts = [r["leaf_count"] for r in results]
    print(f"\n{'='*50}")
    print(f"  Leaf count distribution")
    print(f"{'='*50}")
    vals = np.array(all_leaf_counts)
    print(f"  Mean: {vals.mean():.1f}  Std: {vals.std():.1f}  "
          f"Min: {vals.min():.0f}  Max: {vals.max():.0f}")

    # Recommendation
    recommended_depth, n_cells, grid_size = recommend_depth(all_max_depths)
    p5 = int(np.percentile(all_max_depths, 5))
    p1 = int(np.percentile(all_max_depths, 1))

    print(f"\n{'='*50}")
    print(f"  Recommendation")
    print(f"{'='*50}")
    print(f"  p1  max depth: {p1}  → 4^{p1} = {4**p1} cells")
    print(f"  p5  max depth: {p5}  → 4^{p5} = {4**p5} cells")
    print(f"\n  Recommended target depth: {recommended_depth}")
    print(f"  Grid equivalent:          {grid_size}x{grid_size} = {n_cells} cells")
    print(f"  Rationale: p5 of max depth, capped at 4.")
    print(f"  At this depth, 95% of images reach the target without")
    print(f"  needing parent propagation to fill missing cells.")
    if recommended_depth < p5:
        print(f"  (capped from p5={p5} to avoid {4**p5}-cell feature vectors)")

    # Optional plot
    # Resolve output path
    # Resolve output path — always write to results/depth/ by default
    if args.output.endswith(".png"):
        out_path = args.output
    else:
        out_dir = args.output.rstrip("/")
        out_path = os.path.join(out_dir, "{}_spatial_map.png".format(method))

    if out_path:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#111")
            colors = ["steelblue", "tomato", "mediumseagreen", "orchid"]

            def style_ax(ax):
                ax.set_facecolor("#1a1a1a")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#444")
                ax.grid(True, color="#333", linewidth=0.5, axis="y")

            for ax in axes:
                style_ax(ax)

            # Left: boxplot of all leaf depths per class
            # Each box summarises depths across all leaves in all images of that class.
            # More informative than max depth which is a geometric constant.
            ax = axes[0]
            box_data, box_labels_plot, box_colors = [], [], []
            for label, color in zip(labels, colors):
                flat = []
                for r in results:
                    if r["label"] == label:
                        flat.extend(r["leaf_depths"])
                box_data.append(flat)
                box_labels_plot.append(label)
                box_colors.append(color)

            any_variance = any(len(set(d)) > 1 for d in box_data)
            rng_jitter = np.random.default_rng(0)

            if any_variance:
                parts = ax.violinplot(box_data, positions=range(1, len(box_data)+1),
                                      showmedians=True, showextrema=True)
                for pc, color in zip(parts["bodies"], box_colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)
                parts["cmedians"].set_color("white")
                parts["cmaxes"].set_color("#aaa")
                parts["cmins"].set_color("#aaa")
                parts["cbars"].set_color("#aaa")
            else:
                # All values constant — violin collapses to invisible line.
                # Draw a jittered strip plot with a median line instead.
                for i, (data, color) in enumerate(zip(box_data, box_colors), 1):
                    val = data[0] if data else 0
                    jitter = rng_jitter.uniform(-0.2, 0.2, size=min(len(data), 2000))
                    ax.scatter(i + jitter, np.full(len(jitter), val),
                               color=color, alpha=0.15, s=4, zorder=2)
                    ax.hlines(val, i - 0.3, i + 0.3, colors="white",
                              linewidths=2, zorder=3)

            ax.set_xticks(range(1, len(box_labels_plot)+1))
            ax.set_xticklabels(box_labels_plot, color="white")
            ax.axhline(recommended_depth, color="white", linestyle="--",
                       linewidth=1.2, alpha=0.7,
                       label="target depth={}".format(recommended_depth))
            ax.set_ylabel("Leaf Depth")
            ax.set_title("Leaf Depth Distribution (all leaves, all images)")
            ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444", fontsize=8)

            # Middle: per-image depth IQR per class
            # Shows how much depth varies within a single image.
            # Real images should have wider spread than AI images.
            ax = axes[1]
            for label, color in zip(labels, colors):
                spreads = []
                for r in results:
                    if r["label"] == label:
                        d = np.array(r["leaf_depths"])
                        spreads.append(float(np.percentile(d, 75) - np.percentile(d, 25)))
                spread_bins = np.linspace(max(0.0, min(spreads) - 0.1), max(spreads) + 0.1, 25)
                ax.hist(spreads, bins=spread_bins, alpha=0.6, color=color,
                        label=label, edgecolor="#222")
            ax.set_xlabel("Depth IQR (per image)")
            ax.set_ylabel("Count")
            ax.set_title("Per-Image Depth Spread (IQR of leaf depths)")
            ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")

            # Right: leaf count distribution per class
            ax = axes[2]
            all_lc = [r["leaf_count"] for r in results]
            lc_bins = np.linspace(min(all_lc), max(all_lc), 30)
            for label, color in zip(labels, colors):
                lc = [r["leaf_count"] for r in results if r["label"] == label]
                ax.hist(lc, bins=lc_bins, alpha=0.6, color=color,
                        label=label, edgecolor="#222")
            ax.set_xlabel("Leaf Count")
            ax.set_ylabel("Count")
            ax.set_title("Leaf Count Distribution")
            ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")

            plt.tight_layout()
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111")
            plt.close()
            print("\nSaved plot: {}".format(out_path))
        except ImportError:
            print("\nmatplotlib not available — skipping plot.")


def main():
    args = parse_args()

    if len(args.input) != len(args.labels):
        print("Error: --input and --labels must have the same number of entries.")
        sys.exit(1)

    methods = ["shannon", "compression", "variance"] if args.method == "all" else [args.method]
    for method in methods:
        run_method(args, method)


if __name__ == "__main__":
    main()