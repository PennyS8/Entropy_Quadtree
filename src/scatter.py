"""
scatter.py
----------
Visualize extracted features across a labeled dataset.

Reads a CSV produced by features.py and generates scatter plots
showing how well complexity features separate image classes.

Usage:
    python3 scatter.py features.csv
    python3 scatter.py features.csv --x mean_complexity --y std_complexity
    python3 scatter.py features.csv --auto # generate all feature pair plots
    python3 scatter.py features.csv --auto --output results/scatter/

Output:
    Single plot:    <o>/scatter_<method>.png (or scatter_<method>.png alongside CSV)
    --auto:         <o>/<x>_vs_<y>.png         (or scatter_plots_<method>/<x>_vs_<y>.png)

"""

import argparse
import csv
import os
import numpy as np
from PIL import Image, ImageDraw

from features import FEATURE_FIELDS

# Label colors: real=green, ai=red, photoshopped=orange, unknown=grey
LABEL_COLORS = {
    "real":          (34,  197, 94,  220),
    "ai":            (239, 68,  68,  220),
    "photoshopped":  (249, 115, 22,  220),
    None:            (150, 150, 150, 220),
    "":              (150, 150, 150, 220),
}

NUMERIC_FEATURES = [f for f in FEATURE_FIELDS if f not in ("filename", "label", "leaf_count")]

# Pruned feature pairs — drop mirrors, x_vs_x, dead features,
# and pairs that are known to be redundant from analysis
DEAD = {"max_complexity", "mean_leaf_area", "std_leaf_area", "mean_depth", "std_depth", "leaf_count"}
        

def parse_args():
    parser = argparse.ArgumentParser(description="Scatter plot of image complexity features.")
    parser.add_argument("csv", help="Path to features CSV")
    parser.add_argument("--x", default="mean_complexity",
                        help=f"X axis feature (default: mean_complexity)")
    parser.add_argument("--y", default="std_complexity",
                        help=f"Y axis feature (default: std_complexity)")
    parser.add_argument("--output", default=None,
                        help="Output folder for plot(s). Defaults to alongside the CSV "
                             "for single plots, or scatter_plots_<method>/ for --auto.")
    parser.add_argument("--auto", action="store_true",
                        help="Generate scatter plots for all feature pairs")
    parser.add_argument("--corr-threshold", type=float, default=0.9,
                        help="Skip pairs whose Pearson |r| exceeds this value (default 0.9). "
                        "Lower = more aggressive pruning. Only applies to --auto.")
    return parser.parse_args()


def load_data(csv_path: str) -> list:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Convert numeric fields
    for row in rows:
        for field in NUMERIC_FEATURES + ["leaf_count"]:
            try:
                row[field] = float(row[field])
            except (ValueError, KeyError):
                row[field] = 0.0
    return rows


def filter_correlated_pairs(pairs: list, rows: list, threshold: float) -> tuple:
    """
    Remove pairs where |Pearson r| >= threshold.
    Returns (kept_pairs, skipped_pairs)
    """
    kept, skipped = [], []
    for x_field, y_field in pairs:
        x_vals = np.array([r[x_field] for r in rows], dtype=float)
        y_vals = np.array([r[y_field] for r in rows], dtype=float)
        # Pearson r: guard against zero-variance columns
        if x_vals.std() == 0 or y_vals.std() == 0:
            skipped.append((x_field, y_field, 1.0))
            continue
        r = float(np.corrcoef(x_vals, y_vals, 1.0)[0, 1])
        if abs(r) >= threshold:
            skipped.append((x_field, y_field, r))
        else:
            kept.append((x_field, y_field))
    return kept, skipped


def render_scatter(
    rows: list,
    x_field: str,
    y_field: str,
    width: int = 700,
    height: int = 600,
    margin: int = 70,
) -> Image.Image:
    """
    Render a scatter plot of two features, colored by label.
    Returns a PIL Image.
    """
    x_vals = np.array([r[x_field] for r in rows], dtype=float)
    y_vals = np.array([r[y_field] for r in rows], dtype=float)

    x_min, x_max = x_vals.min(), x_vals.max()
    y_min, y_max = y_vals.min(), y_vals.max()

    # Add 10% padding
    x_pad = (x_max - x_min) * 0.1 or 0.1
    y_pad = (y_max - y_min) * 0.1 or 0.1
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    plot_w = width - margin * 2
    plot_h = height - margin * 2

    def to_px(x, y):
        px = margin + int((x - x_min) / (x_max - x_min) * plot_w)
        py = height - margin - int((y - y_min) / (y_max - y_min) * plot_h)
        return px, py

    img = Image.new("RGB", (width, height), (20, 20, 20))
    draw = ImageDraw.Draw(img, "RGBA")

    # Grid lines
    for i in range(5):
        t = i / 4
        x_grid = margin + int(t * plot_w)
        y_grid = margin + int(t * plot_h)
        draw.line([(x_grid, margin), (x_grid, height - margin)], fill=(60, 60, 60))
        draw.line([(margin, y_grid), (width - margin, y_grid)], fill=(60, 60, 60))

    # Axis labels (tick values)
    for i in range(5):
        t = i / 4
        x_val = x_min + t * (x_max - x_min)
        y_val = y_min + t * (y_max - y_min)
        px = margin + int(t * plot_w)
        py = height - margin - int(t * plot_h)
        draw.text((px - 16, height - margin + 6), f"{x_val:.2f}", fill=(160, 160, 160))
        draw.text((4, py - 6), f"{y_val:.2f}", fill=(160, 160, 160))

    # Axis titles
    draw.text((width // 2 - len(x_field) * 3, height - 18), x_field, fill=(200, 200, 200))
    draw.text((4, height // 2 - 40), y_field, fill=(200, 200, 200))

    # Data points
    radius = 1.5
    for row in rows:
        x, y = to_px(row[x_field], row[y_field])
        label = row.get("label", "").strip() or None
        color = LABEL_COLORS.get(label, LABEL_COLORS[None])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)

    # Legend — only show labels present in the data
    present_labels = sorted({row.get("label", "").strip() for row in rows} - {"", None})
    legend_x = width - margin - 140
    legend_y = margin
    for label in present_labels:
        color = LABEL_COLORS.get(label, LABEL_COLORS[None])
        draw.ellipse([legend_x, legend_y, legend_x + 10, legend_y + 10], fill=color)
        draw.text((legend_x + 14, legend_y - 1), label, fill=(200, 200, 200))
        legend_y += 20

    # Title
    title = f"{x_field}  vs  {y_field}"
    draw.text((margin, 10), title, fill=(220, 220, 220))

    return img


def main():
    args = parse_args()
    rows = load_data(args.csv)
    
    if not rows:
        print("No data found in CSV.")
        return
    
    csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
    csv_stem = csv_stem.removeprefix("features_") # features_shannon -> shannon
    csv_dir  = os.path.dirname(args.csv)
    
    if args.auto:
        useful = [f for f in NUMERIC_FEATURES if f not in DEAD]
        
        # Upper triangle only (no mirrors, no x_vs_x)
        pairs = [(useful[i], useful[j])
                 for i in range(len(useful))
                 for j in range(i + 1, len(useful))]
        
        pairs, skipped = filter_correlated_pairs(pairs, rows, args.corr_threshold)
        
        if skipped:
            print(f"Skipped {len(skipped)} correlated pairs (|r| >= {args.corr_threshold}):")
            for x_field, y_field, r in skipped:
                print(f"  {x_field} vs {y_field}  r={r:+.3f}")
        
        out_dir = args.output or os.path.join(csv_dir, f"scatter_{csv_stem}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating {len(pairs)} plots -> {out_dir}/")
        for x_field, y_field in pairs:
            img = render_scatter(rows, x_field, y_field)
            out = os.path.join(out_dir, f"{x_field}_vs_{y_field}.png")
            img.save(out)
            print(f"\t {x_field} vs {y_field}")
    else:
        img = render_scatter(rows, args.x, args.y)
        out_dir = args.output or csv_dir
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"scatter_{csv_stem}.png")
        img.save(out)
        print(f"Saved: {out}")
        img.show()


if __name__ == "__main__":
    main()