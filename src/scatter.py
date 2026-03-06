"""
scatter.py

Visualize extracted features across a labeled dataset.

Reads a CSV produced by features.py and generates scatter plots
showing how well complexity features separate image classes.

Usage:
    python scatter.py features.csv
    python scatter.py features.csv --x mean_complexity --y std_complexity
    python scatter.py features.csv --all   # generate all feature pair plots
"""

import argparse
import csv
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from features import FEATURE_FIELDS

# Label colors: real=green, ai=red, photoshopped=orange, unknown=grey
LABEL_COLORS = {
    
    "real":          (34,  197, 94,  220),
    "ai":            (239, 68,  68,  220),
    "photoshopped":  (249, 115, 22,  220),
    None:            (150, 150, 150, 220),
    "":              (150, 150, 150, 220)
}

NUMERIC_FEATURES = [f for f in FEATURE_FIELDS if f not in ("filename", "label", "leaf_count")]


def parse_args():
    parser = argparse.ArgumentParser(description="Scatter plot of image complexity features.")
    parser.add_argument("csv", help="Path to features CSV")
    parser.add_argument("--x", default="mean_complexity",
                        help=f"X axis feature (defaultL mean_complexity)")
    parser.add_argument("--y", default="std_complexity",
                        help=f"Y axis feature (default: std_complexity)")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: <csv>_scatter.png)")
    parser.add_argument("--all", action="store_true",
                        help="Generate scatter plots for all feature pairs")
    return parser.parse_args()


def load_data(csv_path: str) -> list:
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    # Convert numberic fields
    for row in rows:
        for field in NUMERIC_FEATURES + ["leaf_count"]:
            try:
                row[field] = float(row[field])
            except (ValueError, KeyError):
                row[field] = 0.0
    
    return rows


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
        t = i/4
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
    radius = 3
    for row in rows:
        x, y = to_px(row[x_field], row[y_field])
        label = row.get("label", "").strip() or None
        color = LABEL_COLORS.get(label, LABEL_COLORS[None])
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=color)
        # Filename label on hover not possible in PIL, show truncated name
        name = row.get("filename:", "")[:12]
        draw.text((x + radius + 2, y - 6), name, fill=(180, 180, 180))
        
    # Legend
    legend_x = width - margin - 140
    legend_y = margin
    for label, color in LABEL_COLORS.items():
        if label in (None, ""):
            continue
        draw.ellipse([legend_x, legend_y, legend_x + 10, legend_y + 10], fill=color)
        draw.text((legend_x + 14, legend_y - 1), label, fill=(200, 200, 200))
        legend_y += 20
    
    # Title
    title = f"{x_field} vs {y_field}"
    draw.text((margin, 10), title, fill=(220, 220, 220))
    
    return img


def main():
    args = parse_args()
    rows = load_data(args.csv)
    
    if not rows:
        print("No data found in CSV.")
        return
    
    base = args.csv.rsplit(".", 1)[0]
    
    if args.all:
        # Generate all unique feature pairs
        pairs = [(NUMERIC_FEATURES[i], NUMERIC_FEATURES[j])
                for i in range(len(NUMERIC_FEATURES))
                for j in range(len(NUMERIC_FEATURES))]
        os.makedirs(f"{base}_scatter_plots", exist_ok=True)
        for x_field, y_field in pairs:
            img = render_scatter(rows, x_field, y_field)
            out = f"{base}_scatter_plots/{x_field}_vs_{y_field}.png"
            img.save(out)
            print(f"Saved: {out}")
    else:
        img = render_scatter(rows, args.x, args.y)
        out = args.output or f"{base}_scatter.png"
        img.save(out)
        print(f"Saved: {out}")
        img.show()


if __name__ == "__main__":
    main()