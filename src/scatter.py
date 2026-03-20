"""
scatter.py
----------
Visualize extracted features across a labeled dataset.

Reads a CSV produced by batch.py and generates scatter plots
showing how well complexity features separate image classes.

Usage:
    python3 src/scatter.py results/features/compression.csv
    python3 src/scatter.py results/features/compression.csv --x mean_complexity --y std_complexity
    python3 src/scatter.py results/features/compression.csv --auto
    python3 src/scatter.py results/features/compression.csv --auto --top-features 15
    python3 src/scatter.py results/features/compression.csv --auto --output results/scatter/compression

--auto mode ranks all features by ANOVA F-statistic (how well each feature
separates the classes) and greedily selects the top N that are also mutually
uncorrelated with each other. This keeps the output focused on the features
that carry independent discriminative signal. Use --top-features to control N.
For unlabeled data, features are ranked by variance instead.

Output:
    Single plot:  results/scatter/{csv_stem}/scatter_{csv_stem}.png
    --auto:       results/scatter/{csv_stem}/{x}_vs_{y}.png
"""

import argparse
import csv
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

# Allow running from project root as: python3 src/scatter.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from features import FEATURE_FIELDS

# Label colors: authentic=green, synthetic=red, manipulated=orange, unknown=grey
LABEL_COLORS = {
    "authentic":     (34,  197, 94,  220),  # green  — genuine camera images
    "synthetic":     (239, 68,  68,  220),  # red    — fully AI-generated
    "manipulated":   (249, 115, 22,  220),  # orange — face-swapped / composited
    # Legacy aliases so old CSVs still render correctly
    "real":          (34,  197, 94,  220),
    "ai":            (239, 68,  68,  220),
    "photoshopped":  (249, 115, 22,  220),
    None:            (150, 150, 150, 220),
    "":              (150, 150, 150, 220),
}

NUMERIC_FEATURES = [f for f in FEATURE_FIELDS if f not in ("filename", "label", "leaf_count")]

# Always exclude — zero or near-zero variance across all images regardless of class
DEAD = {"max_complexity", "mean_leaf_area", "std_leaf_area", "mean_depth", "std_depth", "leaf_count"}


def parse_args():
    parser = argparse.ArgumentParser(description="Scatter plot of image complexity features.")
    parser.add_argument("csv", help="Path to features CSV, e.g. results/features/compression.csv")
    parser.add_argument("--x", default="mean_complexity",
                        help="X axis feature for single plot (default: mean_complexity)")
    parser.add_argument("--y", default="std_complexity",
                        help="Y axis feature for single plot (default: std_complexity)")
    parser.add_argument("--output", default=None,
                        help="Output folder for plot(s). "
                             "Default: results/scatter/{csv_stem}/")
    parser.add_argument("--auto", action="store_true",
                        help="Generate scatter plots for the top N feature pairs, ranked by "
                             "class separability (ANOVA F-statistic). Use --top-features to "
                             "control how many features are included.")
    parser.add_argument("--top-features", type=int, default=10,
                        help="Number of top-ranked features to include in --auto mode. "
                             "Pairs of these features are plotted (default: 10, giving 45 plots). "
                             "Increase for broader coverage, decrease for tighter focus.")
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


def select_top_features(rows: list, feature_cols: list, n: int, corr_threshold: float) -> tuple:
    """
    Select the top N features by class separability (ANOVA F-statistic),
    using a greedy approach that enforces mutual decorrelation during selection
    rather than as a post-filter.

    Algorithm:
        1. Score all features by F-statistic (or variance if single class).
        2. Take the highest-scoring feature unconditionally.
        3. For each remaining feature in score order, only add it if its
           Pearson |r| with every already-selected feature is below
           corr_threshold.
        4. Stop when N features are selected or candidates are exhausted.

    This guarantees the selected set is mutually uncorrelated, so the
    correlation filter in --auto mode will never discard all pairs.

    Returns:
        selected:   list of selected feature names (up to N)
        ranked:     full ranked list of (feature, score) for all candidates
        rank_method: string describing the scoring method used
    """
    from sklearn.feature_selection import f_classif

    labels = [row.get("label", "").strip() for row in rows]
    unique_labels = set(labels) - {""}
    x = np.array([[row[f] for f in feature_cols] for row in rows], dtype=float)

    if len(unique_labels) >= 2:
        y = np.array(labels)
        f_scores, _ = f_classif(x, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
        ranked = sorted(zip(feature_cols, f_scores.tolist()), key=lambda t: -t[1])
        rank_method = "ANOVA F-statistic"
    else:
        variances = np.var(x, axis=0)
        ranked = sorted(zip(feature_cols, variances.tolist()), key=lambda t: -t[1])
        rank_method = "variance (single class — no label separation available)"

    # Precompute column index map for fast lookup
    col_idx = {f: i for i, f in enumerate(feature_cols)}

    selected = []
    for feat, _ in ranked:
        if len(selected) >= n:
            break
        feat_vals = x[:, col_idx[feat]]
        if feat_vals.std() == 0:
            continue
        # Check correlation against every already-selected feature
        correlated = False
        for sel in selected:
            sel_vals = x[:, col_idx[sel]]
            if sel_vals.std() == 0:
                continue
            r = float(np.corrcoef(feat_vals, sel_vals)[0, 1])
            if abs(r) >= corr_threshold:
                correlated = True
                break
        if not correlated:
            selected.append(feat)

    return selected, ranked, rank_method


def filter_correlated_pairs(pairs: list, rows: list, threshold: float) -> tuple:
    """
    Remove pairs where |Pearson r| >= threshold.
    Returns (kept_pairs, skipped_pairs)
    """
    kept, skipped = [], []
    for x_field, y_field in pairs:
        x_vals = np.array([r[x_field] for r in rows], dtype=float)
        y_vals = np.array([r[y_field] for r in rows], dtype=float)
        # Guard against zero-variance columns
        if x_vals.std() == 0 or y_vals.std() == 0:
            skipped.append((x_field, y_field, 1.0))
            continue
        r = float(np.corrcoef(x_vals, y_vals)[0, 1])
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

    if args.auto:
        # Candidates: all numeric features except known-dead ones
        candidates = [f for f in NUMERIC_FEATURES if f not in DEAD]

        # Greedily select top N features that are both high-scoring and
        # mutually uncorrelated — guarantees pairs will exist after selection
        top, ranked, rank_method = select_top_features(
            rows, candidates, args.top_features, args.corr_threshold
        )

        print(f"\nFeature ranking ({rank_method}):")
        top_set = set(top)
        shown = 0
        for feat, score in ranked:
            if feat in top_set:
                shown += 1
                print(f"  {shown:>2}. {feat:<25}  {score:.4f}")
        excluded = len(ranked) - len(top)
        if excluded:
            print(f"  ... {excluded} features excluded (low score or correlated with a higher-ranked feature)\n")

        if not top:
            print("No features selected — try raising --corr-threshold or lowering --top-features.")
            return

        # Upper triangle pairs of selected features
        pairs = [(top[i], top[j])
                 for i in range(len(top))
                 for j in range(i + 1, len(top))]

        out_dir = args.output or os.path.join("results", "scatter", csv_stem)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Generating {len(pairs)} plots -> {out_dir}/")
        for x_field, y_field in pairs:
            img = render_scatter(rows, x_field, y_field)
            out = os.path.join(out_dir, f"{x_field}_vs_{y_field}.png")
            img.save(out)
            print(f"\t {x_field} vs {y_field}")
    else:
        img = render_scatter(rows, args.x, args.y)
        out_dir = args.output or os.path.join("results", "scatter", csv_stem)
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, f"scatter_{csv_stem}.png")
        img.save(out)
        print(f"Saved: {out}")
        img.show()


if __name__ == "__main__":
    main()