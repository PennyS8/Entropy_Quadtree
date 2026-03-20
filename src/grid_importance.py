"""
grid_importance.py

Visualize which spatial cells of the 16x16 tree_grid are most significant
for distinguishing image classes.

Fits a Random Forest on the features CSV, computes permutation importance
for every tree_grid_* feature, and renders them as a 16x16 heatmap —
showing exactly which regions of the face the classifier relies on most.

Optionally overlays the heatmap on a reference image so the spatial
significance map is anchored to real anatomy.

Usage:
    # Heatmap only
    python3 src/grid_importance.py results/features/compression.csv

    # Overlay on a reference face image
    python3 src/grid_importance.py results/features/compression.csv \\
        --image real_00079.jpg

    # Show pairwise class comparison (which cells separate real vs ai)
    python3 src/grid_importance.py results/features/compression.csv \\
        --compare real ai

    # Custom output path
    python3 src/grid_importance.py results/features/compression.csv \\
        --output results/grid_importance/compression.png

Output:
    PNG saved to results/grid_importance/{csv_stem}.png
    (or --output path)
"""

import argparse
import csv
import io
import json
import os
import sys
import zipfile
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import load_csv, FEATURE_FIELDS

KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


def _get_kaggle_credentials():
    for path in (
        os.path.expanduser("~/.config/kaggle/kaggle.json"),
        os.path.expanduser("~/.kaggle/kaggle.json"),
    ):
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            u = cfg.get("username", "").strip()
            k = cfg.get("key", "").strip()
            if u and k:
                return u, k
    print("Error: kaggle.json not found.")
    sys.exit(1)


def _kaggle_session():
    u, k = _get_kaggle_credentials()
    s = requests.Session()
    s.auth = (u, k)
    s.headers["User-Agent"] = "grid_importance/1.0"
    return s


def _download_zip_buf(session, dataset):
    owner, slug = dataset.split("/", 1)
    url  = f"{KAGGLE_API_BASE}/datasets/download/{owner}/{slug}"
    resp = session.get(url, stream=True, timeout=180, allow_redirects=True)
    if resp.status_code == 401:
        print("Error: Kaggle authentication failed.")
        sys.exit(1)
    resp.raise_for_status()
    total = int(resp.headers.get("Content-Length", 0))
    buf   = io.BytesIO()
    done  = 0
    for chunk in resp.iter_content(1 << 20):
        buf.write(chunk)
        done += len(chunk)
        if total:
            print(f"  {done >> 20}/{total >> 20} MB", end="\r")
    print()
    buf.seek(0)
    return buf


def fetch_image_from_kaggle(sources, filename):
    """
    Search for `filename` across multiple Kaggle dataset zips.

    `sources` is a list of (dataset_slug, zip_prefix) tuples, tried in order.
    Each zip is downloaded into memory; the first one containing the file wins.
    Nothing is written to disk.
    """
    session = _kaggle_session()
    for dataset, zip_prefix in sources:
        print(f"  Searching {dataset} for {filename}...")
        buf = _download_zip_buf(session, dataset)
        with zipfile.ZipFile(buf) as zf:
            candidates = [
                n for n in zf.namelist()
                if n.endswith("/" + filename) or n == filename
            ]
            if zip_prefix:
                prefixed = [c for c in candidates if c.startswith(zip_prefix)]
                if prefixed:
                    candidates = prefixed
            if not candidates:
                print(f"  Not found in {dataset}, trying next source...")
                continue
            entry = candidates[0]
            print(f"  Found: {entry}")
            img_bytes = zf.read(entry)
        img = Image.open(io.BytesIO(img_bytes))
        img.load()
        return img.convert("RGB")
    raise FileNotFoundError(
        f"'{filename}' not found in any of the provided datasets: "
        f"{[d for d, _ in sources]}"
    )

GRID_SIZE   = 16   # 16x16 = 256 cells
N_CELLS     = GRID_SIZE * GRID_SIZE
GRID_NAMES  = [f"tree_grid_{i:03d}" for i in range(N_CELLS)]


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize tree_grid spatial importance from a features CSV."
    )
    parser.add_argument("csv", help="Path to features CSV")
    parser.add_argument("--image", default=None,
                        help="Optional reference image to overlay the heatmap on")
    parser.add_argument("--compare", nargs=2, default=None, metavar=("CLASS_A", "CLASS_B"),
                        help="Show mean complexity difference between two classes "
                             "instead of RF importance (e.g. --compare authentic synthetic)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path. Default: results/grid_importance/{csv_stem}.png")
    parser.add_argument("--cell-size", type=int, default=40,
                        help="Pixel size of each grid cell in the output (default: 40)")
    parser.add_argument("--alpha", type=int, default=180,
                        help="Overlay opacity 0-255 (default: 180)")
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance classes before fitting RF (default: True)")
    parser.add_argument("--permutation", action="store_true",
                        help="Use permutation importance instead of impurity-based (slower, sparser)")
    parser.add_argument("--median-image", default=None, metavar="LABEL",
                        help="Use the image closest to the class median for LABEL as overlay reference")
    parser.add_argument("--image-dir", default=".",
                        help="Directory containing images named in the CSV (default: .)")
    parser.add_argument("--download-median", nargs="+", default=None, metavar="DATASET",
                        help="One or more Kaggle dataset slugs to search for the median image. "
                             "Tried in order until the file is found. "
                             "e.g. --download-median ciplab/real-and-fake-face-detection "
                             "wish096/realvsfake-81k-by-wish")
    parser.add_argument("--kaggle-prefix", nargs="+", default=None,
                        help="Zip prefix for each dataset in --download-median (same order). "
                             "Use empty string for no prefix. "
                             "e.g. --kaggle-prefix real_and_fake_face/training_real "
                             "RealVsFake/RealVsFake/Real")
    return parser.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_grid_data(csv_path, balance=True):
    """
    Load the tree_grid features and labels from a CSV.
    Returns (x, y, filenames) where x is (N, 256).
    """
    from sklearn.utils import resample

    rows = load_csv(csv_path)
    x, y, filenames = [], [], []
    for row in rows:
        label = row.get("label", "").strip()
        if not label:
            continue
        try:
            vals = [float(row[g]) for g in GRID_NAMES]
        except (KeyError, ValueError):
            continue
        x.append(vals)
        y.append(label)
        filenames.append(row.get("filename", ""))

    x         = np.array(x, dtype=np.float32)
    y         = np.array(y)
    filenames = np.array(filenames)

    if balance:
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        xb, yb, fb = [], [], []
        for cls in classes:
            mask = y == cls
            xc, yc, fc = resample(x[mask], y[mask], filenames[mask],
                                   n_samples=min_count, random_state=42)
            xb.append(xc); yb.append(yc); fb.append(fc)
        x         = np.vstack(xb)
        y         = np.concatenate(yb)
        filenames = np.concatenate(fb)

    return x, y, filenames


def find_median_image(x, y, filenames, label):
    """
    Return (filename, mean_vector) for the image in `label` whose
    tree_grid vector is closest (L2) to the class mean.
    """
    mask     = y == label
    x_class  = x[mask]
    fn_class = filenames[mask]
    mean_vec = x_class.mean(axis=0)
    dists    = np.linalg.norm(x_class - mean_vec, axis=1)
    best     = int(np.argmin(dists))
    return fn_class[best], mean_vec


# ── Importance computation ────────────────────────────────────────────────────

def compute_rf_importance(x, y, permutation=False):
    """
    Fit a Random Forest and return importances for all 256 cells.

    permutation=False (default): impurity-based importance — distributes
        signal across all 256 cells, giving a dense readable heatmap.
    permutation=True: permutation importance — more statistically reliable
        but concentrates signal on a handful of cells (sparse heatmap).

    Returns a (256,) float array, normalised to [0, 1].
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.inspection import permutation_importance as perm_imp

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(x_tr, y_tr)
    acc = rf.score(x_te, y_te)
    print(f"  RF accuracy on tree_grid features: {acc:.4f}")

    if permutation:
        result = perm_imp(rf, x_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
        imp = result.importances_mean
    else:
        imp = rf.feature_importances_  # impurity-based (Gini)

    imp = np.clip(imp, 0, None)
    if imp.max() > 0:
        imp = imp / imp.max()
    return imp


def compute_class_delta(x, y, class_a, class_b):
    """
    Compute mean complexity difference per cell between two classes.
    Returns a (256,) float array in [-1, 1], where positive = class_a higher.
    """
    mask_a = y == class_a
    mask_b = y == class_b
    if not mask_a.any() or not mask_b.any():
        raise ValueError(f"Classes '{class_a}' or '{class_b}' not found. "
                         f"Available: {sorted(set(y))}")
    delta = x[mask_a].mean(axis=0) - x[mask_b].mean(axis=0)
    # Normalise to [-1, 1]
    max_abs = np.abs(delta).max()
    if max_abs > 0:
        delta = delta / max_abs
    return delta


# ── Colormap ──────────────────────────────────────────────────────────────────

def importance_to_color(v: float) -> tuple:
    """
    Map importance in [0, 1] to RGBA.
    Low importance -> dark blue, high -> bright red/yellow.
    """
    t = max(0.0, min(1.0, v))
    if t < 0.25:
        s = t / 0.25
        return (0, int(s * 100), int(150 + s * 55), 255)
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        return (0, int(100 + s * 155), int(205 - s * 205), 255)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        return (int(s * 255), int(255 - s * 155), 0, 255)
    else:
        s = (t - 0.75) / 0.25
        return (255, int(100 - s * 100), 0, 255)


def delta_to_color(v: float) -> tuple:
    """
    Map signed delta in [-1, 1] to RGBA.
    Negative (class_b higher) -> blue, zero -> grey, positive (class_a higher) -> red.
    """
    t = max(-1.0, min(1.0, v))
    if t < 0:
        s = -t
        return (int(40 * (1 - s)), int(40 * (1 - s)), int(40 + 215 * s), 255)
    else:
        return (int(40 + 215 * t), int(40 * (1 - t)), int(40 * (1 - t)), 255)


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_grid_heatmap(
    values: np.ndarray,
    cell_size: int = 40,
    colormap_fn=importance_to_color,
    title: str = "",
    label_top5: bool = True,
) -> Image.Image:
    """
    Render a 16x16 heatmap from a (256,) values array.
    Cells are drawn in Morton order (NW→NE→SW→SE = left-right, top-bottom).
    Top-5 most significant cells are outlined in white.
    """
    pad    = 40
    w      = GRID_SIZE * cell_size + pad * 2
    h      = GRID_SIZE * cell_size + pad * 2 + (30 if title else 0)

    img  = Image.new("RGB", (w, h), (18, 18, 18))
    draw = ImageDraw.Draw(img)

    top5_idx = set(np.argsort(np.abs(values))[-5:])

    for idx in range(N_CELLS):
        row = idx // GRID_SIZE
        col = idx %  GRID_SIZE
        x0  = pad + col * cell_size
        y0  = pad + row * cell_size + (30 if title else 0)
        x1  = x0 + cell_size - 1
        y1  = y0 + cell_size - 1

        color = colormap_fn(values[idx])
        draw.rectangle([x0, y0, x1, y1], fill=color[:3])

        # White outline for top-5
        if label_top5 and idx in top5_idx:
            draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
            # Cell index label
            rank = sorted(top5_idx, key=lambda i: -abs(values[i])).index(idx) + 1
            draw.text((x0 + 2, y0 + 2), str(rank), fill=(255, 255, 255))

    # Grid lines
    for i in range(GRID_SIZE + 1):
        x = pad + i * cell_size
        y = pad + (30 if title else 0)
        draw.line([(x, y), (x, y + GRID_SIZE * cell_size)], fill=(50, 50, 50), width=1)
        draw.line([(pad, y + i * cell_size), (pad + GRID_SIZE * cell_size, y + i * cell_size)],
                  fill=(50, 50, 50), width=1)

    if title:
        draw.text((pad, 8), title, fill=(220, 220, 220))

    return img


def render_legend(width: int, is_delta: bool = False) -> Image.Image:
    """Render a horizontal colorbar legend."""
    h   = 30
    leg = Image.new("RGB", (width, h + 20), (18, 18, 18))
    draw = ImageDraw.Draw(leg)
    fn = delta_to_color if is_delta else importance_to_color

    for x in range(width):
        t = x / (width - 1)
        v = (t * 2 - 1) if is_delta else t
        c = fn(v)
        draw.line([(x, 0), (x, h - 1)], fill=c[:3])

    if is_delta:
        draw.text((2,      h + 2), "← class_b higher", fill=(150, 150, 150))
        draw.text((width // 2 - 10, h + 2), "equal", fill=(150, 150, 150))
        draw.text((width - 90, h + 2), "class_a higher →", fill=(150, 150, 150))
    else:
        draw.text((2,          h + 2), "Low importance", fill=(150, 150, 150))
        draw.text((width - 80, h + 2), "High →",         fill=(150, 150, 150))

    return leg


def overlay_on_image(heatmap: Image.Image, ref_path, alpha: int = 180) -> Image.Image:
    """
    Composite the heatmap over a reference image.
    The heatmap is resized to match the reference image dimensions.
    """
    if isinstance(ref_path, Image.Image):
        ref = ref_path.convert("RGB")
    else:
        ref = Image.open(ref_path).convert("RGB")
    grid  = heatmap.resize(ref.size, Image.NEAREST).convert("RGBA")

    # Make the heatmap semi-transparent
    r, g, b, a = grid.split()
    a = a.point(lambda x: int(x * alpha / 255))
    grid = Image.merge("RGBA", (r, g, b, a))

    base = ref.convert("RGBA")
    result = Image.alpha_composite(base, grid)
    return result.convert("RGB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found")
        sys.exit(1)

    csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
    out_path = args.output or os.path.join("results", "grid_importance", f"{csv_stem}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Loading {args.csv}...")
    x, y, filenames = load_grid_data(args.csv, balance=args.balance)
    classes = sorted(set(y))
    print(f"Classes: {classes}  Samples: {len(y)}")

    is_delta = args.compare is not None

    if is_delta:
        class_a, class_b = args.compare
        print(f"Computing mean delta: {class_a} vs {class_b}...")
        values = compute_class_delta(x, y, class_a, class_b)
        title  = f"tree_grid  mean({class_a}) − mean({class_b})  [{csv_stem}]"
        colormap_fn = delta_to_color
    else:
        print("Fitting Random Forest and computing permutation importance...")
        values = compute_rf_importance(x, y, permutation=args.permutation)
        mode   = "permutation importance" if args.permutation else "impurity importance"
        title  = f"tree_grid  RF {mode}  [{csv_stem}]"
        colormap_fn = importance_to_color

    # Print top-10 cells
    top10 = np.argsort(np.abs(values))[::-1][:10]
    print(f"\nTop 10 most significant cells (row, col):")
    for rank, idx in enumerate(top10, 1):
        row, col = divmod(idx, GRID_SIZE)
        print(f"  {rank:>2}. cell {idx:>3}  (row {row:>2}, col {col:>2})  "
              f"value={values[idx]:+.4f}")

    # Render heatmap
    heatmap = render_grid_heatmap(
        values,
        cell_size=args.cell_size,
        colormap_fn=colormap_fn,
        title=title,
    )
    legend = render_legend(heatmap.width, is_delta=is_delta)

    # Combine heatmap + legend
    combined = Image.new("RGB", (heatmap.width, heatmap.height + legend.height), (18, 18, 18))
    combined.paste(heatmap, (0, 0))
    combined.paste(legend, (0, heatmap.height))

    ref_image_path = args.image
    ref_image_pil  = None  # in-memory PIL image (used when downloaded from Kaggle)
    if args.median_image:
        median_fn, _ = find_median_image(x, y, filenames, args.median_image)
        print(f"Median image for '{args.median_image}': {median_fn}")
        if args.download_median:
            # Build (dataset, prefix) pairs — pad prefixes with '' if fewer given
            prefixes = args.kaggle_prefix or []
            sources  = [
                (ds, prefixes[i] if i < len(prefixes) else "")
                for i, ds in enumerate(args.download_median)
            ]
            ref_image_pil = fetch_image_from_kaggle(sources, median_fn)
            ref_image_path = "__in_memory__"
        else:
            candidate = os.path.join(args.image_dir, median_fn)
            if os.path.exists(candidate):
                ref_image_path = candidate
            else:
                print(f"Warning: '{median_fn}' not found in --image-dir '{args.image_dir}'")
                print(f"  Use --download-median DATASET_SLUG to fetch it from Kaggle.")
                ref_image_path = None

    if ref_image_path:
        display_path = median_fn if ref_image_pil else ref_image_path
        print(f"Overlaying on {display_path}...")
        # Crop just the grid portion (no padding, no title) for overlay
        pad  = 40
        grid_only = heatmap.crop((pad, pad + 30, pad + GRID_SIZE * args.cell_size,
                                   pad + 30 + GRID_SIZE * args.cell_size))
        ref_arg = ref_image_pil if ref_image_pil is not None else ref_image_path

        # Composite heatmap directly onto the reference image
        overlay = overlay_on_image(grid_only, ref_arg, alpha=args.alpha)

        # Stack: overlay image on top, standalone heatmap+legend below
        # The overlay IS the primary output; the heatmap below gives the
        # unobstructed cell values for reference.
        legend_w = max(overlay.width, combined.width)
        legend   = render_legend(legend_w, is_delta=is_delta)

        total_h = overlay.height + combined.height
        canvas  = Image.new("RGB", (legend_w, total_h), (18, 18, 18))
        canvas.paste(overlay,  (0, 0))
        canvas.paste(combined, (0, overlay.height))
        canvas.save(out_path)
    else:
        combined.save(out_path)

    print(f"\nSaved: {out_path}")
    print('\a')


if __name__ == "__main__":
    main()