"""
tune_threshold.py

Tune the quadtree complexity pruning threshold hyperparameter.
Part of the Quadtree Complexity Analysis for Image Forensics pipeline.

Strategy:
    Build each image's quadtree ONCE, then apply different prune cutoffs
    in memory and re-extract features at each threshold value. This avoids
    rebuilding the tree repeatedly and makes the full sweep as fast as a
    single batch run.

    For each threshold value a RandomForest is cross-validated on the
    resulting feature set. The threshold that maximises mean CV accuracy
    is reported as optimal along with a ready-to-run batch.py command.

Usage:
    # Coarse sweep over the full range
    python3 src/tune_thresholds.py --input real_photos ai_images --labels real ai

    # Recommended starting config for compression
    python3 src/tune_thresholds.py --input real_photos ai_images --labels real ai \
        --method compression --leaf_size 16

    # Custom threshold range
    python3 src/tune_thresholds.py --input real_photos ai_images --labels real ai \
        --thresholds 0 5 10 15 20 25 30 35 40 45 50

    # Fast pilot run — cap images per class before committing to the full set
    python3 src/tune_thresholds.py --input real_photos ai_images --labels real ai \
        --max_images 200

    # Refine around a known peak without rerunning the full range
    python3 src/tune_thresholds.py --input real_photos ai_images --labels real ai \
        --method compression --thresholds 28 30 32 34 35 36 38 40 --append

Required args:
    --input     one or more folders of images (one per class)
    --labels    one label per folder, same order as --input

Optional args:
    --thresholds    percentile values to sweep (default: 0 10 20 30 40 50 60)
    --method        shannon|compression|variance (default: shannon)
    --leaf_size     int (default: 4; use 16 for compression)
    --cv            cross-validation folds (default: 5)
    --max_images    cap images per class — useful for fast pilot runs
    --output        base output folder or CSV path (default: results/tuning/)
                    if a folder is given, file is written as {method}.csv inside it
    --append        merge new threshold values into the existing CSV instead
                    of overwriting. duplicate thresholds are replaced by the
                    new run (last write wins).
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from PIL import Image

# Allow running from project root as: python3 src/tune_thresholds.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from complexity import get_scorer
from quadtree import QuadTree, QuadNode, BG_THRESHOLD
from features import extract_features, FEATURE_FIELDS as _ALL_FEATURE_FIELDS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import config
from config import setup_logging, get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Use the scalar features only (no metadata, no spatial grid) for threshold tuning.
# The spatial grid cells are excluded here because tune_thresholds sweeps threshold
# values in memory by re-pruning the same trees — the grid features are less sensitive
# to the pruning threshold than the scalar features and add noise to the CV signal.
_META = {"filename", "label", "label_detail", "is_real", "dataset_source"}
FEATURE_FIELDS = [
    f for f in _ALL_FEATURE_FIELDS
    if f not in _META and not f.startswith("tree_grid_")
]


# ── Tree helpers ─────────────────────────────────────────────────────────────

def deep_copy_tree(root: QuadNode) -> QuadNode:
    """
    Deep copy a QuadNode tree so we can prune it without destroying the original.
    Uses an explicit stack to avoid Python recursion limits on deep trees.
    """
    # Map old node id -> new node
    mapping = {}
    stack = [root]
    while stack:
        node = stack.pop()
        new_node = QuadNode(
            x=node.x, y=node.y, w=node.w, h=node.h,
            depth=node.depth, complexity=node.complexity,
            background_ratio=node.background_ratio,
        )
        mapping[id(node)] = new_node
        stack.extend(node.children)

    # Wire up children
    stack = [root]
    while stack:
        node = stack.pop()
        mapping[id(node)].children = [mapping[id(c)] for c in node.children]
        stack.extend(node.children)

    return mapping[id(root)]


def prune_tree(root: QuadNode, threshold_pct: float) -> QuadNode:
    """
    Copy the tree and prune leaves below the given percentile.
    threshold_pct=0 means no pruning (returns full tree copy).
    """
    tree_copy = deep_copy_tree(root)

    if threshold_pct == 0:
        return tree_copy

    all_leaves = tree_copy.all_leaves()
    subject_leaves = [n for n in all_leaves if n.background_ratio < BG_THRESHOLD]
    if not subject_leaves:
        return tree_copy

    complexities = np.array([n.complexity for n in subject_leaves])
    cutoff = float(np.percentile(complexities, threshold_pct))

    def _prune(node):
        if node.is_leaf:
            return
        for child in node.children:
            _prune(child)
        if all(c.is_leaf and c.complexity < cutoff for c in node.children):
            node.children.clear()

    _prune(tree_copy)
    return tree_copy


# ── Image loading & tree building ────────────────────────────────────────────

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


def build_tree_for_image(args_tuple):
    """Worker: load image and build the full quadtree (no pruning)."""
    filepath, label, method, leaf_size = args_tuple
    try:
        image_array, alpha = load_image(filepath)
        scorer = get_scorer(method)
        qt = QuadTree(scorer=scorer, leaf_size=leaf_size, threshold=None)
        # normalize=False so raw complexity values are preserved for pruning
        root = qt.build(image_array, alpha=alpha, normalize=False)
        filename = os.path.basename(filepath)
        return filename, label, root, None
    except Exception as e:
        import traceback
        return os.path.basename(filepath), label, None, traceback.format_exc()


# ── Feature extraction at a given threshold ──────────────────────────────────

def extract_at_threshold(built_trees, threshold_pct):
    """
    For a list of (filename, label, root) tuples, prune each tree to the
    given threshold and extract features. Returns X, y arrays.
    """
    X, y = [], []
    for filename, label, root in built_trees:
        pruned = prune_tree(root, threshold_pct)
        feat = extract_features(pruned, filename, label=label)
        feat_dict = feat.to_dict()
        row = [float(feat_dict[col]) for col in FEATURE_FIELDS]
        X.append(row)
        y.append(label)
    return np.array(X, dtype=float), np.array(y)


# ── Classifier evaluation ────────────────────────────────────────────────────

def evaluate_threshold(X, y, cv_folds):
    """Cross-validate a RandomForest on the given feature matrix."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


# ── Leaf count stats (to show what pruning is actually doing) ────────────────

def leaf_stats(built_trees, threshold_pct):
    counts = []
    for _, _, root in built_trees:
        pruned = prune_tree(root, threshold_pct)
        subject = [n for n in pruned.all_leaves() if n.background_ratio < BG_THRESHOLD]
        counts.append(len(subject))
    return float(np.mean(counts)), float(np.std(counts))


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Tune quadtree complexity pruning threshold for optimal feature extraction.")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input folders (one per class)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each folder (same order as --input)")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0, 10, 20, 30, 40, 50, 60],
                        help="Percentile thresholds to sweep (default: 0 10 20 30 40 50 60)")
    parser.add_argument("--method", choices=["shannon", "compression", "variance"],
                        default="shannon",
                        help="Complexity scoring method (default: shannon). "
                             "Use compression with --leaf_size 16 for best results.")
    parser.add_argument("--leaf-size", type=int, default=4,
                        help="Target leaf side length in pixels (default: 4; use 16 for compression)")
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds (default: 5)")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images per class (for fast pilot runs)")
    parser.add_argument("--output", default="results/tuning",
                        help="Output folder or CSV path (default: results/tuning). "
                             "If a folder, file is written as {name}/{method}.csv inside it.")
    parser.add_argument("--name", default=None,
                        help="Dataset identifier used in the output path: "
                             "results/tuning/{name}/{method}.csv. "
                             "Auto-derived from the input path if inputs follow the "
                             "data/sample/{name}/{label}/ convention (default: auto).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing tuning CSV instead of overwriting. "
                             "Duplicate thresholds are replaced by the new run.")
    return parser.parse_args()


def infer_name(input_folders: list) -> str:
    """
    Auto-derive a dataset name from input folder paths.

    Flat convention data/sample/{name}_{label}/:
        data/sample/stylegan_v1_synthetic  ->  stylegan_v1
        data/sample/FFHQ_authentic         ->  FFHQ
        data/sample/deepfacelab_manipulated -> deepfacelab

    Legacy nested convention data/sample/{name}/{label}/:
        data/sample/stylegan_v1/synthetic  ->  stylegan_v1

    Falls back to the basename of the first input if neither pattern matches.
    """
    for folder in input_folders:
        parts = folder.replace("\\", "/").rstrip("/").split("/")
        if "sample" in parts:
            idx = parts.index("sample")
            if idx + 1 < len(parts):
                folder_name = parts[idx + 1]
                # Flat: {name}_{label} — strip known label suffixes
                for suffix in ("_authentic", "_synthetic", "_manipulated"):
                    if folder_name.endswith(suffix):
                        return folder_name[: -len(suffix)]
                # Legacy nested: next part is the name, part after is the label
                if idx + 2 < len(parts):
                    return folder_name
    # Fallback: basename of first input
    return os.path.basename(os.path.abspath(input_folders[0]))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    setup_logging(getattr(args, "verbose", False))
    if len(args.input) != len(args.labels):
        log.error("--input and --labels must have the same number of entries.")
        sys.exit(1)

    # Collect image paths per class
    tasks = []
    class_counts = {}
    for folder, label in zip(args.input, args.labels):
        entries = sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        if args.max_images:
            entries = entries[:args.max_images]
        class_counts[label] = len(entries)
        for fname in entries:
            tasks.append((os.path.join(folder, fname), label, args.method, args.leaf_size))

    # Resolve name early — used in both the startup print and output path
    name = args.name or infer_name(args.input)

    log.info("Threshold Tuning")

    log.info("Name:      %s", name)
    log.info("Method:    %s", args.method)
    log.info("Leaf size: %dpx", args.leaf_size)
    log.info("Thresholds: %s", args.thresholds)
    log.info("CV folds:  %d", args.cv)
    log.info("Classes:   %s", class_counts)
    log.info("Total images: %d", len(tasks))

    # ── Phase 1: Build all trees once ────────────────────────────────────────
    log.info("Phase 1: Building quadtrees...")
    t0 = time.time()

    built_trees = []  # list of (filename, label, root)
    errors = 0

    for i, task in enumerate(tasks, 1):
        fname, label, root, err = build_tree_for_image(task)
        if err:
            log.error("  %s: %s", fname, err.splitlines()[-1])
            errors += 1
        else:
            built_trees.append((fname, label, root))
        if i % 100 == 0 or i == len(tasks):
            log.info("  %d/%d trees built...", i, len(tasks))

    build_time = time.time() - t0
    log.info("Built %d trees in %.1fs  (%d errors)", len(built_trees), build_time, errors)

    if not built_trees:
        log.error("No trees built — check your input paths.")
        sys.exit(1)

    # ── Phase 2: Sweep thresholds ─────────────────────────────────────────────
    log.info("Phase 2: Sweeping %d threshold values...", len(args.thresholds))
    print(f"\n{'Threshold':>10}  {'Mean Leaves':>12}  {'CV Accuracy':>12}  {'± Std':>8}  {'Time':>8}")
    print(f"{'─'*10}  {'─'*12}  {'─'*12}  {'─'*8}  {'─'*8}")

    results = []
    best_acc = -1
    best_threshold = None

    for thr in sorted(args.thresholds):
        t1 = time.time()

        # Extract features at this threshold
        X, y = extract_at_threshold(built_trees, thr)

        # Leaf count stats (diagnostic — shows what pruning is doing)
        mean_leaves, std_leaves = leaf_stats(built_trees, thr)

        # Evaluate
        mean_acc, std_acc = evaluate_threshold(X, y, args.cv)
        elapsed = time.time() - t1

        marker = "  ◀ best" if mean_acc > best_acc else ""
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_threshold = thr

        print(f"{thr:>10.0f}  {mean_leaves:>12.1f}  {mean_acc:>12.4f}  {std_acc:>8.4f}  {elapsed:>7.1f}s{marker}")

        results.append({
            "threshold": thr,
            "mean_leaves": mean_leaves,
            "std_leaves": std_leaves,
            "cv_accuracy_mean": mean_acc,
            "cv_accuracy_std": std_acc,
        })

    # Resolve output path: results/tuning/{name}/{method}.csv
    output_arg = args.output
    if output_arg.endswith(".csv"):
        out_path = output_arg
    else:
        out_path = os.path.join(output_arg.rstrip("/"), name, f"{args.method}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────────

    log.info("Best threshold: %s  (CV accuracy: %.4f)", best_threshold, best_acc)
    log.info("Recommendation:")
    log.info("  python3 src/batch.py --input <folder> --output results/features/ \\")
    log.info("      --method %s --leaf-size %d \\", args.method, args.leaf_size)
    log.info("      --threshold %s", best_threshold)

    # ── Save results ──────────────────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    if args.append and os.path.exists(out_path):
        # Load existing rows, index by threshold so new run overwrites duplicates
        with open(out_path, newline="") as f:
            existing = list(csv.DictReader(f))
        merged = {float(r["threshold"]): r for r in existing}
        for r in results:
            merged[r["threshold"]] = r # new run wins on duplicates
        merged_rows = sorted(merged.values(), key=lambda r: float(r["threshold"]))
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
        log.info("Appended %d entries → %s (%d total)", len(results), out_path, len(merged_rows))
    else:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        log.info("Results saved: %s", out_path)

    # ── Write best threshold to shared JSON ───────────────────────────────────
    # Keyed as {name}_{method} so the Makefile can read thresholds for phase 2
    # without hard-coding them. All tune runs merge into the same file so running
    # tune-stylegan then tune-deepfacelab produces a single complete lookup table.
    import json
    best_path = os.path.join("results", "tuning", "best_thresholds.json")
    try:
        with open(best_path) as f:
            best_all = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        best_all = {}

    key = f"{name}_{args.method}"
    best_all[key] = best_threshold
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    with open(best_path, "w") as f:
        json.dump(best_all, f, indent=2)
    log.info("Best threshold [%s] = %s  →  %s", key, best_threshold, best_path)


if __name__ == "__main__":
    main()