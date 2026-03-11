"""
tune_threshold.py

Tune the quadtree pruning threshold hyperparameter without trial and error.

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
    python3 tune_threshold.py --input real_photos ai_images --labels real ai

    # Recommended starting config for compression
    python3 tune_threshold.py --input real_photos ai_images --labels real ai \
        --method compression --leaf_size 16

    # Custom threshold range
    python3 tune_threshold.py --input real_photos ai_images --labels real ai \
        --thresholds 0 5 10 15 20 25 30 35 40 45 50

    # Fast pilot run — cap images per class before committing to the full set
    python3 tune_threshold.py --input real_photos ai_images --labels real ai \
        --max_images 200

    # Refine around a known peak without rerunning the full range
    python3 tune_threshold.py --input real_photos ai_images --labels real ai \
        --method compression --thresholds 28 30 32 34 35 36 38 40 --append

Required args:
    --input     one or more folders of images (one per class)
    --labels    one label per folder, same order as --input

Optional args:
    --thresholds    percentile values to sweep (default: 0 10 20 30 40 50 60)
    --method        shannon|compression|variance (default: shannon)
    --leaf_size     int (default: 4; use 16 for compression)
    --cv            cross-validation folds (default: 5)
    --workers       parallel image loading workers (default: 4)
    --max_images    cap images per class — useful for fast pilot runs
    --output        base output CSV path (default: tune_results.csv)
                    actual file is written as tune_results_{method}.csv
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

from complexity import get_scorer
from quadtree import QuadTree, QuadNode, BG_THRESHOLD
from features import extract_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

FEATURE_FIELDS = [
    "mean_complexity", "std_complexity", "min_complexity", "max_complexity",
    "complexity_range", "mean_leaf_area", "std_leaf_area", "leaf_count",
    "mean_boundary_delta", "max_boundary_delta", "mean_depth", "std_depth",
    "mean_merge_delta", "max_merge_delta", "std_merge_delta",
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
    parser = argparse.ArgumentParser(description="Tune quadtree pruning threshold.")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input folders (one per class)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for each folder (same order as --input)")
    parser.add_argument("--thresholds", nargs="+", type=float,
                        default=[0, 10, 20, 30, 40, 50, 60],
                        help="Percentile thresholds to sweep (default: 0 10 20 30 40 50 60)")
    parser.add_argument("--method", choices=["shannon", "compression", "variance"],
                        default="shannon")
    parser.add_argument("--leaf_size", type=int, default=4)
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds (default: 5)")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()),
                        help="Parallel workers for image loading (default: 4)")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Max images per class (for fast pilot runs)")
    parser.add_argument("--output", default="tune_results.csv",
                        help="Output CSV path (default: tune_results.csv)")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing features.csv instead of overwriting")
    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if len(args.input) != len(args.labels):
        print("Error: --input and --labels must have the same number of entries.")
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

    print(f"\nThreshold Tuning")
    print(f"{'─'*50}")
    print(f"Method:      {args.method}")
    print(f"Leaf size:   {args.leaf_size}px")
    print(f"Thresholds:  {args.thresholds}")
    print(f"CV folds:    {args.cv}")
    print(f"Workers:     {args.workers}")
    print(f"Classes:     {class_counts}")
    print(f"Total images:{len(tasks)}\n")

    # ── Phase 1: Build all trees once ────────────────────────────────────────
    print(f"Phase 1: Building quadtrees (this is the slow part)...")
    t0 = time.time()

    built_trees = []  # list of (filename, label, root)
    errors = 0

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            for i, (fname, label, root, err) in enumerate(
                pool.imap_unordered(build_tree_for_image, tasks), 1
            ):
                if err:
                    print(f"  ERROR {fname}: {err.splitlines()[-1]}")
                    errors += 1
                else:
                    built_trees.append((fname, label, root))
                if i % 100 == 0 or i == len(tasks):
                    print(f"  {i}/{len(tasks)} trees built...")
    else:
        for i, task in enumerate(tasks, 1):
            fname, label, root, err = build_tree_for_image(task)
            if err:
                print(f"  ERROR {fname}: {err.splitlines()[-1]}")
                errors += 1
            else:
                built_trees.append((fname, label, root))
            if i % 100 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} trees built...")

    build_time = time.time() - t0
    print(f"\nBuilt {len(built_trees)} trees in {build_time:.1f}s  ({errors} errors)\n")

    if not built_trees:
        print("No trees built — check your input paths.")
        sys.exit(1)

    # ── Phase 2: Sweep thresholds ─────────────────────────────────────────────
    print(f"Phase 2: Sweeping {len(args.thresholds)} threshold values...")
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

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"Best threshold: {best_threshold}  (CV accuracy: {best_acc:.4f})")
    print(f"\nRecommendation:")
    print(f"  python batch.py --input <folder> --output results \\")
    print(f"      --method {args.method} --leaf_size {args.leaf_size} \\")
    print(f"      --threshold {best_threshold}")

    # ── Save results ──────────────────────────────────────────────────────────
    fieldnames = list(results[0].keys())
    if args.append and os.path.exists(args.output):
        # Load existing rows, index by threshold so new run overwrites duplicates
        with open(args.output, newline="") as f:
            existing = list(csv.DictReader(f))
        merged = {float(r["threshold"]): r for r in existing}
        for r in results:
            merged[r["threshold"]] = r # new run wins on duplicated
        merged_rows = sorted(merged.values(), key=lambda r: float(r["threshold"]))
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f"\nAppended {len(results)} entries -> {args.output} ({len(merged_rows)} total)")
    else:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()