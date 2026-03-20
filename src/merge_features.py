"""
merge_features.py
-----------------
Merge feature CSVs from two (or more) scoring methods into a single wide CSV
for use with classify.py. Each method's feature columns are prefixed with the
method name, and a set of cross-method derived features is appended.

Why this helps:
    Shannon entropy captures tonal diversity (pixel value histogram).
    Compression ratio captures structural repetition (zlib compressibility).
    They are partially independent: AI-generated regions often have high Shannon
    entropy but low compression ratio because their "noise" is structurally
    repetitive in ways zlib can exploit. The ratio and delta features below
    directly encode this divergence as a signal for the RF.

Usage:
    # Merge shannon + compression (most common case)
    python3 src/merge_features.py \\
        --inputs results/features/shannon.csv results/features/compression.csv \\
        --methods shannon compression \\
        --output  results/features/merged_shannon_compression.csv

    # Three-way merge
    python3 src/merge_features.py \\
        --inputs  results/features/shannon.csv results/features/compression.csv results/features/variance.csv \\
        --methods shannon compression variance \\
        --output  results/features/merged_all.csv

    # Then classify with all features (RF, GB work best here)
    python3 src/classify.py results/features/merged_shannon_compression.csv
    python3 src/classify.py results/features/merged_shannon_compression.csv --fast

Output columns:
    filename, label, label_detail, is_real, dataset_source
    {method_a}_{feature}, {method_b}_{feature}, ...   (prefixed per-method features)
    cross_{a}_{b}_mean_ratio                          (mean_complexity_a / mean_complexity_b)
    cross_{a}_{b}_std_ratio                           (std_complexity_a  / std_complexity_b)
    cross_{a}_{b}_mean_delta                          (mean_complexity_a - mean_complexity_b)
    cross_{a}_{b}_std_delta                           (std_complexity_a  - std_complexity_b)
    cross_{a}_{b}_boundary_ratio                      (mean_boundary_delta_a / mean_boundary_delta_b)
    cross_{a}_{b}_merge_ratio                         (mean_merge_delta_a    / mean_merge_delta_b)

The cross_* features are the highest-value additions: the ratio of Shannon to
compression complexity directly measures how much of the region's "complexity"
is structural repetition vs. true entropy — a strong AI-vs-real discriminator.
"""

import argparse
import csv
import os
import sys

import numpy as np

# Allow running from project root as: python3 src/merge_features.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Metadata columns that are shared — join key + pass-through fields
META_COLS = {"filename", "label", "label_detail", "is_real", "dataset_source"}

# Feature columns to prefix and cross-compare (skip tree_grid_* by default
# because 256 spatial cells × N methods = very wide; use --include-grid to opt in)
CROSS_TARGETS = [
    "mean_complexity",
    "std_complexity",
    "mean_boundary_delta",
    "mean_merge_delta",
    "std_merge_delta",
    "max_merge_delta",
    "complexity_range",
]

EPS = 1e-9


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge feature CSVs from multiple methods into a wide CSV."
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="Feature CSVs to merge, one per method (same order as --methods)"
    )
    parser.add_argument(
        "--methods", nargs="+", required=True,
        help="Short method name for each CSV (e.g. shannon compression variance)"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for merged CSV"
    )
    parser.add_argument(
        "--include-grid", action="store_true",
        help="Include tree_grid_* spatial cells from every method (adds 256×N columns)"
    )
    parser.add_argument(
        "--no-cross", action="store_true",
        help="Skip cross-method derived features"
    )
    parser.add_argument(
        "--join", choices=["inner", "outer"], default="inner",
        help="How to handle filenames present in only some CSVs (default: inner)"
    )
    return parser.parse_args()


def load_csv(path: str) -> dict:
    """Load CSV into {filename: row_dict}."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        fn = row.get("filename", "").strip()
        if fn:
            result[fn] = row
    return result


def get_feature_cols(row: dict, include_grid: bool) -> list:
    """Return the non-meta feature column names from a sample row."""
    cols = []
    for k in row:
        if k in META_COLS:
            continue
        if not include_grid and k.startswith("tree_grid_"):
            continue
        cols.append(k)
    return cols


def safe_float(val) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def build_cross_features(merged_row: dict, pairs: list) -> dict:
    """
    Compute cross-method ratio and delta features for every (method_a, method_b) pair.

    Returns a dict of new column_name -> float entries.
    """
    cross = {}
    for a, b in pairs:
        for feat in CROSS_TARGETS:
            col_a = f"{a}_{feat}"
            col_b = f"{b}_{feat}"
            val_a = safe_float(merged_row.get(col_a, 0.0))
            val_b = safe_float(merged_row.get(col_b, 0.0))

            prefix = f"cross_{a}_{b}_{feat}"
            cross[f"{prefix}_ratio"] = val_a / (val_b + EPS)
            cross[f"{prefix}_delta"] = val_a - val_b

    return cross


def main():
    args = parse_args()

    if len(args.inputs) != len(args.methods):
        print("Error: --inputs and --methods must have the same number of entries.")
        sys.exit(1)

    # Load all CSVs
    datasets = {}
    for path, method in zip(args.inputs, args.methods):
        if not os.path.exists(path):
            print(f"Error: file not found: {path}")
            sys.exit(1)
        data = load_csv(path)
        datasets[method] = data
        print(f"Loaded {len(data)} rows  [{method}]  {path}")

    # Determine filename universe
    all_filenames = [set(d.keys()) for d in datasets.values()]
    if args.join == "inner":
        filenames = sorted(set.intersection(*all_filenames))
        dropped = sum(len(s) for s in all_filenames) - len(filenames) * len(datasets)
        if dropped:
            print(f"Inner join: {dropped} row(s) dropped (not present in all CSVs)")
    else:
        filenames = sorted(set.union(*all_filenames))
        print(f"Outer join: {len(filenames)} unique filenames")

    if not filenames:
        print("Error: no filenames survive the join. Check that CSVs share filenames.")
        sys.exit(1)

    print(f"Merging {len(filenames)} images across {len(datasets)} methods...")

    # Determine feature columns per method (from first non-empty row)
    method_feat_cols = {}
    for method, data in datasets.items():
        sample = next(iter(data.values())) if data else {}
        method_feat_cols[method] = get_feature_cols(sample, args.include_grid)

    # Build output fieldnames
    out_fields = ["filename", "label", "label_detail", "is_real", "dataset_source"]
    for method, feat_cols in method_feat_cols.items():
        for col in feat_cols:
            out_fields.append(f"{method}_{col}")

    # Cross-method pair names (all upper-triangle pairs)
    methods_list = args.methods
    pairs = [
        (methods_list[i], methods_list[j])
        for i in range(len(methods_list))
        for j in range(i + 1, len(methods_list))
    ]

    if not args.no_cross and pairs:
        # Pre-compute cross field names from a dummy merged row
        dummy = {f"{m}_{f}": 0.0 for m, fcols in method_feat_cols.items() for f in fcols}
        cross_dummy = build_cross_features(dummy, pairs)
        out_fields.extend(sorted(cross_dummy.keys()))

    # Merge rows
    merged_rows = []
    for fn in filenames:
        out_row = {"filename": fn}

        # Meta: take from whichever method has this file
        for method, data in datasets.items():
            row = data.get(fn)
            if row:
                out_row.setdefault("label",          row.get("label", ""))
                out_row.setdefault("label_detail",   row.get("label_detail", ""))
                out_row.setdefault("is_real",        row.get("is_real", ""))
                out_row.setdefault("dataset_source", row.get("dataset_source", ""))
                break

        # Prefixed features
        for method, feat_cols in method_feat_cols.items():
            row = datasets[method].get(fn, {})
            for col in feat_cols:
                out_row[f"{method}_{col}"] = safe_float(row.get(col, 0.0)) if row else 0.0

        # Cross features
        if not args.no_cross and pairs:
            cross = build_cross_features(out_row, pairs)
            out_row.update(cross)

        merged_rows.append(out_row)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({k: row.get(k, "") for k in out_fields})

    n_feat = len(out_fields) - len(META_COLS)
    print(f"\nSaved: {args.output}")
    print(f"  Rows     : {len(merged_rows)}")
    print(f"  Features : {n_feat}  ({len(out_fields)} total columns)")
    if not args.no_cross and pairs:
        n_cross = len(cross_dummy)
        print(f"  Cross    : {n_cross} derived features from {len(pairs)} method pair(s)")
    print()
    print("Next step:")
    print(f"  python3 src/classify.py {args.output}")
    print(f"  python3 src/classify.py {args.output} --prune-features 0.0")


if __name__ == "__main__":
    main()