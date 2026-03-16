"""
classify.py
-----------
Train and evaluate classifiers on extracted quadtree features.

Default — four classifiers:
  1. Logistic Regression  — linear baseline, interpretable weights
  2. Linear SVM           — fast linear separator, scales to large datasets
  3. Random Forest        — handles feature interactions, gives importances
  4. Gradient Boosting    — highest accuracy ceiling for tabular data

--fast mode — two fast classifiers only (seconds, not minutes):
  1. Logistic Regression
  2. Random Forest

Usage:
    python3 src/classify.py results/features/compression.csv
    python3 src/classify.py results/features/compression.csv --fast
    python3 src/classify.py results/features/compression.csv --test-size 0.2
    python3 src/classify.py results/features/compression.csv --features mean_complexity std_complexity
    python3 src/classify.py results/features/compression.csv --balance
    python3 src/classify.py results/features/compression.csv --permutation-importance
    python3 src/classify.py results/features/compression.csv --prune-features 0.0
    python3 src/classify.py results/features/compression.csv --prune-features 0.0005

--prune-features THRESHOLD
    Fits a Random Forest on the full feature set, computes permutation
    importance, then discards any feature whose mean permutation importance
    is below THRESHOLD (default 0.0 drops only features that actively hurt
    accuracy). The remaining classifiers then train on the pruned set.

Output:
    Console report: accuracy, precision, recall, F1, confusion matrix
    results/classify/{csv_stem}.txt  (e.g. results/classify/compression.txt)
"""

import argparse
import os
import sys
import numpy as np
import joblib

# Allow running from project root as: python3 src/classify.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
from sklearn.utils import resample

from features import load_csv, FEATURE_FIELDS

FEATURE_COLS = [f for f in FEATURE_FIELDS if f not in ("filename", "label")]

CLASSIFIERS_FULL = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM":          LinearSVC(max_iter=2000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

CLASSIFIERS_FAST = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on quadtree features.")
    parser.add_argument("csv", help="Path to features CSV, e.g. results/features/compression.csv")
    parser.add_argument("--output", default=None,
                        help="Path to save results text file. "
                             "Default: results/classify/{csv_stem}.txt")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--features", nargs="+", default=FEATURE_COLS,
                        help=f"Feature columns to use (default: {FEATURE_COLS})")
    parser.add_argument("--fast", action="store_true",
                        help="Run only Logistic Regression and Random Forest (seconds, not minutes)")
    parser.add_argument("--balance", action="store_true",
                        help="Undersample majority class to balance dataset")
    parser.add_argument("--cv", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--permutation-importance", action="store_true",
                       help="Compute permutation importance for Random Forest in addition "
                       "to impurity-based importance. Slower but more reliable for "
                       "correlated features.")
    parser.add_argument("--prune-features", type=float, default=None, metavar="THRESHOLD",
                       help="Fit a Random Forest on the full feature set, compute permutation "
                       "importance, then drop any feature whose mean importance is below "
                       "THRESHOLD before running the main classifiers. "
                       "Use 0.0 to drop only features that actively hurt accuracy; "
                       "higher values (e.g. 0.0005) prune more aggressively.")
    return parser.parse_args()


def detect_outliers(x: np.ndarray, feature_cols: list):
    n = len(x)
    outlier_mask = np.zeros(n, dtype=bool)
    lines = ["Outlier Detection (3*IQR beyond 1st/99th percentile)"]
    lines.append(f"  {chr(39)+'Feature'+chr(39):<25} {chr(39)+'Mean'+chr(39):>8} {chr(39)+'Std'+chr(39):>8} {chr(39)+'Min'+chr(39):>8} {chr(39)+'Max'+chr(39):>8} {chr(39)+'Outliers'+chr(39):>10}")
    lines.append(" " + "-" * 75)
    for i, col in enumerate(feature_cols):
        vals = x[:, i]
        q1, q3 = np.percentile(vals, [1, 99])
        iqr = q3 - q1
        lo = q1 - 3 * iqr
        hi = q3 + 3 * iqr
        col_outliers = (vals < lo) | (vals > hi)
        outlier_mask |= col_outliers
        lines.append(f"  {col:<25} {vals.mean():>8.4f} {vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f} {col_outliers.sum():>10}")
    total = outlier_mask.sum()
    lines.append(f"Total outlier rows removed: {total} / {n} ({100*total/n:.2f}%)")
    return outlier_mask, "\n".join(lines)


def load_data(csv_path: str, feature_cols: list, balance: bool):
    rows = load_csv(csv_path)

    x, y = [], []
    skipped = 0
    for row in rows:
        label = row.get("label", "").strip()
        if not label:
            skipped += 1
            continue
        try:
            feats = [float(row[f]) for f in feature_cols]
        except (KeyError, ValueError):
            skipped += 1
            continue
        x.append(feats)
        y.append(label)

    x = np.array(x)
    y = np.array(y)

    if skipped:
        print(f"Skipped {skipped} rows with missing labels or features.")

    # Outlier detection and removal
    outlier_mask, outlier_report = detect_outliers(x, feature_cols)
    x = x[~outlier_mask]
    y = y[~outlier_mask]
    print(outlier_report)
    
    if balance:
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        x_bal, y_bal = [], []
        for cls in classes:
            mask = y == cls
            x_cls, y_cls = resample(x[mask], y[mask], n_samples=min_count, random_state=42)
            x_bal.append(x_cls)
            y_bal.append(y_cls)
        x = np.vstack(x_bal)
        y = np.concatenate(y_bal)
        print(f"Balanced to {min_count} samples per class.")

    return x, y, outlier_report


def format_confusion_matrix(cm, labels):
    col_w = max(len(l) for l in labels) + 2
    header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels) + "  ← predicted"
    rows = []
    for i, label in enumerate(labels):
        row = f"{label:>{col_w}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(len(labels)))
        rows.append(row)
    return "\n".join([header] + rows)


def evaluate(name, clf,
    x_train, x_test,
    y_train, y_test,
    feature_cols, cv_scores,
    show_permutation_importance=False
):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    labels = sorted(set(y_test))

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=labels)

    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {name}")
    lines.append(f"{'='*60}")
    lines.append(f"  Accuracy:  {acc:.4f}")
    lines.append(f"  Precision: {prec:.4f}  (weighted)")
    lines.append(f"  Recall:    {rec:.4f}  (weighted)")
    lines.append(f"  F1:        {f1:.4f}  (weighted)")
    lines.append(f"  CV ({len(cv_scores)}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    lines.append(f"\n  Confusion matrix:")
    lines.append(format_confusion_matrix(cm, labels))

    # Feature importances (Random Forest / Gradient Boosting only)
    if hasattr(clf, "feature_importances_"):
        lines.append(f"\n  Feature importances:")
        importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
        for feat, imp in importances:
            bar = "█" * int(imp * 40)
            lines.append(f"    {feat:<25} {imp:.4f}  {bar}")
    
    # Permutation importance (opt-in, RF and GB only)
    if show_permutation_importance and hasattr(clf, "feature_importances_"):
        lines.append(f"\n  Permutation importances (mean decrease in accuracy, 10 repeats):")
        result = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        perm_sorted = sorted(
            zip(feature_cols, result.importances_mean, result.importances_std),
            key=lambda x: -x[1]
        )
        for feat, mean, std in perm_sorted:
            bar = "█" * max(0, int(mean * 40))
            lines.append(f"    {feat:<25} {mean:+.4f} ± {std:.4f}  {bar}")
    
    # Logistic regression coefficients
    if hasattr(clf, "coef_"):
        lines.append(f"\n  Coefficients:")
        for label, coefs in zip(clf.classes_, clf.coef_):
            lines.append(f"    [{label}]")
            for feat, coef in zip(feature_cols, coefs):
                lines.append(f"      {feat:<25} {coef:+.4f}")

    return "\n".join(lines), acc


def prune_features(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list,
    threshold: float,
    n_repeats: int = 10,
) -> tuple:
    """
    Fit a Random Forest, compute permutation importance, and return
    (kept_cols, dropped_cols, report).

    Features with mean permutation importance >= threshold are kept.
    Use threshold=0.0 to drop only features that actively hurt accuracy.
    """
    print(f"Pruning features (threshold={threshold})...")
    probe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    probe.fit(x_train, y_train)
    result = permutation_importance(
        probe, x_test, y_test,
        n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    pairs = sorted(
        zip(feature_cols, result.importances_mean, result.importances_std),
        key=lambda t: -t[1]
    )

    kept_cols, dropped_cols = [], []
    for feat, mean, _ in pairs:
        if mean >= threshold:
            kept_cols.append(feat)
        else:
            dropped_cols.append(feat)

    lines = [
        f"\nFeature pruning  (RF permutation importance, threshold={threshold})",
        f"  Kept {len(kept_cols)} / {len(feature_cols)} features  "
        f"(dropped {len(dropped_cols)})",
    ]
    if dropped_cols:
        lines.append("  Dropped:")
        for feat, mean, std in pairs:
            if feat in dropped_cols:
                lines.append(f"    {feat:<25}  {mean:+.4f} ± {std:.4f}")

    # Preserve original column order
    kept_ordered = [f for f in feature_cols if f in set(kept_cols)]
    return kept_ordered, dropped_cols, "\n".join(lines)


def main():
    args = parse_args()

    # WSL cannot fork the memory-mapped worker processes joblib uses by default.
    # Force threading backend globally so all n_jobs=-1 calls share memory.
    joblib.parallel_backend("threading", n_jobs=-1)

    if not os.path.exists(args.csv):
        print(f"Error: file not found: {args.csv}")
        sys.exit(1)

    print(f"Loading {args.csv}...")
    x, y, outlier_report = load_data(args.csv, args.features, args.balance)

    classes, counts = np.unique(y, return_counts=True)
    print(f"Classes: {dict(zip(classes, counts))}")
    print(f"Features: {args.features}")
    print(f"Total samples: {len(y)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # Scale features — required for LR and SVM, harmless for tree methods
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s  = scaler.transform(x_test)

    # --- Optional feature pruning -----------------------------------------
    prune_report = ""
    feature_cols = list(args.features)

    if args.prune_features is not None:
        kept, dropped, prune_report = prune_features(
            x_train, x_test, y_train, y_test,
            feature_cols, threshold=args.prune_features
        )
        print(prune_report)
        if not kept:
            print("Error: all features were pruned — lower the threshold.")
            sys.exit(1)

        kept_idx   = [feature_cols.index(f) for f in kept]
        x_train    = x_train[:, kept_idx]
        x_test     = x_test[:, kept_idx]
        x_train_s  = x_train_s[:, kept_idx]
        x_test_s   = x_test_s[:, kept_idx]
        feature_cols = kept
        print(f"Continuing with {len(kept)} features.\n")
    # ----------------------------------------------------------------------

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"\nTrain: {len(y_train)}  Test: {len(y_test)}\n")

    classifiers = CLASSIFIERS_FAST if args.fast else CLASSIFIERS_FULL
    mode = "fast" if args.fast else "full"

    all_lines = [
        f"Classification Report ({mode} mode)",
        f"CSV: {args.csv}",
        f"Features ({len(feature_cols)}): {feature_cols}",
        f"Train/test split: {1-args.test_size:.0%} / {args.test_size:.0%}",
        f"Balanced: {args.balance}",
        "",
        outlier_report,
    ]
    if prune_report:
        all_lines.append(prune_report)

    results = []
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        print(f"Training {name}...", end=" ", flush=True)
        # Tree methods don't need scaling
        if name in ("Random Forest", "Gradient Boosting"):
            x_tr, x_te = x_train, x_test
        else:
            x_tr, x_te = x_train_s, x_test_s
        cv_scores = cross_val_score(clf, x_tr, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        perm_imp = getattr(args, "permutation_importance", False)
        report, acc = evaluate(name, clf, x_tr, x_te, y_train, y_test, feature_cols, cv_scores, perm_imp)
        print(f"accuracy={acc:.4f}")
        all_lines.append(report)
        results.append((name, acc))
    
    # Summary ranking
    all_lines.append(f"\n{'='*60}")
    all_lines.append("  SUMMARY (by accuracy)")
    all_lines.append(f"{'='*60}")
    for name, acc in sorted(results, key=lambda x: -x[1]):
        bar = "█" * int(acc * 40)
        all_lines.append(f"  {name:<25} {acc:.4f}  {bar}")
    
    report_text = "\n".join(all_lines)
    print(report_text)
    
    if args.output:
        out_path = args.output
    else:
        # Derive name from the CSV stem and write into results/classify/
        # e.g. results/features/compression_spatial_map.csv -> results/classify/compression_spatial_map.txt
        csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join("results", "classify", f"{csv_stem}.txt")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()