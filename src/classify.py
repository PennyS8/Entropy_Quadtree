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

    # Save the best classifier for use with predict.py
    python3 src/classify.py results/features/ciplab_faces_shannon.csv \\
        --fast --balance --save-model ciplab_faces_shannon \\
        --method shannon --leaf-size 4
    # → results/models/ciplab_faces_shannon.joblib

    # Cross-dataset generalization test: train on ciplab, test on wish096
    python3 src/classify.py results/features/ciplab_faces_shannon.csv \\
        --test-csv results/features/wish096_faces_shannon.csv --fast --balance

--test-csv PATH
    Cross-dataset evaluation mode. Trains on the entire CSV (no random split),
    then tests on a completely separate dataset CSV. Both CSVs must share the
    same feature columns (same --method and --leaf_size).

    Output: results/classify/{train_stem}_vs_{test_stem}.txt

    If accuracy is much lower than within-dataset accuracy, the classifier is
    memorizing dataset-specific patterns (likely the spatial grid cells) rather
    than learning general AI-detection features.

--prune-features THRESHOLD
    Fits a Random Forest on the full feature set, computes permutation
    importance, then discards any feature whose mean permutation importance
    is below THRESHOLD (default 0.0 drops only features that actively hurt
    accuracy). The remaining classifiers then train on the pruned set.
    In cross-dataset mode, pruning uses an internal 80/20 split of the train
    set so the test CSV is never seen during feature selection.

Output:
    Console report: accuracy, precision, recall, F1, confusion matrix
    results/classify/{csv_stem}.txt  (e.g. results/classify/compression.txt)
    results/classify/{train_stem}_vs_{test_stem}.txt  (cross-dataset mode)
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

FEATURE_COLS = [f for f in FEATURE_FIELDS if f not in ("filename", "label", "label_detail", "is_real", "dataset_source")]

# Columns that are never numeric features regardless of CSV schema
_META_COLS = {"filename", "label", "label_detail", "is_real", "dataset_source"}

# Pure numeric feature names (no metadata) — used to detect schema mismatch
_NUMERIC_FEATURE_COLS = [f for f in FEATURE_COLS if f not in _META_COLS]


def infer_feature_cols(csv_path: str) -> list:
    """
    Read the CSV header and return all columns that look like numeric features.
    Used automatically when the CSV schema doesn't match FEATURE_COLS,
    e.g. merged multi-method CSVs with prefixed columns like shannon_mean_complexity.
    """
    import csv as _csv
    with open(csv_path, newline="") as f:
        header = next(_csv.reader(f))
    return [col for col in header if col not in _META_COLS]

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
    parser.add_argument("--test-csv", default=None, metavar="PATH",
                       help="Cross-dataset evaluation: train on the full CSV, test on this "
                       "separate CSV. Bypasses the random train/test split entirely. "
                       "Both CSVs must share the same feature columns (same method, same "
                       "leaf size). --test-size and --balance are ignored for the test set. "
                       "Output: results/classify/{train_stem}_vs_{test_stem}.txt")
    parser.add_argument("--save-model", default=None, metavar="NAME",
                       help="After training, serialize the best classifier to "
                       "results/models/{NAME}.joblib. The bundle includes the fitted "
                       "classifier, scaler, feature columns, method, and leaf size — "
                       "everything predict.py needs to run on new images. "
                       "Requires --method and --leaf-size to be set so the quadtree "
                       "can be rebuilt identically at inference time.")
    parser.add_argument("--method", default=None,
                       choices=["shannon", "compression", "variance"],
                       help="Scoring method used to produce the CSV. Required when "
                       "using --save-model so the method is embedded in the model bundle.")
    parser.add_argument("--leaf-size", type=int, default=None, metavar="N",
                       help="Leaf size used to produce the CSV. Required when using "
                       "--save-model so the quadtree can be rebuilt identically at "
                       "inference time.")
    parser.add_argument("--resize", type=int, default=None, metavar="N",
                       help="Resize applied when producing the CSV (NxN pixels). "
                       "Required when using --save-model if --resize was passed to "
                       "batch.py or stream_batch.py, so predict.py can apply the "
                       "same resize before feature extraction.")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for train/test split, classifiers, and "
                       "permutation importance. Set to any integer for a different "
                       "reproducible run, or pass different values to measure variance. "
                       "(default: 42)")
    return parser.parse_args()


def detect_outliers(x: np.ndarray, feature_cols: list):
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(
            f"No valid samples loaded (x.shape={x.shape}). "
            "The feature column names don't match the CSV header. "
            "Pass --features explicitly, or omit it to let classify.py auto-detect."
        )
    n = len(x)
    outlier_mask = np.zeros(n, dtype=bool)
    lines = ["Outlier Detection (3*IQR beyond 1st/99th percentile)"]
    lines.append(f"  {chr(39)+'Feature'+chr(39):<25} {chr(39)+'Mean'+chr(39):>8} {chr(39)+'Std'+chr(39):>8} {chr(39)+'Min'+chr(39):>8} {chr(39)+'Max'+chr(39):>8} {chr(39)+'Outliers'+chr(39):>10}")
    lines.append(" " + "-" * 75)
    tg_outlier_total = 0
    for i, col in enumerate(feature_cols):
        vals = x[:, i]
        q1, q3 = np.percentile(vals, [1, 99])
        iqr = q3 - q1
        lo = q1 - 3 * iqr
        hi = q3 + 3 * iqr
        col_outliers = (vals < lo) | (vals > hi)
        outlier_mask |= col_outliers
        if col.startswith("tree_grid_"):
            tg_outlier_total += int(col_outliers.sum())
        else:
            lines.append(f"  {col:<25} {vals.mean():>8.4f} {vals.std():>8.4f} {vals.min():>8.4f} {vals.max():>8.4f} {col_outliers.sum():>10}")
    tg_cols = [c for c in feature_cols if c.startswith("tree_grid_")]
    if tg_cols:
        lines.append(f"  tree_grid_000..255        ({len(tg_cols)} spatial cells, {tg_outlier_total} outliers)")
    total = outlier_mask.sum()
    lines.append(f"Total outlier rows removed: {total} / {n} ({100*total/n:.2f}%)")
    return outlier_mask, "\n".join(lines)


def load_data(csv_path: str, feature_cols: list, balance: bool, seed: int = 42):
    rows = load_csv(csv_path)

    x, y, filenames = [], [], []
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
        filenames.append(row.get("filename", ""))

    x = np.array(x)
    y = np.array(y)
    filenames = np.array(filenames)

    if skipped:
        print(f"Skipped {skipped} rows with missing labels or features.")

    # Outlier detection and removal
    outlier_mask, outlier_report = detect_outliers(x, feature_cols)
    x = x[~outlier_mask]
    y = y[~outlier_mask]
    filenames = filenames[~outlier_mask]
    print(outlier_report)
    
    if balance:
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()
        x_bal, y_bal, fn_bal = [], [], []
        for cls in classes:
            mask = y == cls
            x_cls, y_cls, fn_cls = resample(x[mask], y[mask], filenames[mask], n_samples=min_count, random_state=42)
            x_bal.append(x_cls)
            y_bal.append(y_cls)
            fn_bal.append(fn_cls)
        x = np.vstack(x_bal)
        y = np.concatenate(y_bal)
        filenames = np.concatenate(fn_bal)
        print(f"Balanced to {min_count} samples per class.")

    return x, y, filenames, outlier_report


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
    show_permutation_importance=False,
    fn_test=None
):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    labels = sorted(set(y_test))

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=labels)

    # Misclassified files (printed to console, not saved to report)
    if fn_test is not None:
        for true_label in sorted(set(y_test)):
            for pred_label in sorted(set(y_test)):
                if true_label == pred_label:
                    continue
                mask = (np.array(y_test) == true_label) & (np.array(y_pred) == pred_label)
                files = [fn_test[i] for i in range(len(fn_test)) if mask[i]]
                if files:
                    print(f"  [{name}] true={true_label} predicted={pred_label} ({len(files)} files):")
                    for fn in files:
                        print(f"    {fn}")

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
        lines.append(f"\n  Feature importances (non-spatial only):")
        importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: -x[1])
        tg_imp_total = sum(imp for feat, imp in importances if feat.startswith("tree_grid_"))
        for feat, imp in importances:
            if feat.startswith("tree_grid_"):
                continue
            bar = "█" * int(imp * 40)
            lines.append(f"    {feat:<25} {imp:.4f}  {bar}")
        if tg_imp_total > 0:
            bar = "█" * int(tg_imp_total * 40)
            lines.append(f"    {'tree_grid_* (256 cells)':<25} {tg_imp_total:.4f}  {bar}  (combined)")
    
    # Permutation importance (opt-in, RF and GB only)
    if show_permutation_importance and hasattr(clf, "feature_importances_"):
        lines.append(f"\n  Permutation importances (mean decrease in accuracy, 10 repeats):")
        result = permutation_importance(clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)  # seed fixed for consistency
        perm_sorted = sorted(
            zip(feature_cols, result.importances_mean, result.importances_std),
            key=lambda x: -x[1]
        )
        tg_perm_mean = sum(m for f, m, _ in perm_sorted if f.startswith("tree_grid_"))
        tg_perm_std  = np.mean([s for f, _, s in perm_sorted if f.startswith("tree_grid_")]) if any(f.startswith("tree_grid_") for f, _, _ in perm_sorted) else 0.0
        for feat, mean, std in perm_sorted:
            if feat.startswith("tree_grid_"):
                continue
            bar = "█" * max(0, int(mean * 40))
            lines.append(f"    {feat:<25} {mean:+.4f} ± {std:.4f}  {bar}")
        if any(f.startswith("tree_grid_") for f, _, _ in perm_sorted):
            bar = "█" * max(0, int(tg_perm_mean * 40))
            lines.append(f"    {'tree_grid_* (256 cells)':<25} {tg_perm_mean:+.4f} ± {tg_perm_std:.4f}  {bar}  (combined)")
    
    # Logistic regression coefficients
    if hasattr(clf, "coef_"):
        lines.append(f"\n  Coefficients (non-spatial only):")
        for label, coefs in zip(clf.classes_, clf.coef_):
            lines.append(f"    [{label}]")
            tg_coef_sum = sum(abs(c) for f, c in zip(feature_cols, coefs) if f.startswith("tree_grid_"))
            for feat, coef in zip(feature_cols, coefs):
                if feat.startswith("tree_grid_"):
                    continue
                lines.append(f"      {feat:<25} {coef:+.4f}")
            if tg_coef_sum > 0:
                lines.append(f"      {'tree_grid_* (256 cells)':<25} (sum |coef|={tg_coef_sum:.4f})")

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
    probe = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    probe.fit(x_train, y_train)
    result = permutation_importance(
        probe, x_test, y_test,
        n_repeats=n_repeats, random_state=seed, n_jobs=-1
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


def save_model(
    clf,
    clf_name: str,
    scaler,
    feature_cols: list,
    classes: list,
    method: str,
    leaf_size: int,
    resize: int,
    trained_on: str,
    accuracy: float,
    name: str,
    seed: int = 42,
) -> str:
    """
    Serialize a fitted classifier + everything predict.py needs to a .joblib bundle.

    Bundle schema:
        clf           — fitted sklearn classifier
        clf_name      — human-readable classifier name (e.g. "Random Forest")
        scaler        — fitted StandardScaler (None for tree-based classifiers)
        feature_cols  — ordered list of feature column names the model expects
        classes       — list of class labels in classifier order
        method        — quadtree scoring method ("shannon"/"compression"/"variance")
        leaf_size     — quadtree leaf size in pixels
        resize        — image resize applied before feature extraction (None = no resize)
        trained_on    — path to the training CSV
        accuracy      — test-set accuracy at save time
        saved_at      — ISO timestamp

    Args:
        name: output stem — saved to results/models/{name}.joblib
    
    Returns:
        Absolute path of the saved file.
    """
    import datetime
    needs_scaling = not hasattr(clf, "feature_importances_")  # True for LR / SVM

    bundle = {
        "clf":          clf,
        "clf_name":     clf_name,
        "scaler":       scaler if needs_scaling else None,
        "feature_cols": feature_cols,
        "classes":      list(classes),
        "method":       method,
        "leaf_size":    leaf_size,
        "resize":       resize,
        "seed":         seed,
        "trained_on":   trained_on,
        "accuracy":     accuracy,
        "saved_at":     datetime.datetime.now().isoformat(timespec="seconds"),
    }

    out_dir = os.path.join("results", "models")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(bundle, out_path)
    return out_path


def main():
    args = parse_args()

    # WSL cannot fork the memory-mapped worker processes joblib uses by default.
    # Force threading backend globally so all n_jobs=-1 calls share memory.
    joblib.parallel_backend("threading", n_jobs=-1)

    if not os.path.exists(args.csv):
        print(f"Error: file not found: {args.csv}")
        sys.exit(1)

    if args.save_model:
        if not args.method:
            print("Error: --method is required when using --save-model "
                  "(it is embedded in the bundle so predict.py can rebuild the quadtree).")
            sys.exit(1)
        if not args.leaf_size:
            print("Error: --leaf-size is required when using --save-model "
                  "(it is embedded in the bundle so predict.py can rebuild the quadtree).")
            sys.exit(1)

    print(f"Loading {args.csv}...")

    # Auto-detect feature columns when the CSV schema doesn't match FEATURE_COLS.
    # The check uses _NUMERIC_FEATURE_COLS (pure numeric, no metadata fields) so
    # columns like label_detail/is_real that appear in both schemas don't cause
    # a false "schema matches" result.
    feature_cols_arg = args.features
    if feature_cols_arg == FEATURE_COLS:
        import csv as _csv
        with open(args.csv, newline="") as _f:
            csv_header = set(next(_csv.reader(_f)))
        if not any(col in csv_header for col in _NUMERIC_FEATURE_COLS):
            feature_cols_arg = infer_feature_cols(args.csv)
            print(f"Auto-detected {len(feature_cols_arg)} feature columns from CSV header "
                  "(merged/non-standard schema). Pass --features explicitly to override.\n")

    x, y, filenames, outlier_report = load_data(args.csv, feature_cols_arg, args.balance)

    classes, counts = np.unique(y, return_counts=True)
    print(f"Classes: {dict(zip(classes, counts))}")
    visible = [f for f in feature_cols_arg if not f.startswith("tree_grid_")]
    tg_count = sum(1 for f in feature_cols_arg if f.startswith("tree_grid_"))
    print(f"Features: {visible}" + (f" + {tg_count} tree_grid spatial cells" if tg_count else ""))
    print(f"Total samples: {len(y)}")

    # ── Split strategy ────────────────────────────────────────────────────────
    if args.test_csv:
        # Cross-dataset mode: train on entire CSV, test on a separate dataset.
        # Validates that both CSVs share the same feature columns before loading.
        if not os.path.exists(args.test_csv):
            print(f"Error: --test-csv file not found: {args.test_csv}")
            sys.exit(1)

        import csv as _csv
        with open(args.test_csv, newline="") as _f:
            test_header = set(next(_csv.reader(_f)))
        missing = [c for c in feature_cols_arg if c not in test_header]
        if missing:
            print(f"Error: {len(missing)} feature column(s) in train CSV are missing from "
                  f"--test-csv. Make sure both were produced with the same method and "
                  f"leaf_size.\nMissing (first 5): {missing[:5]}")
            sys.exit(1)

        print(f"\nCross-dataset mode:")
        print(f"  Train: {args.csv}  ({len(y)} samples)")
        print(f"  Test:  {args.test_csv}")
        x_test_raw, y_test, fn_test, test_outlier_report = load_data(
            args.test_csv, feature_cols_arg, balance=False, seed=args.seed
        )
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        print(f"  Test classes: {dict(zip(test_classes, test_counts))}")

        x_train, y_train, fn_train = x, y, filenames
        x_test = x_test_raw

        cross_dataset = True
        split_desc = (f"Train: {args.csv}\n"
                      f"Test (cross-dataset): {args.test_csv}")
    else:
        # Standard mode: random within-dataset split
        if args.test_csv is None and args.test_size >= 1.0:
            print("Error: --test-size must be < 1.0")
            sys.exit(1)
        x_train, x_test, y_train, y_test, fn_train, fn_test = train_test_split(
            x, y, filenames, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        cross_dataset = False
        split_desc = f"Train/test split: {1-args.test_size:.0%} / {args.test_size:.0%}"
        test_outlier_report = ""

    # Scale features — required for LR and SVM, harmless for tree methods
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s  = scaler.transform(x_test)

    # --- Optional feature pruning -----------------------------------------
    prune_report = ""
    feature_cols = list(feature_cols_arg)

    if args.prune_features is not None:
        # In cross-dataset mode, use an internal split of the train set for
        # the probe so the test CSV remains completely unseen during pruning.
        if cross_dataset:
            x_pr_tr, x_pr_te, y_pr_tr, y_pr_te = train_test_split(
                x_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
            )
        else:
            x_pr_tr, x_pr_te, y_pr_tr, y_pr_te = x_train, x_test, y_train, y_test

        kept, dropped, prune_report = prune_features(
            x_pr_tr, x_pr_te, y_pr_tr, y_pr_te,
            feature_cols, threshold=args.prune_features, seed=args.seed
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

    visible_cols = [f for f in feature_cols if not f.startswith("tree_grid_")]
    tg_n = len(feature_cols) - len(visible_cols)
    suffix = f" + {tg_n} tree_grid spatial cells" if tg_n else ""
    print(f"Features ({len(feature_cols)}): {visible_cols}{suffix}")
    print(f"\nTrain: {len(y_train)}  Test: {len(y_test)}\n")

    classifiers = make_classifiers_fast(args.seed) if args.fast else make_classifiers(args.seed)
    import datetime
    mode = "fast" if args.fast else "full"
    mode_str = f"{mode} mode, cross-dataset" if cross_dataset else f"{mode} mode"

    all_lines = [
        f"Classification Report ({mode_str})",
        f"Timestamp: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"Seed:      {args.seed}",
        split_desc,
        f"Features ({len(feature_cols)}): {feature_cols}",
        f"Balanced: {args.balance}",
        "",
        outlier_report,
    ]
    if cross_dataset and test_outlier_report:
        all_lines.append(f"\nTest set outlier detection:\n{test_outlier_report}")
    if prune_report:
        all_lines.append(prune_report)

    results = []
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    for name, clf in classifiers.items():
        print(f"Training {name}...", end=" ", flush=True)
        # Tree methods don't need scaling
        if name in ("Random Forest", "Gradient Boosting"):
            x_tr, x_te = x_train, x_test
        else:
            x_tr, x_te = x_train_s, x_test_s
        cv_scores = cross_val_score(clf, x_tr, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        perm_imp = getattr(args, "permutation_importance", False)
        report, acc = evaluate(name, clf, x_tr, x_te, y_train, y_test, feature_cols, cv_scores, perm_imp, fn_test=fn_test)
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

    # --- Save best model --------------------------------------------------
    if args.save_model:
        best_name, best_acc = max(results, key=lambda x: x[1])
        best_clf = classifiers[best_name]
        model_path = save_model(
            clf=best_clf,
            clf_name=best_name,
            scaler=scaler,
            feature_cols=feature_cols,
            classes=sorted(set(y_train)),
            method=args.method,
            leaf_size=args.leaf_size,
            resize=args.resize,
            trained_on=", ".join(args.csv),
            accuracy=best_acc,
            name=args.save_model,
            seed=args.seed,
        )
        model_line = (f"\nModel saved: {model_path}"
                      f"  [{best_name}, acc={best_acc:.4f}]"
                      f"  method={args.method}  leaf_size={args.leaf_size}"
                      f"  resize={args.resize or 'off'}")
        all_lines.append(model_line)
        print(model_line)
    # ----------------------------------------------------------------------
    
    report_text = "\n".join(all_lines)
    print(report_text)
    
    if args.output:
        out_path = args.output
    else:
        train_stem = os.path.splitext(os.path.basename(args.csv))[0]
        if cross_dataset:
            test_stem = os.path.splitext(os.path.basename(args.test_csv))[0]
            out_path = os.path.join("results", "classify", f"{train_stem}_vs_{test_stem}.txt")
        else:
            out_path = os.path.join("results", "classify", f"{train_stem}.txt")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()