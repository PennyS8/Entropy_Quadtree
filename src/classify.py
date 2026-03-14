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
    python3 classify.py results/features.csv
    python3 classify.py results/features.csv --fast
    python3 classify.py results/features.csv --test-size 0.2
    python3 classify.py results/features.csv --features mean_complexity std_complexity
    python3 classify.py results/features.csv --balance

Output:
    Console report: accuracy, precision, recall, F1, confusion matrix
    classify_results.txt saved alongside the CSV
"""

import argparse
import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
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
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

CLASSIFIERS_FAST = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on quadtree features.")
    parser.add_argument("csv", help="Path to features.csv produced by batch.py")
    parser.add_argument("--output", default=None,
                        help="Path to save results text file. Defaults to classify_results.txt "
                             "alongside the CSV.")
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


def evaluate(name, clf, x_train, x_test, y_train, y_test, feature_cols, cv_scores):
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

    # Logistic regression coefficients
    if hasattr(clf, "coef_"):
        lines.append(f"\n  Coefficients:")
        for label, coefs in zip(clf.classes_, clf.coef_):
            lines.append(f"    [{label}]")
            for feat, coef in zip(feature_cols, coefs):
                lines.append(f"      {feat:<25} {coef:+.4f}")

    return "\n".join(lines), acc


def main():
    args = parse_args()

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

    print(f"\nTrain: {len(y_train)}  Test: {len(y_test)}\n")

    classifiers = CLASSIFIERS_FAST if args.fast else CLASSIFIERS_FULL
    mode = "fast" if args.fast else "full"

    all_lines = [
        f"Classification Report ({mode} mode)",
        f"CSV: {args.csv}",
        f"Features: {args.features}",
        f"Train/test split: {1-args.test_size:.0%} / {args.test_size:.0%}",
        f"Balanced: {args.balance}",
        "",
        outlier_report
    ]

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
        report, acc = evaluate(name, clf, x_tr, x_te, y_train, y_test, args.features, cv_scores)
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
        out_path = os.path.join(os.path.dirname(args.csv), "classify_results.txt")
    
    with open(out_path, "w") as f:
        f.write(report_text)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()