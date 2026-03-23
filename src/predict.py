"""
predict.py
----------
Run a saved image forensics classifier on new images without re-training.
Part of the Quadtree Complexity Analysis for Image Forensics pipeline.

Loads a model bundle produced by classify.py --save-model, rebuilds the
quadtree complexity pipeline using the stored parameters, extracts features,
and outputs a predicted class and confidence score for each image.

Usage:
    # Single image
    python3 src/predict.py results/models/ciplab_faces_shannon.joblib photo.jpg

    # Multiple images
    python3 src/predict.py results/models/ciplab_faces_shannon.joblib *.jpg

    # Whole folder
    python3 src/predict.py results/models/ciplab_faces_shannon.joblib my_images/

    # Save results to CSV
    python3 src/predict.py results/models/ciplab_faces_shannon.joblib my_images/ \\
        --output results/predictions/my_images.csv

    # Show the overlay for each image as well
    python3 src/predict.py results/models/ciplab_faces_shannon.joblib photo.jpg \\
        --overlay --overlay-dir results/overlays/

    # Quiet: only print the CSV, no per-image console output
    python3 src/predict.py results/models/ciplab_faces_shannon.joblib my_images/ \\
        --output results/predictions/my_images.csv --quiet

    # Show bundle metadata and exit
    python3 src/predict.py results/models/ciplab_faces_shannon.joblib --info

Output columns (CSV):
    filename        — image filename
    predicted       — predicted class label (e.g. "authentic", "synthetic")
    confidence      — confidence in the prediction:
                        predict_proba models (LR, RF, GB): probability of predicted class
                        LinearSVC: raw decision score (not a probability)
    {class}_score   — one column per class with individual scores
    method          — quadtree method used (from bundle)
    model           — model bundle filename
"""

import argparse
import csv
import os
import sys

import numpy as np
import joblib
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from complexity import get_scorer
from quadtree import QuadTree
from features import extract_features, FEATURE_FIELDS
from visualizer import save_result
import config
from config import setup_logging, get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

DEFAULT_LEAF_SIZES = {"shannon": 4, "compression": 16, "variance": 4}


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify images using a model bundle saved by classify.py --save-model."
    )
    parser.add_argument("model", help="Path to .joblib model bundle")
    parser.add_argument("images", nargs="*",
                        help="Image file(s) or folder(s) to classify. "
                             "Omit when using --info.")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Save predictions to a CSV file. "
                             "Default: print to console only.")
    parser.add_argument("--overlay", action="store_true",
                        help="Save a complexity overlay PNG for each image.")
    parser.add_argument("--overlay-dir", default="results/overlays/", metavar="DIR",
                        help="Folder for overlay PNGs (default: results/overlays/)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-image console output. "
                             "Summary and errors are always printed.")
    parser.add_argument("--info", action="store_true",
                        help="Print model bundle metadata and exit.")
    return parser.parse_args()


# ── Bundle loading and inspection ─────────────────────────────────────────────

def load_bundle(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Error: model bundle not found: {path}")
        sys.exit(1)
    bundle = joblib.load(path)
    required = {"clf", "feature_cols", "method", "leaf_size", "classes"}
    missing = required - set(bundle.keys())
    if missing:
        print(f"Error: bundle is missing keys: {missing}. "
              f"Re-save with a current version of classify.py.")
        sys.exit(1)
    return bundle


def print_bundle_info(bundle: dict, path: str):
    print(f"\nModel bundle: {path}")
    print(f"  Classifier : {bundle.get('clf_name', type(bundle['clf']).__name__)}")
    print(f"  Classes    : {bundle['classes']}")
    print(f"  Method     : {bundle['method']}")
    print(f"  Leaf size  : {bundle['leaf_size']}px")
    resize = bundle.get("resize")
    print(f"  Resize     : {f'{resize}x{resize}px' if resize else 'off'}")
    n_feat = len(bundle['feature_cols'])
    tg = sum(1 for f in bundle['feature_cols'] if f.startswith("tree_grid_"))
    scalar = n_feat - tg
    print(f"  Features   : {n_feat}  ({scalar} scalar + {tg} spatial grid cells)")
    print(f"  Trained on : {bundle.get('trained_on', 'unknown')}")
    print(f"  Accuracy   : {bundle.get('accuracy', 'unknown')}")
    print(f"  Seed       : {bundle.get('seed', 42)}")
    print(f"  Saved at   : {bundle.get('saved_at', 'unknown')}")
    has_proba = hasattr(bundle['clf'], 'predict_proba')
    print(f"  Confidence : {'probability (predict_proba)' if has_proba else 'decision score (LinearSVC)'}")


# ── Image discovery ────────────────────────────────────────────────────────────

def collect_images(inputs: list) -> list:
    """Expand a mix of file paths and folders into a flat list of image paths."""
    paths = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fname in sorted(os.listdir(inp)):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                    paths.append(os.path.join(inp, fname))
        elif os.path.isfile(inp):
            if os.path.splitext(inp)[1].lower() in SUPPORTED_EXTENSIONS:
                paths.append(inp)
            else:
                print(f"Warning: skipping unsupported file type: {inp}")
        else:
            print(f"Warning: path not found: {inp}")
    return paths


# ── Image loading ──────────────────────────────────────────────────────────────

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


# ── Feature extraction ─────────────────────────────────────────────────────────

def resize_image(image_array: np.ndarray, size: int) -> np.ndarray:
    """Resize to size x size using Lanczos, preserving RGB or RGBA mode."""
    mode = "RGBA" if image_array.ndim == 3 and image_array.shape[2] == 4 else "RGB"
    img = Image.fromarray(image_array, mode=mode)
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


def extract(image_path: str, scorer, leaf_size: int, resize: int = None) -> tuple:
    """
    Load image, optionally resize, build quadtree, extract features.

    Returns:
        (image_array, root, features_dict)
    """
    image_array, alpha = load_image(image_path)

    if resize is not None:
        image_array = resize_image(image_array, resize)
        # Re-extract alpha from resized array if RGBA
        if image_array.ndim == 3 and image_array.shape[2] == 4:
            alpha = image_array[:, :, 3]
        else:
            alpha = None
    qt = QuadTree(scorer=scorer, leaf_size=leaf_size)
    root = qt.build(image_array, alpha=alpha, normalize=True)
    feat = extract_features(
        root,
        filename=os.path.basename(image_path),
        image=image_array,
        scorer=scorer,
        img_shape=image_array.shape[:2],
    )
    return image_array, root, feat.to_dict()


# ── Prediction ─────────────────────────────────────────────────────────────────

def predict_one(feat_dict: dict, bundle: dict) -> dict:
    """
    Run inference on a single feature dict.

    Returns a dict with keys: predicted, confidence, {class}_score
    """
    clf          = bundle["clf"]
    scaler       = bundle.get("scaler")
    feature_cols = bundle["feature_cols"]
    classes      = bundle["classes"]

    # Build feature vector — missing columns default to 0.0
    x = np.array([[float(feat_dict.get(f, 0.0)) for f in feature_cols]])

    # Scale for LR / SVM (scaler is None for tree methods)
    if scaler is not None:
        x = scaler.transform(x)

    predicted = clf.predict(x)[0]

    # Confidence scores
    scores = {}
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(x)[0]
        for cls, p in zip(clf.classes_, proba):
            scores[f"{cls}_score"] = float(p)
        confidence = float(proba[list(clf.classes_).index(predicted)])
    else:
        # LinearSVC: use decision_function (not a probability)
        decision = clf.decision_function(x)[0]
        if decision.ndim == 0:
            # Binary case — decision is a scalar
            confidence = float(decision)
            for cls in classes:
                scores[f"{cls}_score"] = float(decision) if cls == predicted else float(-decision)
        else:
            for cls, score in zip(clf.classes_, decision):
                scores[f"{cls}_score"] = float(score)
            confidence = float(decision[list(clf.classes_).index(predicted)])

    return {"predicted": predicted, "confidence": confidence, **scores}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    bundle     = load_bundle(args.model)
    model_name = os.path.basename(args.model)

    if args.info:
        print_bundle_info(bundle, args.model)
        return

    if not args.images:
        print("Error: provide at least one image path or folder, or use --info.")
        sys.exit(1)

    image_paths = collect_images(args.images)
    if not image_paths:
        print("No supported images found.")
        sys.exit(1)

    method    = bundle["method"]
    leaf_size = bundle["leaf_size"]
    resize    = bundle.get("resize")
    classes   = bundle["classes"]
    scorer    = get_scorer(method)

    print_bundle_info(bundle, args.model)
    log.info("Classifying {len(image_paths)} image(s)  "
          f"[method={method}, leaf_size={leaf_size}px"
          f"{f', resize={resize}x{resize}px' if resize else ''}]\n")

    if args.overlay:
        os.makedirs(args.overlay_dir, exist_ok=True)

    # CSV output setup
    score_cols = [f"{cls}_score" for cls in classes]
    csv_fields = ["filename", "predicted", "confidence"] + score_cols + ["method", "model"]
    rows       = []
    errors     = 0

    for i, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        try:
            image_array, root, feat_dict = extract(img_path, scorer, leaf_size, resize)
            result = predict_one(feat_dict, bundle)

            conf_str = f"{result['confidence']:.4f}"
            score_str = "  ".join(
                f"{cls}={result.get(f'{cls}_score', 0.0):.3f}" for cls in classes
            )

            if not args.quiet:
                print(f"  [{i}/{len(image_paths)}] {filename:<40} "
                      f"→ {result['predicted']:<14} conf={conf_str}  {score_str}")

            if args.overlay:
                out_name = os.path.splitext(filename)[0] + f"_{method}_overlay.png"
                overlay_path = os.path.join(args.overlay_dir, out_name)
                save_result(
                    image=image_array,
                    root=root,
                    output_path=overlay_path,
                    fill_alpha=120,
                    show_borders=False,
                    include_legend=True,
                )

            rows.append({
                "filename":   filename,
                "predicted":  result["predicted"],
                "confidence": result["confidence"],
                **{col: result.get(col, 0.0) for col in score_cols},
                "method":     method,
                "model":      model_name,
            })

        except Exception as e:
            import traceback
            print(f"  [{i}/{len(image_paths)}] ERROR {filename}: {e}")
            if not args.quiet:
                traceback.print_exc()
            errors += 1

    # Summary
    log.info("Done. {len(rows)} classified, {errors} errors.")

    if rows:
        from collections import Counter
        tally = Counter(r["predicted"] for r in rows)
        for cls, n in sorted(tally.items()):
            pct = n / len(rows) * 100
            bar = "█" * int(pct / 2)
            print(f"  {cls:<20} {n:>5}  ({pct:5.1f}%)  {bar}")

    # Save CSV
    if args.output and rows:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(rows)
        log.info("Saved: %s", args.output)



if __name__ == "__main__":
    main()