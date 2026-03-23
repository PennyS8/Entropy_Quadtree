"""
batch.py

Run the Quadtree Complexity Analysis for Image Forensics pipeline on every
image in one or more folders. Extracts complexity features and writes them to
a CSV file for use with classify.py. Overlay rendering is disabled;
use main.py for single-image complexity overlays.

Usage:
    # Single folder
    python3 src/batch.py --input my_images --labels unlabeled --output results/features/

    # Multiple folders in one run — builds a labeled dataset in a single pass
    python3 src/batch.py --input real_photos ai_images --labels authentic synthetic \
        --output results/features/

    # Name the output so it doesn't overwrite results from another dataset
    python3 src/batch.py --input real_photos ai_images --labels authentic synthetic \
        --output results/features/ --name my_dataset
    # -> results/features/my_dataset_shannon.csv

    # Append a new class to an existing CSV
    python3 src/batch.py --input photoshopped --labels manipulated \
        --output results/features/compression.csv --append

Required args:
    --input     one or more input folders of images
    --labels    one label per folder, same order as --input
                (authentic / synthetic / manipulated)
    --output    path for the output CSV file, or a folder — if a folder or path ending
                in '/' is given, the file is auto-named {name}_{method}.csv inside it

Optional args:
    --method      shannon|compression|variance  (default: shannon)
    --leaf-size   int     target leaf side length in pixels (default: 4)
    --threshold   float   percentile pruning cutoff, default off
    --resize      int     resize every image to NxN before processing.
                          Required when combining datasets with different resolutions —
                          the quadtree leaf size is in pixels so resolution differences
                          directly affect tree depth and spatial features.
    --name        str     dataset identifier prepended to the output filename:
                          {name}_{method}.csv  e.g. --name my_dataset
                          Required when running multiple datasets to avoid overwriting results.
    --append              append to existing CSV instead of overwriting
    --max-images  int     max images per input folder (default: all)
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from PIL import Image

# Allow running from project root as: python3 src/batch.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from complexity import get_scorer
from quadtree import QuadTree
from visualizer import save_result
from features import extract_features, save_csv, load_csv, FEATURE_FIELDS
import config
from config import setup_logging, get_logger

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = config.SUPPORTED_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(description="Batch quadtree complexity feature extraction for image forensics.")
    parser.add_argument("--input", nargs="+", required=True,
                        help="Input folder(s) of images")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Label for each input folder (same order as --input)")
    parser.add_argument("--output", required=True,
                        help="Output path for features CSV, or a folder. "
                        "If a folder (or path ending in /), auto-names to "
                        "{method}.csv inside it. e.g. results/features/")
    parser.add_argument("--method", choices=["shannon", "compression", "variance"], default="shannon")
    parser.add_argument("--leaf-size", type=int, default=4,
                        help="Target leaf side length in pixels. (default: 4)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV instead of overwriting")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Max images to process per input folder (default: all). "
                             "Useful for fast pilot runs before committing to the full dataset.")
    parser.add_argument("--name", default=None,
                        help="Dataset identifier prepended to output filename: "
                             "{name}_{method}.csv  e.g. --name ciplab_faces "
                             "-> results/features/ciplab_faces_shannon.csv. "
                             "Required when running multiple datasets to avoid "
                             "overwriting previous results. (default: {method}.csv)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--resize", type=int, default=None, metavar="N",
                        help="Resize every image to NxN before processing using Lanczos "
                             "resampling. Required when combining datasets with different "
                             "resolutions — the quadtree leaf size is in pixels so resolution "
                             "differences directly affect tree depth and spatial features. "
                             "Must match the resize used when training any model you intend "
                             "to use for inference.")
    return parser.parse_args()


def resolve_output_path(output_arg: str, method: str, name: str = None) -> str:
    """
    Resolve --output to a concrete CSV path.

    Filename format:
        name given  ->  {name}_{method}.csv   e.g. ciplab_faces_shannon.csv
        no name     ->  {method}.csv           (legacy behaviour)

    Path rules:
        Ends in .csv          -> use as-is (name/method already embedded by caller)
        Ends in / or is a dir -> place file inside the folder
        Anything else         -> use as-is
    """
    stem = f"{name}_{method}" if name else method
    if output_arg.endswith(".csv"):
        return output_arg
    stripped = output_arg.rstrip("/").rstrip(os.sep)
    if output_arg.endswith("/") or output_arg.endswith(os.sep) or os.path.isdir(output_arg):
        return os.path.join(stripped, f"{stem}.csv")
    return output_arg


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


def resize_image(image_array: np.ndarray, size: int) -> np.ndarray:
    """
    Resize image to size x size using Lanczos resampling.
    Preserves mode (RGB or RGBA). Alpha channel is resized alongside RGB.
    """
    mode = "RGBA" if image_array.shape[2] == 4 else "RGB"
    img = Image.fromarray(image_array, mode=mode)
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)


# ---------------------------------------------------------------------------
# Worker — must be a top-level function to be picklable by multiprocessing
# ---------------------------------------------------------------------------

def process_image(args_tuple):
    """
    Process a single image. Called by both single and multi-worker paths.

    Returns:
        (filename, feat, error_string) — feat is None on error
    """
    (filename, input_path, output_path, method, leaf_size, threshold,
     label, no_overlay, fill_alpha, show_borders, include_legend, resize) = args_tuple

    try:
        image_array, alpha = load_image(input_path)

        if resize is not None:
            image_array = resize_image(image_array, resize)
            # Re-extract alpha after resize (it's embedded in channel 3 of RGBA)
            if image_array.ndim == 3 and image_array.shape[2] == 4:
                alpha = image_array[:, :, 3]
            else:
                alpha = None

        scorer = get_scorer(method)
        DEFAULT_LEAF_SIZE = {"shannon": 4, "compression": 16, "variance": 4}
        effective_leaf_size = leaf_size if leaf_size is not None else DEFAULT_LEAF_SIZE[method]
        qt = QuadTree(
            scorer=scorer,
            leaf_size=effective_leaf_size,
            threshold=threshold
        )

        root = qt.build(image_array, alpha=alpha, normalize=not no_overlay)
        feat = extract_features(root, filename, label=label, image=image_array,
                                scorer=scorer, img_shape=image_array.shape[:2])

        if not no_overlay:
            save_result(
                image=image_array,
                root=root,
                output_path=output_path,
                fill_alpha=fill_alpha,
                show_borders=show_borders,
                include_legend=include_legend,
            )

        return filename, feat, None

    except Exception as e:
        import traceback
        return filename, None, traceback.format_exc()


def main():
    args = parse_args()
    
    setup_logging(getattr(args, "verbose", False))
    if len(args.input) != len(args.labels):
        log.error("--input and --labels must have the same number of entries.")
        sys.exit(1)
    
    features_path = resolve_output_path(args.output, args.method, args.name)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    log.info("Output: %s", features_path)
    leaf_size_display = args.leaf_size or "auto"
    
    log.info("Method: %s | Leaf size: %s | Threshold: %s | Resize: %s",
             args.method, leaf_size_display, args.threshold or "off",
             f"{args.resize}x{args.resize}px" if args.resize else "off")
    
    # Build task list across all folders
    tasks = []
    for folder, label in zip(args.input, args.labels):
        entries = sorted([
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])
        if not entries:
            log.warning("No supported images found in %r, skipping.", folder)
            continue
        if args.max_images:
            entries = entries[:args.max_images]
        log.info("Found %d image(s) in %r  (label: %s)", len(entries), folder, label)
        for filename in entries:
            input_path = os.path.join(folder, filename)
            tasks.append((
                filename, input_path, None,
                args.method, args.leaf_size, args.threshold,
                label, True, 120, False, False, args.resize
            ))
    
    if not tasks:
        log.error("No images found across all input folders.")
        sys.exit(1)
    
    print(f"\nTotal: {len(tasks)} images\n")
    
    # Load existing features if appending
    if args.append and os.path.exists(features_path):
        existing_dicts = load_csv(features_path)
        log.info("Appending to existing %d entries.", len(existing_dicts))
    else:
        existing_dicts = []

    # Process images
    new_features = []
    errors = 0
    t0 = time.time()

    for i, task in enumerate(tasks, 1):
        filename, feat, err = process_image(task)
        if err:
            print(f"\n  ERROR {filename}: {err.splitlines()[-1]}")
            errors += 1
        else:
            new_features.append(feat)
        pct  = i / len(tasks) * 100
        rate = i / max(time.time() - t0, 0.001)
        print(f"  {pct:5.1f}%  {i} / {len(tasks)}  ({rate:.0f} img/s)    ", end="\r")
    print()  # newline after final \r

    # Save features CSV
    if existing_dicts:
        new_dicts = [f.to_dict() for f in new_features]
        combined = existing_dicts + new_dicts
        with open(features_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
            writer.writeheader()
            for row in combined:
                writer.writerow({k: row.get(k, "") for k in FEATURE_FIELDS})
        log.info("Appended %d entries → %s", len(new_features), features_path)
    else:
        save_csv(new_features, features_path)

    # Write sidecar metadata JSON
    import json, datetime
    meta = {
        "method": args.method, "leaf_size": args.leaf_size,
        "threshold": args.threshold, "resize": args.resize,
        "name": args.name, "inputs": args.input, "labels": args.labels,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "total": len(new_features), "errors": errors,
    }
    with open(features_path + ".json", "w") as _jf:
        json.dump(meta, _jf, indent=2)

    log.info("Done. %d processed, %d errors.", len(new_features), errors)


if __name__ == "__main__":
    main()