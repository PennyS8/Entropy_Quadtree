"""
batch.py
--------
Run the entropy visualizer on every image in a folder.
Produces overlay images and a features.csv for scatter plot analysis.

Usage:
    python batch.py --input my_images --output results

    # With labels for AI detection analysis
    python batch.py --input real_photos  --output results --label real
    python batch.py --input ai_images    --output results --label ai          --append
    python batch.py --input photoshopped --output results --label photoshopped --append

    # Parallel processing
    python batch.py --input my_images --output results --label real --workers 8

Optional args:
    --method      shannon | compression  (default: compression)
    --max-depth   int                    (default: 6)
    --threshold   float                  (default: off)
    --alpha       int                    (default: 120)
    --label       str                    label for all images in this batch
    --append                             append to existing features.csv
    --no-overlay                         skip overlay rendering, extract features only
    --workers     int                    number of parallel workers (default: 1)
    --no-borders
    --no-legend
"""

import argparse
import csv
import os
import sys
import numpy as np
from PIL import Image
from multiprocessing import Pool, cpu_count

from complexity import get_scorer
from quadtree import QuadTree
from visualizer import save_result
from features import extract_features, save_csv, load_csv, FEATURE_FIELDS

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch entropy visualization and feature extraction.")
    parser.add_argument("--input",    required=True, help="Input folder of images")
    parser.add_argument("--output",   required=True, help="Output folder for results")
    parser.add_argument("--method",   choices=["shannon", "compression"], default="compression")
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-size",  type=int, default=8)
    parser.add_argument("--alpha",     type=int, default=120)
    parser.add_argument("--label",     type=str, default=None,
                        help="Label for all images in this batch (e.g. real, ai, photoshopped)")
    parser.add_argument("--append",    action="store_true",
                        help="Append to existing features.csv instead of overwriting")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip overlay image rendering, extract features only")
    parser.add_argument("--no-borders", action="store_true")
    parser.add_argument("--no-legend",  action="store_true")
    parser.add_argument("--workers",   type=int, default=1,
                        help=f"Number of parallel workers (default: 1, max: {cpu_count()})")
    return parser.parse_args()


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


# ---------------------------------------------------------------------------
# Worker — must be a top-level function to be picklable by multiprocessing
# ---------------------------------------------------------------------------

def process_image(args_tuple):
    """
    Process a single image. Called by both single and multi-worker paths.

    Returns:
        (filename, feat, error_string) — feat is None on error
    """
    (filename, input_path, output_path, method, max_depth, threshold,
     min_size, label, no_overlay, fill_alpha, show_borders, include_legend) = args_tuple

    try:
        image_array, alpha = load_image(input_path)

        scorer = get_scorer(method)
        qt = QuadTree(
            scorer=scorer,
            max_depth=max_depth,
            threshold=threshold,
            min_size=min_size,
        )

        root = qt.build(image_array, alpha=alpha)
        feat = extract_features(root, filename, label=label)

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
    os.makedirs(args.output, exist_ok=True)

    entries = sorted([
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ])

    if not entries:
        print(f"No supported images found in '{args.input}'.")
        return

    workers = min(args.workers, cpu_count())
    max_depth = args.max_depth if args.max_depth > 0 else None

    print(f"Found {len(entries)} image(s) in '{args.input}'")
    print(f"Method: {args.method} | Max depth: {max_depth} | "
          f"Threshold: {args.threshold or 'off'} | "
          f"Label: {args.label or 'none'} | Workers: {workers}\n")

    # Build task list
    tasks = []
    for filename in entries:
        input_path = os.path.join(args.input, filename)
        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output, f"{stem}_entropy.png")
        tasks.append((
            filename, input_path, output_path,
            args.method, max_depth, args.threshold, args.min_size,
            args.label, args.no_overlay, args.alpha,
            not args.no_borders, not args.no_legend,
        ))

    # Load existing features if appending
    features_path = os.path.join(args.output, "features.csv")
    if args.append and os.path.exists(features_path):
        existing_dicts = load_csv(features_path)
        print(f"Appending to existing {len(existing_dicts)} entries in features.csv\n")
    else:
        existing_dicts = []

    # Process images
    new_features = []
    errors = 0

    if workers > 1:
        with Pool(processes=workers) as pool:
            for i, (filename, feat, err) in enumerate(
                pool.imap_unordered(process_image, tasks), 1
            ):
                if err:
                    print(f"  [{i}/{len(tasks)}] ERROR {filename}: {err.splitlines()[-1]}")
                    errors += 1
                else:
                    new_features.append(feat)
                    print(f"  [{i}/{len(tasks)}] {filename}  "
                          f"mean: {feat.mean_complexity:.4f}  "
                          f"std: {feat.std_complexity:.4f}  "
                          f"boundary_delta: {feat.mean_boundary_delta:.4f}")
    else:
        for i, task in enumerate(tasks, 1):
            filename, feat, err = process_image(task)
            if err:
                print(f"  [{i}/{len(tasks)}] ERROR {filename}: {err.splitlines()[-1]}")
                errors += 1
            else:
                new_features.append(feat)
                print(f"  [{i}/{len(tasks)}] {filename}  "
                      f"mean: {feat.mean_complexity:.4f}  "
                      f"std: {feat.std_complexity:.4f}  "
                      f"boundary_delta: {feat.mean_boundary_delta:.4f}")

    # Save features CSV
    if existing_dicts:
        new_dicts = [f.to_dict() for f in new_features]
        combined = existing_dicts + new_dicts
        with open(features_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
            writer.writeheader()
            for row in combined:
                writer.writerow({k: row.get(k, "") for k in FEATURE_FIELDS})
        print(f"\nAppended {len(new_features)} entries → {features_path}")
    else:
        save_csv(new_features, features_path)

    print(f"Done. {len(new_features)} processed, {errors} errors.")


if __name__ == "__main__":
    main()