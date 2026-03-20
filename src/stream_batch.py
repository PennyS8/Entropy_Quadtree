"""
stream_batch.py

Stream images directly from a Kaggle dataset into memory, process them,
and write features to CSV — without ever storing images on disk.

Downloads the dataset zip into memory, iterates entries with zipfile — each
image is decompressed into a BytesIO buffer, processed through the quadtree
pipeline, and discarded. No temp files are written at any point.

New in this version:
    --methods          multiple scorers in one pass (one download, N CSVs)
    --resize           normalise resolution before processing
    --max_per_dataset  per-dataset cap to prevent large datasets dominating
    --label_detail     descriptive sublabel alongside the general class label

Setup:
    pip install kaggle requests Pillow numpy

Usage:
    # Explore dataset structure first
    python3 src/stream_batch.py \\
        --dataset ciplab/real-and-fake-face-detection \\
        --explore

    # Single method
    python3 src/stream_batch.py \\
        --dataset ciplab/real-and-fake-face-detection \\
        --classes training_fake training_real \\
        --labels manipulated authentic \\
        --label_detail gan_faceswap authentic_portrait \\
        --prefix real_and_fake_face \\
        --methods compression \\
        --resize 256

    # Two methods in one pass (one download, two CSVs)
    python3 src/stream_batch.py \\
        --dataset ciplab/real-and-fake-face-detection \\
        --classes training_fake training_real \\
        --labels manipulated authentic \\
        --label_detail gan_faceswap authentic_portrait \\
        --prefix real_and_fake_face \\
        --methods compression shannon \\
        --resize 256 --workers 8

    # Append wish096 to the same CSVs
    python3 src/stream_batch.py \\
        --dataset wish096/realvsfake-81k-by-wish \\
        --classes Fake Real \\
        --labels synthetic authentic \\
        --label_detail gan_portrait authentic_portrait \\
        --prefix RealVsFake/RealVsFake \\
        --methods compression shannon \\
        --resize 256 --max_per_dataset 2000 --workers 8 \\
        --append

Required args:
    --dataset       Kaggle dataset slug
    --classes       folder name(s) within --prefix, one per class
    --labels        general label for each class (e.g. authentic synthetic manipulated)
    --prefix        path inside the zip containing the class folders

Optional args:
    --label_detail  descriptive sublabel per class (same order as --labels)
                    e.g. --label_detail gan_faceswap authentic_portrait
    --output        output folder or CSV path (default: results/features/)
    --methods       one or more of: shannon compression variance
                    each produces its own CSV; all share one zip download
                    (default: compression)
    --leaf_size     int overrides per-method defaults
                    (compression=16, shannon=4, variance=4)
    --resize        int resize every image to NxN before processing
                    strongly recommended when mixing datasets
    --threshold     float percentile pruning cutoff (default: off)
    --max_images    int max images per class (default: 500)
    --max_per_dataset  hard cap per (dataset, class) — prevents large
                    datasets dominating when appending multiple sources
    --workers       int parallel workers (default: 4)
    --append        append to existing CSV instead of overwriting
    --dry_run       show what would be processed without downloading
    --explore       print folder structure and exit
    --depth         depth for --explore (default: 2)
"""

import argparse
import csv
import io
import json
import os
import sys
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from complexity import get_scorer
from features import extract_features, FEATURE_FIELDS
from quadtree import QuadTree

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
KAGGLE_API_BASE      = "https://www.kaggle.com/api/v1"
DEFAULT_LEAF_SIZES   = {"shannon": 4, "compression": 16, "variance": 4}


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream a Kaggle dataset zip through the quadtree pipeline — no disk storage."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--labels", nargs="+",
                        help="General label per class e.g. authentic synthetic manipulated")
    parser.add_argument("--label_detail", nargs="+", default=None,
                        help="Descriptive sublabel per class e.g. gan_faceswap real_portrait")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output", default="results/features/")
    parser.add_argument("--methods", nargs="+",
                        choices=["shannon", "compression", "variance"],
                        default=["compression"],
                        help="Scoring methods — each produces a separate CSV, "
                             "all share one zip download. (default: compression)")
    parser.add_argument("--leaf_size", type=int, default=None,
                        help="Leaf size override (defaults: compression=16, shannon=4, variance=4)")
    parser.add_argument("--resize", type=int, default=None, metavar="N",
                        help="Resize every image to NxN before processing. "
                             "Recommended when combining datasets with different resolutions.")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_images", type=int, default=500,
                        help="Max images per class (default: 500)")
    parser.add_argument("--max_per_dataset", type=int, default=None,
                        help="Hard cap per (dataset, class). Prevents large datasets "
                             "dominating when appending multiple sources. "
                             "Defaults to --max_images if not set.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--depth", type=int, default=2,
                        help="Directory depth for --explore (default: 2)")
    return parser.parse_args()


# ── Kaggle auth ───────────────────────────────────────────────────────────────

def get_credentials():
    for path in (
        os.path.expanduser("~/.config/kaggle/kaggle.json"),
        os.path.expanduser("~/.kaggle/kaggle.json"),
    ):
        if os.path.exists(path):
            with open(path) as f:
                cfg = json.load(f)
            u = cfg.get("username", "").strip()
            k = cfg.get("key",      "").strip()
            if u and k:
                return u, k
            print(f"Error: {path} is missing 'username' or 'key'.")
            sys.exit(1)
    print("Error: kaggle.json not found.")
    sys.exit(1)


def make_session(username, key):
    s = requests.Session()
    s.auth = (username, key)
    s.headers["User-Agent"] = "stream_batch/4.0"
    return s


# ── Zip download ──────────────────────────────────────────────────────────────

def download_zip(session, dataset, verbose=True):
    """Download the full dataset zip into a BytesIO buffer. Nothing written to disk."""
    owner, slug = dataset.split("/", 1)
    url = f"{KAGGLE_API_BASE}/datasets/download/{owner}/{slug}"

    if verbose:
        print(f"Downloading zip: {url}")

    resp = session.get(url, stream=True, timeout=120, allow_redirects=True)

    if resp.status_code == 401:
        print("Error: authentication failed.")
        sys.exit(1)
    if resp.status_code == 403:
        print(f"Error: access denied. Accept the dataset license at:")
        print(f"  https://www.kaggle.com/datasets/{dataset}")
        sys.exit(1)
    if resp.status_code == 404:
        print(f"Error: dataset '{dataset}' not found.")
        sys.exit(1)
    resp.raise_for_status()

    total = int(resp.headers.get("Content-Length", 0))
    buf   = io.BytesIO()
    done  = 0
    t0    = time.time()

    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        done += len(chunk)
        if verbose and total:
            pct   = done / total * 100
            speed = done / max(time.time() - t0, 0.001) / (1 << 20)
            print(f"  {pct:5.1f}%  {done >> 20} / {total >> 20} MB  "
                  f"({speed:.1f} MB/s)    ", end="\r")

    if verbose:
        print(f"  100.0%  {done >> 20} MB downloaded in {time.time()-t0:.1f}s          ")

    buf.seek(0)
    return buf


# ── Zip exploration ───────────────────────────────────────────────────────────

def explore_zip(zf, depth=2):
    all_names = [n for n in zf.namelist() if not n.endswith("/")]
    dirs = sorted(set(
        "/".join(n.split("/")[:depth])
        for n in all_names if n.count("/") >= depth - 1
    ))
    print(f"\nTotal files in zip: {len(all_names)}")
    print(f"\nDirectory structure (first {depth} levels):")
    for d in dirs[:60]:
        members = [n for n in all_names if n.startswith(d + "/")]
        exts    = {os.path.splitext(n)[1].lower() for n in members}
        print(f"  {d}/  ({len(members)} files  types: {', '.join(sorted(exts)) or 'none'})")
    if len(dirs) > 60:
        print(f"  ... and {len(dirs) - 60} more directories")
    print()
    print("Use these with --prefix and --classes:")
    print("  --prefix  <the common parent directory>")
    print("  --classes <the per-class subfolder names>")


# ── File selection ────────────────────────────────────────────────────────────

def select_class_entries(zf, prefix, class_folder, max_images):
    class_prefix = "/".join(filter(None, [prefix, class_folder])) + "/"
    entries = [
        n for n in zf.namelist()
        if n.startswith(class_prefix)
        and not n.endswith("/")
        and os.path.splitext(n)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    return entries[:max_images]


# ── Image loading ─────────────────────────────────────────────────────────────

def load_from_bytes(data):
    """Open image from raw bytes. Returns (image_array, alpha)."""
    buf = io.BytesIO(data)
    img = Image.open(buf)
    img.load()
    alpha = None
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img         = img.convert("RGBA")
        alpha       = np.array(img.split()[3])
        image_array = np.array(img)
    else:
        img         = img.convert("RGB")
        image_array = np.array(img)
    return image_array, alpha


def resize_image(image_array, size):
    """
    Resize to size x size using Lanczos resampling.

    Resizing to a fixed resolution before processing ensures:
      - Consistent leaf counts across datasets with different native resolutions
      - Comparable tree_grid spatial maps (each cell covers the same face region)
      - Resolution-independent scalar features (mean_leaf_area, leaf_count, mean_depth)
    """
    has_alpha   = image_array.ndim == 3 and image_array.shape[2] == 4
    mode        = "RGBA" if has_alpha else "RGB"
    img         = Image.fromarray(image_array, mode)
    img         = img.resize((size, size), Image.LANCZOS)
    image_array = np.array(img)
    alpha       = image_array[:, :, 3] if has_alpha else None
    return image_array, alpha


# ── Processing ────────────────────────────────────────────────────────────────

def process_entry(entry_name, image_data, label, methods, leaf_sizes,
                  threshold, resize=None, label_detail=None, dataset_source=""):
    """
    Decode one image and run every requested scoring method in a single pass.
    The image bytes are decoded once; each method scores the same numpy array.

    Returns:
        (filename, {method: ImageFeatures} | None, error_string | None)
    """
    filename = os.path.basename(entry_name)
    try:
        image_array, alpha = load_from_bytes(image_data)

        if resize:
            image_array, alpha = resize_image(image_array, resize)

        results = {}
        for method in methods:
            scorer = get_scorer(method)
            qt     = QuadTree(
                scorer=scorer,
                leaf_size=leaf_sizes[method],
                threshold=threshold,
            )
            root = qt.build(image_array, alpha=alpha, normalize=False)
            feat = extract_features(
                root, filename,
                label=label,
                image=image_array,
                scorer=scorer,
                img_shape=image_array.shape[:2],
                label_detail=label_detail,
                dataset_source=dataset_source,
            )
            results[method] = feat

        return filename, results, None
    except Exception:
        return filename, None, traceback.format_exc()


# ── CSV helpers ───────────────────────────────────────────────────────────────

def resolve_output_path(output_arg, method):
    """
    Resolve --output to a concrete .csv path for a given method.

    Folder path  ->  {folder}/{method}.csv
    Explicit .csv -> insert method before extension if not already there
                     so multiple methods never overwrite each other
    """
    if output_arg.endswith(".csv"):
        stem = output_arg[:-4]
        if stem.endswith(method) or stem.endswith("_" + method):
            return output_arg
        return stem + "_" + method + ".csv"
    stripped = output_arg.rstrip("/").rstrip(os.sep)
    if output_arg.endswith("/") or output_arg.endswith(os.sep) or os.path.isdir(output_arg):
        return os.path.join(stripped, method + ".csv")
    return output_arg


def append_to_csv(features, path, write_header):
    mode = "w" if write_header else "a"
    with open(path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        if write_header:
            writer.writeheader()
        for feat in features:
            row = feat.to_dict()
            writer.writerow({k: row.get(k, "") for k in FEATURE_FIELDS})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.explore and (not args.classes or not args.labels):
        print("Error: --classes and --labels are required unless --explore is used.")
        sys.exit(1)
    if args.classes and args.labels and len(args.classes) != len(args.labels):
        print("Error: --classes and --labels must have the same number of entries.")
        sys.exit(1)
    if args.label_detail and len(args.label_detail) != len(args.labels):
        print("Error: --label_detail must have the same number of entries as --labels.")
        sys.exit(1)

    username, key = get_credentials()
    session       = make_session(username, key)
    print(f"Kaggle authenticated as: {username}")

    zip_buf = download_zip(session, args.dataset)

    with zipfile.ZipFile(zip_buf) as zf:

        if args.explore:
            explore_zip(zf, depth=args.depth)
            return

        # ── Setup ─────────────────────────────────────────────────────────────
        methods    = list(dict.fromkeys(args.methods))
        leaf_sizes = {m: args.leaf_size or DEFAULT_LEAF_SIZES[m] for m in methods}
        max_cap    = args.max_per_dataset or args.max_images

        output_paths  = {m: resolve_output_path(args.output, m) for m in methods}
        write_headers = {
            m: not (args.append and os.path.exists(output_paths[m]))
            for m in methods
        }
        for path in output_paths.values():
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        label_details = args.label_detail or ([None] * len(args.labels))

        print(f"\nStream Batch — in-memory zip, no local image storage")
        print("─" * 56)
        print(f"Dataset:        {args.dataset}")
        print(f"Classes:        {list(zip(args.classes, args.labels))}")
        if args.label_detail:
            print(f"Label detail:   {args.label_detail}")
        print(f"Max/class:      {args.max_images}  |  Max/dataset: {max_cap}")
        print(f"Methods:        {methods}")
        print(f"Leaf sizes:     {leaf_sizes}")
        if args.resize:
            print(f"Resize:         {args.resize}x{args.resize}px")
        print(f"Threshold:      {args.threshold or 'off'}")
        for m, p in output_paths.items():
            print(f"Output [{m}]:  {p}")
        print(f"Workers:        {args.workers}  |  Append: {args.append}\n")

        total_processed = 0
        total_errors    = 0

        for class_folder, label, detail in zip(args.classes, args.labels, label_details):
            print(f"\n{'─' * 56}")
            print(f"Class: {label}  detail: {detail or '—'}  (folder: {class_folder})")

            entries = select_class_entries(zf, args.prefix, class_folder, max_cap)
            if not entries:
                print("  Warning: no images found. Run --explore to check paths.")
                continue

            print(f"  {len(entries)} images selected")

            if args.dry_run:
                print("  [dry run] First 5:")
                for e in entries[:5]:
                    print(f"    {e}")
                if len(entries) > 5:
                    print(f"    ... and {len(entries) - 5} more")
                continue

            print("  Reading from zip...")
            t_read     = time.time()
            entry_data = [(name, zf.read(name)) for name in entries]
            print(f"  Read {len(entry_data)} images in {time.time()-t_read:.1f}s")

            class_features = {m: [] for m in methods}
            class_errors   = 0
            t0             = time.time()
            n_total        = len(entry_data)

            def _task(item):
                name, data = item
                return process_entry(
                    name, data, label, methods, leaf_sizes,
                    args.threshold,
                    resize=args.resize,
                    label_detail=detail,
                    dataset_source=args.dataset,
                )

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_task, item): item[0] for item in entry_data}
                done    = 0
                for future in as_completed(futures):
                    done += 1
                    filename, results, err = future.result()
                    if err:
                        class_errors += 1
                        print(f"  [{done}/{n_total}] ERROR {filename}: "
                              f"{err.strip().splitlines()[-1]}")
                    else:
                        first = results[methods[0]]
                        print(
                            f"  [{done}/{n_total}] {filename}  "
                            f"mean: {first.mean_complexity:.4f}  "
                            f"std: {first.std_complexity:.4f}  "
                            f"bd: {first.mean_boundary_delta:.4f}"
                        )
                        for m in methods:
                            class_features[m].append(results[m])

            for m in methods:
                feats = class_features[m]
                if feats:
                    append_to_csv(feats, output_paths[m], write_header=write_headers[m])
                    write_headers[m] = False

            n_ok    = len(class_features[methods[0]])
            elapsed = time.time() - t0
            print(f"\n  '{label}': {n_ok} processed, {class_errors} errors  ({elapsed:.1f}s)")
            total_processed += n_ok
            total_errors    += class_errors

    print(f"\n{'─' * 56}")
    if args.dry_run:
        print("Dry run complete — nothing processed.")
    else:
        print(f"Done.  {total_processed} processed, {total_errors} errors.")
        if total_processed > 0:
            for m, p in output_paths.items():
                print(f"  [{m}] -> {p}")
    print('\a')


if __name__ == "__main__":
    main()