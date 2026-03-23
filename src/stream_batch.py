"""
stream_batch.py

Stream images directly from a Kaggle dataset into memory, extract quadtree
complexity features, and write them to CSV — without storing images to disk.
Part of the Quadtree Complexity Analysis for Image Forensics pipeline.

Downloads the dataset zip into memory once, then iterates entries in chunks —
each image is decompressed, processed, and freed before the next chunk loads.
Peak memory is bounded by --chunk-size rather than the full dataset size.

Setup:
    pip install kaggle requests Pillow numpy

Usage:
    # Explore dataset structure before processing
    python3 src/stream_batch.py \
        --dataset ciplab/real-and-fake-face-detection \
        --explore

    # Extract two methods in one pass (one download, two CSVs)
    python3 src/stream_batch.py \
        --dataset ciplab/real-and-fake-face-detection \
        --classes training_fake training_real \
        --labels manipulated authentic \
        --label-detail gan_faceswap authentic_portrait \
        --prefix real_and_fake_face \
        --methods compression shannon \
        --resize 256 --name ciplab_faces
    # -> results/features/ciplab_faces_shannon.csv
    # -> results/features/ciplab_faces_compression.csv

    # Extract with tuned per-method thresholds and universal authentic CSV
    python3 src/stream_batch.py \
        --dataset kshitizbhargava/deepfake-face-images \
        --prefix "Final Dataset" \
        --classes Fake --labels synthetic \
        --label-detail stylegan_v1v2_portrait \
        --methods shannon compression \
        --thresholds 36 29 \
        --resize 256 --name stylegan_v1v2 \
        --pair-with results/features/FFHQ_shannon.csv \
                    results/features/FFHQ_compression.csv

Required args:
    --dataset       Kaggle dataset slug
    --classes       folder name(s) within --prefix, one per class
    --labels        general label for each class: authentic synthetic manipulated
    --prefix        path inside the zip containing the class folders

Optional args:
    --label-detail  descriptive sublabel per class (same order as --labels)
    --output        output folder or CSV path (default: results/features/)
    --methods       one or more of: shannon compression variance
                    each produces its own CSV; all share one zip download
                    (default: shannon)
    --leaf-size     int overrides per-method defaults
                    (compression=16, shannon=4, variance=4)
    --resize        int resize every image to NxN before processing (default: 256)
    --threshold     float percentile pruning cutoff applied to all methods (default: off)
    --thresholds    one float per method, same order as --methods
                    e.g. --methods shannon compression --thresholds 36 29
    --max-images    int max images per class (default: 500)
    --max-per-dataset  hard cap per (dataset, class)
    --chunk-size    int images per processing chunk, controls peak memory (default: 1000)
    --append        append to existing CSV instead of overwriting
    --dry-run       show what would be processed without downloading
    --explore       print folder structure and exit
    --depth         depth for --explore (default: 2)
    --save-sample   int  save the first N images per class to
                    data/sample/{name}_{label}/ for use with tune_thresholds.py
    --pair-with     one CSV per method to prepend before extraction —
                    ensures output always contains at least two classes
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

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from config import setup_logging, get_logger
from complexity import get_scorer
from features import extract_features, FEATURE_FIELDS
from quadtree import QuadTree

log = get_logger(__name__)

SUPPORTED_EXTENSIONS = config.SUPPORTED_EXTENSIONS
KAGGLE_API_BASE      = "https://www.kaggle.com/api/v1"
DEFAULT_LEAF_SIZES   = config.DEFAULT_LEAF_SIZES


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream a Kaggle dataset through the quadtree complexity pipeline — no disk storage."
    )
    parser.add_argument("--dataset", required=True,
                        help="Kaggle dataset slug e.g. xhlulu/140k-real-and-fake-faces")
    parser.add_argument("--classes", nargs="+",
                        help="Folder name(s) within --prefix, one per class")
    parser.add_argument("--labels", nargs="+",
                        help="General label per class: authentic synthetic manipulated")
    parser.add_argument("--label-detail", nargs="+", default=None,
                        help="Descriptive sublabel per class e.g. stylegan_v1_portrait real_portrait")
    parser.add_argument("--prefix", default="",
                        help="Path inside the zip containing the class folders")
    parser.add_argument("--output", default=config.DIRS["features"],
                        help=f"Output folder or CSV path (default: {config.DIRS['features']})")
    parser.add_argument("--methods", nargs="+",
                        choices=config.METHODS,
                        default=[config.DEFAULT_METHOD],
                        help="Scoring methods — each produces a separate CSV, "
                             "all share one zip download. (default: shannon)")
    parser.add_argument("--leaf-size", type=int, default=None,
                        help="Leaf size override in pixels "
                             "(defaults: compression=16, shannon=4, variance=4)")
    parser.add_argument("--resize", type=int, default=config.DEFAULT_RESIZE, metavar="N",
                        help=f"Resize every image to NxN before processing "
                             f"(default: {config.DEFAULT_RESIZE})")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Percentile pruning cutoff applied to all methods. "
                             "Use --thresholds for per-method values.")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None,
                        metavar="T",
                        help="Per-method pruning cutoffs, one per --methods entry "
                             "(same order). Overrides --threshold. "
                             "e.g. --methods shannon compression --thresholds 36 29")
    parser.add_argument("--max-images", type=int, default=500,
                        help="Max images per class (default: 500)")
    parser.add_argument("--max-per-dataset", type=int, default=None,
                        help="Hard cap per (dataset, class). Prevents large datasets "
                             "dominating when appending multiple sources. "
                             "Defaults to --max-images if not set.")
    parser.add_argument("--append", action="store_true",
                        help="Append to existing CSV instead of overwriting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be processed without downloading")
    parser.add_argument("--explore", action="store_true",
                        help="Print folder structure and exit")
    parser.add_argument("--depth", type=int, default=2,
                        help="Directory depth for --explore (default: 2)")
    parser.add_argument("--name", default=None,
                        help="Dataset identifier prepended to output filename: "
                             "{name}_{method}.csv  e.g. --name FFHQ "
                             "→ results/features/FFHQ_shannon.csv. "
                             "Auto-derived from --dataset slug if not set.")
    parser.add_argument("--save-sample", type=int, default=None, metavar="N",
                        help="Save the first N images per class to data/sample/{name}_{label}/. "
                             "Use the saved folders with tune_thresholds.py to find optimal "
                             "per-method thresholds without a separate download.")
    parser.add_argument("--pair-with", nargs="+", default=None, metavar="CSV",
                        help="One existing CSV per method (same order as --methods) to "
                             "prepend into each output CSV before extraction. "
                             "Ensures every output CSV contains at least two classes — "
                             "pass the universal authentic CSV so single-class datasets "
                             "are immediately ready for classify.py without a merge step.")
    parser.add_argument("--progress", action="store_true", default=True,
                        help="Show a rolling progress line (default: on)")
    parser.add_argument("--chunk-size", type=int, default=1000, metavar="N",
                        help="Process images in batches of N to limit peak memory usage. "
                             "Each chunk is read from zip, processed, written to CSV, then "
                             "freed before the next chunk is loaded. Lower this if you hit "
                             "memory errors; raise it for slightly better throughput. "
                             "(default: 1000)")
    parser.add_argument("--no-progress", dest="progress", action="store_false",
                        help="Suppress the progress line")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
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
            log.error(f"{path} is missing 'username' or 'key'.")
            sys.exit(1)
    log.error("kaggle.json not found.")
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
        log.info("Downloading zip: %s", url)

    resp = session.get(url, stream=True, timeout=120, allow_redirects=True)

    if resp.status_code == 401:
        log.error("Authentication failed.")
        sys.exit(1)
    if resp.status_code == 403:
        log.error("Access denied. Accept the dataset license at:")
        log.error("  https://www.kaggle.com/datasets/%s", dataset)
        sys.exit(1)
    if resp.status_code == 404:
        log.error("Dataset not found: %s", dataset)
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

    buf.seek(0)
    return buf


# ── Zip exploration ───────────────────────────────────────────────────────────

def explore_zip(zf, depth=2):
    all_names = [n for n in zf.namelist() if not n.endswith("/")]
    dirs = sorted(set(
        "/".join(n.split("/")[:depth])
        for n in all_names if n.count("/") >= depth - 1
    ))
    log.info("Total files in zip: %d", len(all_names))
    log.info("Directory structure (first %d levels):", depth)
    for d in dirs[:60]:
        members = [n for n in all_names if n.startswith(d + "/")]
        exts    = {os.path.splitext(n)[1].lower() for n in members}
        log.info("  %s/  (%d files  types: %s)", d, len(members), ", ".join(sorted(exts)) or "none")
    if len(dirs) > 60:
        log.info("  ... and %d more directories", len(dirs) - 60)
    log.info("\nUse these with --prefix and --classes:")
    log.info("  --prefix  <the common parent directory>")
    log.info("  --classes <the per-class subfolder names>")


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
                  thresholds, resize=None, label_detail=None, dataset_source=""):
    """
    Decode one image and run every requested scoring method in a single pass.
    The image bytes are decoded once; each method scores the same numpy array.

    Args:
        thresholds: dict mapping method -> threshold float (or None for no pruning)

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
                threshold=thresholds.get(method),
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

def resolve_output_path(output_arg, method, name=None):
    """
    Resolve --output to a concrete .csv path for a given method.

    Filename format:
        name given  ->  {name}_{method}.csv   e.g. ciplab_faces_shannon.csv
        no name     ->  {method}.csv           (legacy behaviour)

    Path rules:
        Folder path   ->  {folder}/{name}_{method}.csv
        Explicit .csv ->  used as-is (name/method assumed already embedded)
    """
    stem = f"{name}_{method}" if name else method
    if output_arg.endswith(".csv"):
        return output_arg
    stripped = output_arg.rstrip("/").rstrip(os.sep)
    if output_arg.endswith("/") or output_arg.endswith(os.sep) or os.path.isdir(output_arg):
        return os.path.join(stripped, f"{stem}.csv")
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


def slug_to_name(dataset_slug: str) -> str:
    """
    Convert a Kaggle dataset slug to a safe filename prefix.

    ciplab/real-and-fake-face-detection  ->  ciplab_real_and_fake_face_detection
    wish096/realvsfake-81k-by-wish       ->  wish096_realvsfake_81k_by_wish
    """
    import re
    return re.sub(r"[^a-zA-Z0-9]+", "_", dataset_slug).strip("_")


def save_sample_images(entry_data: list, label: str, name: str, n: int) -> str:
    """
    Write the first N raw image bytes to data/sample/{name}_{label}/.
    Called once per class during processing — images are already in memory
    from the zip read so no extra download is needed.

    Returns the output directory path.
    """
    out_dir = os.path.join("data", "sample", f"{name}_{label}")
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for entry_name, data in entry_data[:n]:
        filename = os.path.basename(entry_name)
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "wb") as f:
            f.write(data)
        saved += 1
    log.info("Sample: saved %d images → %s", saved, out_dir)
    return out_dir


def prepend_pair_csv(pair_path: str, output_path: str):
    """
    Copy all rows from pair_path into output_path as the first rows,
    before any extracted features are written.

    Called once per method at the start of a run so the output CSV
    always contains at least two classes even if only one is extracted.
    """
    if not os.path.exists(pair_path):
        log.error("--pair-with file not found: %s", pair_path)
        sys.exit(1)

    with open(pair_path, newline="") as f:
        reader = csv.DictReader(f)
        pair_fields = reader.fieldnames
        rows = list(reader)

    if not rows:
        log.warning("--pair-with file is empty: %s", pair_path)
        return 0

    expected = set(FEATURE_FIELDS)
    actual   = set(pair_fields or [])
    if expected != actual:
        missing = expected - actual
        if missing:
            log.warning("--pair-with %s is missing %d fields — check features.py version",
                        pair_path, len(missing))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FEATURE_FIELDS})

    return len(rows)


def _process_entry_packed(args_tuple):
    """
    Thin wrapper around process_entry that accepts a flat tuple.
    Keeps the call site in the processing loop clean.
    """
    entry_name, image_data, label, methods, leaf_sizes, thresholds, resize, label_detail, dataset_source = args_tuple
    return process_entry(entry_name, image_data, label, methods, leaf_sizes,
                         thresholds, resize=resize, label_detail=label_detail,
                         dataset_source=dataset_source)


def write_sidecar_json(csv_path: str, meta: dict) -> None:
    """
    Write a sidecar JSON file alongside a CSV with extraction metadata.
    Stored at {csv_path}.json — e.g. FFHQ_shannon.csv → FFHQ_shannon.csv.json
    """
    sidecar_path = csv_path + ".json"
    with open(sidecar_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.debug("Sidecar written: %s", sidecar_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    setup_logging(args.verbose)

    if not args.explore and (not args.classes or not args.labels):
        log.error("--classes and --labels are required unless --explore is used.")
        sys.exit(1)
    if args.classes and args.labels and len(args.classes) != len(args.labels):
        log.error("--classes and --labels must have the same number of entries.")
        sys.exit(1)
    if args.label_detail and len(args.label_detail) != len(args.labels):
        log.error("--label-detail must have the same number of entries as --labels.")
        sys.exit(1)

    username, key = get_credentials()
    session       = make_session(username, key)
    log.info("Kaggle authenticated as: %s", username)

    zip_buf = download_zip(session, args.dataset)

    with zipfile.ZipFile(zip_buf) as zf:

        if args.explore:
            explore_zip(zf, depth=args.depth)
            return

        # ── Setup ─────────────────────────────────────────────────────────────
        methods    = list(dict.fromkeys(args.methods))
        leaf_sizes = {m: args.leaf_size or DEFAULT_LEAF_SIZES[m] for m in methods}
        max_cap    = args.max_per_dataset or args.max_images

        if args.thresholds is not None:
            if len(args.thresholds) != len(methods):
                log.error("--thresholds must have %d entries (one per method), got %d",
                          len(methods), len(args.thresholds))
                sys.exit(1)
            thresholds = {m: t for m, t in zip(methods, args.thresholds)}
        else:
            thresholds = {m: args.threshold for m in methods}

        if args.pair_with is not None:
            if len(args.pair_with) != len(methods):
                log.error("--pair-with must have %d entries (one per method), got %d",
                          len(methods), len(args.pair_with))
                sys.exit(1)
            pair_csvs = {m: p for m, p in zip(methods, args.pair_with)}
        else:
            pair_csvs = {}

        name         = args.name or slug_to_name(args.dataset)
        output_paths = {m: resolve_output_path(args.output, m, name) for m in methods}

        for path in output_paths.values():
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        write_headers = {}
        for m in methods:
            if m in pair_csvs:
                n_pair = prepend_pair_csv(pair_csvs[m], output_paths[m])
                log.info("Paired %d rows from %s → %s", n_pair, pair_csvs[m], output_paths[m])
                write_headers[m] = False
            else:
                write_headers[m] = not (args.append and os.path.exists(output_paths[m]))

        sample_base   = config.DIRS["sample"] if args.save_sample else None
        label_details = args.label_detail or ([None] * len(args.labels))

        log.info("Stream Batch — in-memory zip, no local image storage")
        log.info("Dataset:   %s  (name: %s)", args.dataset, name)
        log.info("Classes:   %s", list(zip(args.classes, args.labels)))
        log.info("Methods:   %s  leaf sizes: %s", methods, leaf_sizes)
        log.info("Resize:    %spx", args.resize)
        log.info("Thresholds: %s", thresholds)
        if pair_csvs:
            for m, p in pair_csvs.items():
                log.info("Pair-with [%s]: %s", m, p)
        for m, p in output_paths.items():
            log.info("Output [%s]: %s", m, p)
        log.info("Processing:  sequential  chunk: %d  max/class: %d",
                 args.chunk_size, args.max_images)

        total_processed = 0
        total_errors    = 0

        for class_folder, label, detail in zip(args.classes, args.labels, label_details):
            log.info("── Class: %s  (folder: %s)", label, class_folder)

            entries = select_class_entries(zf, args.prefix, class_folder, max_cap)
            if not entries:
                log.warning("No images found — run --explore to check paths.")
                continue

            log.info("%d images selected", len(entries))

            if args.dry_run:
                log.info("[dry run] First 5: %s%s",
                         entries[:5], f" ... and {len(entries)-5} more" if len(entries) > 5 else "")
                continue

            class_features = {m: [] for m in methods}
            class_errors   = 0
            t0             = time.time()
            n_total        = len(entries)
            done           = 0
            chunk_size     = args.chunk_size

            log.info("Processing %d images in chunks of %d", n_total, chunk_size)

            for chunk_start in range(0, n_total, chunk_size):
                chunk_entries = entries[chunk_start:chunk_start + chunk_size]

                # Read only this chunk from zip — previous chunk is already freed
                chunk_data = [(ep, zf.read(ep)) for ep in chunk_entries]

                # Save samples from first chunk only
                if sample_base and args.save_sample and chunk_start == 0:
                    save_sample_images(chunk_data, label, name, args.save_sample)

                chunk_features = {m: [] for m in methods}

                for ep, data in chunk_data:
                    done += 1
                    filename, results, err = _process_entry_packed(
                        (ep, data, label, methods, leaf_sizes, thresholds,
                         args.resize, detail, args.dataset)
                    )
                    if err:
                        class_errors += 1
                        log.error("%-40s %s", filename, err.strip().splitlines()[-1])
                    else:
                        for m in methods:
                            chunk_features[m].append(results[m])
                    if args.progress and n_total:
                        pct  = done / n_total * 100
                        rate = done / max(time.time() - t0, 0.001)
                        print(f"  {pct:5.1f}%  {done} / {n_total}  ({rate:.0f} img/s)    ",
                              end="\r")

                # Flush chunk to CSV and free memory before next chunk
                for m in methods:
                    feats = chunk_features[m]
                    if feats:
                        append_to_csv(feats, output_paths[m], write_header=write_headers[m])
                        write_headers[m] = False
                        class_features[m].extend(feats)

                del chunk_data, chunk_features  # free before next chunk

            print()  # newline after final \r

            n_ok    = len(class_features[methods[0]])
            elapsed = time.time() - t0
            log.info("'%s': %d processed, %d errors  (%.1fs)", label, n_ok, class_errors, elapsed)
            total_processed += n_ok
            total_errors    += class_errors

        # ── Sidecar JSON ──────────────────────────────────────────────────────
        import datetime
        meta_base = {
            "dataset":    args.dataset,
            "name":       name,
            "classes":    list(zip(args.classes, args.labels)),
            "resize":     args.resize,
            "thresholds": thresholds,
            "leaf_sizes": leaf_sizes,
            "max_images": args.max_images,
            "pair_with":  pair_csvs,
            "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
            "total":      total_processed,
            "errors":     total_errors,
        }
        for m, path in output_paths.items():
            write_sidecar_json(path, {**meta_base, "method": m, "leaf_size": leaf_sizes[m],
                                      "threshold": thresholds.get(m)})

    if args.dry_run:
        log.info("Dry run complete — nothing processed.")
    else:
        log.info("Done.  %d processed, %d errors.", total_processed, total_errors)
        for m, p in output_paths.items():
            log.info("  [%s] → %s", m, p)

        if sample_base and args.save_sample:
            label_dirs = " ".join(
                os.path.join("data", "sample", f"{name}_{lbl}") for lbl in args.labels
            )
            labels_str = " ".join(args.labels)
            log.info("\nSamples saved to: data/sample/%s_{label}/", name)
            log.info("Run tune_thresholds.py to find optimal thresholds:")
            for m in methods:
                ls = leaf_sizes[m]
                log.info(
                    "\n  python3 src/tune_thresholds.py \\\n"
                    "      --input %s \\\n"
                    "      --labels %s \\\n"
                    "      --method %s --leaf-size %d \\\n"
                    "      --max-images %d",
                    label_dirs, labels_str, m, ls, args.save_sample
                )


if __name__ == "__main__":
    main()