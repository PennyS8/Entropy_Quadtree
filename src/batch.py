"""
batch.py
--------
Run the entropy visualizer on every image in a folder.
Produces overlay images and a features.csv for scatter plot analysis.

Usage:
    python batch.py --input my_images --output results

    # With labels for AI detection analysis
    python batch.py --input real_photos --output results --label real
    python batch.py --input ai_images   --output results --label ai      --append
    python batch.py --input photoshopped --output results --label photoshopped --append

Optional args:
    --method      shannon | compression  (default: compression)
    --max-depth   int                    (default: 6)
    --threshold   float                  (default: off)
    --alpha       int                    (default: 120)
    --label       str                    label for all images in this batch
    --append                             append to existing features.csv
    --no-overlay                         skip overlay rendering, extract features only
    --no-borders
    --no-legend
"""

import argparse
import os
import numpy as np
from PIL import Image

from complexity import get_scorer
from quadtree import QuadTree, tree_stats
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
    parser.add_argument("--append", action="store_true",
                        help="Append to existing features.csv instead of overwriting")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Skip overlay image rendering, extract features only")
    parser.add_argument("--no-borders", action="store_true")
    parser.add_argument("--no-legend",  action="store_true")
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
    
    print(f"Found {len(entries)} image(s) in '{args.input}'")
    print(f"Method: {args.method} | Max depth: {args.max_depth} | "
          f"Threshold: {args.threshold or 'off'} | Label: {args.label or 'none'}\n")
    
    scorer = get_scorer(args.method)
    max_depth = args.max_depth if args.max_depth > 0 else None
    
    qt = QuadTree(
        scorer=scorer,
        max_depth=max_depth,
        threshold=args.threshold,
        min_size=args.min_size
    )
    
    features_path = os.path.join(args.output, "features.csv")
    all_features = []
    
    #Load existing features if appending
    if args.append and os.path.exists(features_path):
        existing = load_csv(features_path)
        print(f"Appending to existing {len(existing)} entries in features.csv\n")
        # Convert back to ImageFeatures-like dicts, kept as dicts for CSV rewrite
        all_features_dicts = existing
    else:
        all_features_dicts = []
    
    new_features = []
    
    for i, filename in enumerate(entries, 1):
        input_path = os.path.join(args.input, filename)
        stem = os.path.splitext(filename)[0]
        output_path = os.path.join(args.output, f"{stem}_entropy.png")
        
        print(f"[{i}/{len(entries)}] {filename}")

        try:
            image_array, alpha = load_image(input_path)
            root = qt.build(image_array, alpha=alpha)
            
            # Extract features
            feat = extract_features(root, filename, label=args.label)
            new_features.append(feat)
            
            stats = tree_stats(root)
            print(f"subject leaves: {stats['subject_leaf_count']}"
                  f"mean: {stats['mean_leaf_complexity']:4f}"
                  f"std: {stats['std_leaf_complexity']:4f}"
                  f"boundry_delta: {feat.mean_boundry_delta:.4f}")
            
            # Render overlay
            if not args.no_overlay:
                save_result(
                    image=image_array,
                    root=root,
                    output_path=output_path,
                    fill_alpha=args.alpha,
                    show_borders=not args.no_borders,
                    include_legend=not args.no_legend,
                )
        
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
    
    # Save features CSV
    if args.append and all_features_dicts:
        # Write existing + new together
        import csv
        new_dicts = [f.to_dict() for f in new_features]
        combined = all_features_dicts + new_dicts
        with open(features_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
            writer.writeheader()
            for row in combined:
                writer.writerow({k: row.get(k, "") for k in FEATURE_FIELDS})
        print(f"\nAppended {len(new_features)} entries -> {features_path}")
    else:
        save_csv(new_features, features_path)
    
    print(f"Done. Results saved to '{args.output}'")

if __name__ == "__main__":
    main()