"""
main.py

CLI entry point for the entropy visualizer. Processes a single image and
saves an overlay PNG showing the quadtree complexity map.

Usage:
    python3 src/main.py photo.jpg
    python3 src/main.py photo.jpg --method compression
    python3 src/main.py photo.jpg --method shannon --leaf-size 4
    python3 src/main.py photo.jpg --method compression --threshold 0.05
    python3 src/main.py photo.jpg -o result.png --no-legend --no-borders

Optional args:
    --method      shannon|compression|variance (default: shannon)
                  shannon:     pixel value distribution — works at any leaf size
                  compression: zlib compressibility — reliable at leaf-size >= 16
                  variance:    fastest — single np.var call, good compression proxy
    --leaf_size   int                   (default: 4)
    --threshold   float                 (default: off)
    --alpha       int                   (default: 120)
    --borders                           show quadrant border lines
    --legend                            show the colorbar legend
"""

import argparse
import numpy as np
from PIL import Image
import os
import sys

# Allow running from project root as: python3 src/main.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from complexity import get_scorer
from quadtree import QuadTree, tree_stats
from visualizer import save_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize image complexity as a quadtree display"
    )
    parser.add_argument("image", help="Path to input image file")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path  (default: <input>_entropy.png)"
    )
    parser.add_argument(
        "--method",
        choices=["shannon", "compression", "variance"],
        default="shannon",
        help="Complexity scoring method (default: shannon)"
    )
    parser.add_argument(
        "--leaf_size",
        type=int,
        default=4,
        help="Target leaf side length in pixels. Smaller = finer detail. (default: 4)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Adaptive threshold: stop splitting below this complexity. Default: off"
    )
    parser.add_argument(
        "--bg-threshold",
        type=float,
        default=0.95,
        help="Background mask threshold 0-1. Nodes with this fraction of transparent pixels are excluded. (default: 0.95)"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=120,
        help="Overlay opacity 0-255. Default: 120"
    )
    parser.add_argument(
        "--borders",
        action="store_true",
        help="Show quadrant border lines"
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Show the colorbar legend"
    )
    return parser.parse_args()

def load_image(path: str) -> np.ndarray:
    """
    Load image, preserving transparency.

    For RGBA images the full RGBA array is returned so transparent regions
    remain transparent in the output. The alpha channel is also returned
    separately for background masking and complexity scoring.

    Returns:
        image_array: (H, W, 4) RGBA or (H, W, 3) RGB uint8 numpy array
        alpha:       (H, W) uint8 array, or None if no alpha channel
        img:         PIL Image (for width/height reporting)
    """
    img = Image.open(path)
    alpha = None
    
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        alpha = np.array(img.split()[3])
        image_array = np.array(img)
    else:
        img = img.convert("RGB")
        image_array = np.array(img)
    
    return image_array, alpha, img

def main():
    args = parse_args()
    
    # Load image
    image_array, alpha, img = load_image(args.image)    
    print(f"Image: {args.image} ({img.width}x{img.height})")
    print(f"Alpha channel: {'yes' if alpha is not None else 'no'}")
    
    # Configure scorer
    scorer = get_scorer(args.method)
    print(f"Method: {args.method}")
    
    # Configure quadtree
    # Use method-appropriate default if user did not explicitly set leaf_size
    DEFAULT_LEAF_SIZE = {"shannon": 4, "compression": 16, "variance": 4}
    leaf_size = args.leaf_size if args.leaf_size is not None else DEFAULT_LEAF_SIZE[args.method]
    print(f"Leaf size: {args.leaf_size}px")
    print(f"Threshold: {args.threshold or 'off'}")
    
    qt = QuadTree(
        scorer = scorer,
        leaf_size = leaf_size,
        threshold = args.threshold
    )
    
    # Build tree
    print("Building quadtree...")
    root = qt.build(image_array, alpha=alpha, normalize=True)
    
    # Print stats
    stats = tree_stats(root)
    print("\nQuadtree Stats (subject only):")
    for k, v in stats.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    print()
    
    # Render and save
    output_path = args.output or args.image.rsplit(".", 1)[0] + "_entropy.png"
    save_result(
        image = image_array,
        root = root,
        output_path = output_path,
        fill_alpha = args.alpha,
        show_borders = args.borders,
        include_legend = args.legend
    )


if __name__ == "__main__":
    main()