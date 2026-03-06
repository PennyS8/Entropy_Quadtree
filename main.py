"""
main.py

CLI entry point for the entropy visualizer.

Usage examples:

    # Shannon entropy, fixed depth 5, overlay on image
    python main.py photo.jpg --method shannon --max-depth 5
    
    # Compression entropy, adaptive splitting at threshold 0.3
    python main.py photo.jpg --method compression --threshold 0.3
    
    # Both stopping conditions active
    python main.py photo.jpg -method compression --max-depth 6 --threshold 0.2
    
    # Save to custom path, no legend
    python main.py photo.jpg -o result.png --no-legend
"""

import argparse
import numpy as np
from PIL import Image

from complexity import get_scorer
from quadtree import QuadTree, tree_stats
from visualizer import save_result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize image complexity as a quadtree display"
    )
    parser.add_argument("image", help="Path to input image file")
    parser.add_argument(
        "-0", "--output",
        default=None,
        help="Output file path  (default: <input>_entropy.png)"
    )
    parser.add_argument(
        "--method",
        choices=["shannon", "compression"],
        default="shannon",
        help="Complexity scoring method (default: shannon)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=6,
        help="Maximum quadtree depth. Set to 0 to disable (adaptive only). Default: 6"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Adaptive threshold: stop splitting below this complexity. Default: off"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=8,
        help="Minimum region side length in pixels. Default: 8"
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=120,
        help="Overlay opacity 0-255. Default: 120"
    )
    parser.add_argument(
        "--no-borders",
        action="store_true",
        help="Hide quadrant border lines"
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Omit the colorbar legend"
    )
    return parser.parse_args()

def load_image(path: str) -> np.ndarray:
    """
    Load an image and return as an RGB numpy array
    Transparent images (RGBA, LA, P with transparency) are composited
    onto a white background before conversion.
    """
    img = Image.open(path)
    
    if img.mode in ("RGB", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[3]) # alpha channel as mask
        img = background
    else:
        img = img.convert("RGBA")
    
    return np.array(img), img

def main():
    args = parse_args()
    
    # Load image
    image_array, img = load_image(args.image)
    print(f"Image: {args.image} ({img.width}x{img.height})")
    
    # Configure scorer
    scorer = get_scorer(args.method)
    print(f"Method: {args.method}")
    
    # Configure quadtree
    max_depth = args.max_depth if args.max_depth > 0 else None
    print(f"Max depth: {max_depth or 'unlimited'}")
    print(f"Threshold: {args.threshold or 'off'}")
    
    qt = QuadTree(
        scorer = scorer,
        max_depth = max_depth,
        threshold = args.threshold,
        min_size = args.min_size
    )
    
    # Build tree
    print("Building quadtree...")
    root = qt.build(image_array)
    
    # Print stats
    stats = tree_stats(root)
    print("\nQuadtree Stats:")
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
        show_borders = not args.no_borders,
        include_legend = not args.no_legend
    )


if __name__ == "__main__":
    main()