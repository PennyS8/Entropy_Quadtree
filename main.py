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
        "--bg-threshold",
        type=float,
        default=0.5,
        help="Background mask threshold 0-1. Nodes with this fraction of transparent pixels are excluded. (default: 0.5)"
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
    print(f"DEBUG mode={img.mode}, condition={img.mode in ('RGBA', 'LA')}")
    alpha = None
    
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        alpha = np.array(img.split()[3]) # extract alpha before compositing
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    
    return np.array(img), alpha, img

def main():
    args = parse_args()
    
    # Load image
    image_array, alpha, img = load_image(args.image)
    print(f"Alpha: {alpha is not None}, transparent pixels: {(alpha < 128).sum() if alpha is not None else 'N/A'}")
    
    print(f"Image: {args.image} ({img.width}x{img.height})")
    print(f"Alpha channel: {'yes' if alpha is not None else 'no'}")
    
    # Configure scorer
    scorer = get_scorer(args.method)
    print(f"Method: {args.method}")
    
    # Configure quadtree
    max_depth = args.max_depth if args.max_depth > 0 else None
    print(f"Max depth: {max_depth or 'unlimited'}")
    print(f"Threshold: {args.threshold or 'off'}")
    print(f"BG threshold: {args.bg_threshold}")
    
    qt = QuadTree(
        scorer = scorer,
        max_depth = max_depth,
        threshold = args.threshold,
        min_size = args.min_size,
        bg_threshold = args.bg_threshold
    )
    
    # Build tree
    print("Building quadtree...")
    root = qt.build(image_array, alpha=alpha)
    
    # Print stats
    stats = tree_stats(root, bg_threshold=args.bg_threshold)
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
        alpha = alpha,
        fill_alpha = args.alpha,
        show_borders = not args.no_borders,
        include_legend = not args.no_legend,
        bg_threshold = args.bg_threshold
    )


if __name__ == "__main__":
    main()