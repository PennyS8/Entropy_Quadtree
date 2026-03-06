"""
visualizer.py

Renders the complexity quadtree as an overlay on the original image.

Each leaf node is drawn as a rectangle colored by its complexity score
using a heatmap colormap (blue = low complexity, red = high complexity).
Internal node boundries are optionally drawn as thin lines.
"""

import numpy as np
from PIL import Image, ImageDraw
from quadtree import QuadNode


# Colormap

def complexity_to_color(complexity:float, alpha:int=100) -> tuple:
    """
    Map a complexity score in [0, 1] to an RGBA color.
    
    Low complexity  -> cool blue
    Mid complexity  -> green/yellow
    High complexity -> hot red

    Args:
        complexity: float in [0, 1]
        alpha: opacity of the overlay (0=transparent, 255=opaque)

    Returns:
        (R, G, B, A) tuple
    """
    # Simple blue -> cyan -> green -> yellow -> red gradient
    t = max(0.0, min(1.0, complexity))
    
    if t < 0.25:
        s = t / 0.25
        r, g, b = int(0 + s * 0),   int(0 + s * 128),   int(200 + s * 55)
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        r, g, b = int(0 + s * 0),   int(128 + s * 127), int(255 - s * 255)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        r, g, b = int(0 + s * 255), int(255 - s * 128), int(0)
    else:
        s = (t - 0.75) / 0.25
        r, g, b = 255,              int(127 - s * 127), int(0)
    
    return (r, g, b, alpha)


# Overlay renderer

def render_overlay(
    image: np.ndarray,
    root: QuadNode,
    fill_alpha: int = 120,
    border_color: tuple = (255, 255, 255),
    border_width: int = 1,
    show_borders: bool = True
) -> Image.Image:
    """
    Render the quadtree complexity overlay on top of the original image.

    Args:
        image: numpy array (H, W, C) uint8, RGB
        root: root QuadNode from QuadTree.build()
        fill_alpha: opacity of the complexity color filll (0-255)
        border_color: RGBA color for quadrant borders
        border_width: pixel width of borders
        show_borders: whether to draw borders lines between quadrants

    Returns:
        PIL image in RGBA mode with overlay applied
    """
    base = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")
    
    leaves = root.all_leaves()
    
    for node in leaves:
        if node.w <= 0 or node.h <= 0:
            continue
        
        r, g, b, _ = complexity_to_color(node.complexity, alpha=255)
        fill = (r, g, b, fill_alpha)
        outline = border_color + (220,) if show_borders else None
        
        draw.rectangle(
            [node.x, node.y, node.x + node.w - 1, node.y + node.h - 1],
            fill = fill,
            outline = outline,
            width = border_width
        )
    
    return base


# Legend

def render_legend(height:int = 30, width:int = 300) -> Image.Image:
    """
    Render a horizontal colorbar legend showing low -> high complexity.

    Returns:
        PIL Image (RGB)
    """
    legend = Image.new("RGB", (width, height + 20), (30, 30, 30))
    draw = ImageDraw.Draw(legend)
    
    for x in range(width):
        t = x / (width - 1)
        r, g, b, _ = complexity_to_color(t, alpha=255)
        draw.line([(x, 0), (x, height - 1)], fill=(r, g, b))
    
    draw.text((2, height + 2),          "Low",  fill=(200, 200, 200))
    draw.text((width - 26, height + 2), "High", fill=(200, 200, 200))
    
    return legend


# Save helper

def save_result(
    image: np.ndarray,
    root: QuadNode,
    output_path: str,
    fill_alpha: int = 120,
    show_borders: bool = True,
    include_legend: bool = True
) -> None:
    """
    Render and save the overlay image to disk.

    Args:
        image:          numpy array (H, W, C) uint8 RGB
        root:           built QuadNode root
        output_path:    file path to save (e.g. "output.png")
        fill_alpha:     opacity of complexity fill
        show_borders:   draw quadrant borders
        include_legend: append a colorbar legend below the image
    """
    result = render_overlay(image, root, fill_alpha=fill_alpha, show_borders=show_borders)
    
    if include_legend:
        legend = render_legend(width=result.width)
        combined = Image.new("RGB", (result.width, result.height + legend.height))
        combined.paste(result, (0, 0))
        combined.paste(legend, (0, result.height))
        combined.save(output_path)
    else:
        result.save(output_path)
    
    print(f"Saved: {output_path}")