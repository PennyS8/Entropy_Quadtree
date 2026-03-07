"""
visualizer.py

Renders the complexity quadtree as an overlay on the original image.

Transparency is preserved — RGBA images remain RGBA in the output.
Background nodes (background_ratio >= BG_THRESHOLD) are drawn as neutral
grey. Subject nodes are colored by complexity score (blue=low, red=high).
"""

import numpy as np
from PIL import Image, ImageDraw
from quadtree import QuadNode, BG_THRESHOLD


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
        r, g, b = 0, int(s * 128), int(200 + s * 55)
    elif t < 0.5:
        s = (t - 0.25) / 0.25
        r, g, b = 0, int(128 + s * 127), int(255 - s * 255)
    elif t < 0.75:
        s = (t - 0.5) / 0.25
        r, g, b = int(s * 255), int(255 - s * 128), 0
    else:
        s = (t - 0.75) / 0.25
        r, g, b = 255, int(127 - s * 127), 0
    
    return (r, g, b, alpha)


# Overlay renderer

def render_overlay(
    image: np.ndarray,
    root: QuadNode,
    fill_alpha: int = 120,
    border_color: tuple = (255, 255, 255),
    border_width: int = 1,
    show_borders: bool = True,
) -> Image.Image:
    """
    Render the quadtree complexity overlay on top of the original image.
    
    Draws onto a separate RGBA overlay layer then composites onto the base,
    so the original image's alpha channel is never corrupted.
    
    RGBA images remain RGBA — transparent regions stay transparent.
    Background nodes (background_ratio >= BG_THRESHOLD) are drawn as
    neutral grey. Subject nodes are colored by complexity score.
    
    Args:
        image:          numpy array (H, W, C) uint8, RGB
        root:           root QuadNode from QuadTree.build()
        fill_alpha:     opacity of the complexity color filll (0-255)
        border_color:   RGBA color for quadrant borders
        border_width:   pixel width of borders
        show_borders:   whether to draw borders lines between quadrants

    Returns:
        PIL image in RGBA mode with overlay applied
    """
    has_alpha = image.ndim == 3 and image.shape[2] == 4
    
    if has_alpha:
        base = Image.fromarray(image, mode="RGBA")
    else:
        base = Image.fromarray(image).convert("RGB")
    
    # Draw onto a blank RGBA overlay — never touch the base directly
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    for node in root.all_leaves():
        if node.w <= 0 or node.h <= 0:
            continue
        
        rect = [node.x, node.y, node.x + node.w - 1, node.y + node.h - 1]

        if node.background_ratio >= BG_THRESHOLD:
            draw.rectangle(rect,
                           fill=(128, 128, 128, 80),
                           outline=(100, 100, 100, 120) if show_borders else None,
                           width=border_width)
        else:
            r, g, b, _ = complexity_to_color(node.complexity, alpha=255)
            draw.rectangle(
                rect,
                fill=(r, g, b, fill_alpha),
                outline=border_color + (220,) if show_borders else None,
                width=border_width
            )
    
    # Composite overlay onto base — preserves base alpha
    result = Image.alpha_composite(base, overlay)

    # Return RGB if original had no alpha
    if not has_alpha:
        result = result.convert("RGB")

    return result


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
    include_legend: bool = True,
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
        bg_threshold:   nodes with background_ratio >= this are skipped
    """
    result = render_overlay(image, root, fill_alpha=fill_alpha, show_borders=show_borders)
    
    if include_legend:
        legend = render_legend(width=result.width)
        if result.mode == "RGBA":
            legend = legend.convert("RGBA")
            combined = Image.new("RGBA", (result.width, result.height + legend.height))
        else:
            combined = Image.new("RGB", (result.width, result.height + legend.height))
        combined.paste(result, (0, 0))
        combined.paste(legend, (0, result.height))
        combined.save(output_path)
    else:
        result.save(output_path)
    
    print(f"Saved: {output_path}")