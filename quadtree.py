"""
quadtree.py

Builds a complexity quadtree over an image.

Splitting stops when either:
    max_depth is reached (fixed mode)
    region complexity falls below threshold (adaptive mode)
    region becomes too small to split further (min_size guard)

Both stopping conditions can be active simultaneously (configurable).
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np


# Data structure

@dataclass
class QuadNode:
    x: int # left edge (pixels)
    y: int # top edge (pixels)
    w: int # width (pixels)
    h: int # height (pixels)
    depth: int # 0 = root
    complexity: float # score in [0, 1]
    children: list = field(default_factory=list) # list[QuadNode], empty if leaf
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def all_leaves(self) -> list:
        """Return all leaf nodes in the subtree rooted here."""
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.all_leaves())
        return leaves
    
    def all_nodes(self) -> list:
        """Return every node (internal + leaf) in the subtree."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.all_nodes())
        return nodes


# Builder

class QuadTree:
    """
    Builds a complexity quadtree over a numpy image array.
    
    Args:
        scorer:     Callable(np.ndarray) -> float, see complexity.py
        max_depth:  Maximum split depth. None = unlimited (adaptive only).
        threshold:  Stop splitting when complexity < threshold.
                    None = always split to max_depth.
        min_size:   Minimum region side length in pixels (guard against
                    splitting into sub_pixel regions).
    """
    
    def __init__(
        self,
        scorer: Callable[[np.ndarray], float],
        max_depth: Optional[int] = 6,
        threshold: Optional[float] = None,
        min_size: int = 8
    ):
        self.scorer = scorer
        self.max_depth = max_depth
        self.threshold = threshold
        self.min_size = min_size
    
    def build(self, image: np.ndarray) -> QuadNode:
        """
        Build the quadtree for the given image.

        Args:
            image: numpy array (H, W) or (H, W, C), dtype uint8

        Returns:
            Root QuadNode
        """
        h, w = image.shape[:2]
        root_complexity = self.scorer(image)
        root = QuadNode(x=0, y=0, w=w, h=h, depth=0, complexity=root_complexity)
        self._split(root, image)
        return root
    
    def _split(self, node: QuadNode, image: np.ndarray) -> None:
        """Recursively split a node if stopping condition aren't met."""
        
        # Stopping conditions
        
        # Too small to split
        if node.w < self.min_size * 2 or node.h < self.min_size * 2:
            return
        
        # Max depth reached
        if self.max_depth is not None and node.depth >= self.max_depth:
            return
        
        # Complexity below threshold (adaptive)
        if self.threshold is not None and node.complexity < self.threshold:
            return
        
        # Split into four quadrants
        half_w = node.w // 2
        half_h = node.h // 2
        
        quadrants = [
            (node.x,            node.y,             half_w,             half_h),            # top left
            (node.x + half_w,   node.y,             node.w - half_w,    half_h),            # top right
            (node.x,            node.y + half_h,    half_w,             node.h - half_h),   # bottom left
            (node.x + half_w,   node.y + half_h,    node.w - half_w,    node.h - half_h)    # bottom right
        ]
        
        for(x, y, w, h) in quadrants:
            if w <= 0 or h <= 0:
                continue
            region = image[y:y+h, x:x+w]
            complexity = self.scorer(region)
            child = QuadNode(x=x, y=y, w=w, h=h, depth=node.depth + 1, complexity=complexity)
            node.children.append(child)
            self._split(child, image)



# Convenience stats

def tree_stats(root: QuadNode) -> dict:
    """
    Return summary statistics for a built quadtree.

    Useful for comparing images or tuning thresholds
    """
    leaves = root.all_leaves()
    complexities = [n.complexity for n in leaves]
    depths = [n.depth for n in leaves]
    
    return {
        "total_nodes": len(root.all_nodes()),
        "leaf_count": len(leaves),
        "max_depth_reached": max(depths),
        "mean_leaf_complexity": float(np.mean(complexities)),
        "std_leaf_complexity": float(np.std(complexities)),
        "min_leaf_complexity": float(np.min(complexities)),
        "max_leaf_complexity": float(np.max(complexities)),
    }