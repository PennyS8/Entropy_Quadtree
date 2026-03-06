"""
quadtree.py

Builds a complexity quadtree over an image.

Splitting stops when either:
    Max_depth is reached (fixed mode).
    Region complexity falls below threshold (adaptive mode).
    Region becomes too small to split further (min_size guard).
    Region is almost entirely transparent (bg_split_threshold).

Background masking: if an alpha channel is provided:
    Compexity scoring excludes transparent pixels so boundry regions
    aren't dragged down by easily-compressed background.
    Each node is tagged with background_ratio for use in stats/visualization
    bg_threshold controls classification (stats + overlay exclusion)
    bg_split_threshold controls early split stopping (performance only)
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
    background_ratio: float = 0.0
    
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
        scorer:             Callable(np.ndarray) -> float, see complexity.py
        max_depth:          Maximum split depth. None = unlimited (adaptive only).
        threshold:          Stop splitting when complexity < threshold.
                            None = always split to max_depth.
        min_size:           Minimum region side length in pixels (guard against
                            splitting into sub_pixel regions).
        bg_threshold:       Fraction of transparent pixels above which a node is
                            cosidered background (default 0.5 = majority transparent).
        bg_split_threshold: Nodes with background_ratio >= this stop splitting
                            early. Should be high (0.95)
    """
    
    def __init__(
        self,
        scorer: Callable[[np.ndarray], float],
        max_depth: Optional[int] = 6,
        threshold: Optional[float] = None,
        min_size: int = 8,
        bg_threshold: float = 0.5,          # classify as background (stats + overlay)
        bg_split_threshold: float = 0.95    # stop splitting
    ):
        self.scorer = scorer
        self.max_depth = max_depth
        self.threshold = threshold
        self.min_size = min_size
        self.bg_threshold = bg_threshold
        self.bg_split_threshold = bg_split_threshold
    
    def build(self, image: np.ndarray, alpha: Optional[np.ndarray] = None) -> QuadNode:
        """
        Build the quadtree for the given image.

        Args:
            image:  numpy array (H, W) or (H, W, C), dtype uint8
            alpha:  optional numpy array (H, W) unit8, 0=transparent 255=opaque.
                    If provided, each node is tagged with background_ratio.

        Returns:
            Root QuadNode
        """
        h, w = image.shape[:2]
        bg_ratio = self._bg_ratio(alpha, 0, 0, w, h) if alpha is not None else 0.0
        mask = self._make_mask(alpha, 0, 0, w, h) if alpha is not None else None
        root_complexity = self.scorer(image, mask)
        root = QuadNode(x=0, y=0, w=w, h=h, depth=0, complexity=root_complexity, background_ratio=bg_ratio)
        self._split(root, image, alpha)
        return root
    
    def _bg_ratio(self, alpha: np.ndarray, x: int, y: int, w: int, h: int) -> float:
        """Return the fraction of pixels in this region that are transparent (alpha < 128)."""
        region = alpha[y:y+h, x:x+w]
        return float((region < 128).mean())
    
    def _make_mask(self, alpha: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Boolean mask (H, W), True where pixel is opaque (alpha >= 128)."""
        return alpha[y:y+h, x:x+w] >= 128
    
    def _score_region(self, region: np.ndarray, mask: np.ndarray) -> float:
        """
        Score a region using only opaque pixels.
        Strips the alpha channel if present so it does not corrupt the score.
        If no opaque pixels exist, return 0.
        """
        # Drop alpha channel, score only RGB data
        if region.ndim == 3 and region.shape[2] == 4:
            region = region[:, :, :3]
        if mask is not None and mask.sum() == 0:
            return 0.0
        return self.scorer(region, mask)
    
    def _split(self, node: QuadNode, image: np.ndarray, alpha: Optional[np.ndarray] = None) -> None:
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
        
        # Stop splitting only if almost entirely transparent
        if node.background_ratio >= self.bg_split_threshold:
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
            bg_ratio = self._bg_ratio(alpha, x, y, w, h) if alpha is not None else 0.0
            mask = self._make_mask(alpha, x, y, w, h) if alpha is not None else None
            complexity = self._score_region(region, mask)
            
            child = QuadNode(x=x, y=y, w=w, h=h, depth=node.depth + 1, complexity=complexity, background_ratio=bg_ratio)
            node.children.append(child)
            self._split(child, image, alpha)


# Convenience stats

def tree_stats(root: QuadNode, bg_threshold: float = 0.5) -> dict:
    """
    Return summary statistics for a built quadtree.

    Useful for comparing images or tuning thresholds
    """
    all_leaves = root.all_leaves()
    subject_leaves = [n for n in all_leaves if n.background_ratio < bg_threshold]
    
    if not subject_leaves:
        # Fallback if entire image is transparent
        subject_leaves = all_leaves
    
    complexities = [n.complexity for n in subject_leaves]
    depths = [n.depth for n in subject_leaves]
    
    return {
        "total_nodes":              len(root.all_nodes()),
        "leaf_count":               len(all_leaves),
        "subject_leaf_count":       len(subject_leaves),
        "background_leaf_count":    len(all_leaves) - len(subject_leaves),
        "max_depth_reached":        max(depths),
        "mean_leaf_complexity":     float(np.mean(complexities)),
        "std_leaf_complexity":      float(np.std(complexities)),
        "min_leaf_complexity":      float(np.min(complexities)),
        "max_leaf_complexity":      float(np.max(complexities)),
    }