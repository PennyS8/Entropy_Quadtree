"""
quadtree.py

Builds an adaptive complexity quadtree over an image.
Core data structure for the Quadtree Complexity Analysis for Image Forensics pipeline.

Splitting stops when either:
    leaf_size is reached (fixed mode).
    Region complexity falls below threshold (adaptive mode).
    Region is almost entirely transparent (bg_split_threshold).

Background masking: if an alpha channel is provided:
    Compexity scoring excludes transparent pixels so boundary regions
    aren't dragged down by easily-compressed background.
    Each node is tagged with background_ratio for use in stats/visualization
    bg_threshold controls classification (stats + overlay exclusion)
    bg_split_threshold controls early split stopping (performance only)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np

# Nodes with this fraction of transparent pixels are treated as background:
# excluded from stats, shown as grey in the overlay, and not split further.
BG_THRESHOLD = 0.95


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
        scorer:     Callable(np.ndarray) -> float, see complexity.py
        leaf_size:  Maximum split depth. None = unlimited (adaptive only).
        threshold:  Percentile (0-100) below which leaves are pruned after the
                    full tree is built. E.g. threshold=25 collapses the least
                    complex 25% of leaves back into their parent. Content-adaptive,
                    percentile is computed from the image itself. Default None.
    """
    
    def __init__(
        self,
        scorer: Callable[[np.ndarray], float],
        leaf_size: Optional[int] = 4,
        threshold: Optional[float] = None
    ):
        self.scorer = scorer
        self.leaf_size = leaf_size
        self.threshold = threshold
    
    def build(self, image: np.ndarray, alpha: Optional[np.ndarray] = None, normalize: bool = False) -> QuadNode:
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
        
        # Per-image normalization (overlay only)
        # Stretch the complexity range to [0, 1] using the 1st and 99th
        # percentile of subject leaves so the full colormap is always used
        # regardless of scoring method. Variance needs this because its raw
        # values exceed 1; Shannon and compression benefit from it too because
        # their values rarely span the full [0, 1] range in practice.
        if normalize:
            subject_leaves = [n for n in root.all_leaves() if n.background_ratio < BG_THRESHOLD]
            if subject_leaves:
                raw_vals = np.array([n.complexity for n in subject_leaves])
                p01 = float(np.percentile(raw_vals, 1))
                p99 = float(np.percentile(raw_vals, 99))
                span = p99 - p01
                if span < 1e-6:
                    span = 1e-6
                # Only apply sqrt gamma for variance — its raw values are
                # heavily right-skewed (unbounded), so sqrt brings the low end
                # up into the visible colour range. Shannon and compression
                # are already roughly uniform in [0,1] so gamma would
                # over-amplify background noise.
                needs_gamma = p99 > 1.0
                for node in self._all_nodes(root):
                    normed = float(np.clip((node.complexity - p01) / span, 0.0, 1.0))
                    node.complexity = float(np.sqrt(normed)) if needs_gamma else normed
        
        # Prune below percentile threshold
        if self.threshold is not None:
            all_leaves = root.all_leaves()
            subject_leaves = [n for n in all_leaves if n.background_ratio < BG_THRESHOLD]
            if subject_leaves:
                complexities = np.array([n.complexity for n in subject_leaves])
                cutoff = float(np.percentile(complexities, self.threshold))
                self._prune(root, cutoff)
        
        return root
    
    def _all_nodes(self, root: QuadNode):
        """Yield every node in the tree (internal + leaves)."""
        queue = [root]
        while queue:
            node = queue.pop()
            yield node
            queue.extend(node.children)
    
    def _prune(self, node: QuadNode, cutoff: float) -> None:
        """
        Recursively collapse any internal node whose children are all leaves
        and all score below the cutoff complexity.
        """
        if node.is_leaf:
            return
        # Recurse into children first (bottom-up)
        for child in node.children:
            self._prune(child, cutoff)
        # Collapse this node if all children are now leaves below cutoff
        if all(c.is_leaf and c.complexity < cutoff for c in node.children):
            node.children.clear()
    
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
        
        # Leaf size reached — children would be at or below target size
        if node.w // 2 <= self.leaf_size or node.h // 2 <= self.leaf_size:
            return

        # Background — don't split nearly-transparent regions
        if node.background_ratio >= BG_THRESHOLD:
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
            
            # # Relative threshold, skip recursing if child complexity didn't
            # # change enough from parent to justify further splitting
            # if self.threshold is not None:
            #     delta = abs(complexity - node.complexity)
            #     if delta < self.threshold:
            #         continue
            
            self._split(child, image, alpha)


# Convenience stats

def tree_stats(root: QuadNode) -> dict:
    """Summary statistics for a built quadtree. Background nodes are excluded using BG_THRESHOLD."""
    all_leaves = root.all_leaves()
    subject_leaves = [n for n in all_leaves if n.background_ratio < BG_THRESHOLD]
    
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