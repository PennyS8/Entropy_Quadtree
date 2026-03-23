"""
features.py

Extracts a complexity feature vector from a built quadtree for use in
image forensics classification.

Part of the Quadtree Complexity Analysis for Image Forensics pipeline.
Features characterise the spatial distribution of image complexity to
distinguish authentic photographs from synthetic (AI-generated) and
manipulated (face-swapped/composited) images.

Output can be saved as CSV for use in scatter plots or classifiers.
"""

import os
import csv
import numpy as np
from dataclasses import dataclass, field, make_dataclass
from typing import Optional
from quadtree import QuadNode, BG_THRESHOLD


# Tree-grid spatial map
# Extract complexity at a fixed depth in the quadtree rather than a fixed pixel grid.
# At depth D the tree has up to 4^D nodes covering the image in a regular spatial
# arrangement. Resolution-independent: cell size = image_size / 2^D regardless of
# image resolution, because the tree was built with a fixed leaf_size.
#
# Traversal order: NW, NE, SW, SE at each split (matches quadtree.py quadrant order),
# which is Morton order — top-left to bottom-right, row-major within each quadrant.
#
# Cells where the tree stopped splitting before target_depth (pruned, background,
# or leaf_size reached) are filled by propagating the parent node's complexity
# downward. This produces a fixed-length vector for every image.
TREE_GRID_DEPTH = 4                    # 4^4 = 256 cells, 16x16 equivalent
TREE_GRID_SIZE  = 4 ** TREE_GRID_DEPTH # 256
TREE_GRID_NAMES = ["tree_grid_{:03d}".format(i) for i in range(TREE_GRID_SIZE)]


# Feature vector

# imageFeatures is generated dynamically so the 256 tree_grid_* fields don't
# need to be written out by hand. make_dataclass produces an identical class to
# what a hand-written @dataclass would give — same constructor, same attributes,
# same repr, same to_dict — the only difference is the class body is 5 lines
# instead of 300.
#
# Field layout (dataclass rules: required fields must precede fields with defaults):
#   1. Required (no default): filename + 9 scalar metrics
#   2. Optional float=0.0:    per-channel features, then 256 tree_grid cells
#   3. Optional metadata:     label, label_detail, is_real, dataset_source
#
# Removed as confirmed-dead (zero permutation importance across all runs):
#   mean_leaf_area, std_leaf_area, leaf_count  — constant for fixed leaf_size
#   mean_depth, std_depth                      — constant for fixed leaf_size
#   max_complexity                             — almost always exactly 1.0

def _to_dict(self) -> dict:
    always = {"label", "label_detail", "is_real", "dataset_source"}
    return {k: v for k, v in self.__dict__.items() if v is not None or k in always}

imageFeatures = make_dataclass(
    "imageFeatures",
    fields=[
        # --- required (no default) ---
        ("filename",             str),
        # Global complexity distribution (subject nodes only)
        ("mean_complexity",      float),   # average complexity across subject leaves
        ("std_complexity",       float),   # variance of complexity — low in AI images
        ("min_complexity",       float),   # floor of complexity
        ("complexity_range",     float),   # max - min — spread of the distribution
        # Boundary signal
        ("mean_boundary_delta",  float),   # average |parent - child| complexity at splits
        ("max_boundary_delta",   float),   # maximum boundary jump, manipulation signal
        # Merge-step delta — information gain at each split
        ("mean_merge_delta",     float),   # average std(children)/(parent+eps) across splits
        ("max_merge_delta",      float),   # peak merge delta — highlights manipulation boundaries
        ("std_merge_delta",      float),   # variance of merge deltas — high in real, low in AI
        # --- optional float=0.0 ---
        # Per-channel (requires image passed to extract_features)
        ("mean_complexity_r",    float,    field(default=0.0)),  # mean complexity, red channel
        ("mean_complexity_g",    float,    field(default=0.0)),  # mean complexity, green channel
        ("mean_complexity_b",    float,    field(default=0.0)),  # mean complexity, blue channel
        ("complexity_delta_rg",  float,    field(default=0.0)),  # |mean_r - mean_g|
        ("complexity_delta_rb",  float,    field(default=0.0)),  # |mean_r - mean_b|
        ("complexity_delta_gb",  float,    field(default=0.0)),  # |mean_g - mean_b|
        # Tree-grid spatial map — 256 cells in Morton order (tree_grid_000..255)
        *[(name, float, field(default=0.0)) for name in TREE_GRID_NAMES],
        # --- optional metadata ---
        ("label",          Optional[str],  field(default=None)),  # "authentic"/"synthetic"/"manipulated"
        ("label_detail",   Optional[str],  field(default=None)),  # e.g. "diffusion", "real_portrait"
        ("is_real",        int,            field(default=-1)),     # 1=authentic, 0=not, -1=unknown
        ("dataset_source", str,            field(default="")),     # e.g. "ciplab/real-and-fake-face-detection"
    ],
    namespace={"to_dict": _to_dict},
)


# Extractor

def extract_features(root: QuadNode, filename: str, label: str = None,
                    image: np.ndarray = None, scorer=None,
                    img_shape: tuple = None,
                    label_detail: str = None,
                    dataset_source: str = "") -> imageFeatures:
    """
    Extract feature vector from a built quadtree.
    
    Args:
        root:       root QuadNode from QuadTree.build()
        filename:   image filename for identification in CSV
        label:      optional ground truth label ("authentic", "synthetic", "manipulated")
        image:      optional numpy array (H, W, C) uint8 — if provided, per-channel
                    complexity features are computed. Must be the same image used
                    to build the tree so leaf coordinates are valid.
        scorer:     optional scoring function — should match the method used to build
                    the tree. Required if image is provided.
        img_shape:  optional (H, W) tuple. Inferred from image if not provided.
            
    Returns:
        ImageFeatures dataclass
    """
    all_leaves = root.all_leaves()
    subject_leaves = [n for n in all_leaves if n.background_ratio < BG_THRESHOLD]
    
    if not subject_leaves:
        subject_leaves = all_leaves
    
    complexities    = np.array([n.complexity    for n in subject_leaves])
    
    # Boundary delta: for every internal node, measure complexity jump to children
    boundary_deltas = _compute_boundary_deltas(root)
    merge_deltas    = _compute_merge_deltas(root)
    
    channel_features = _compute_channel_features(subject_leaves, image, scorer)
    
    spatial_features = _compute_tree_grid(root, TREE_GRID_DEPTH)
    
    return imageFeatures(
        filename=os.path.basename(filename),

        mean_complexity=float(np.mean(complexities)),
        std_complexity=float(np.std(complexities)),
        min_complexity=float(np.min(complexities)),
        complexity_range=float(np.max(complexities) - np.min(complexities)),

        mean_boundary_delta=float(np.mean(boundary_deltas)) if len(boundary_deltas) else 0.0,
        max_boundary_delta=float(np.max(boundary_deltas))  if len(boundary_deltas) else 0.0,

        mean_merge_delta=float(np.mean(merge_deltas)) if len(merge_deltas) else 0.0,
        max_merge_delta=float(np.max(merge_deltas))  if len(merge_deltas) else 0.0,
        std_merge_delta=float(np.std(merge_deltas))  if len(merge_deltas) else 0.0,
        
        **channel_features,
        **spatial_features,
        
        label=label,
        label_detail=label_detail,
        is_real=1 if label == "authentic" else (0 if label is not None else -1),
        dataset_source=dataset_source,
    )


def _compute_boundary_deltas(root: QuadNode) -> list:
    """
    For every internal node, compute the absolute complexity difference
    between the parent and each of its children.

    High deltas at a boundary = abrupt complexity discontinuity = manipulation signal.
    """
    deltas = []
    queue = [root]
    while queue:
        node = queue.pop()
        if not node.is_leaf:
            subject_children = [c for c in node.children if c.background_ratio < BG_THRESHOLD]
            for child in subject_children:
                deltas.append(abs(node.complexity - child.complexity))
            queue.extend(node.children)
    return deltas


def _compute_merge_deltas(root:QuadNode) -> list:
    """
    Compute merge-step delta for every internal subject node.
    
    Delta = std(child_complexities) / (parent_complexity + eps)
    
    This measures the information gained at each split, how much
    heterogeneity the split revealed relative to the parent score.
    
    High delta: split revealed genuine internal structure (real photo signal).
    Low delta: split revealed homogeneous subregions (AI generator singal).
    Peak delta: manipulation boundary where generated content meets natural image.
    """
    EPS = 1e-6
    deltas = []
    queue = [root]
    while queue:
        node = queue.pop()
        if not node.is_leaf:
            subject_children = [c for c in node.children if c.background_ratio < BG_THRESHOLD]
            if len(subject_children) >= 2:
                child_complexities = np.array([c.complexity for c in subject_children])
                delta = float(np.std(child_complexities) / (node.complexity + EPS))
                deltas.append(delta)
            queue.extend(node.children)
    return deltas


def _compute_channel_features(subject_leaves: list, image: np.ndarray, scorer) -> dict:
    """
    Score each RGB channel independently across all subject leaf regions and
    return per-channel mean complexity and inter-channel delta features.
    
    Args:
        subject_leaves: list of subject QuadNodes (background already filtered)
        image:          numpy array (H, W, C) uint8 — the original image
        scorer:         complexity scoring function, same as used to build the tree
        
    Returns:
        dict with keys: mean_complexity_r/g/b, complexity_delta_rg/rb/gb.
        All zeros if image or scorer is None, or if image has fewer than 3 channels.
    """
    zero = {
        "mean_complexity_r": 0.0, "mean_complexity_g": 0.0, "mean_complexity_b": 0.0,
        "complexity_delta_rg": 0.0, "complexity_delta_rb": 0.0, "complexity_delta_gb": 0.0
    }
    
    if image is None or scorer is None:
        return zero
    if image.ndim < 3 or image.shape[2] < 3:
        return zero
    if not subject_leaves:
        return zero
    
    scores_r, scores_g, scores_b = [], [], []
    
    for node in subject_leaves:
        region = image[node.y:node.y + node.h, node.x:node.x + node.w]
        if region.size == 0:
            continue
        # Score each channel slice independently — no mask needed since
        # subject_leaves are already filtered for background_ratio
        scores_r.append(scorer(region[:, :, 0]))
        scores_g.append(scorer(region[:, :, 1]))
        scores_b.append(scorer(region[:, :, 2]))
    
    if not scores_r:
        return zero
    
    mean_r = float(np.mean(scores_r))
    mean_g = float(np.mean(scores_g))
    mean_b = float(np.mean(scores_b))
    
    return {
        "mean_complexity_r":   mean_r,
        "mean_complexity_g":   mean_g,
        "mean_complexity_b":   mean_b,
        "complexity_delta_rg": float(abs(mean_r - mean_g)),
        "complexity_delta_rb": float(abs(mean_r - mean_b)),
        "complexity_delta_gb": float(abs(mean_g - mean_b))
    }


def _compute_tree_grid(root, target_depth: int) -> dict:
    """
    Extract a fixed-length complexity vector by traversing the quadtree to
    target_depth and reading node complexity values in Morton order.

    Morton order: NW, NE, SW, SE at each split, matching quadtree.py's quadrant
    order. This gives a top-left to bottom-right, row-major spatial ordering
    at each level, so tree_grid_000 is always the top-left cell and
    tree_grid_255 is always the bottom-right cell.

    Nodes that stopped splitting before target_depth (due to leaf_size,
    background, or pruning) propagate their complexity value down to fill
    all 4^(target_depth - node.depth) positions they cover. This ensures
    the output vector is always exactly 4^target_depth elements.

    Args:
        root:         root QuadNode
        target_depth: depth at which to read the grid (4 gives 256 cells)

    Returns:
        dict with keys tree_grid_000 through tree_grid_{4^target_depth - 1}
    """
    n_cells = 4 ** target_depth
    values = np.zeros(n_cells, dtype=np.float32)

    def _fill(node, idx_start, n_slots):
        """Recursively fill values[idx_start:idx_start+n_slots] from this node."""
        if n_slots == 1 or node.is_leaf or node.depth >= target_depth:
            # This node covers n_slots cells — fill all with its complexity
            values[idx_start:idx_start + n_slots] = node.complexity
            return
        # Split into 4 child slots (NW, NE, SW, SE — Morton order)
        child_slots = n_slots // 4
        children = node.children  # always 4 or 0
        if len(children) == 4:
            for i, child in enumerate(children):
                _fill(child, idx_start + i * child_slots, child_slots)
        else:
            # Node has no children despite not being at target depth
            # (background or pruned) — fill with parent complexity
            values[idx_start:idx_start + n_slots] = node.complexity

    _fill(root, 0, n_cells)

    return {"tree_grid_{:03d}".format(i): float(values[i]) for i in range(n_cells)}


# CSV export

FEATURE_FIELDS = [
    "filename", "label", "label_detail", "is_real", "dataset_source",
    "mean_complexity", "std_complexity", "min_complexity", "complexity_range",
    "mean_boundary_delta", "max_boundary_delta",
    "mean_merge_delta", "max_merge_delta", "std_merge_delta",
    "mean_complexity_r", "mean_complexity_g", "mean_complexity_b",
    "complexity_delta_rg", "complexity_delta_rb", "complexity_delta_gb",
    *TREE_GRID_NAMES,
]

def save_csv(features_list: list, output_path: str) -> None:
    """
    Save a list of ImageFeatures to a CSV file.

    Args:
        features_list: list of ImageFeatures
        output_path:   path to write CSV
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        for feat in features_list:
            row = feat.to_dict()
            writer.writerow({k: row.get(k, "") for k in FEATURE_FIELDS})
    print(f"Saved features: {output_path}")


def load_csv(path: str) -> list:
    """Load a CSV of features back into a list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))