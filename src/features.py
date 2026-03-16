"""
features.py

Extracts a feature vector from a built quadtree for downstream analysis.

Features are designed to distinguish real photographs, AI-generated images,
and photoshopped images based on complexity distribution properties.

Output can be saved as CSV for use in scatter plots or classifiers.
"""

import os
import csv
import numpy as np
from dataclasses import dataclass
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

@dataclass
class imageFeatures:
    filename: str
    
    # Global complexity distribution (subject nodes only)
    mean_complexity: float      # average complexity across subject leaves
    std_complexity: float       # variance of complexity — low in AI images
    min_complexity: float       # floor of complexity
    max_complexity: float       # ceiling of complexity
    complexity_range: float     # max - min — spread of the distribution
    
    # Spacial structure
    mean_leaf_area: float       # average subject leaf area in pixels
    std_leaf_area: float        # variance in leaf area, high = adaptive splits
    leaf_count: int             # total subjet leaf count
    
    # Boundary signal, complexity delta between parent and children
    mean_boundary_delta: float  # average |parent - child| complexity at splits
    max_boundary_delta: float   # maximum boundary jump, manipulation signal
    
    # Depth distribution
    mean_depth: float           # how deep the tree goes on average
    std_depth: float            # variance in depth
    
    # Merge-step delta, information gain at each split
    mean_merge_delta: float     # average std(children) / (parent + eps) across all splits
    max_merge_delta: float      # peak merge delta — highlights manipulation boundaries
    std_merge_delta: float      # variance of merge deltas — high in real, low in AI
    
    # Per-channel (optional, requires image passed to extract_features)
    # Raw means track per-channel complexity; deltas capture inter-channel divergence.
    mean_complexity_r: float = 0.0 # mean complexity of red channel across subject leaves
    mean_complexity_g: float = 0.0 # mean complexity of green channel across subject leaves
    mean_complexity_b: float = 0.0 # mean complexity of blue channel across subject leaves
    complexity_delta_rg: float = 0.0 # |mean_r - mean_g|, inter-channel divergence
    complexity_delta_rb: float = 0.0 # |mean_r - mean_b|
    complexity_delta_gb: float = 0.0 # |mean_g - mean_b|
    
    # Tree-grid spatial map — complexity at each cell of a 4^4=256 node grid.
    # Cells are named tree_grid_000 through tree_grid_255 in Morton order.
    # Filled by propagating parent complexity for nodes that stopped early.
    tree_grid_000: float = 0.0
    tree_grid_001: float = 0.0
    tree_grid_002: float = 0.0
    tree_grid_003: float = 0.0
    tree_grid_004: float = 0.0
    tree_grid_005: float = 0.0
    tree_grid_006: float = 0.0
    tree_grid_007: float = 0.0
    tree_grid_008: float = 0.0
    tree_grid_009: float = 0.0
    tree_grid_010: float = 0.0
    tree_grid_011: float = 0.0
    tree_grid_012: float = 0.0
    tree_grid_013: float = 0.0
    tree_grid_014: float = 0.0
    tree_grid_015: float = 0.0
    tree_grid_016: float = 0.0
    tree_grid_017: float = 0.0
    tree_grid_018: float = 0.0
    tree_grid_019: float = 0.0
    tree_grid_020: float = 0.0
    tree_grid_021: float = 0.0
    tree_grid_022: float = 0.0
    tree_grid_023: float = 0.0
    tree_grid_024: float = 0.0
    tree_grid_025: float = 0.0
    tree_grid_026: float = 0.0
    tree_grid_027: float = 0.0
    tree_grid_028: float = 0.0
    tree_grid_029: float = 0.0
    tree_grid_030: float = 0.0
    tree_grid_031: float = 0.0
    tree_grid_032: float = 0.0
    tree_grid_033: float = 0.0
    tree_grid_034: float = 0.0
    tree_grid_035: float = 0.0
    tree_grid_036: float = 0.0
    tree_grid_037: float = 0.0
    tree_grid_038: float = 0.0
    tree_grid_039: float = 0.0
    tree_grid_040: float = 0.0
    tree_grid_041: float = 0.0
    tree_grid_042: float = 0.0
    tree_grid_043: float = 0.0
    tree_grid_044: float = 0.0
    tree_grid_045: float = 0.0
    tree_grid_046: float = 0.0
    tree_grid_047: float = 0.0
    tree_grid_048: float = 0.0
    tree_grid_049: float = 0.0
    tree_grid_050: float = 0.0
    tree_grid_051: float = 0.0
    tree_grid_052: float = 0.0
    tree_grid_053: float = 0.0
    tree_grid_054: float = 0.0
    tree_grid_055: float = 0.0
    tree_grid_056: float = 0.0
    tree_grid_057: float = 0.0
    tree_grid_058: float = 0.0
    tree_grid_059: float = 0.0
    tree_grid_060: float = 0.0
    tree_grid_061: float = 0.0
    tree_grid_062: float = 0.0
    tree_grid_063: float = 0.0
    tree_grid_064: float = 0.0
    tree_grid_065: float = 0.0
    tree_grid_066: float = 0.0
    tree_grid_067: float = 0.0
    tree_grid_068: float = 0.0
    tree_grid_069: float = 0.0
    tree_grid_070: float = 0.0
    tree_grid_071: float = 0.0
    tree_grid_072: float = 0.0
    tree_grid_073: float = 0.0
    tree_grid_074: float = 0.0
    tree_grid_075: float = 0.0
    tree_grid_076: float = 0.0
    tree_grid_077: float = 0.0
    tree_grid_078: float = 0.0
    tree_grid_079: float = 0.0
    tree_grid_080: float = 0.0
    tree_grid_081: float = 0.0
    tree_grid_082: float = 0.0
    tree_grid_083: float = 0.0
    tree_grid_084: float = 0.0
    tree_grid_085: float = 0.0
    tree_grid_086: float = 0.0
    tree_grid_087: float = 0.0
    tree_grid_088: float = 0.0
    tree_grid_089: float = 0.0
    tree_grid_090: float = 0.0
    tree_grid_091: float = 0.0
    tree_grid_092: float = 0.0
    tree_grid_093: float = 0.0
    tree_grid_094: float = 0.0
    tree_grid_095: float = 0.0
    tree_grid_096: float = 0.0
    tree_grid_097: float = 0.0
    tree_grid_098: float = 0.0
    tree_grid_099: float = 0.0
    tree_grid_100: float = 0.0
    tree_grid_101: float = 0.0
    tree_grid_102: float = 0.0
    tree_grid_103: float = 0.0
    tree_grid_104: float = 0.0
    tree_grid_105: float = 0.0
    tree_grid_106: float = 0.0
    tree_grid_107: float = 0.0
    tree_grid_108: float = 0.0
    tree_grid_109: float = 0.0
    tree_grid_110: float = 0.0
    tree_grid_111: float = 0.0
    tree_grid_112: float = 0.0
    tree_grid_113: float = 0.0
    tree_grid_114: float = 0.0
    tree_grid_115: float = 0.0
    tree_grid_116: float = 0.0
    tree_grid_117: float = 0.0
    tree_grid_118: float = 0.0
    tree_grid_119: float = 0.0
    tree_grid_120: float = 0.0
    tree_grid_121: float = 0.0
    tree_grid_122: float = 0.0
    tree_grid_123: float = 0.0
    tree_grid_124: float = 0.0
    tree_grid_125: float = 0.0
    tree_grid_126: float = 0.0
    tree_grid_127: float = 0.0
    tree_grid_128: float = 0.0
    tree_grid_129: float = 0.0
    tree_grid_130: float = 0.0
    tree_grid_131: float = 0.0
    tree_grid_132: float = 0.0
    tree_grid_133: float = 0.0
    tree_grid_134: float = 0.0
    tree_grid_135: float = 0.0
    tree_grid_136: float = 0.0
    tree_grid_137: float = 0.0
    tree_grid_138: float = 0.0
    tree_grid_139: float = 0.0
    tree_grid_140: float = 0.0
    tree_grid_141: float = 0.0
    tree_grid_142: float = 0.0
    tree_grid_143: float = 0.0
    tree_grid_144: float = 0.0
    tree_grid_145: float = 0.0
    tree_grid_146: float = 0.0
    tree_grid_147: float = 0.0
    tree_grid_148: float = 0.0
    tree_grid_149: float = 0.0
    tree_grid_150: float = 0.0
    tree_grid_151: float = 0.0
    tree_grid_152: float = 0.0
    tree_grid_153: float = 0.0
    tree_grid_154: float = 0.0
    tree_grid_155: float = 0.0
    tree_grid_156: float = 0.0
    tree_grid_157: float = 0.0
    tree_grid_158: float = 0.0
    tree_grid_159: float = 0.0
    tree_grid_160: float = 0.0
    tree_grid_161: float = 0.0
    tree_grid_162: float = 0.0
    tree_grid_163: float = 0.0
    tree_grid_164: float = 0.0
    tree_grid_165: float = 0.0
    tree_grid_166: float = 0.0
    tree_grid_167: float = 0.0
    tree_grid_168: float = 0.0
    tree_grid_169: float = 0.0
    tree_grid_170: float = 0.0
    tree_grid_171: float = 0.0
    tree_grid_172: float = 0.0
    tree_grid_173: float = 0.0
    tree_grid_174: float = 0.0
    tree_grid_175: float = 0.0
    tree_grid_176: float = 0.0
    tree_grid_177: float = 0.0
    tree_grid_178: float = 0.0
    tree_grid_179: float = 0.0
    tree_grid_180: float = 0.0
    tree_grid_181: float = 0.0
    tree_grid_182: float = 0.0
    tree_grid_183: float = 0.0
    tree_grid_184: float = 0.0
    tree_grid_185: float = 0.0
    tree_grid_186: float = 0.0
    tree_grid_187: float = 0.0
    tree_grid_188: float = 0.0
    tree_grid_189: float = 0.0
    tree_grid_190: float = 0.0
    tree_grid_191: float = 0.0
    tree_grid_192: float = 0.0
    tree_grid_193: float = 0.0
    tree_grid_194: float = 0.0
    tree_grid_195: float = 0.0
    tree_grid_196: float = 0.0
    tree_grid_197: float = 0.0
    tree_grid_198: float = 0.0
    tree_grid_199: float = 0.0
    tree_grid_200: float = 0.0
    tree_grid_201: float = 0.0
    tree_grid_202: float = 0.0
    tree_grid_203: float = 0.0
    tree_grid_204: float = 0.0
    tree_grid_205: float = 0.0
    tree_grid_206: float = 0.0
    tree_grid_207: float = 0.0
    tree_grid_208: float = 0.0
    tree_grid_209: float = 0.0
    tree_grid_210: float = 0.0
    tree_grid_211: float = 0.0
    tree_grid_212: float = 0.0
    tree_grid_213: float = 0.0
    tree_grid_214: float = 0.0
    tree_grid_215: float = 0.0
    tree_grid_216: float = 0.0
    tree_grid_217: float = 0.0
    tree_grid_218: float = 0.0
    tree_grid_219: float = 0.0
    tree_grid_220: float = 0.0
    tree_grid_221: float = 0.0
    tree_grid_222: float = 0.0
    tree_grid_223: float = 0.0
    tree_grid_224: float = 0.0
    tree_grid_225: float = 0.0
    tree_grid_226: float = 0.0
    tree_grid_227: float = 0.0
    tree_grid_228: float = 0.0
    tree_grid_229: float = 0.0
    tree_grid_230: float = 0.0
    tree_grid_231: float = 0.0
    tree_grid_232: float = 0.0
    tree_grid_233: float = 0.0
    tree_grid_234: float = 0.0
    tree_grid_235: float = 0.0
    tree_grid_236: float = 0.0
    tree_grid_237: float = 0.0
    tree_grid_238: float = 0.0
    tree_grid_239: float = 0.0
    tree_grid_240: float = 0.0
    tree_grid_241: float = 0.0
    tree_grid_242: float = 0.0
    tree_grid_243: float = 0.0
    tree_grid_244: float = 0.0
    tree_grid_245: float = 0.0
    tree_grid_246: float = 0.0
    tree_grid_247: float = 0.0
    tree_grid_248: float = 0.0
    tree_grid_249: float = 0.0
    tree_grid_250: float = 0.0
    tree_grid_251: float = 0.0
    tree_grid_252: float = 0.0
    tree_grid_253: float = 0.0
    tree_grid_254: float = 0.0
    tree_grid_255: float = 0.0
    
    label: Optional[str] = None # "real", "ai", "photoshopped" (set manually)
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "label"}


# Extractor

def extract_features(root: QuadNode, filename: str, label: str = None,
                    image: np.ndarray = None, scorer=None,
                    img_shape: tuple = None) -> imageFeatures:
    """
    Extract feature vector from a built quadtree.
    
    Args:
        root:       root QuadNode from QuadTree.build()
        filename:   image filename for identification in CSV
        label:      optional ground truth label ("real", "ai", "photoshopped")
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
    areas           = np.array([n.w * n.h       for n in subject_leaves], dtype=float)
    depths          = np.array([n.depth         for n in subject_leaves], dtype=float)
    
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
        max_complexity=float(np.max(complexities)),
        complexity_range=float(np.max(complexities) - np.min(complexities)),

        mean_leaf_area=float(np.mean(areas)),
        std_leaf_area=float(np.std(areas)),
        leaf_count=len(subject_leaves),

        mean_boundary_delta=float(np.mean(boundary_deltas)) if len(boundary_deltas) else 0.0,
        max_boundary_delta=float(np.max(boundary_deltas))  if len(boundary_deltas) else 0.0,

        mean_depth=float(np.mean(depths)),
        std_depth=float(np.std(depths)),

        mean_merge_delta=float(np.mean(merge_deltas)) if len(merge_deltas) else 0.0,
        max_merge_delta=float(np.max(merge_deltas))  if len(merge_deltas) else 0.0,
        std_merge_delta=float(np.std(merge_deltas))  if len(merge_deltas) else 0.0,
        
        **channel_features,
        **spatial_features,
        
        label=label
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
    "filename", "label",
    "mean_complexity", "std_complexity", "min_complexity", "max_complexity", "complexity_range",
    "mean_leaf_area", "std_leaf_area", "leaf_count",
    "mean_boundary_delta", "max_boundary_delta",
    "mean_depth", "std_depth",
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