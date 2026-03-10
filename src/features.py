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
    
    label: Optional[str] = None # "real", "ai", "photoshopped" (set manually)
    
    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None or k == "label"}


# Extractor

def extract_features(root: QuadNode, filename: str, label: str = None) -> imageFeatures:
    """
    Extract feature vector from a built quadtree.

    Args:
        root:     root QuadNode from QuadTree.build()
        filename: image filename for identification in CSV
        label:    optional ground truth label ("real", "ai", "photoshopped")

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

        label=label,
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


# CSV export

FEATURE_FIELDS = [
    "filename", "label",
    "mean_complexity", "std_complexity", "min_complexity", "max_complexity", "complexity_range",
    "mean_leaf_area", "std_leaf_area", "leaf_count",
    "mean_boundary_delta", "max_boundary_delta",
    "mean_depth", "std_depth",
    "mean_merge_delta", "max_merge_delta", "std_merge_delta",
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