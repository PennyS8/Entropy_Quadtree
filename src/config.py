"""
config.py

Central configuration for the Entropy Quadtree pipeline.

All sripts import their defaults from here so that changing a value
in one place propagates everywhere. No script should hard-code a
default leaf size, method name, resize value, or directory path.
"""

import os
import logging


# Quadtree defaults

# Default leaf side length in pixels per scoring method
# Compression needs larger leaves (16px) to have enough bytes for zlib to
# produce a meaningful compression ratio. Shannon and variance work well at 4px.
DEFAULT_LEAF_SIZES: dict[str, int] = {
    "shannon":      4,
    "compression":  16,
    "variance":     4
}

# All supported scoring methods.
METHODS = list(DEFAULT_LEAF_SIZES.keys())

# Default resize applied to images before processing.
# All datasets are normalized to this resolution so that leaf sizes and
# spatial grid cells are comparable across sources.
DEFAULT_RESIZE: int = 256

# Default method for single-method scripts (main.py, batch.py, tune_thresholds.py).
DEFAULT_METHOD: str = "shannon"

# Background mask threshold: nodes with this fraction of transparent pixelse
# are excluded from scoring and statistics.
BG_THRESHOLD: float = 0.95


# Tree-grid spatial map

# Depth at which to read the spatial grid. 4^4 = 256 cells (16x16 equivalent).
TREE_GRID_DEPTH: int = 4


# Supported image formats

SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# Directory structure

# All paths are relative to the project root (where you run scripts from).

DIRS = {
    # Downloaded image samples for threshold tuning.
    # Format: data/sample/{name}_{label}/
    "sample":       os.path.join("data", "sample"),
 
    # Extracted feature CSVs and sidecar JSON metadata.
    # Format: results/features/{name}_{method}.csv + .json
    "features":     os.path.join("results", "features"),
 
    # Saved model bundles (.joblib).
    # Format: results/models/{name}.joblib
    "models":       os.path.join("results", "models"),
 
    # Threshold tuning CSVs and plots.
    # Format: results/tuning/{name}/{method}.csv + .png
    "tuning":       os.path.join("results", "tuning"),
 
    # Within-dataset classification reports.
    # Format: results/classify/within/{name}.txt
    "classify_within": os.path.join("results", "classify", "within"),
 
    # Cross-dataset evaluation reports.
    # Format: results/classify/cross/{train}_vs_{test}.txt
    "classify_cross":  os.path.join("results", "classify", "cross"),
 
    # Scatter plots.
    # Format: results/scatter/{name}/
    "scatter":      os.path.join("results", "scatter"),
 
    # Spatial grid importance heatmaps.
    # Format: results/spatial/{name}.png
    "spatial":      os.path.join("results", "spatial"),
 
    # Tree depth distribution plots.
    # Format: results/depth/{method}_spatial_map.png
    "depth":        os.path.join("results", "depth"),
 
    # Complexity overlay images.
    # Format: results/overlays/{name}/
    "overlays":     os.path.join("results", "overlays"),
 
    # Prediction output CSVs.
    # Format: results/predictions/{name}.csv
    "predictions":  os.path.join("results", "predictions"),
 
    # Example overlays referenced in README — tracked in git.
    "examples":     os.path.join("results", "examples"),
}


# Logging

LOG_FORMAT  = "%(asctime)s  %(levelname)-8s  %(message)s"
LOG_DATEFMT = "%H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """
    Configure the root logger for console output.
    
    Call once at the top of each script's main().

    Args:
        verbose: if True, show DEBUG messages; otherwise INFO and above.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT
    )
    # Suppress overly chatty third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Use as: log = config.get_logger(__name__)"""
    return logging.getLogger(name)