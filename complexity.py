"""
complexity.py

Scoring algorithms for image region complexity.

Two methods:
    1. Shannon entropy: fast, based on pixel value distribution
    2. Compression ratio: slower, captures repetition (fixes A+A problem)

Both return a float in [0, 1] where 1 = maximally complex
"""

import zlib
import numpy as np
from PIL import Image


# Shannon Entropy

def shannon_entropy(region: np.ndarray) -> float:
    """
    Compute normalized Shannon entropy of pixel values in a region.
    
    Treats each channel independently then averages, so RGB images
    are handled naturally.

    Args:
        region: numpy array of shape (H, W) or (H, W, C), dtype uint8

    Returns:
        Float in [0, 1]. 0 = perfectly uniform, 1 = maximally random.
    """
    if region.size == 0:
        return 0.0

    # Flatten to 1D per channel
    if region.ndim == 2:
        channels = [region.flatten()]
    else:
        channels = [region[:, :, c].flatten() for c in range(region.shape[2])]
    
    entropies = []
    for channel in channels:
        # Count occurrences of each pixel value (0-255)
        counts = np.bincount(channel, minlength=256)
        probs = counts / counts.sum()
        
        # Remove zero probabilities before log
        probs = probs[probs > 0]
        
        # Shannon entropy: -sum(p * log2(p)), max is log2(256) = 8 bits
        entropy = -np.sum(probs * np.log2(probs))
        entropies.append(entropy / 8.0) # normalize to [0, 1]
    
    return float(np.mean(entropies))


# Compression Ratio

def compression_entropy(region: np.ndarray) -> float:
    """
    Estimate complexity via zlib compression ratio.
    
    Key property: compression_entropy(A+A) < compression(A)
    because repeated content compresses more. Shannon entropy
    would give the same score for both.
    
    Args:
        region: numpy array of shape (H, W) or (H, W, C), dtype uint8

    Returns:
        Float in [0, 1]. 0 = perfectly compressible, 1 = incompressible.
    """
    raw = region.tobytes()
    if len(raw) == 0:
        return 0.0
    
    compressed = zlib.compress(raw, level=6)
    
    # Ratio of compressed to original size
    ratio = len(compressed) / len(raw)
    
    # Clamp to [0, 1], compressed can rarely exceed orgional for tiny regions
    return float(min(ratio, 1.0))


# Scorer factory

SCORERS = {
    "shannon": shannon_entropy,
    "compression": compression_entropy
}

def get_scorer(method: str):
    """
    Return a scoring function by name

    Args:
        method: "shannon" or "compression"
    
    Returns:
        Callable(np.ndarray) -> float
    """
    if method not in SCORERS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(SCORERS.keys())}")
    return SCORERS[method]