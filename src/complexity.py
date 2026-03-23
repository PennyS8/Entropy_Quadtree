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


# Shannon Entropy

def shannon_entropy(region: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Compute normalized Shannon entropy of pixel values in a region.
    
    Treats each channel independently then averages, so RGB images
    are handled naturally.

    Args:
        region: numpy array of shape (H, W) or (H, W, C), dtype uint8
        mask:   optional boolean array (H, W), True = include pixel.
                If provided, only opaque pixels contribute to the score.


    Returns:
        Float in [0, 1]. 0 = perfectly uniform, 1 = maximally random.
    """
    if region.size == 0:
        return 0.0
    
    if mask is not None:
        if region.ndim == 2:
            pixels = region[mask]
        else:
            pixels = region[np.broadcast_to(mask[:, :, None], region.shape)].flatten()
    else:
        pixels = region.flatten()
    
    if pixels.size == 0:
        return 0.0
    
    # Count occurrences of each pixel value (0-255)
    counts = np.bincount(pixels, minlength=256)
    probs = counts / counts.sum()
    
    # Shannon entropy: -sum(p * log2(p)), max is log2(256) = 8 bits
    safe_log = np.where(probs > 0, np.log2(np.where(probs > 0, probs, 1)), 0)
    entropy = -np.sum(probs * safe_log)
    # entropies.append(entropy / 8.0) # normalize to [0, 1]
    
    return float(entropy / 8.0) # normalize: max is log2(256) = 8 bits


# Compression Ratio

def compression_entropy(region: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Estimate complexity via zlib compression ratio.
    
    Args:
        region: numpy array of shape (H, W) or (H, W, C), dtype uint8
        mask:   optional boolean array (H, W), True = include pixel.
                If provided, only opaque pixels are compressed; prevents
                transparent background pixels from delating the score.

    Returns:
        Float in [0, 1]. 0 = perfectly compressible, 1 = incompressible.
    """
    if mask is not None:
        # Extract only the opaque pixels for compression
        if region.ndim == 2:
            raw = region[mask].tobytes()
        else:
            # Flatten each opaque pixel's channels together
            raw = region[mask].tobytes()
    else:
        raw = region.tobytes()
    
    if len(raw) == 0:
        return 0.0
    
    compressed = zlib.compress(raw, level=1)
    
    # Ratio of compressed to original size
    ratio = len(compressed) / len(raw)
    
    # Clamp to [0, 1], compressed can rarely exceed orgional for tiny regions
    return float(min(ratio, 1.0))

def variance_complexity(region: np.ndarray, mask: np.ndarray = None) -> float:
    """
    Estimate complexity as normalised pixel variance
    
    Much faster than Shannon or compression, a single np.var call.
    Caputers spatial spread without histogram or zlib overhead.

    Args:
        region: numpy array (H, W) or (H, W, C), dtype uint8
        mask: optional boolean array (H, W), True = include pixel.

    Returns:
        Float in [0, 1]. 0 = uniform, 1 = maximaly varied.
    """
    if region.size == 0:
        return 0.0
    
    if mask is not None:
        if region.ndim == 2:
            pixels = region[mask].astype(np.float32)
        else:
            pixels = region[np.broadcast_to(mask[:, :, None], region.shape)].astype(np.float32)
    else:
        pixels = region.flatten().astype(np.float32)
    
    if pixels.size == 0:
        return 0.0
    
    # Max possible variance for uint8 is when half pixels=0, half=255 -> var=16256.25
    # MAX_VAR = 16256.25
    # return float(min(np.sqrt(np.var(pixels) / MAX_VAR), 1.0))
    
    # Return raw variance — normalization is handled adaptively in QuadTree
    # using the 99th percentile of the image's own variance distribution.
    return float(np.var(pixels))


# Scorer factory

SCORERS = {
    "shannon": shannon_entropy,
    "compression": compression_entropy,
    "variance": variance_complexity
}

def get_scorer(method: str):
    """
    Return a scoring function by name

    Args:
        method: "shannon" or "compression" or "variance"
    
    Returns:
        Callable(np.ndarray) -> float
    """
    if method not in SCORERS:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(SCORERS.keys())}")
    return SCORERS[method]