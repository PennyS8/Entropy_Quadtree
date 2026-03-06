"""
features.py
-----------
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