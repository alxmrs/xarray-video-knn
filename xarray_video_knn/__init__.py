"""
XArray Video KNN - A library for k-nearest neighbors classification using video compression.

This library implements the compression-based classification method from
"Low-Resource Text Classification: A Parameter-Free Classification Method with Compressors"
adapted for multidimensional xarray data using video compression instead of gzip.
"""

from .classifier import XArrayVideoKNNClassifier
from .utils import create_conversion_rules, estimate_compression_size

__version__ = "0.1.0"
__author__ = "Alex Merose"
__email__ = "al@merose.com"

__all__ = [
    "XArrayVideoKNNClassifier",
    "create_conversion_rules",
    "estimate_compression_size"
]