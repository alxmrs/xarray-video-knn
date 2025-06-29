"""
Main classifier module for XArray Video KNN.
"""

import logging
import os
import tempfile
from collections import Counter
from typing import Any, Optional

import numpy as np
import xarray as xr
from xarrayvideo import xarray2video
from .utils import create_conversion_rules

logger = logging.getLogger(__name__)


class XArrayVideoKNNClassifier:
  """
  K-Nearest Neighbors classifier using video compression for xarray data.

  This classifier implements the Normalized Compression Distance (NCD) method
  using video compression to measure similarity between multidimensional datasets.

  Based on: "Low-Resource Text Classification: A Parameter-Free Classification
  Method with Compressors" but adapted for xarray data with video compression.

  Parameters
  ----------
  k : int, default=1
      Number of nearest neighbors to consider for classification.
  compression_params : dict, optional
      Video compression parameters. If None, uses lossless FFV1 codec.
  temp_dir : str, optional
      Directory for temporary files. Uses system temp directory if None.
  use_lossless : bool, default=True
      Whether to use lossless compression (recommended for classification).
  cleanup_temp_files : bool, default=True
      Whether to automatically clean up temporary files.

  Attributes
  ----------
  training_data_ : list[xr.Dataset]
      Training datasets after fitting.
  training_labels_ : list[Any]
      Training labels after fitting.
  is_fitted_ : bool
      Whether the classifier has been fitted.
  """

  def __init__(
      self,
      k: int = 3,
      compression_params: Optional[dict] = None,
      temp_dir: Optional[str] = None,
      use_lossless: bool = False,
      cleanup_temp_files: bool = True,
      verbose=False,
  ):
    self.k = k
    self.temp_dir = temp_dir or tempfile.gettempdir()
    self.use_lossless = use_lossless
    self.cleanup_temp_files = cleanup_temp_files
    self.verbose = verbose

    # Set compression parameters based on video compression best practices
    if compression_params is None:
      if use_lossless:
        # FFV1 with optimal settings for multidimensional data
        self.compression_params = {
          'c:v': 'ffv1',
          'level': '3',  # FFV1 version 3 for better compression
          'slices': '4',  # Parallel processing
          'slicecrc': '1',  # Error detection
          'context': '1'   # Better compression for structured data
        }
      else:
        # Optimized lossy compression preserving classification-relevant features
        # Use libx264 with high444 profile for 4:4:4 chroma subsampling support
        self.compression_params = {
          'c:v': 'libx264',
          'preset': 'veryslow',  # Better compression efficiency
          'crf': '18',  # Very high quality for classification accuracy
          'tune': 'psnr',  # Optimize for signal fidelity
          'profile:v': 'high444',  # Support for 4:4:4 chroma subsampling
          'pix_fmt': 'yuv444p'  # 4:4:4 pixel format for floating-point data
        }
    else:
      self.compression_params = compression_params

    # Initialize state
    self.training_data_ = []
    self.training_labels_ = []
    self.is_fitted_ = False

  def _validate_dataset(self, dataset: xr.Dataset) -> None:
    """Validate that a dataset is suitable for compression."""
    if not isinstance(dataset, xr.Dataset):
      raise TypeError("Input must be an xarray Dataset")

    if len(dataset.data_vars) == 0:
      raise ValueError("Dataset must contain at least one data variable")

    # Check for required dimensions
    for var_name, var in dataset.data_vars.items():
      if len(var.dims) < 2:
        logger.warning(f"Variable '{var_name}' has only {len(var.dims)} "
                       f"dimensions, may not compress well")

  def _get_compression_size(self, dataset: xr.Dataset) -> int:
    """
    Get the compressed size of an xarray Dataset using video compression.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to compress

    Returns
    -------
    int
        Compressed size in bytes
    """
    self._validate_dataset(dataset)

    # Create conversion rules for all data variables
    conversion_rules = create_conversion_rules(dataset, self.compression_params)

    # Generate unique array_id
    array_id = f"temp_{abs(hash(str(dataset)))}"
    output_path = self.temp_dir

    # Compress using xarrayvideo
    result = xarray2video(
      dataset,
      array_id,
      conversion_rules,
      output_path=output_path,
      compute_stats=False,  # Disable metrics computation to avoid torchmetrics errors
      loglevel='quiet' if not self.verbose else 'debug',
      verbose=False,  # This turns off plotting!
    )

    # Extract compression size
    if isinstance(result, dict) and 'compression_stats' in result:
      size = result['compression_stats'].get('total_compressed_size', 0)
      logger.debug(f"Compressed dataset to {size} bytes")
      return size
    else:
      # Fallback: estimate size from files
      import glob
      files = glob.glob(os.path.join(output_path, f"*{array_id}*"))
      total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))

      # Clean up if requested
      if self.cleanup_temp_files:
        for f in files:
          try:
            os.unlink(f)
          except OSError:
            pass

      logger.debug(f"Estimated compressed size: {total_size} bytes")
      return total_size

  def _concatenate_datasets(self, x1: xr.Dataset, x2: xr.Dataset) -> xr.Dataset:
    """
    Concatenate two xarray Datasets for joint compression analysis.
    
    Uses video compression principles to optimize concatenation strategy:
    - Temporal concatenation preserves motion patterns
    - Spatial interleaving can reveal structural similarities
    - Choose strategy based on data characteristics

    Parameters
    ----------
    x1, x2 : xr.Dataset
        Datasets to concatenate

    Returns
    -------
    xr.Dataset
        Combined dataset optimized for compression analysis
    """
    # Strategy 1: Temporal concatenation for time-series data
    if 'time' in x1.dims and 'time' in x2.dims:
      # Extend temporal sequence for better motion compression
      return xr.concat([x1, x2], dim='time')
    
    # Strategy 2: Spatial interleaving for spatial data
    elif self._has_spatial_dims(x1) and self._has_spatial_dims(x2):
      # Create synthetic time dimension for spatial interleaving
      # This helps video codecs detect spatial patterns between datasets
      x1_expanded = x1.expand_dims('time_synthetic').assign_coords(time_synthetic=[0])
      x2_expanded = x2.expand_dims('time_synthetic').assign_coords(time_synthetic=[1])
      return xr.concat([x1_expanded, x2_expanded], dim='time_synthetic')
    
    # Strategy 3: Fallback to simple concatenation
    else:
      return xr.concat([x1, x2], dim='concat')
      
  def _has_spatial_dims(self, dataset: xr.Dataset) -> bool:
    """
    Check if dataset has spatial-like dimensions that benefit from interleaving.
    """
    spatial_indicators = ['x', 'y', 'lat', 'lon', 'latitude', 'longitude']
    return any(dim in dataset.dims for dim in spatial_indicators)

  def _normalized_compression_distance(self, x1: xr.Dataset, x2: xr.Dataset) -> float:
    """
    Calculate Normalized Compression Distance (NCD) between two datasets.

    NCD(x1,x2) = (C(x1x2) - min(C(x1), C(x2))) / max(C(x1), C(x2))

    Parameters
    ----------
    x1, x2 : xr.Dataset
        Datasets to compare

    Returns
    -------
    float
        Normalized compression distance (0 = identical, higher = more different)
    """
    # Get individual compression sizes
    cx1 = self._get_compression_size(x1)
    cx2 = self._get_compression_size(x2)

    # Get joint compression size
    x1x2 = self._concatenate_datasets(x1, x2)
    cx1x2 = self._get_compression_size(x1x2)

    # Calculate NCD with safeguards
    max_size = max(cx1, cx2)
    if max_size == 0:
      logger.warning("Both datasets compressed to 0 bytes")
      return 0.0

    ncd = (cx1x2 - min(cx1, cx2)) / max_size
    ncd = max(0.0, ncd)  # Ensure non-negative

    logger.debug(f"NCD calculation: C(x1)={cx1}, C(x2)={cx2}, "
                 f"C(x1x2)={cx1x2}, NCD={ncd:.4f}")

    return ncd

  def fit(self, X: list[xr.Dataset], y: list[Any]) -> 'XArrayVideoKNNClassifier':
    """
    Fit the classifier with training data.

    Parameters
    ----------
    X : list[xr.Dataset]
        Training datasets
    y : list[Any]
        Training labels

    Returns
    -------
    self : XArrayVideoKNNClassifier
        Returns self for method chaining
    """
    if len(X) != len(y):
      raise ValueError("X and y must have the same length")

    if len(X) == 0:
      raise ValueError("Training data cannot be empty")

    if self.k > len(X):
      raise ValueError(f"k ({self.k}) cannot be larger than training set size ({len(X)})")

    # Validate all datasets
    for i, dataset in enumerate(X):
      try:
        self._validate_dataset(dataset)
      except Exception as e:
        raise ValueError(f"Invalid dataset at index {i}: {e}")

    self.training_data_ = X.copy()
    self.training_labels_ = y.copy()
    self.is_fitted_ = True

    logger.debug(f"Fitted classifier with {len(X)} training samples, "
                 f"{len(set(y))} unique classes")

    return self

  def predict_single(self, x: xr.Dataset) -> Any:
    """
    Predict the class of a single dataset.

    Parameters
    ----------
    x : xr.Dataset
        Dataset to classify

    Returns
    -------
    Any
        Predicted class label
    """
    if not self.is_fitted_:
      raise ValueError("Classifier must be fitted before prediction")

    self._validate_dataset(x)

    distances = []

    # Calculate NCD with each training sample
    for i, train_sample in enumerate(self.training_data_):
      try:
        ncd = self._normalized_compression_distance(x, train_sample)
        distances.append(ncd)
        logger.debug(f"Distance to training sample {i}: {ncd:.4f}")
      except Exception as e:
        logger.error(f"Failed to calculate distance to training sample {i}: {e}")
        distances.append(float('inf'))  # Assign infinite distance on error

    # Get k nearest neighbors
    distances_array = np.array(distances)
    k_nearest_indices = np.argsort(distances_array)[:self.k]

    # Get labels of k nearest neighbors
    k_nearest_labels = [self.training_labels_[i] for i in k_nearest_indices]
    k_nearest_distances = [distances_array[i] for i in k_nearest_indices]

    logger.debug(f"K nearest neighbors: indices={k_nearest_indices.tolist()}, "
                 f"labels={k_nearest_labels}, distances={k_nearest_distances}")

    # Return prediction
    if self.k == 1:
      return k_nearest_labels[0]
    else:
      # Vote among k neighbors
      label_counts = Counter(k_nearest_labels)
      predicted_label = label_counts.most_common(1)[0][0]
      logger.debug(f"Voting result: {dict(label_counts)}")
      return predicted_label

  def predict(self, X: list[xr.Dataset]) -> list[Any]:
    """
    Predict classes for multiple datasets.

    Parameters
    ----------
    X : list[xr.Dataset]
        Datasets to classify

    Returns
    -------
    list[Any]
        Predicted class labels
    """
    if not self.is_fitted_:
      raise ValueError("Classifier must be fitted before prediction")

    if len(X) == 0:
      return []

    predictions = []
    for i, dataset in enumerate(X):
      try:
        pred = self.predict_single(dataset)
        predictions.append(pred)
        logger.debug(f"Predicted '{pred}' for test sample {i}")
      except Exception as e:
        logger.error(f"Failed to predict for sample {i}: {e}")
        # Could raise or append None/default - depends on desired behavior
        raise

    return predictions

  def get_params(self) -> dict:
    """Get classifier parameters."""
    return {
      'k': self.k,
      'compression_params': self.compression_params,
      'temp_dir': self.temp_dir,
      'use_lossless': self.use_lossless,
      'cleanup_temp_files': self.cleanup_temp_files
    }

  def set_params(self, **params) -> 'XArrayVideoKNNClassifier':
    """Set classifier parameters."""
    for key, value in params.items():
      if hasattr(self, key):
        setattr(self, key, value)
      else:
        raise ValueError(f"Invalid parameter: {key}")
    return self