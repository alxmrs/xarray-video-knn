"""
Main classifier module for XArray Video KNN.
"""

import os
import tempfile
import numpy as np
import xarray as xr
from typing import List, Any, Dict, Optional, Union
from collections import Counter
import logging

from .utils import create_conversion_rules, estimate_compression_size

# Set up logging
logger = logging.getLogger(__name__)

# Check for xarrayvideo availability
try:
  from xarrayvideo import xarray2video, video2xarray

  XARRAYVIDEO_AVAILABLE = True
  logger.info("xarrayvideo library loaded successfully")
except ImportError:
  XARRAYVIDEO_AVAILABLE = False
  logger.warning("xarrayvideo not available, using fallback compression")


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
  training_data_ : List[xr.Dataset]
      Training datasets after fitting.
  training_labels_ : List[Any]
      Training labels after fitting.
  is_fitted_ : bool
      Whether the classifier has been fitted.
  """

  def __init__(
      self,
      k: int = 1,
      compression_params: Optional[Dict] = None,
      temp_dir: Optional[str] = None,
      use_lossless: bool = True,
      cleanup_temp_files: bool = True
  ):
    self.k = k
    self.temp_dir = temp_dir or tempfile.gettempdir()
    self.use_lossless = use_lossless
    self.cleanup_temp_files = cleanup_temp_files

    # Set compression parameters
    if compression_params is None:
      self.compression_params = {'c:v': 'ffv1'} if use_lossless else {
        'c:v': 'libx265',
        'preset': 'medium',
        'crf': 51,
        'x265-params': 'qpmin=0:qpmax=0.01',
        'tune': 'psnr',
      }
    else:
      self.compression_params = compression_params

    # Initialize state
    self.training_data_ = []
    self.training_labels_ = []
    self.is_fitted_ = False

    logger.info(f"Initialized XArrayVideoKNNClassifier with k={k}, "
                f"lossless={use_lossless}")

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

    if not XARRAYVIDEO_AVAILABLE:
      logger.debug("Using fallback compression")
      return estimate_compression_size(dataset, self.temp_dir)

    # Create conversion rules for all data variables
    conversion_rules = create_conversion_rules(dataset, self.compression_params)

    # Generate unique array_id
    array_id = f"temp_{abs(hash(str(dataset)))}"
    output_path = self.temp_dir

    try:
      # Compress using xarrayvideo
      result = xarray2video(
        dataset,
        array_id,
        conversion_rules,
        output_path=output_path,
        compute_stats=True,
        loglevel='error'  # Reduce verbosity
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

    except Exception as e:
      logger.warning(f"Video compression failed ({e}), using fallback")
      return estimate_compression_size(dataset, self.temp_dir)

  def _concatenate_datasets(self, x1: xr.Dataset, x2: xr.Dataset) -> xr.Dataset:
    """
    Concatenate two xarray Datasets for joint compression analysis.

    Parameters
    ----------
    x1, x2 : xr.Dataset
        Datasets to concatenate

    Returns
    -------
    xr.Dataset
        Combined dataset
    """
    try:
      # Combine datasets by concatenating along a new 'concat' dimension
      combined = xr.concat([x1, x2], dim='concat')
      return combined
    except Exception as e:
      logger.error(f"Failed to concatenate datasets: {e}")
      raise

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

  def fit(self, X: List[xr.Dataset], y: List[Any]) -> 'XArrayVideoKNNClassifier':
    """
    Fit the classifier with training data.

    Parameters
    ----------
    X : List[xr.Dataset]
        Training datasets
    y : List[Any]
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

    logger.info(f"Fitted classifier with {len(X)} training samples, "
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

  def predict(self, X: List[xr.Dataset]) -> List[Any]:
    """
    Predict classes for multiple datasets.

    Parameters
    ----------
    X : List[xr.Dataset]
        Datasets to classify

    Returns
    -------
    List[Any]
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

  def get_params(self) -> Dict:
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