"""
Utility functions for XArray Video KNN.
"""

import os
import tempfile
import numpy as np
import xarray as xr
from typing import Any, Hashable
import logging

logger = logging.getLogger(__name__)


def create_conversion_rules(
    dataset: xr.Dataset,
    compression_params: dict
) -> dict[Hashable, tuple[tuple[Hashable, ...], tuple[str, ...], int, dict, int]]:
  """
  Create conversion rules for xarrayvideo based on dataset structure.

  Parameters
  ----------
  dataset : xr.Dataset
      Dataset to create rules for
  compression_params : dict
      Compression parameters to use

  Returns
  -------
  dict
      Conversion rules in xarrayvideo format
  """
  conversion_rules = {}

  for var_name in dataset.data_vars:
    var = dataset[var_name]
    dims = list(var.dims)

    # Ensure temporal dimension comes first for video encoding
    if 'time' in dims:
      dims = ['time'] + [d for d in dims if d != 'time']

    # Determine bit depth based on data type
    if var.dtype == np.uint8:
      bit_depth = 8
    elif var.dtype == np.uint16:
      bit_depth = 16
    elif var.dtype in [np.float32, np.float64]:
      bit_depth = 16  # Convert floats to 16-bit
    else:
      bit_depth = 16  # Default

    conversion_rules[var_name] = (
      (var_name,),  # variables to include
      tuple(dims),  # dimension order
      0,  # offset
      compression_params,  # compression parameters
      bit_depth  # bit depth
    )

    logger.debug(f"Created conversion rule for '{var_name}': dims={dims}, "
                 f"bit_depth={bit_depth}")

  return conversion_rules


def estimate_compression_size(dataset: xr.Dataset, temp_dir: str) -> int:
  """
  Estimate compression size using numpy's compressed format as fallback.

  Parameters
  ----------
  dataset : xr.Dataset
      Dataset to estimate compression for
  temp_dir : str
      Temporary directory for files

  Returns
  -------
  int
      Estimated compressed size in bytes
  """
  try:
    # Create temporary file
    with tempfile.NamedTemporaryFile(
        suffix='.npz',
        dir=temp_dir,
        delete=False
    ) as temp_file:
      temp_path = temp_file.name

    # Save as compressed numpy arrays
    data_dict = {}
    for var_name, var in dataset.data_vars.items():
      data_dict[var_name] = var.values

    np.savez_compressed(temp_path, **data_dict)

    # Get file size
    size = os.path.getsize(temp_path)

    # Clean up
    os.unlink(temp_path)

    logger.debug(f"Estimated compression size: {size} bytes")
    return size

  except Exception as e:
    logger.error(f"Failed to estimate compression size: {e}")
    # Return rough estimate based on data size
    total_elements = sum(var.size for var in dataset.data_vars.values())
    estimated_size = total_elements * 2  # Rough estimate
    logger.debug(f"Using rough size estimate: {estimated_size} bytes")
    return estimated_size


def validate_datasets_compatible(datasets: list) -> bool:
  """
  Check if datasets have compatible structure for comparison.

  Parameters
  ----------
  datasets : list
      List of xarray Datasets to check

  Returns
  -------
  bool
      True if datasets are compatible
  """
  if len(datasets) < 2:
    return True

  # Check that all datasets have the same variables
  first_vars = set(datasets[0].data_vars.keys())

  for i, dataset in enumerate(datasets[1:], 1):
    current_vars = set(dataset.data_vars.keys())
    if current_vars != first_vars:
      logger.warning(f"Dataset {i} has different variables: "
                     f"{current_vars} vs {first_vars}")
      return False

  # Check dimensions are compatible
  for var_name in first_vars:
    first_dims = datasets[0][var_name].dims

    for i, dataset in enumerate(datasets[1:], 1):
      current_dims = dataset[var_name].dims
      if current_dims != first_dims:
        logger.warning(f"Variable '{var_name}' in dataset {i} has "
                       f"different dimensions: {current_dims} vs {first_dims}")
        return False

  return True


def get_dataset_info(dataset: xr.Dataset) -> dict[str, Any]:
  """
  Get summary information about a dataset.

  Parameters
  ----------
  dataset : xr.Dataset
      Dataset to analyze

  Returns
  -------
  dict
      Dictionary with dataset information
  """
  info = {
    'variables': list(dataset.data_vars.keys()),
    'coordinates': list(dataset.coords.keys()),
    'dimensions': dict(dataset.dims),
    'total_size': dataset.nbytes,
  }

  # Add per-variable info
  var_info = {}
  for var_name, var in dataset.data_vars.items():
    var_info[var_name] = {
      'shape': var.shape,
      'dtype': str(var.dtype),
      'dimensions': list(var.dims),
      'size_bytes': var.nbytes
    }

  info['variable_details'] = var_info

  return info


def memory_usage_mb(dataset: xr.Dataset) -> float:
  """
  Calculate memory usage of dataset in MB.

  Parameters
  ----------
  dataset : xr.Dataset
      Dataset to analyze

  Returns
  -------
  float
      Memory usage in megabytes
  """
  return dataset.nbytes / (1024 * 1024)