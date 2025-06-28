"""
Tests for XArray Video KNN Classifier.
"""

import pytest
import numpy as np
import xarray as xr
import tempfile
import os
from unittest.mock import patch, MagicMock

from xarray_video_knn import XArrayVideoKNNClassifier
# Assuming the library is installed or in path
from xarray_video_knn.utils import (
  create_conversion_rules,
  estimate_compression_size,
  validate_datasets_compatible,
  get_dataset_info,
  memory_usage_mb
)


class TestXArrayVideoKNNClassifier:
  """Test cases for the main classifier."""

  @pytest.fixture
  def sample_datasets(self):
    """Create sample datasets for testing."""
    np.random.seed(42)

    # Create synthetic datasets
    time = np.arange(10)
    y = np.arange(5)
    x = np.arange(5)

    datasets = []
    labels = []

    # Class A datasets
    for i in range(3):
      data = np.random.randn(len(time), len(y), len(x)) + i
      mask = (data > 0).astype(np.uint8)

      dataset = xr.Dataset({
        'temperature': (['time', 'y', 'x'], data),
        'mask': (['time', 'y', 'x'], mask)
      }, coords={'time': time, 'y': y, 'x': x})

      datasets.append(dataset)
      labels.append('class_a')

    # Class B datasets
    for i in range(3):
      data = -np.random.randn(len(time), len(y), len(x)) - i
      mask = (data < 0).astype(np.uint8)

      dataset = xr.Dataset({
        'temperature': (['time', 'y', 'x'], data),
        'mask': (['time', 'y', 'x'], mask)
      }, coords={'time': time, 'y': y, 'x': x})

      datasets.append(dataset)
      labels.append('class_b')

    return datasets, labels

  def test_init(self):
    """Test classifier initialization."""
    classifier = XArrayVideoKNNClassifier(k=3)
    assert classifier.k == 3
    assert not classifier.is_fitted_
    assert classifier.use_lossless == True

    # Test with custom parameters
    params = {'c:v': 'libx264'}
    classifier = XArrayVideoKNNClassifier(
      k=1,
      compression_params=params,
      use_lossless=False
    )
    assert classifier.compression_params == params
    assert classifier.use_lossless == False

  def test_fit(self, sample_datasets):
    """Test fitting the classifier."""
    datasets, labels = sample_datasets
    classifier = XArrayVideoKNNClassifier(k=1)

    # Test successful fit
    result = classifier.fit(datasets, labels)
    assert result is classifier  # Should return self
    assert classifier.is_fitted_
    assert len(classifier.training_data_) == len(datasets)
    assert len(classifier.training_labels_) == len(labels)

  def test_fit_validation(self, sample_datasets):
    """Test input validation in fit method."""
    datasets, labels = sample_datasets
    classifier = XArrayVideoKNNClassifier(k=1)

    # Test mismatched lengths
    with pytest.raises(ValueError, match="X and y must have the same length"):
      classifier.fit(datasets, labels[:-1])

    # Test empty data
    with pytest.raises(ValueError, match="Training data cannot be empty"):
      classifier.fit([], [])

    # Test k too large
    classifier_big_k = XArrayVideoKNNClassifier(k=10)
    with pytest.raises(ValueError, match="k.*cannot be larger than training set size"):
      classifier_big_k.fit(datasets, labels)

    # Test invalid dataset
    with pytest.raises(ValueError, match="Invalid dataset"):
      classifier.fit([None], ['label'])

  def test_predict_not_fitted(self, sample_datasets):
    """Test prediction before fitting."""
    datasets, _ = sample_datasets
    classifier = XArrayVideoKNNClassifier(k=1)

    with pytest.raises(ValueError, match="Classifier must be fitted"):
      classifier.predict_single(datasets[0])

    with pytest.raises(ValueError, match="Classifier must be fitted"):
      classifier.predict(datasets[:1])

  @patch('xarray_video_knn.utils.estimate_compression_size')
  def test_predict_single(self, mock_compression, sample_datasets):
    """Test single prediction."""
    datasets, labels = sample_datasets

    # Mock compression sizes to make prediction deterministic
    # Make class_a datasets compress smaller with each other
    def mock_size_func(dataset, temp_dir):
      # Check if it's a concatenated dataset or single
      if 'concat' in dataset.dims:
        return 50  # Joint compression
      else:
        # Return different sizes based on data pattern
        temp_data = dataset['temperature'].values
        if np.mean(temp_data) > 0:
          return 100  # class_a
        else:
          return 120  # class_b

    mock_compression.side_effect = mock_size_func

    classifier = XArrayVideoKNNClassifier(k=1)
    classifier.fit(datasets[:4], labels[:4])  # 2 per class

    # Test prediction
    pred = classifier.predict_single(datasets[0])  # Should be class_a
    assert pred in ['class_a', 'class_b']

  def test_predict_multiple(self, sample_datasets):
    """Test multiple predictions."""
    datasets, labels = sample_datasets
    classifier = XArrayVideoKNNClassifier(k=1)
    classifier.fit(datasets[:4], labels[:4])

    predictions = classifier.predict(datasets[4:])
    assert len(predictions) == 2
    assert all(pred in ['class_a', 'class_b'] for pred in predictions)

    # Test empty input
    assert classifier.predict([]) == []

  def test_get_set_params(self):
    """Test parameter getting and setting."""
    classifier = XArrayVideoKNNClassifier(k=1, use_lossless=True)

    params = classifier.get_params()
    assert params['k'] == 1
    assert params['use_lossless'] == True

    # Test setting parameters
    classifier.set_params(k=3, use_lossless=False)
    assert classifier.k == 3
    assert classifier.use_lossless == False

    # Test invalid parameter
    with pytest.raises(ValueError, match="Invalid parameter"):
      classifier.set_params(invalid_param=123)

  def test_validate_dataset(self):
    """Test dataset validation."""
    classifier = XArrayVideoKNNClassifier()

    # Valid dataset
    valid_ds = xr.Dataset({
      'temp': (['time', 'x'], np.random.randn(10, 5))
    }, coords={'time': range(10), 'x': range(5)})

    classifier._validate_dataset(valid_ds)  # Should not raise

    # Invalid type
    with pytest.raises(TypeError, match="Input must be an xarray Dataset"):
      classifier._validate_dataset("not a dataset")

    # Empty dataset
    empty_ds = xr.Dataset()
    with pytest.raises(ValueError, match="Dataset must contain at least one data variable"):
      classifier._validate_dataset(empty_ds)


class TestUtils:
  """Test utility functions."""

  @pytest.fixture
  def sample_dataset(self):
    """Create a sample dataset."""
    time = np.arange(5)
    y = np.arange(3)
    x = np.arange(3)

    return xr.Dataset({
      'temperature': (['time', 'y', 'x'], np.random.randn(5, 3, 3)),
      'humidity': (['time', 'y', 'x'], np.random.randn(5, 3, 3)),
      'mask': (['time', 'y', 'x'], np.random.randint(0, 2, (5, 3, 3), dtype=np.uint8))
    }, coords={'time': time, 'y': y, 'x': x})

  def test_create_conversion_rules(self, sample_dataset):
    """Test conversion rules creation."""
    params = {'c:v': 'ffv1'}
    rules = create_conversion_rules(sample_dataset, params)

    assert len(rules) == 3  # Three variables
    assert 'temperature' in rules
    assert 'humidity' in rules
    assert 'mask' in rules

    # Check rule structure
    temp_rule = rules['temperature']
    assert temp_rule[0] == ('temperature',)  # variables
    assert 'time' in temp_rule[1]  # dimensions
    assert temp_rule[2] == 0  # offset
    assert temp_rule[3] == params  # compression params
    assert temp_rule[4] == 16  # bit depth for float

    # Check uint8 bit depth
    mask_rule = rules['mask']
    assert mask_rule[4] == 8  # bit depth for uint8