#!/usr/bin/env python3
"""
Demo script for XArray Video KNN Classifier.

This script creates mock xarray datasets with different patterns and
demonstrates the KNN classification capabilities using video compression.

Requirements:
- ffmpeg must be installed and available in PATH
  - macOS: brew install ffmpeg
  - Ubuntu: sudo apt install ffmpeg
  - Windows: download from https://ffmpeg.org/
"""

import shutil
import sys

import numpy as np
import xarray as xr
from sklearn.metrics import accuracy_score, classification_report

from xarray_video_knn import XArrayVideoKNNClassifier


def check_ffmpeg():
    """Check if ffmpeg is available in PATH."""
    if shutil.which('ffmpeg') is None:
        print("ERROR: ffmpeg not found in PATH")
        print("Please install ffmpeg:")
        print("  - macOS: brew install ffmpeg")
        print("  - Ubuntu: sudo apt install ffmpeg")
        print("  - Windows: download from https://ffmpeg.org/")
        return False
    return True


def create_mock_dataset(pattern_type: str, size: tuple = (50, 50), time_steps: int = 10) -> xr.Dataset:
    """
    Create mock xarray datasets with different patterns.
    
    Parameters
    ----------
    pattern_type : str
        Type of pattern to generate ('sine', 'checkerboard', 'random', 'linear')
    size : tuple
        Spatial dimensions (height, width)
    time_steps : int
        Number of time steps
        
    Returns
    -------
    xr.Dataset
        Generated dataset with specified pattern
    """
    height, width = size
    
    # Create coordinate arrays
    x = np.linspace(0, 2*np.pi, width)
    y = np.linspace(0, 2*np.pi, height)
    time = np.arange(time_steps)
    
    # Create meshgrids
    X, Y = np.meshgrid(x, y)
    
    if pattern_type == 'sine':
        # Sine wave pattern that evolves over time
        data = np.zeros((time_steps, height, width))
        for t in range(time_steps):
            data[t] = np.sin(X + 0.5*t) * np.cos(Y + 0.3*t)
            
    elif pattern_type == 'checkerboard':
        # Checkerboard pattern with time-varying intensity
        data = np.zeros((time_steps, height, width))
        for t in range(time_steps):
            checker = ((X // (2*np.pi/8)).astype(int) + (Y // (2*np.pi/8)).astype(int)) % 2
            data[t] = checker * (0.5 + 0.5 * np.sin(0.5*t))
            
    elif pattern_type == 'random':
        # Random pattern (should be hardest to classify consistently)
        np.random.seed(42)  # For reproducibility
        data = np.random.randn(time_steps, height, width)
        
    elif pattern_type == 'linear':
        # Linear gradient pattern
        data = np.zeros((time_steps, height, width))
        for t in range(time_steps):
            data[t] = (X + Y) * (1 + 0.1*t)
            
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # Create xarray Dataset
    dataset = xr.Dataset({
        'temperature': (['time', 'y', 'x'], data)
    }, coords={
        'time': time,
        'x': ('x', x),
        'y': ('y', y)
    })
    
    return dataset


def generate_training_data(n_samples_per_class: int = 5) -> tuple[list[xr.Dataset], list[str]]:
    """Generate training datasets for each pattern type."""
    patterns = ['sine', 'checkerboard', 'linear']
    X_train = []
    y_train = []
    
    print(f"Generating {n_samples_per_class} samples per class for training...")
    
    for pattern in patterns:
        for i in range(n_samples_per_class):
            # Add some variation by changing size and time steps
            size = (40 + i*2, 40 + i*2)  # Slight size variation
            time_steps = 8 + i  # Slight temporal variation
            
            dataset = create_mock_dataset(pattern, size=size, time_steps=time_steps)
            X_train.append(dataset)
            y_train.append(pattern)
            
    return X_train, y_train


def generate_test_data(n_samples_per_class: int = 3) -> tuple[list[xr.Dataset], list[str]]:
    """Generate test datasets for each pattern type."""
    patterns = ['sine', 'checkerboard', 'linear']
    X_test = []
    y_test = []
    
    print(f"Generating {n_samples_per_class} samples per class for testing...")
    
    for pattern in patterns:
        for i in range(n_samples_per_class):
            # Use different parameters than training to test generalization
            size = (45 + i, 45 + i)
            time_steps = 12 + i
            
            dataset = create_mock_dataset(pattern, size=size, time_steps=time_steps)
            X_test.append(dataset)
            y_test.append(pattern)
            
    return X_test, y_test


def main():
    """Run the demo."""
    print("XArray Video KNN Classifier Demo")
    print("=" * 40)
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        sys.exit(1)
    
    # Generate training and test data
    X_train, y_train = generate_training_data(n_samples_per_class=3)  # Small dataset for demo
    X_test, y_test = generate_test_data(n_samples_per_class=2)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Classes: {set(y_train)}")
    print()
    
    # Create classifier with optimized defaults
    print("Creating XArrayVideoKNNClassifier...")
    classifier = XArrayVideoKNNClassifier(
        # Using optimized defaults: k=3, use_lossless=True
        cleanup_temp_files=True
    )
    
    # Fit the classifier
    print("Fitting classifier on training data...")
    classifier.fit(X_train, y_train)
    print("✓ Classifier fitted successfully")
    print()
    
    # Make predictions
    print("Making predictions on test data...")
    y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Results:")
    print("-" * 20)
    print(f"Accuracy: {accuracy:.2%}")
    print()
    
    print("Detailed Results:")
    for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
        status = "✓" if true_label == pred_label else "✗"
        print(f"Sample {i+1}: True={true_label}, Predicted={pred_label} {status}")
    
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Print classifier parameters
    print("Classifier Parameters:")
    params = classifier.get_params()
    for key, value in params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()