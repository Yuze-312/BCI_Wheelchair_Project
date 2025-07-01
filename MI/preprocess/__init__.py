"""
Preprocessing Module for EEG/MI Data

This module contains all preprocessing functions for:
- Loading EEG data from CSV files
- Temporal filtering (bandpass, notch)
- Spatial filtering (CAR, Laplacian)
- Artifact detection and rejection
- Baseline correction
- Epoch extraction
- Feature extraction (CSP)
- Complete processing pipeline
"""

from .preprocess import MIPreprocessor
from .epoch_extraction import EpochExtractor
from .load_data import MIDataLoader
from .load_and_preprocessing import MIProcessingPipeline

__all__ = ['MIPreprocessor', 'EpochExtractor', 'MIDataLoader', 'MIProcessingPipeline']