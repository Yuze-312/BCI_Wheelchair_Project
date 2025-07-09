"""
ErrP (Error-Related Potential) Processing Module

This module provides tools for processing EEG data to extract and analyze
error-related potentials from BCI experiments.

Main components:
- ErrPDataLoader: Load EEG data and event logs
- ErrPPreprocessor: Filter and preprocess EEG signals
- ErrPEpochExtractor: Extract epochs around error events
- ErrPAnalyzer: Detect and analyze error potentials
- ErrPPipeline: Complete processing pipeline
"""

from .data_loader import ErrPDataLoader
from .preprocessor import ErrPPreprocessor
from .epoch_extractor import ErrPEpochExtractor
from .analyzer import ErrPAnalyzer
from .pipeline import ErrPPipeline

__all__ = [
    'ErrPDataLoader',
    'ErrPPreprocessor',
    'ErrPEpochExtractor',
    'ErrPAnalyzer',
    'ErrPPipeline'
]

__version__ = '0.1.0'