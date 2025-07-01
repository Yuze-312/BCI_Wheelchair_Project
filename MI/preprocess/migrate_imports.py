"""
Migration script to update imports after reorganization

This script helps update import statements to use the new preprocess folder structure.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("PREPROCESSING MODULE REORGANIZATION")
print("=" * 60)

print("\nThe preprocessing files have been reorganized into MI/preprocess folder:")
print("  - MI/preprocess/load_data.py (data loading from CSV)")
print("  - MI/preprocess/load_and_preprocessing.py (complete pipeline with MIProcessingPipeline)")
print("  - MI/preprocess/preprocess.py (main preprocessing functions)")
print("  - MI/preprocess/epoch_extraction.py (epoch extraction)")
print("  - MI/preprocess/full_preprocessing_pipeline.py (alternative pipeline)")
print("  - MI/preprocess/README.md (documentation)")

print("\n" + "=" * 60)
print("HOW TO UPDATE YOUR IMPORTS")
print("=" * 60)

print("\nOLD import style:")
print("  from MI.data_processing.preprocess import MIPreprocessor")
print("  from MI.data_processing.epoch_extraction import EpochExtractor")
print("  from MI.data_processing.pipeline import MIProcessingPipeline")

print("\nNEW import style (if in project root):")
print("  from MI.preprocess.preprocess import MIPreprocessor")
print("  from MI.preprocess.epoch_extraction import EpochExtractor")

print("\nOR use the convenient __init__.py imports:")
print("  from MI.preprocess import MIPreprocessor, EpochExtractor, MIDataLoader, MIProcessingPipeline")

print("\n" + "=" * 60)
print("MAINTAINING BACKWARD COMPATIBILITY")
print("=" * 60)

print("\nThe MI/data_processing/ folder has been REMOVED.")
print("All preprocessing functionality is now in MI/preprocess/.")

print("\nIMPORTANT: Update your imports to use the new location!")

print("\n" + "=" * 60)
print("EXAMPLE USAGE")
print("=" * 60)

print("\n# Basic preprocessing")
print("from MI.preprocess import MIPreprocessor")
print("preprocessor = MIPreprocessor(sampling_rate=512)")
print("filtered_data = preprocessor.bandpass_filter(raw_data)")

print("\n# Data loading and processing pipeline")
print("from MI.preprocess import MIProcessingPipeline")
print("pipeline = MIProcessingPipeline(base_path='MI/EEG data')")
print("processed_data = pipeline.process_participant('T-005')")

print("\n# Alternative full pipeline")
print("from MI.preprocess.full_preprocessing_pipeline import FullPreprocessingPipeline")
print("pipeline = FullPreprocessingPipeline()")
print("features, csp, info = pipeline.process_raw_to_features(raw_data, labels)")

print("\n" + "=" * 60)