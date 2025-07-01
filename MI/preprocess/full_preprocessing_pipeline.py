"""
Full Preprocessing Pipeline for MI/ErrP Data

This script provides a complete preprocessing pipeline from raw data to classifier-ready features.
"""

import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .preprocess import MIPreprocessor
from .epoch_extraction import EpochExtractor
from ..data_processing.load_data import MIDataLoader
from mne.decoding import CSP


class FullPreprocessingPipeline:
    """Complete preprocessing pipeline for MI/ErrP data"""
    
    def __init__(self, sampling_rate=512):
        self.sampling_rate = sampling_rate
        self.preprocessor = MIPreprocessor(sampling_rate)
        self.epoch_extractor = EpochExtractor(sampling_rate)
        
    def process_raw_to_features(self, raw_data, labels, preprocess_params=None, csp_params=None):
        """
        Complete pipeline from raw data to features ready for classification
        
        Args:
            raw_data: Raw EEG data (n_samples, n_channels)
            labels: Class labels
            preprocess_params: Dict of preprocessing parameters
            csp_params: Dict of CSP parameters
            
        Returns:
            features: Extracted features
            csp: Fitted CSP object
            preprocessing_info: Dict with preprocessing details
        """
        if preprocess_params is None:
            preprocess_params = {
                'bandpass_low': 8.0,
                'bandpass_high': 30.0,
                'notch_freq': 50.0,
                'spatial_filter': 'car',
                'artifact_threshold': 100.0
            }
            
        if csp_params is None:
            csp_params = {
                'n_components': 4,
                'reg': 0.01,
                'log': True,
                'norm_trace': False
            }
        
        preprocessing_info = {
            'original_shape': raw_data.shape,
            'sampling_rate': self.sampling_rate,
            'preprocessing_params': preprocess_params,
            'csp_params': csp_params
        }
        
        # Step 1: Temporal Filtering
        print("Step 1: Applying temporal filters...")
        
        # Notch filter
        data = self.preprocessor.notch_filter(
            raw_data, 
            freq=preprocess_params['notch_freq']
        )
        
        # Bandpass filter
        data = self.preprocessor.bandpass_filter(
            data,
            low_freq=preprocess_params['bandpass_low'],
            high_freq=preprocess_params['bandpass_high']
        )
        
        # Step 2: Spatial Filtering
        print("Step 2: Applying spatial filter...")
        if preprocess_params['spatial_filter'] == 'car':
            data = self.preprocessor.apply_car(data)
        elif preprocess_params['spatial_filter'] == 'laplacian':
            data = self.preprocessor.apply_laplacian(data)
        
        # Step 3: Artifact Detection
        print("Step 3: Detecting artifacts...")
        artifact_mask = self.preprocessor.detect_artifacts(
            data, 
            threshold=preprocess_params['artifact_threshold']
        )
        preprocessing_info['artifact_percentage'] = 100 * (1 - np.mean(artifact_mask))
        print(f"  Artifact percentage: {preprocessing_info['artifact_percentage']:.1f}%")
        
        # Step 4: Baseline Correction
        print("Step 4: Applying baseline correction...")
        baseline_samples = int(1.0 * self.sampling_rate)  # 1 second baseline
        data = self.preprocessor.baseline_correction(data, baseline_samples)
        
        # Step 5: Epoch Extraction (if continuous data)
        # This step would depend on your specific trial structure
        
        # Step 6: CSP Feature Extraction
        print("Step 5: Extracting CSP features...")
        
        # Reshape data for CSP (n_epochs, n_channels, n_times)
        # Assuming data is already epoched
        if len(data.shape) == 2:
            # If continuous, create pseudo-epochs
            epoch_length = 2 * self.sampling_rate  # 2 second epochs
            n_epochs = data.shape[0] // epoch_length
            data = data[:n_epochs * epoch_length].reshape(n_epochs, epoch_length, -1)
            data = np.transpose(data, (0, 2, 1))
        
        # Fit CSP
        csp = CSP(**csp_params)
        csp.fit(data, labels[:len(data)])
        
        # Transform to features
        features = csp.transform(data)
        preprocessing_info['feature_shape'] = features.shape
        
        print(f"\nPreprocessing complete!")
        print(f"  Input shape: {preprocessing_info['original_shape']}")
        print(f"  Output shape: {features.shape}")
        
        return features, csp, preprocessing_info
    
    def save_preprocessing_config(self, config, filepath):
        """Save preprocessing configuration for reproducibility"""
        import json
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
            
    def load_preprocessing_config(self, filepath):
        """Load preprocessing configuration"""
        import json
        with open(filepath, 'r') as f:
            return json.load(f)


def main():
    """Example usage"""
    print("Full Preprocessing Pipeline Example")
    print("=" * 50)
    
    # Create pipeline
    pipeline = FullPreprocessingPipeline(sampling_rate=512)
    
    # Example: Create dummy data
    n_samples = 5120  # 10 seconds at 512 Hz
    n_channels = 16
    n_trials = 50
    
    # Generate dummy data
    np.random.seed(42)
    raw_data = np.random.randn(n_samples * n_trials, n_channels) * 30  # μV scale
    labels = np.repeat([0, 1], n_trials // 2)  # Binary labels
    
    # Process data
    features, csp, info = pipeline.process_raw_to_features(raw_data, labels)
    
    print(f"\nProcessing complete!")
    print(f"Features ready for classification: {features.shape}")
    
    # Save configuration
    pipeline.save_preprocessing_config(info, 'preprocessing_config.json')
    print(f"Configuration saved to preprocessing_config.json")


if __name__ == "__main__":
    main()