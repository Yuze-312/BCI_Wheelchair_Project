"""
ErrP Preprocessor Module

Handles preprocessing of EEG data for ErrP detection including:
- Bandpass filtering for ErrP-relevant frequencies
- Notch filtering for powerline noise
- Spatial filtering (CAR, Laplacian)
- Artifact detection and rejection
- Baseline correction
"""

import numpy as np
import pandas as pd
from scipy import signal
from typing import Dict, List, Optional, Tuple, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPPreprocessor:
    """Preprocess EEG data for ErrP analysis"""
    
    def __init__(self, sampling_rate: int = 512):
        """
        Initialize the ErrP preprocessor
        
        Args:
            sampling_rate: EEG sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        
        # ErrP-specific frequency bands
        # ErrPs typically appear in 1-10 Hz range
        self.errp_band = (1.0, 10.0)
        
        # Default preprocessing parameters
        self.params = {
            'bandpass_low': 1.0,
            'bandpass_high': 10.0,
            'notch_freq': 60.0,  # 60 Hz for US, 50 Hz for EU
            'notch_quality': 30,
            'artifact_threshold': 100.0,  # μV
            'baseline_window': (-0.2, 0.0)  # 200ms before stimulus
        }
        
    def preprocess(self, eeg_data: pd.DataFrame, 
                  channel_names: Optional[List[str]] = None,
                  apply_spatial_filter: str = 'car',
                  remove_powerline: bool = True) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline to EEG data
        
        Args:
            eeg_data: DataFrame with EEG channels and timestamp
            channel_names: List of channel column names
            apply_spatial_filter: Spatial filter type ('car', 'laplacian', or None)
            remove_powerline: Whether to apply notch filter
            
        Returns:
            Preprocessed EEG data
        """
        if channel_names is None:
            # Auto-detect channel columns
            channel_names = [col for col in eeg_data.columns 
                           if col.startswith('Channel') or col.startswith('Ch')]
        
        logger.info(f"Preprocessing {len(channel_names)} channels")
        
        # Create a copy to avoid modifying original
        processed_data = eeg_data.copy()
        
        # Extract channel data
        channel_data = processed_data[channel_names].values
        
        # 1. Remove DC offset
        channel_data = self._remove_dc_offset(channel_data)
        
        # 2. Apply spatial filter
        if apply_spatial_filter:
            logger.info(f"Applying {apply_spatial_filter} spatial filter")
            if apply_spatial_filter == 'car':
                channel_data = self._apply_car(channel_data)
            elif apply_spatial_filter == 'laplacian':
                channel_data = self._apply_laplacian(channel_data)
        
        # 3. Apply bandpass filter for ErrP frequencies
        logger.info(f"Applying bandpass filter: {self.params['bandpass_low']}-{self.params['bandpass_high']} Hz")
        channel_data = self._bandpass_filter(
            channel_data,
            self.params['bandpass_low'],
            self.params['bandpass_high']
        )
        
        # 4. Remove powerline noise
        if remove_powerline:
            logger.info(f"Removing {self.params['notch_freq']} Hz powerline noise")
            channel_data = self._notch_filter(
                channel_data,
                self.params['notch_freq'],
                self.params['notch_quality']
            )
        
        # 5. Detect and mark artifacts
        artifact_mask = self._detect_artifacts(channel_data)
        if artifact_mask.any():
            logger.warning(f"Detected artifacts in {artifact_mask.sum()} samples")
        
        # Update the processed data
        processed_data[channel_names] = channel_data
        processed_data['artifact'] = artifact_mask
        
        return processed_data
    
    def _remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """Remove DC offset from each channel"""
        return data - np.mean(data, axis=0)
    
    def _apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR) spatial filter
        
        Args:
            data: EEG data array (samples x channels)
            
        Returns:
            CAR-filtered data
        """
        # Calculate average across all channels for each time point
        car_reference = np.mean(data, axis=1, keepdims=True)
        
        # Subtract average from each channel
        return data - car_reference
    
    def _apply_laplacian(self, data: np.ndarray, 
                        channel_neighbors: Optional[Dict[int, List[int]]] = None) -> np.ndarray:
        """
        Apply Laplacian spatial filter
        
        Args:
            data: EEG data array (samples x channels)
            channel_neighbors: Dictionary mapping channel index to neighbor indices
            
        Returns:
            Laplacian-filtered data
        """
        n_samples, n_channels = data.shape
        filtered_data = np.zeros_like(data)
        
        # If no neighbor map provided, use simple adjacent channels
        if channel_neighbors is None:
            # Simple 1D Laplacian (assuming linear electrode arrangement)
            for ch in range(n_channels):
                neighbors = []
                if ch > 0:
                    neighbors.append(ch - 1)
                if ch < n_channels - 1:
                    neighbors.append(ch + 1)
                
                if neighbors:
                    filtered_data[:, ch] = data[:, ch] - np.mean(data[:, neighbors], axis=1)
                else:
                    filtered_data[:, ch] = data[:, ch]
        else:
            # Use provided neighbor map
            for ch, neighbors in channel_neighbors.items():
                if neighbors:
                    filtered_data[:, ch] = data[:, ch] - np.mean(data[:, neighbors], axis=1)
                else:
                    filtered_data[:, ch] = data[:, ch]
        
        return filtered_data
    
    def _bandpass_filter(self, data: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """
        Apply bandpass filter to EEG data
        
        Args:
            data: EEG data array (samples x channels)
            low_freq: Lower cutoff frequency
            high_freq: Upper cutoff frequency
            
        Returns:
            Bandpass filtered data
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        sos = signal.butter(4, [low, high], btype='band', output='sos')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = signal.sosfiltfilt(sos, data[:, ch])
        
        return filtered_data
    
    def _notch_filter(self, data: np.ndarray, freq: float, quality: float) -> np.ndarray:
        """
        Apply notch filter to remove powerline noise
        
        Args:
            data: EEG data array (samples x channels)
            freq: Frequency to notch out (Hz)
            quality: Quality factor of the notch filter
            
        Returns:
            Notch filtered data
        """
        nyquist = self.sampling_rate / 2
        w0 = freq / nyquist
        
        # Design notch filter
        b, a = signal.iirnotch(w0, quality)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = signal.filtfilt(b, a, data[:, ch])
        
        return filtered_data
    
    def _detect_artifacts(self, data: np.ndarray, 
                         threshold: Optional[float] = None) -> np.ndarray:
        """
        Detect artifacts in EEG data using amplitude threshold
        
        Args:
            data: EEG data array (samples x channels)
            threshold: Amplitude threshold in μV
            
        Returns:
            Boolean mask indicating artifact samples
        """
        if threshold is None:
            threshold = self.params['artifact_threshold']
        
        # Check if any channel exceeds threshold
        artifact_mask = np.any(np.abs(data) > threshold, axis=1)
        
        # Extend artifact periods by 100ms on each side
        extension_samples = int(0.1 * self.sampling_rate)
        extended_mask = np.zeros_like(artifact_mask)
        
        artifact_indices = np.where(artifact_mask)[0]
        for idx in artifact_indices:
            start = max(0, idx - extension_samples)
            end = min(len(artifact_mask), idx + extension_samples + 1)
            extended_mask[start:end] = True
        
        return extended_mask
    
    def apply_baseline_correction(self, epochs: np.ndarray, 
                                baseline_window: Optional[Tuple[float, float]] = None,
                                epoch_start_time: float = -0.2) -> np.ndarray:
        """
        Apply baseline correction to epoched data
        
        Args:
            epochs: Epoched data (n_epochs x n_samples x n_channels)
            baseline_window: Time window for baseline in seconds (start, end)
            epoch_start_time: Time of epoch start relative to event (seconds)
            
        Returns:
            Baseline-corrected epochs
        """
        if baseline_window is None:
            baseline_window = self.params['baseline_window']
        
        # Convert time window to sample indices
        baseline_start_idx = int((baseline_window[0] - epoch_start_time) * self.sampling_rate)
        baseline_end_idx = int((baseline_window[1] - epoch_start_time) * self.sampling_rate)
        
        # Ensure indices are within bounds
        baseline_start_idx = max(0, baseline_start_idx)
        baseline_end_idx = min(epochs.shape[1], baseline_end_idx)
        
        # Calculate baseline for each epoch and channel
        baseline = np.mean(epochs[:, baseline_start_idx:baseline_end_idx, :], axis=1, keepdims=True)
        
        # Subtract baseline
        corrected_epochs = epochs - baseline
        
        return corrected_epochs
    
    def get_preprocessing_info(self) -> Dict:
        """
        Get information about preprocessing parameters
        
        Returns:
            Dictionary with preprocessing information
        """
        return {
            'sampling_rate': self.sampling_rate,
            'errp_band': self.errp_band,
            'parameters': self.params,
            'methods': {
                'spatial_filter': 'Common Average Reference (CAR) or Laplacian',
                'temporal_filter': 'Butterworth bandpass (order 4)',
                'notch_filter': 'IIR notch filter',
                'artifact_detection': 'Amplitude threshold with 100ms extension'
            }
        }