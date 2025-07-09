"""
ErrP Epoch Extractor Module

Extracts epochs around error events for ErrP analysis.
Handles:
- Epoch extraction around error/correct events
- Time window selection for ErrP components
- Epoch validation and quality control
- Feature extraction from epochs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPEpochExtractor:
    """Extract epochs around error events for ErrP analysis"""
    
    def __init__(self, sampling_rate: int = 512):
        """
        Initialize the epoch extractor
        
        Args:
            sampling_rate: EEG sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        
        # Default epoch parameters for ErrP
        # ErrP components typically appear 200-600ms after error
        self.epoch_params = {
            'epoch_start': -0.2,  # 200ms before event
            'epoch_end': 0.8,     # 800ms after event
            'baseline_start': -0.2,
            'baseline_end': 0.0,
            'min_epoch_length': 0.8,  # Minimum valid epoch length
        }
        
        # ErrP component windows (in seconds relative to event)
        self.component_windows = {
            'ERN': (0.0, 0.15),      # Error-Related Negativity (0-150ms)
            'Pe': (0.2, 0.5),        # Error Positivity (200-500ms)
            'P300': (0.25, 0.6),     # P300-like component (250-600ms)
            'late': (0.5, 0.8)       # Late positive component (500-800ms)
        }
        
    def extract_epochs(self, eeg_data: pd.DataFrame, 
                      events: pd.DataFrame,
                      event_types: Optional[List[int]] = None,
                      channel_names: Optional[List[str]] = None) -> Dict:
        """
        Extract epochs around specified events
        
        Args:
            eeg_data: Preprocessed EEG data with channels and timestamp
            events: Event log with event types and sample indices
            event_types: List of event types to extract (default: error events)
            channel_names: List of channel names to use
            
        Returns:
            Dictionary containing:
                - epochs: Array of shape (n_epochs, n_samples, n_channels)
                - labels: Event type for each epoch
                - metadata: Information about each epoch
                - times: Time vector for epochs
        """
        if event_types is None:
            # Default to error events (9: feedback error, 10: primary ErrP)
            event_types = [9, 10]
        
        if channel_names is None:
            # Auto-detect channel columns
            channel_names = [col for col in eeg_data.columns 
                           if col.startswith('Channel') or col.startswith('Ch')]
        
        logger.info(f"Extracting epochs for event types: {event_types}")
        
        # Filter events
        selected_events = events[events['event'].isin(event_types)].copy()
        logger.info(f"Found {len(selected_events)} events")
        
        # Calculate epoch samples
        # Use round instead of int to avoid off-by-one errors
        epoch_start_samples = int(np.round(self.epoch_params['epoch_start'] * self.sampling_rate))
        epoch_end_samples = int(np.round(self.epoch_params['epoch_end'] * self.sampling_rate))
        epoch_length = epoch_end_samples - epoch_start_samples
        
        # Extract channel data
        channel_data = eeg_data[channel_names].values
        n_channels = len(channel_names)
        
        # Initialize storage
        epochs = []
        labels = []
        metadata = []
        rejected_count = 0
        
        # Extract epochs
        for _, event in selected_events.iterrows():
            sample_idx = event['sample_idx']
            
            # Calculate epoch boundaries
            try:
                start_idx = int(sample_idx + epoch_start_samples)
                end_idx = int(sample_idx + epoch_end_samples)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid sample index {sample_idx}: {e}")
                rejected_count += 1
                continue
            
            # Check boundaries
            if start_idx < 0 or end_idx > len(channel_data):
                logger.warning(f"Epoch at sample {sample_idx} out of bounds (start={start_idx}, end={end_idx}, data_len={len(channel_data)}), skipping")
                rejected_count += 1
                continue
            
            # Extract epoch data
            epoch_data = channel_data[start_idx:end_idx, :]
            
            # Check for artifacts if artifact column exists
            if 'artifact' in eeg_data.columns:
                artifact_mask = eeg_data['artifact'].iloc[start_idx:end_idx].values
                if artifact_mask.any():
                    logger.debug(f"Epoch at sample {sample_idx} contains artifacts, skipping")
                    rejected_count += 1
                    continue
            
            # Validate epoch shape
            if epoch_data.shape[0] != epoch_length:
                logger.warning(f"Epoch at sample {sample_idx} has incorrect length, skipping")
                rejected_count += 1
                continue
            
            # Store epoch
            epochs.append(epoch_data)
            labels.append(event['event'])
            
            # Store metadata
            meta = {
                'event_sample': sample_idx,
                'event_time': event['timestamp'],
                'event_type': event['event'],
                'epoch_start_sample': start_idx,
                'epoch_end_sample': end_idx
            }
            
            # Add additional event information if available
            for col in ['correct', 'expected_action', 'actual_action']:
                if col in event.index:
                    meta[col] = event[col]
            
            metadata.append(meta)
        
        # Convert to arrays
        if epochs:
            epochs = np.array(epochs)
            labels = np.array(labels)
        else:
            logger.warning("No valid epochs extracted!")
            epochs = np.empty((0, epoch_length, n_channels))
            labels = np.array([])
        
        # Create time vector
        times = np.arange(epoch_start_samples, epoch_end_samples) / self.sampling_rate
        
        logger.info(f"Extracted {len(epochs)} valid epochs, rejected {rejected_count}")
        
        return {
            'epochs': epochs,
            'labels': labels,
            'metadata': metadata,
            'times': times,
            'channel_names': channel_names,
            'sampling_rate': self.sampling_rate,
            'n_rejected': rejected_count
        }
    
    def extract_comparison_epochs(self, eeg_data: pd.DataFrame,
                                 events: pd.DataFrame,
                                 channel_names: Optional[List[str]] = None) -> Dict:
        """
        Extract both error and correct epochs for comparison
        
        Args:
            eeg_data: Preprocessed EEG data
            events: Event log
            channel_names: Channel names to use
            
        Returns:
            Dictionary with error and correct epochs
        """
        # Based on simple_markers.py:
        # Error events: 6 (RESPONSE_ERROR), 12 (NATURAL_ERROR), 13 (FORCED_ERROR)
        # Correct events: 5 (RESPONSE_CORRECT), 10 (NATURAL_CORRECT), 11 (FORCED_CORRECT)
        
        error_epochs = self.extract_epochs(
            eeg_data, events, 
            event_types=[6, 12, 13],  # Error response markers
            channel_names=channel_names
        )
        
        correct_epochs = self.extract_epochs(
            eeg_data, events,
            event_types=[5, 10, 11],  # Correct response markers
            channel_names=channel_names
        )
        
        return {
            'error': error_epochs,
            'correct': correct_epochs
        }
    
    def _extract_epochs_from_events(self, eeg_data: pd.DataFrame,
                                   events: pd.DataFrame,
                                   channel_names: Optional[List[str]] = None) -> Dict:
        """
        Internal method to extract epochs from pre-filtered events
        
        Args:
            eeg_data: Preprocessed EEG data
            events: Pre-filtered event DataFrame
            channel_names: Channel names to use
            
        Returns:
            Dictionary with epochs
        """
        if channel_names is None:
            channel_names = [col for col in eeg_data.columns 
                           if col.startswith('Channel') or col.startswith('Ch')]
        
        # Calculate epoch samples
        # Use round instead of int to avoid off-by-one errors
        epoch_start_samples = int(np.round(self.epoch_params['epoch_start'] * self.sampling_rate))
        epoch_end_samples = int(np.round(self.epoch_params['epoch_end'] * self.sampling_rate))
        epoch_length = epoch_end_samples - epoch_start_samples
        
        # Extract channel data
        channel_data = eeg_data[channel_names].values
        n_channels = len(channel_names)
        
        # Initialize storage
        epochs = []
        labels = []
        metadata = []
        rejected_count = 0
        
        logger.info(f"Extracting epochs from {len(events)} events")
        
        # Extract epochs
        for _, event in events.iterrows():
            # Get sample index
            if 'sample_idx' in event and pd.notna(event['sample_idx']):
                sample_idx = event['sample_idx']
            else:
                sample_idx = event['timestamp'] * self.sampling_rate
            
            # Calculate epoch boundaries
            try:
                start_idx = int(sample_idx + epoch_start_samples)
                end_idx = int(sample_idx + epoch_end_samples)
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid sample index {sample_idx}: {e}")
                rejected_count += 1
                continue
            
            # Check boundaries
            if start_idx < 0 or end_idx > len(channel_data):
                logger.debug(f"Epoch out of bounds: start={start_idx}, end={end_idx}, data_len={len(channel_data)}")
                rejected_count += 1
                continue
            
            # Extract epoch data
            epoch_data = channel_data[start_idx:end_idx, :]
            
            # Check for artifacts if artifact column exists
            if 'artifact' in eeg_data.columns:
                artifact_mask = eeg_data['artifact'].iloc[start_idx:end_idx].values
                if artifact_mask.any():
                    rejected_count += 1
                    continue
            
            # Validate epoch shape
            if epoch_data.shape[0] != epoch_length:
                rejected_count += 1
                continue
            
            # Store epoch
            epochs.append(epoch_data)
            labels.append(event.get('event', 5))  # Default to event 5 for coin collection
            
            # Store metadata
            meta = {
                'event_sample': sample_idx,
                'event_time': event['timestamp'],
                'event_type': event.get('event', 5),
                'is_error': event.get('is_error', None),
                'gt': event.get('gt', None),
                'classifier_out': event.get('classifier_out', None),
                'confidence': event.get('confidence', None)
            }
            metadata.append(meta)
        
        # Convert to arrays
        if epochs:
            epochs = np.array(epochs)
            labels = np.array(labels)
        else:
            logger.warning("No valid epochs extracted!")
            epochs = np.empty((0, epoch_length, n_channels))
            labels = np.array([])
        
        # Create time vector
        times = np.arange(epoch_start_samples, epoch_end_samples) / self.sampling_rate
        
        logger.info(f"Extracted {len(epochs)} valid epochs, rejected {rejected_count}")
        
        return {
            'epochs': epochs,
            'labels': labels,
            'metadata': metadata,
            'times': times,
            'channel_names': channel_names,
            'sampling_rate': self.sampling_rate,
            'n_rejected': rejected_count
        }
    
    def compute_erp(self, epochs: np.ndarray, 
                   baseline_correction: bool = True) -> np.ndarray:
        """
        Compute Event-Related Potential (average across epochs)
        
        Args:
            epochs: Epoch array (n_epochs x n_samples x n_channels)
            baseline_correction: Whether to apply baseline correction
            
        Returns:
            ERP array (n_samples x n_channels)
        """
        if len(epochs) == 0:
            logger.warning("No epochs to average")
            return np.array([])
        
        if baseline_correction:
            # Apply baseline correction first
            baseline_start_idx = 0
            baseline_end_idx = int(-self.epoch_params['epoch_start'] * self.sampling_rate)
            baseline = np.mean(epochs[:, baseline_start_idx:baseline_end_idx, :], 
                             axis=1, keepdims=True)
            epochs = epochs - baseline
        
        # Compute average
        erp = np.mean(epochs, axis=0)
        
        return erp
    
    def extract_features(self, epochs: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Extract ErrP-specific features from epochs
        
        Args:
            epochs: Epoch array (n_epochs x n_samples x n_channels)
            times: Time vector for epochs
            
        Returns:
            Feature array (n_epochs x n_features)
        """
        n_epochs, n_samples, n_channels = epochs.shape
        features_list = []
        
        for epoch_idx in range(n_epochs):
            epoch = epochs[epoch_idx]
            epoch_features = []
            
            # Extract features for each channel
            for ch in range(n_channels):
                channel_data = epoch[:, ch]
                
                # 1. Peak features in component windows
                for comp_name, (start_time, end_time) in self.component_windows.items():
                    # Get time window indices
                    time_mask = (times >= start_time) & (times <= end_time)
                    window_data = channel_data[time_mask]
                    
                    if len(window_data) > 0:
                        # Peak amplitude
                        peak_amp = np.max(np.abs(window_data))
                        epoch_features.append(peak_amp)
                        
                        # Mean amplitude
                        mean_amp = np.mean(window_data)
                        epoch_features.append(mean_amp)
                        
                        # Peak latency
                        peak_idx = np.argmax(np.abs(window_data))
                        peak_latency = times[time_mask][peak_idx]
                        epoch_features.append(peak_latency)
                    else:
                        epoch_features.extend([0, 0, 0])
                
                # 2. Area under curve for positive/negative deflections
                positive_area = np.trapz(np.maximum(channel_data, 0), times)
                negative_area = np.trapz(np.minimum(channel_data, 0), times)
                epoch_features.extend([positive_area, negative_area])
                
                # 3. Zero-crossing rate
                zero_crossings = np.sum(np.diff(np.sign(channel_data)) != 0)
                epoch_features.append(zero_crossings)
            
            features_list.append(epoch_features)
        
        features = np.array(features_list)
        
        logger.info(f"Extracted {features.shape[1]} features from {n_epochs} epochs")
        
        return features
    
    def get_feature_names(self, n_channels: int) -> List[str]:
        """
        Get names for all extracted features
        
        Args:
            n_channels: Number of EEG channels
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        for ch in range(n_channels):
            # Component window features
            for comp_name in self.component_windows:
                feature_names.extend([
                    f'Ch{ch+1}_{comp_name}_peak_amp',
                    f'Ch{ch+1}_{comp_name}_mean_amp',
                    f'Ch{ch+1}_{comp_name}_peak_latency'
                ])
            
            # Area features
            feature_names.extend([
                f'Ch{ch+1}_positive_area',
                f'Ch{ch+1}_negative_area',
                f'Ch{ch+1}_zero_crossings'
            ])
        
        return feature_names