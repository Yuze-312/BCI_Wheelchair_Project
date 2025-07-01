"""
Motor Imagery Data Preprocessing Module
Handles filtering, artifact rejection, and preprocessing
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

class MIPreprocessor:
    def __init__(self, sampling_rate: int = 512):
        """
        Initialize preprocessor
        
        Args:
            sampling_rate: Sampling frequency (default 512Hz)
        """
        self.sampling_rate = sampling_rate
        
    def bandpass_filter(self, data: np.ndarray, low_freq: float = 8.0, 
                       high_freq: float = 30.0, order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to extract mu (8-13Hz) and beta (13-30Hz) rhythms
        
        Args:
            data: EEG data (samples x channels)
            low_freq: Lower cutoff frequency
            high_freq: Upper cutoff frequency
            order: Filter order
            
        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
            
        return filtered_data
    
    def notch_filter(self, data: np.ndarray, freq: float = 50.0, 
                    quality: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line interference
        
        Args:
            data: EEG data (samples x channels)
            freq: Notch frequency (50Hz for Europe, 60Hz for US)
            quality: Quality factor
            
        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        w0 = freq / nyquist
        
        b, a = iirnotch(w0, quality)
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[1]):
            filtered_data[:, ch] = filtfilt(b, a, data[:, ch])
            
        return filtered_data
    
    def apply_car(self, data: np.ndarray) -> np.ndarray:
        """
        Apply Common Average Reference (CAR) spatial filter
        
        Args:
            data: EEG data (samples x channels)
            
        Returns:
            CAR filtered data
        """
        # Calculate average across channels for each sample
        car_ref = np.mean(data, axis=1, keepdims=True)
        
        # Subtract from each channel
        return data - car_ref
    
    def apply_laplacian(self, data: np.ndarray, 
                       channel_neighbors: Optional[Dict] = None) -> np.ndarray:
        """
        Apply Laplacian spatial filter (simplified version)
        
        Args:
            data: EEG data (samples x channels)
            channel_neighbors: Dictionary mapping channels to their neighbors
            
        Returns:
            Laplacian filtered data
        """
        if channel_neighbors is None:
            # Simple approximation: each channel minus average of all others
            filtered_data = np.zeros_like(data)
            for ch in range(data.shape[1]):
                others = np.mean(np.delete(data, ch, axis=1), axis=1)
                filtered_data[:, ch] = data[:, ch] - others
            return filtered_data
        else:
            # Use provided neighbor structure
            filtered_data = np.zeros_like(data)
            for ch, neighbors in channel_neighbors.items():
                if neighbors:
                    neighbor_avg = np.mean(data[:, neighbors], axis=1)
                    filtered_data[:, ch] = data[:, ch] - neighbor_avg
                else:
                    filtered_data[:, ch] = data[:, ch]
            return filtered_data
    
    def detect_artifacts(self, data: np.ndarray, threshold: float = 100.0) -> np.ndarray:
        """
        Detect artifacts based on amplitude threshold
        
        Args:
            data: EEG data (samples x channels)
            threshold: Maximum allowed amplitude in μV
            
        Returns:
            Boolean mask (True = clean, False = artifact)
        """
        # Check if any channel exceeds threshold at each time point
        artifact_mask = np.all(np.abs(data) < threshold, axis=1)
        
        # Extend artifact rejection window (±100ms around detected artifact)
        window = int(0.1 * self.sampling_rate)  # 100ms
        artifact_indices = np.where(~artifact_mask)[0]
        
        for idx in artifact_indices:
            start = max(0, idx - window)
            end = min(len(artifact_mask), idx + window)
            artifact_mask[start:end] = False
            
        return artifact_mask
    
    def baseline_correction(self, data: np.ndarray, baseline_samples: int) -> np.ndarray:
        """
        Apply baseline correction by subtracting pre-stimulus mean
        
        Args:
            data: EEG data (samples x channels)
            baseline_samples: Number of samples to use for baseline
            
        Returns:
            Baseline corrected data
        """
        if baseline_samples > data.shape[0]:
            baseline_samples = data.shape[0] // 4
            
        baseline_mean = np.mean(data[:baseline_samples, :], axis=0, keepdims=True)
        return data - baseline_mean
    
    def preprocess_trial(self, trial: Dict, spatial_filter: str = 'car',
                        remove_powerline: bool = True) -> Dict:
        """
        Apply full preprocessing pipeline to a single trial
        
        Args:
            trial: Trial dictionary from data loader
            spatial_filter: Type of spatial filter ('car', 'laplacian', or None)
            remove_powerline: Whether to apply notch filter
            
        Returns:
            Preprocessed trial dictionary
        """
        data = trial['data'].copy()
        
        # 1. Notch filter for power line noise
        if remove_powerline:
            data = self.notch_filter(data, freq=50.0)  # Adjust to 60Hz if needed
        
        # 2. Bandpass filter (8-30 Hz for motor imagery)
        data = self.bandpass_filter(data, low_freq=8.0, high_freq=30.0)
        
        # 3. Spatial filtering
        if spatial_filter == 'car':
            data = self.apply_car(data)
        elif spatial_filter == 'laplacian':
            data = self.apply_laplacian(data)
        
        # 4. Baseline correction (using first second as baseline)
        baseline_samples = int(1.0 * self.sampling_rate)
        data = self.baseline_correction(data, baseline_samples)
        
        # 5. Artifact detection
        artifact_mask = self.detect_artifacts(data, threshold=100.0)
        
        # Create preprocessed trial
        preprocessed_trial = trial.copy()
        preprocessed_trial['data'] = data
        preprocessed_trial['artifact_mask'] = artifact_mask
        preprocessed_trial['artifact_percentage'] = 100 * (1 - np.mean(artifact_mask))
        preprocessed_trial['is_valid'] = preprocessed_trial['artifact_percentage'] < 20.0
        
        return preprocessed_trial
    
    def preprocess_session(self, session_trials: Dict, **kwargs) -> Dict:
        """
        Preprocess all trials in a session
        
        Args:
            session_trials: Dictionary of runs with trials
            **kwargs: Arguments passed to preprocess_trial
            
        Returns:
            Preprocessed trials
        """
        preprocessed_session = {}
        
        for run_name, trials in session_trials.items():
            preprocessed_trials = []
            valid_count = 0
            
            for trial in trials:
                preprocessed = self.preprocess_trial(trial, **kwargs)
                preprocessed_trials.append(preprocessed)
                
                if preprocessed['is_valid']:
                    valid_count += 1
            
            preprocessed_session[run_name] = preprocessed_trials
            print(f"{run_name}: {valid_count}/{len(trials)} valid trials after preprocessing")
            
        return preprocessed_session
    
    def plot_preprocessing_comparison(self, trial: Dict, channel: int = 0):
        """
        Plot comparison of raw vs preprocessed data
        
        Args:
            trial: Original trial dictionary
            channel: Channel index to plot
        """
        # Preprocess the trial
        preprocessed = self.preprocess_trial(trial)
        
        # Time axis
        time = np.arange(trial['data'].shape[0]) / self.sampling_rate - 1.0
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Raw data
        axes[0].plot(time, trial['data'][:, channel], 'b', alpha=0.7)
        axes[0].set_ylabel('Raw (μV)')
        axes[0].set_title(f'Channel {channel+1} - Class {trial["label"]}')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Preprocessed data
        axes[1].plot(time, preprocessed['data'][:, channel], 'g', alpha=0.7)
        axes[1].set_ylabel('Preprocessed (μV)')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Artifact mask
        axes[2].fill_between(time, 0, preprocessed['artifact_mask'], 
                           alpha=0.5, label='Clean')
        axes[2].set_ylabel('Artifact mask')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='MI onset')
        axes[2].legend()
        
        plt.tight_layout()
        return fig
    
    def calculate_snr(self, data: np.ndarray, 
                     signal_window: Tuple[int, int],
                     noise_window: Tuple[int, int]) -> float:
        """
        Calculate signal-to-noise ratio
        
        Args:
            data: EEG data (samples x channels)
            signal_window: (start, end) samples for signal
            noise_window: (start, end) samples for noise
            
        Returns:
            SNR in dB
        """
        signal_power = np.mean(data[signal_window[0]:signal_window[1], :]**2)
        noise_power = np.mean(data[noise_window[0]:noise_window[1], :]**2)
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = np.inf
            
        return snr_db


def main():
    """Example usage"""
    try:
        from .load_data import MIDataLoader
    except ImportError:
        from load_data import MIDataLoader
    
    # Load some data
    loader = MIDataLoader("/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI")
    t005_data = loader.get_participant_data("T-005")
    
    # Initialize preprocessor
    preprocessor = MIPreprocessor(sampling_rate=512)
    
    # Get first trial
    if t005_data['sessions']:
        first_session = list(t005_data['sessions'].values())[0]
        if first_session:
            first_run = list(first_session.values())[0]
            if first_run:
                trial = first_run[0]
                
                # Plot preprocessing comparison
                print("Plotting preprocessing comparison...")
                fig = preprocessor.plot_preprocessing_comparison(trial)
                plt.show()
                
                # Preprocess all trials in first session
                print("\nPreprocessing first session...")
                preprocessed = preprocessor.preprocess_session(first_session)
                
                # Calculate statistics
                total_trials = sum(len(run) for run in first_session.values())
                valid_trials = sum(sum(1 for t in run if t['is_valid']) 
                                 for run in preprocessed.values())
                
                print(f"\nPreprocessing summary:")
                print(f"Total trials: {total_trials}")
                print(f"Valid trials: {valid_trials} ({100*valid_trials/total_trials:.1f}%)")


if __name__ == "__main__":
    main()