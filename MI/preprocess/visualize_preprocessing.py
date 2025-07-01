"""
Visualization of Preprocessing Steps

This script shows the effect of each preprocessing step on EEG data.
"""

import numpy as np
import matplotlib.pyplot as plt
from .preprocess import MIPreprocessor
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_eeg(duration=5, sampling_rate=512, n_channels=16):
    """Generate synthetic EEG data with artifacts"""
    n_samples = int(duration * sampling_rate)
    time = np.arange(n_samples) / sampling_rate
    
    # Base EEG signal (alpha + beta rhythms)
    data = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        # Alpha rhythm (8-13 Hz)
        alpha_freq = 10 + np.random.rand() * 3
        alpha_amp = 20 + np.random.rand() * 10
        data[:, ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * time)
        
        # Beta rhythm (13-30 Hz)
        beta_freq = 20 + np.random.rand() * 10
        beta_amp = 10 + np.random.rand() * 5
        data[:, ch] += beta_amp * np.sin(2 * np.pi * beta_freq * time)
        
        # Background noise
        data[:, ch] += np.random.randn(n_samples) * 5
    
    # Add 50 Hz powerline noise
    powerline_amp = 15
    data += powerline_amp * np.sin(2 * np.pi * 50 * time)[:, np.newaxis]
    
    # Add some artifacts (eye blinks)
    artifact_times = [1.0, 2.5, 4.0]  # seconds
    for t in artifact_times:
        idx = int(t * sampling_rate)
        if idx < n_samples - 50:
            artifact = 150 * np.exp(-np.arange(50) / 10)
            data[idx:idx+50, :n_channels//2] += artifact[:, np.newaxis]
    
    return data, time


def plot_preprocessing_steps():
    """Visualize each preprocessing step"""
    # Generate synthetic data
    print("Generating synthetic EEG data...")
    sampling_rate = 512
    data, time = generate_synthetic_eeg(duration=5, sampling_rate=sampling_rate)
    
    # Initialize preprocessor
    preprocessor = MIPreprocessor(sampling_rate=sampling_rate)
    
    # Select channel to visualize
    ch = 0
    
    # Create figure
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('EEG Preprocessing Steps Visualization', fontsize=16)
    
    # 1. Raw data
    axes[0].plot(time, data[:, ch], 'b-', linewidth=0.5)
    axes[0].set_ylabel('Raw\n(μV)')
    axes[0].set_title('Step 1: Raw EEG Data')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Notch filter
    data_notch = preprocessor.notch_filter(data, freq=50.0)
    axes[1].plot(time, data_notch[:, ch], 'g-', linewidth=0.5)
    axes[1].set_ylabel('Notch\n(μV)')
    axes[1].set_title('Step 2: After 50Hz Notch Filter')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Bandpass filter
    data_bandpass = preprocessor.bandpass_filter(data_notch, low_freq=8.0, high_freq=30.0)
    axes[2].plot(time, data_bandpass[:, ch], 'r-', linewidth=0.5)
    axes[2].set_ylabel('Bandpass\n(μV)')
    axes[2].set_title('Step 3: After 8-30Hz Bandpass Filter')
    axes[2].grid(True, alpha=0.3)
    
    # 4. CAR spatial filter
    data_car = preprocessor.apply_car(data_bandpass)
    axes[3].plot(time, data_car[:, ch], 'm-', linewidth=0.5)
    axes[3].set_ylabel('CAR\n(μV)')
    axes[3].set_title('Step 4: After Common Average Reference')
    axes[3].grid(True, alpha=0.3)
    
    # 5. Baseline correction
    baseline_samples = int(1.0 * sampling_rate)
    data_baseline = preprocessor.baseline_correction(data_car, baseline_samples)
    axes[4].plot(time, data_baseline[:, ch], 'c-', linewidth=0.5)
    axes[4].set_ylabel('Baseline\n(μV)')
    axes[4].set_title('Step 5: After Baseline Correction')
    axes[4].grid(True, alpha=0.3)
    
    # 6. Artifact detection
    artifact_mask = preprocessor.detect_artifacts(data_baseline, threshold=100.0)
    data_clean = data_baseline.copy()
    data_clean[~artifact_mask, :] = np.nan
    axes[5].plot(time, data_clean[:, ch], 'k-', linewidth=0.5)
    axes[5].plot(time[~artifact_mask], data_baseline[~artifact_mask, ch], 'r.', markersize=1)
    axes[5].set_ylabel('Clean\n(μV)')
    axes[5].set_title('Step 6: After Artifact Detection (red = artifacts)')
    axes[5].set_xlabel('Time (s)')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create frequency spectrum comparison
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
    fig2.suptitle('Frequency Spectrum Comparison', fontsize=16)
    
    # Compute power spectral density
    from scipy import signal
    freqs, psd_raw = signal.welch(data[:, ch], fs=sampling_rate, nperseg=1024)
    freqs, psd_clean = signal.welch(data_baseline[:, ch], fs=sampling_rate, nperseg=1024)
    
    # Plot PSDs
    axes2[0].semilogy(freqs, psd_raw, 'b-', label='Raw')
    axes2[0].semilogy(freqs, psd_clean, 'k-', label='Preprocessed')
    axes2[0].set_xlim([0, 100])
    axes2[0].set_ylabel('PSD (μV²/Hz)')
    axes2[0].set_title('Power Spectral Density')
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)
    axes2[0].axvline(50, color='r', linestyle='--', alpha=0.5, label='50Hz')
    axes2[0].axvspan(8, 30, alpha=0.2, color='green', label='MI band')
    
    # Zoom in on MI frequencies
    axes2[1].semilogy(freqs, psd_raw, 'b-', label='Raw')
    axes2[1].semilogy(freqs, psd_clean, 'k-', label='Preprocessed')
    axes2[1].set_xlim([5, 35])
    axes2[1].set_xlabel('Frequency (Hz)')
    axes2[1].set_ylabel('PSD (μV²/Hz)')
    axes2[1].set_title('Motor Imagery Frequency Band (8-30 Hz)')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    axes2[1].axvspan(8, 13, alpha=0.2, color='orange', label='Mu')
    axes2[1].axvspan(13, 30, alpha=0.2, color='cyan', label='Beta')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nPreprocessing Statistics:")
    print(f"  Original data range: [{np.min(data):.1f}, {np.max(data):.1f}] μV")
    print(f"  Preprocessed range: [{np.nanmin(data_clean):.1f}, {np.nanmax(data_clean):.1f}] μV")
    print(f"  Artifacts detected: {100 * (1 - np.mean(artifact_mask)):.1f}%")
    print(f"  50Hz power reduction: {(psd_raw[freqs == 50] / psd_clean[freqs == 50])[0]:.1f}x")


if __name__ == "__main__":
    print("EEG Preprocessing Visualization")
    print("=" * 50)
    plot_preprocessing_steps()