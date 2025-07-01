#!/usr/bin/env python
"""
Plot raw EEG data for participant T-005 from test_data folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob

def load_latest_eeg_data():
    """Load the latest EEG data from test_data"""
    test_data_dir = "test_data"
    
    # Find latest BrainFlow file
    brainflow_files = glob.glob(os.path.join(test_data_dir, "BrainFlow-RAW*.csv"))
    if brainflow_files:
        brainflow_file = sorted(brainflow_files)[-1]
        print(f"Loading BrainFlow data from: {brainflow_file}")
        
        # Load tab-separated data
        df = pd.read_csv(brainflow_file, sep='\t', header=None)
        
        # Extract EEG channels (columns 1-16)
        eeg_data = df.iloc[:, 1:17].values
        
        # Try to find timestamps
        timestamps = None
        if df.shape[1] > 30:
            # Check for timestamp column (usually second to last)
            timestamp_col = df.iloc[:, -2].values
            if np.all(timestamp_col > 1e9):  # Unix timestamp
                timestamps = timestamp_col
            else:
                # Create synthetic timestamps at 250Hz
                fs = 250
                timestamps = np.arange(len(eeg_data)) / fs
        
        return eeg_data, timestamps, "BrainFlow"
    
    # Try OpenBCI format
    openbci_files = glob.glob(os.path.join(test_data_dir, "OpenBCI-RAW*.txt"))
    if openbci_files:
        openbci_file = sorted(openbci_files)[-1]
        print(f"Loading OpenBCI data from: {openbci_file}")
        
        # Skip header lines
        df = pd.read_csv(openbci_file, skiprows=lambda x: x < 5, header=0)
        
        # Extract EEG channels
        eeg_cols = [col for col in df.columns if 'EXG Channel' in col]
        if eeg_cols:
            eeg_data = df[eeg_cols].values
            
            # Get timestamps
            if 'Timestamp' in df.columns:
                timestamps = df['Timestamp'].values
            else:
                fs = 125  # OpenBCI default
                timestamps = np.arange(len(eeg_data)) / fs
                
            return eeg_data, timestamps, "OpenBCI"
    
    raise FileNotFoundError("No EEG data files found in test_data/")

def plot_raw_eeg(eeg_data, timestamps, data_source, duration=30, start_time=0):
    """
    Plot raw EEG data
    
    Args:
        eeg_data: EEG data array (samples x channels)
        timestamps: Timestamp array
        data_source: Source of data (for title)
        duration: Duration to plot in seconds
        start_time: Start time in seconds
    """
    
    # Calculate sampling rate
    if timestamps[0] > 1e9:  # Unix timestamps
        fs = 1.0 / np.median(np.diff(timestamps))
        time_vec = timestamps - timestamps[0]  # Start from 0
    else:
        time_vec = timestamps
        fs = 1.0 / np.median(np.diff(timestamps))
    
    print(f"Sampling rate: {fs:.1f} Hz")
    
    # Select time window
    start_idx = int(start_time * fs)
    end_idx = min(int((start_time + duration) * fs), len(eeg_data))
    
    # Extract window
    data_window = eeg_data[start_idx:end_idx, :]
    time_window = time_vec[start_idx:end_idx]
    
    # Scale factor (convert to more reasonable units)
    scale_factor = 0.001  # Assuming we need to convert to mV
    
    # Create figure with multiple subplots
    n_channels = data_window.shape[1]
    fig, axes = plt.subplots(n_channels, 1, figsize=(15, 2*n_channels), sharex=True)
    
    # Plot each channel
    for ch in range(n_channels):
        ch_data = data_window[:, ch] * scale_factor
        
        # Remove DC offset
        ch_data = ch_data - np.mean(ch_data)
        
        axes[ch].plot(time_window, ch_data, 'b-', linewidth=0.5)
        axes[ch].set_ylabel(f'Ch{ch+1}\n(mV)', rotation=0, labelpad=20)
        axes[ch].grid(True, alpha=0.3)
        axes[ch].set_ylim([-50, 50])  # Adjust based on data
        
        # Add channel statistics
        ch_std = np.std(ch_data)
        ch_range = np.ptp(ch_data)
        axes[ch].text(0.02, 0.95, f'SD: {ch_std:.1f} mV, Range: {ch_range:.1f} mV',
                     transform=axes[ch].transAxes, va='top', fontsize=8,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'Raw EEG Data - {data_source} Recording\n'
                 f'{duration}s window starting at {start_time}s', fontsize=14)
    plt.tight_layout()
    
    # Save
    output_file = f'test_data/raw_eeg_005_{int(start_time)}s_{int(duration)}s.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Create a more detailed view of selected channels
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    
    # Select channels with different characteristics
    selected_channels = [2, 3, 6, 7]  # Channels that showed better quality
    
    for idx, ch in enumerate(selected_channels):
        ch_data = data_window[:, ch] * scale_factor
        ch_data = ch_data - np.mean(ch_data)
        
        axes[idx].plot(time_window, ch_data, 'b-', linewidth=0.8)
        axes[idx].set_ylabel(f'Channel {ch+1}\n(mV)')
        axes[idx].grid(True, alpha=0.3)
        
        # Add frequency content indicator
        if len(ch_data) > fs:
            freqs, psd = signal.welch(ch_data, fs=fs, nperseg=min(len(ch_data)//4, int(fs)))
            
            # Find dominant frequency
            dominant_freq = freqs[np.argmax(psd)]
            
            # Calculate band powers
            alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
            beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
            
            alpha_power = np.sum(psd[alpha_idx])
            beta_power = np.sum(psd[beta_idx])
            total_power = np.sum(psd)
            
            info_text = (f'Dom. freq: {dominant_freq:.1f} Hz\n'
                        f'Alpha: {100*alpha_power/total_power:.1f}%\n'
                        f'Beta: {100*beta_power/total_power:.1f}%')
            
            axes[idx].text(0.98, 0.95, info_text,
                          transform=axes[idx].transAxes, va='top', ha='right',
                          fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    axes[-1].set_xlabel('Time (s)')
    plt.suptitle(f'Selected Channels - Raw EEG Data\n{duration}s window', fontsize=14)
    plt.tight_layout()
    
    output_file = 'test_data/raw_eeg_005_selected_channels.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_file}")
    
    # Create power spectrum plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate PSD for all channels
    for ch in range(n_channels):
        ch_data = eeg_data[:, ch] * scale_factor
        ch_data = ch_data - np.mean(ch_data)
        
        if len(ch_data) > fs * 2:  # At least 2 seconds
            freqs, psd = signal.welch(ch_data, fs=fs, nperseg=int(fs*2))
            
            # Plot up to 50 Hz
            freq_mask = freqs <= 50
            
            # Color based on channel quality
            if ch in [2, 3, 6, 7]:
                ax1.semilogy(freqs[freq_mask], psd[freq_mask], alpha=0.8, linewidth=2, label=f'Ch{ch+1}')
            else:
                ax1.semilogy(freqs[freq_mask], psd[freq_mask], alpha=0.3, linewidth=1)
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power Spectral Density (mV²/Hz)')
    ax1.set_title('Power Spectrum - All Channels')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, 50])
    
    # Averaged spectrum for good channels
    good_channels = [2, 3, 6, 7]
    all_psd = []
    
    for ch in good_channels:
        ch_data = eeg_data[:, ch] * scale_factor
        ch_data = ch_data - np.mean(ch_data)
        
        if len(ch_data) > fs * 2:
            freqs, psd = signal.welch(ch_data, fs=fs, nperseg=int(fs*2))
            all_psd.append(psd)
    
    if all_psd:
        mean_psd = np.mean(all_psd, axis=0)
        std_psd = np.std(all_psd, axis=0)
        
        freq_mask = freqs <= 50
        ax2.semilogy(freqs[freq_mask], mean_psd[freq_mask], 'b-', linewidth=2)
        ax2.fill_between(freqs[freq_mask], 
                        mean_psd[freq_mask] - std_psd[freq_mask],
                        mean_psd[freq_mask] + std_psd[freq_mask],
                        alpha=0.3)
        
        # Mark frequency bands
        ax2.axvspan(8, 13, alpha=0.2, color='green', label='Alpha (8-13 Hz)')
        ax2.axvspan(13, 30, alpha=0.2, color='orange', label='Beta (13-30 Hz)')
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density (mV²/Hz)')
        ax2.set_title('Average Power Spectrum - Good Channels')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim([0, 50])
    
    plt.tight_layout()
    plt.savefig('test_data/raw_eeg_005_power_spectrum.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: test_data/raw_eeg_005_power_spectrum.png")

def create_eeg_montage_plot(eeg_data, timestamps, segment_duration=10):
    """Create a clinical-style EEG montage plot"""
    
    # Calculate sampling rate
    fs = 1.0 / np.median(np.diff(timestamps)) if len(timestamps) > 1 else 250
    
    # Select a segment
    segment_samples = int(segment_duration * fs)
    start_idx = len(eeg_data) // 4  # Start 1/4 into recording
    end_idx = start_idx + segment_samples
    
    if end_idx > len(eeg_data):
        end_idx = len(eeg_data)
        start_idx = max(0, end_idx - segment_samples)
    
    # Extract segment
    data_segment = eeg_data[start_idx:end_idx, :]
    time_segment = np.arange(len(data_segment)) / fs
    
    # Scale and offset for display
    scale_factor = 0.001  # to mV
    channel_offset = 100  # mV between channels
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot channels with offset
    for ch in range(data_segment.shape[1]):
        ch_data = data_segment[:, ch] * scale_factor
        ch_data = ch_data - np.mean(ch_data)
        
        # Limit amplitude to prevent overlap
        ch_data = np.clip(ch_data, -40, 40)
        
        # Add offset
        y_offset = (data_segment.shape[1] - ch - 1) * channel_offset
        
        ax.plot(time_segment, ch_data + y_offset, 'k-', linewidth=0.5)
        
        # Add channel label
        ax.text(-0.5, y_offset, f'Ch{ch+1}', ha='right', va='center', fontsize=10)
    
    # Add time grid
    for t in range(0, int(segment_duration) + 1):
        ax.axvline(t, color='gray', alpha=0.3, linestyle='--')
        ax.text(t, -channel_offset/2, f'{t}s', ha='center', va='top', fontsize=9)
    
    # Add amplitude scale
    scale_y = data_segment.shape[1] * channel_offset
    ax.plot([segment_duration - 0.5, segment_duration - 0.5], 
            [scale_y - 50, scale_y], 'k-', linewidth=2)
    ax.text(segment_duration - 0.3, scale_y - 25, '50 mV', ha='left', va='center', fontsize=9)
    
    ax.set_xlim([-1, segment_duration + 0.5])
    ax.set_ylim([-channel_offset, data_segment.shape[1] * channel_offset])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title(f'EEG Montage View - {segment_duration}s segment', fontsize=14)
    
    # Remove y-axis
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('test_data/raw_eeg_005_montage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: test_data/raw_eeg_005_montage.png")

def main():
    """Main function"""
    
    try:
        # Load data
        print("Loading EEG data...")
        eeg_data, timestamps, data_source = load_latest_eeg_data()
        
        print(f"Data shape: {eeg_data.shape}")
        print(f"Duration: {len(eeg_data) / 250:.1f} seconds (assuming 250 Hz)")
        
        # Check data statistics
        print("\nData Statistics:")
        print(f"Overall range: {np.min(eeg_data):.1f} to {np.max(eeg_data):.1f}")
        print(f"Mean amplitude: {np.mean(eeg_data):.1f}")
        print(f"Std deviation: {np.std(eeg_data):.1f}")
        
        # Plot different views
        print("\nGenerating plots...")
        
        # 1. Raw data plot - first 30 seconds
        plot_raw_eeg(eeg_data, timestamps, data_source, duration=30, start_time=0)
        
        # 2. Raw data plot - middle section
        middle_time = len(eeg_data) / 250 / 2  # Middle of recording
        plot_raw_eeg(eeg_data, timestamps, data_source, duration=10, start_time=middle_time)
        
        # 3. Clinical montage view
        create_eeg_montage_plot(eeg_data, timestamps)
        
        print("\nAll plots generated successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()