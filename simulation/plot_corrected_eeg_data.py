#!/usr/bin/env python
"""
Plot EEG data with correct scaling (removing 24x gain)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob

# SCALING FACTOR - Remove 24x gain
GAIN_FACTOR = 24

def load_and_scale_data():
    """Load EEG data and apply correct scaling"""
    test_data_dir = "test_data"
    
    # Find latest files
    brainflow_files = glob.glob(os.path.join(test_data_dir, "BrainFlow-RAW*.csv"))
    brainflow_file = sorted(brainflow_files)[-1] if brainflow_files else None
    
    print(f"Loading: {brainflow_file}")
    
    # Load data
    df = pd.read_csv(brainflow_file, sep='\t', header=None)
    eeg_data_raw = df.iloc[:, 1:17].values  # 16 channels
    
    # Apply scaling - divide by gain
    eeg_data_scaled = eeg_data_raw / GAIN_FACTOR
    
    # Create time vector
    fs = 250  # Hz
    time_vec = np.arange(len(eeg_data_scaled)) / fs
    
    return eeg_data_scaled, time_vec, fs

def plot_scaled_raw_data(eeg_data, time_vec, fs, window_start=0, window_duration=10):
    """Plot properly scaled raw EEG data"""
    
    # Select window
    start_idx = int(window_start * fs)
    end_idx = int((window_start + window_duration) * fs)
    
    data_window = eeg_data[start_idx:end_idx, :]
    time_window = time_vec[start_idx:end_idx]
    
    # Create figure
    fig, axes = plt.subplots(8, 2, figsize=(15, 16))
    axes = axes.flatten()
    
    for ch in range(16):
        ax = axes[ch]
        ch_data = data_window[:, ch]
        
        # Remove NaN values
        valid_mask = ~np.isnan(ch_data)
        if np.sum(valid_mask) == 0:
            ax.text(0.5, 0.5, f'Channel {ch+1}\nNo valid data', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_xlim(0, window_duration)
            continue
        
        ch_data_valid = ch_data[valid_mask]
        time_valid = time_window[valid_mask]
        
        # Remove DC offset
        ch_data_valid = ch_data_valid - np.mean(ch_data_valid)
        
        # Plot
        ax.plot(time_valid, ch_data_valid, 'b-', linewidth=0.5)
        
        # Add statistics
        ch_std = np.std(ch_data_valid)
        ch_range = np.ptp(ch_data_valid)
        
        stats_text = f'SD: {ch_std:.1f} µV\nRange: {ch_range:.1f} µV'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Formatting
        ax.set_ylabel(f'Ch{ch+1} (µV)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-200, 200])  # Standard EEG range
        
        # Mark if channel is bad
        if ch_range > 1000 or ch_range < 1:
            ax.set_facecolor('#ffeeee')
            ax.text(0.98, 0.98, 'BAD', transform=ax.transAxes, 
                   va='top', ha='right', color='red', fontweight='bold')
    
    # Only show x-label on bottom plots
    for i in range(14, 16):
        axes[i].set_xlabel('Time (s)')
    
    plt.suptitle(f'Corrected EEG Data (Gain-Compensated)\n{window_duration}s window from {window_start}s', 
                 fontsize=16)
    plt.tight_layout()
    plt.savefig('test_data/corrected_eeg_all_channels.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: test_data/corrected_eeg_all_channels.png")

def plot_clinical_montage(eeg_data, time_vec, fs, window_start=10, window_duration=10):
    """Create clinical-style montage with corrected scaling"""
    
    # Select window
    start_idx = int(window_start * fs)
    end_idx = int((window_start + window_duration) * fs)
    
    data_window = eeg_data[start_idx:end_idx, :]
    time_window = time_vec[start_idx:end_idx] - time_vec[start_idx]
    
    # Select good channels only
    good_channels = [2, 3, 5, 6, 7, 9, 10, 11, 14, 15]  # 0-indexed
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Channel spacing
    channel_offset = 150  # µV between channels
    
    # Plot channels
    for i, ch in enumerate(good_channels):
        ch_data = data_window[:, ch]
        
        # Remove NaN and DC offset
        valid_mask = ~np.isnan(ch_data)
        if np.sum(valid_mask) > 0:
            ch_data = ch_data[valid_mask]
            time_valid = time_window[valid_mask]
            ch_data = ch_data - np.mean(ch_data)
            
            # Clip to prevent overlap
            ch_data = np.clip(ch_data, -100, 100)
            
            # Add offset
            y_offset = (len(good_channels) - i - 1) * channel_offset
            
            ax.plot(time_valid, ch_data + y_offset, 'k-', linewidth=0.8)
            
            # Channel label
            ax.text(-0.5, y_offset, f'Ch{ch+1}', ha='right', va='center', fontsize=10)
    
    # Time markers
    for t in range(0, int(window_duration) + 1):
        ax.axvline(t, color='gray', alpha=0.3, linestyle='--')
        ax.text(t, -50, f'{t}s', ha='center', va='top', fontsize=9)
    
    # Scale bar
    scale_y = len(good_channels) * channel_offset
    ax.plot([window_duration - 1, window_duration - 1], 
            [scale_y - 100, scale_y], 'k-', linewidth=2)
    ax.text(window_duration - 0.8, scale_y - 50, '100 µV', ha='left', va='center')
    
    # Formatting
    ax.set_xlim([-1, window_duration + 0.5])
    ax.set_ylim([-100, len(good_channels) * channel_offset + 50])
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_title('Clinical Montage - Corrected EEG (Good Channels Only)', fontsize=14)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('test_data/corrected_eeg_montage.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: test_data/corrected_eeg_montage.png")

def plot_power_spectrum_comparison(eeg_data_scaled, eeg_data_raw, fs):
    """Compare power spectra before and after scaling"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Select a good channel
    ch_idx = 2  # Channel 3
    
    # Raw data spectrum
    ax = axes[0, 0]
    ch_data_raw = eeg_data_raw[:, ch_idx]
    valid_mask = ~np.isnan(ch_data_raw)
    if np.sum(valid_mask) > fs * 2:
        ch_data_raw = ch_data_raw[valid_mask]
        freqs, psd_raw = signal.welch(ch_data_raw, fs=fs, nperseg=int(fs*2))
        ax.semilogy(freqs[freqs <= 50], psd_raw[freqs <= 50], 'r-', linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (raw units²/Hz)')
    ax.set_title('Raw Data (with 24x gain)')
    ax.grid(True, alpha=0.3)
    
    # Scaled data spectrum
    ax = axes[0, 1]
    ch_data_scaled = eeg_data_scaled[:, ch_idx]
    valid_mask = ~np.isnan(ch_data_scaled)
    if np.sum(valid_mask) > fs * 2:
        ch_data_scaled = ch_data_scaled[valid_mask]
        ch_data_scaled = ch_data_scaled - np.mean(ch_data_scaled)
        freqs, psd_scaled = signal.welch(ch_data_scaled, fs=fs, nperseg=int(fs*2))
        ax.semilogy(freqs[freqs <= 50], psd_scaled[freqs <= 50], 'b-', linewidth=2)
    
    # Mark frequency bands
    ax.axvspan(8, 13, alpha=0.2, color='green', label='Alpha')
    ax.axvspan(13, 30, alpha=0.2, color='orange', label='Beta')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (µV²/Hz)')
    ax.set_title('Corrected Data (gain removed)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Time domain comparison
    # Raw signal
    ax = axes[1, 0]
    t_start = 1000
    t_end = 3500  # 10 seconds at 250Hz
    time_seg = np.arange(t_end - t_start) / fs
    
    sig_raw = eeg_data_raw[t_start:t_end, ch_idx]
    ax.plot(time_seg, sig_raw, 'r-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (raw units)')
    ax.set_title('Raw Signal')
    ax.grid(True, alpha=0.3)
    
    # Scaled signal
    ax = axes[1, 1]
    sig_scaled = eeg_data_scaled[t_start:t_end, ch_idx]
    sig_scaled = sig_scaled - np.nanmean(sig_scaled)
    ax.plot(time_seg, sig_scaled, 'b-', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Corrected Signal')
    ax.set_ylim([-200, 200])
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Before and After Gain Correction - Channel {ch_idx+1}', fontsize=16)
    plt.tight_layout()
    plt.savefig('test_data/scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: test_data/scaling_comparison.png")

def plot_grand_averages_corrected():
    """Re-run grand average analysis with corrected scaling"""
    
    # Load event data
    test_data_dir = "test_data"
    event_files = glob.glob(os.path.join(test_data_dir, "subway_errp*.csv"))
    event_file = sorted(event_files)[-1] if event_files else None
    
    if not event_file:
        print("No event file found for epoch analysis")
        return
    
    events_df = pd.read_csv(event_file)
    
    # Load and scale EEG data
    eeg_data, time_vec, fs = load_and_scale_data()
    
    # Extract epochs with corrected data
    from analyze_epochs_grand_average_v2 import create_simple_epochs
    
    epochs_by_type, epoch_time_vec = create_simple_epochs(eeg_data, events_df, fs)
    
    if not epochs_by_type:
        print("No epochs extracted")
        return
    
    # Plot corrected grand averages
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {'cue_left': 'blue', 'cue_right': 'red', 
              'feedback_correct': 'green', 'feedback_error': 'orange'}
    
    # Use good channels
    good_channels = [2, 3, 6, 7]
    
    for event_type, epochs in epochs_by_type.items():
        # Average across good channels
        grand_avg_all_ch = []
        
        for ch in good_channels:
            ch_avg = np.mean(epochs[:, :, ch], axis=0)
            grand_avg_all_ch.append(ch_avg)
        
        # Grand average across channels
        grand_avg = np.mean(grand_avg_all_ch, axis=0)
        sem = np.std(grand_avg_all_ch, axis=0) / np.sqrt(len(good_channels))
        
        # Plot
        color = colors.get(event_type, 'black')
        ax.plot(epoch_time_vec, grand_avg, color=color, linewidth=2, 
                label=f'{event_type} (n={len(epochs)})')
        ax.fill_between(epoch_time_vec, grand_avg - sem, grand_avg + sem, 
                       color=color, alpha=0.3)
    
    # Formatting
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.set_title('Grand Average ERPs - Corrected Scaling\n(Average of channels 3, 4, 7, 8)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim([-50, 50])
    
    plt.tight_layout()
    plt.savefig('test_data/grand_average_corrected.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: test_data/grand_average_corrected.png")

def main():
    """Main analysis with corrected scaling"""
    
    print("Loading and scaling EEG data...")
    print(f"Applying scaling factor: ÷{GAIN_FACTOR} (removing hardware gain)")
    
    # Load both raw and scaled data
    test_data_dir = "test_data"
    brainflow_files = glob.glob(os.path.join(test_data_dir, "BrainFlow-RAW*.csv"))
    brainflow_file = sorted(brainflow_files)[-1]
    
    df = pd.read_csv(brainflow_file, sep='\t', header=None)
    global eeg_data_raw
    eeg_data_raw = df.iloc[:, 1:17].values
    
    # Get scaled data
    eeg_data_scaled, time_vec, fs = load_and_scale_data()
    
    print(f"\nData shape: {eeg_data_scaled.shape}")
    print(f"Duration: {len(eeg_data_scaled)/fs:.1f} seconds")
    
    # Check new amplitude ranges
    print("\nCorrected amplitude ranges (µV):")
    for ch in range(min(8, eeg_data_scaled.shape[1])):
        ch_data = eeg_data_scaled[:, ch]
        ch_data = ch_data[~np.isnan(ch_data)]
        if len(ch_data) > 0:
            print(f"  Channel {ch+1}: {np.ptp(ch_data):.1f} µV")
    
    # Generate all plots
    print("\nGenerating corrected plots...")
    
    # 1. All channels view
    plot_scaled_raw_data(eeg_data_scaled, time_vec, fs, window_start=5, window_duration=10)
    
    # 2. Clinical montage
    plot_clinical_montage(eeg_data_scaled, time_vec, fs, window_start=10, window_duration=10)
    
    # 3. Comparison plots
    plot_power_spectrum_comparison(eeg_data_scaled, eeg_data_raw, fs)
    
    # 4. Grand averages with correct scaling
    try:
        plot_grand_averages_corrected()
    except Exception as e:
        print(f"Could not create grand averages: {e}")
    
    print("\nAll corrected plots generated successfully!")

if __name__ == "__main__":
    main()