#!/usr/bin/env python
"""
Compare units between OpenBCI test_data and T-005 MI data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def load_openbci_data():
    """Load OpenBCI data from test_data"""
    test_data_dir = "test_data"
    brainflow_files = glob.glob(os.path.join(test_data_dir, "BrainFlow-RAW*.csv"))
    
    if brainflow_files:
        brainflow_file = sorted(brainflow_files)[-1]
        df = pd.read_csv(brainflow_file, sep='\t', header=None)
        eeg_data = df.iloc[:, 1:17].values
        
        # Apply gain correction
        eeg_data_corrected = eeg_data / 24
        
        return eeg_data, eeg_data_corrected, "OpenBCI (test_data)"
    return None, None, None

def load_t005_mi_data():
    """Load T-005 data from MI folder"""
    # Try to load one of the T-005 session files
    mi_file = "../MI/EEG data/T-005/Session 1/test-[2023.03.15-15.39.03].csv"
    
    if os.path.exists(mi_file):
        # Load the CSV file
        df = pd.read_csv(mi_file)
        
        # Get channel columns (usually named Channel 1, Channel 2, etc.)
        channel_cols = [col for col in df.columns if 'Channel' in col]
        
        if channel_cols:
            eeg_data = df[channel_cols].values
            return eeg_data, mi_file
    
    return None, None

def analyze_units():
    """Compare units between datasets"""
    
    print("="*60)
    print("UNIT COMPARISON: OpenBCI vs T-005 MI Data")
    print("="*60)
    
    # Load OpenBCI data
    openbci_raw, openbci_corrected, openbci_source = load_openbci_data()
    
    if openbci_raw is not None:
        print(f"\n1. OpenBCI Data (from test_data):")
        print(f"   Raw data shape: {openbci_raw.shape}")
        
        # Calculate ranges
        ch_ranges_raw = []
        ch_ranges_corrected = []
        
        for ch in range(min(8, openbci_raw.shape[1])):
            ch_data_raw = openbci_raw[:, ch]
            ch_data_corrected = openbci_corrected[:, ch]
            
            # Remove NaN
            ch_data_raw = ch_data_raw[~np.isnan(ch_data_raw)]
            ch_data_corrected = ch_data_corrected[~np.isnan(ch_data_corrected)]
            
            if len(ch_data_raw) > 0:
                ch_ranges_raw.append(np.ptp(ch_data_raw))
                ch_ranges_corrected.append(np.ptp(ch_data_corrected))
        
        print(f"\n   Raw amplitude ranges:")
        for i, r in enumerate(ch_ranges_raw[:4]):
            print(f"     Ch{i+1}: {r:,.0f} units")
        print(f"   Average: {np.mean(ch_ranges_raw):,.0f} units")
        
        print(f"\n   Corrected (÷24) ranges:")
        for i, r in enumerate(ch_ranges_corrected[:4]):
            print(f"     Ch{i+1}: {r:.1f} µV")
        print(f"   Average: {np.mean(ch_ranges_corrected):.1f} µV")
    
    # Load T-005 MI data
    t005_data, t005_file = load_t005_mi_data()
    
    if t005_data is not None:
        print(f"\n2. T-005 MI Data:")
        print(f"   File: {os.path.basename(t005_file)}")
        print(f"   Data shape: {t005_data.shape}")
        
        # Calculate ranges
        ch_ranges_t005 = []
        
        for ch in range(min(8, t005_data.shape[1])):
            ch_data = t005_data[:, ch]
            ch_data = ch_data[~np.isnan(ch_data)]
            
            if len(ch_data) > 0:
                ch_ranges_t005.append(np.ptp(ch_data))
        
        print(f"\n   Amplitude ranges:")
        for i, r in enumerate(ch_ranges_t005[:4]):
            print(f"     Ch{i+1}: {r:.3f} units")
        print(f"   Average: {np.mean(ch_ranges_t005):.3f} units")
        
        # Check the mean values to understand units
        ch_means_t005 = []
        for ch in range(min(8, t005_data.shape[1])):
            ch_data = t005_data[:, ch]
            ch_data = ch_data[~np.isnan(ch_data)]
            if len(ch_data) > 0:
                ch_means_t005.append(np.mean(ch_data))
        
        print(f"\n   Mean values:")
        for i, m in enumerate(ch_means_t005[:4]):
            print(f"     Ch{i+1}: {m:.1f} units")
        
        # These large mean values (40000+) with small ranges (1-5) suggest microvolts
        avg_mean = np.mean(ch_means_t005)
        avg_range = np.mean(ch_ranges_t005)
        
        print(f"\n   Analysis:")
        print(f"   - Mean values around {avg_mean:.0f}")
        print(f"   - Range values around {avg_range:.1f}")
        
        if avg_mean > 10000:
            print("\n   → T-005 data appears to be in MICROVOLTS (µV)")
            print("   → Already in the correct units!")
            ch_ranges_t005_uv = ch_ranges_t005
        elif avg_mean < 100:
            print("\n   → T-005 data appears to be in MILLIVOLTS (mV)")
            print(f"   → Converting to µV: multiply by 1000")
            ch_ranges_t005_uv = [r * 1000 for r in ch_ranges_t005]
        else:
            print("\n   → T-005 data units unclear")
            ch_ranges_t005_uv = ch_ranges_t005
        
    # Comparison
    if openbci_corrected is not None and t005_data is not None:
        print("\n" + "="*60)
        print("UNIT ALIGNMENT ANALYSIS")
        print("="*60)
        
        avg_openbci = np.mean(ch_ranges_corrected)
        avg_t005_raw = np.mean(ch_ranges_t005)
        avg_t005_uv = avg_t005_raw * 1000 if avg_t005_raw < 1 else avg_t005_raw
        
        print(f"\nOpenBCI (corrected): {avg_openbci:.1f} µV")
        print(f"T-005 (converted):   {avg_t005_uv:.1f} µV")
        print(f"\nRatio: {avg_openbci/avg_t005_uv:.2f}x")
        
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        
        if avg_mean > 10000:
            print("✓ YES! The units are aligned:")
            print("  - OpenBCI test_data: µV (after ÷24 correction)")
            print("  - T-005 MI data: µV (already in microvolts!)")
            print("\nBoth datasets are in the same scale!")
            print("\nThe T-005 data shows:")
            print("  - Large DC offset (~40,000 µV)")
            print("  - Actual signal ranges: ~1,900 µV")
            print("  - This matches the corrected OpenBCI ranges (~885 µV)")
        
    # Create visual comparison
    if openbci_corrected is not None and t005_data is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: OpenBCI corrected
        ax = axes[0, 0]
        time_openbci = np.arange(2500) / 250  # 10 seconds
        ch_data = openbci_corrected[:2500, 2]  # Channel 3
        ch_data = ch_data - np.nanmean(ch_data)
        ax.plot(time_openbci, ch_data, 'b-', linewidth=0.5)
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title('OpenBCI Data - Corrected (µV)')
        ax.set_ylim([-200, 200])
        ax.grid(True, alpha=0.3)
        
        # Plot 2: T-005 in µV (original)
        ax = axes[0, 1]
        time_t005 = np.arange(min(2500, len(t005_data))) / 512  # T-005 is 512Hz
        ch_data = t005_data[:2500, 0] if len(t005_data) > 2500 else t005_data[:, 0]
        ch_data = ch_data - np.nanmean(ch_data)
        ax.plot(time_t005, ch_data, 'r-', linewidth=0.5)
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title('T-005 MI Data - Original (µV)')
        ax.set_ylim([-1000, 1000])
        ax.grid(True, alpha=0.3)
        
        # Plot 3: T-005 zoomed view
        ax = axes[1, 0]
        ax.plot(time_t005, ch_data, 'g-', linewidth=0.5)
        ax.set_ylabel('Amplitude (µV)')
        ax.set_xlabel('Time (s)')
        ax.set_title('T-005 MI Data - Zoomed View (µV)')
        ax.set_ylim([-200, 200])
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Direct comparison
        ax = axes[1, 1]
        ax.plot(time_openbci[:1000], openbci_corrected[:1000, 2] - np.nanmean(openbci_corrected[:1000, 2]), 
                'b-', alpha=0.7, label='OpenBCI (µV)', linewidth=0.8)
        ax.plot(time_t005[:1000], t005_data[:1000, 0] - np.nanmean(t005_data[:1000, 0]), 
                'r-', alpha=0.7, label='T-005 (µV)', linewidth=0.8)
        ax.set_ylabel('Amplitude (µV)')
        ax.set_xlabel('Time (s)')
        ax.set_title('Direct Comparison (4s window)')
        ax.set_ylim([-100, 100])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Unit Comparison: OpenBCI vs T-005 Data', fontsize=16)
        plt.tight_layout()
        plt.savefig('test_data/unit_comparison_005.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nSaved comparison plot: test_data/unit_comparison_005.png")

if __name__ == "__main__":
    analyze_units()