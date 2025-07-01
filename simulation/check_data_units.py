#!/usr/bin/env python
"""
Check what units the EEG data is in based on OpenBCI vertical scale
"""

import numpy as np
import pandas as pd
import glob
import os

def check_units():
    """Determine the units of the recorded data"""
    
    # Load latest data
    test_data_dir = "test_data"
    brainflow_files = glob.glob(os.path.join(test_data_dir, "BrainFlow-RAW*.csv"))
    
    if not brainflow_files:
        print("No BrainFlow files found!")
        return
        
    brainflow_file = sorted(brainflow_files)[-1]
    print(f"Analyzing: {brainflow_file}")
    
    # Load data
    df = pd.read_csv(brainflow_file, sep='\t', header=None)
    eeg_data = df.iloc[:, 1:17].values  # 16 EEG channels
    
    # Get a clean channel (avoid saturated ones)
    ch_idx = 2  # Channel 3
    ch_data = eeg_data[:1000, ch_idx]  # First 1000 samples
    ch_data = ch_data[~np.isnan(ch_data)]
    
    if len(ch_data) == 0:
        print("No valid data found!")
        return
    
    # Calculate statistics
    data_range = np.ptp(ch_data)
    data_std = np.std(ch_data)
    data_mean = np.mean(ch_data)
    
    print("\n" + "="*60)
    print("DATA UNIT ANALYSIS")
    print("="*60)
    print(f"\nChannel 3 statistics (first 1000 samples):")
    print(f"Range: {data_range:.1f}")
    print(f"Std Dev: {data_std:.1f}")
    print(f"Mean: {data_mean:.1f}")
    
    print("\n" + "="*60)
    print("UNIT DETERMINATION")
    print("="*60)
    
    # OpenBCI conversion factors
    ADS1299_VREF = 4.5  # Reference voltage
    ADS1299_GAIN = 24   # Assuming 24x gain
    SCALE_FACTOR_COUNTS_TO_VOLTS = ADS1299_VREF / (2**23 - 1) / ADS1299_GAIN
    
    print(f"\nIf vertical scale in GUI = 200 µV:")
    print("This means each channel shows ±200 µV vertically")
    
    print("\nChecking different unit hypotheses:")
    
    # Hypothesis 1: Raw counts
    if data_range > 10000:
        uV_from_counts = data_range * SCALE_FACTOR_COUNTS_TO_VOLTS * 1e6
        print(f"\n1. If data is in COUNTS:")
        print(f"   {data_range:.0f} counts → {uV_from_counts:.1f} µV")
        print(f"   This would be {uV_from_counts/200:.1f}x your GUI scale")
        
    # Hypothesis 2: Microvolts (but gain not compensated)
    print(f"\n2. If data is in µV (but includes 24x gain):")
    print(f"   {data_range:.0f} µV recorded → {data_range/24:.1f} µV actual")
    print(f"   This would be {data_range/24/200:.1f}x your GUI scale")
    
    # Hypothesis 3: Already in microvolts
    print(f"\n3. If data is already in proper µV:")
    print(f"   {data_range:.0f} µV is {data_range/200:.1f}x your GUI scale")
    print("   This seems too large for properly scaled EEG")
    
    # Most likely scenario
    print("\n" + "="*60)
    print("MOST LIKELY SCENARIO")
    print("="*60)
    
    if data_range > 1000:
        if data_range > 10000:
            print("\nYour data is most likely in:")
            print("→ RAW ADC COUNTS or µV WITHOUT GAIN COMPENSATION")
            print(f"\nTo convert to proper µV:")
            print(f"- If counts: multiply by {SCALE_FACTOR_COUNTS_TO_VOLTS * 1e6:.4f}")
            print(f"- If µV with gain: divide by {ADS1299_GAIN}")
            
            # Show what the signal would look like
            counts_to_uV = data_range * SCALE_FACTOR_COUNTS_TO_VOLTS * 1e6
            gain_corrected_uV = data_range / ADS1299_GAIN
            
            print(f"\nYour {data_range:.0f} units would become:")
            print(f"- As counts → {counts_to_uV:.1f} µV")
            print(f"- As µV/gain → {gain_corrected_uV:.1f} µV ← Most likely!")
        else:
            print("\nYour data might already be in µV")
            print("But the amplitude is still higher than typical EEG")
    
    # Visual check
    print("\n" + "="*60)
    print("VISUAL SCALE CHECK")
    print("="*60)
    print("\nIn OpenBCI GUI with 200 µV scale:")
    print("- Clean EEG should fill about 1/4 to 1/2 of channel height")
    print("- Eye blinks might fill the full height")
    print("- If signal is clipping at top/bottom → scale is too small")
    print("- If signal is a thin line → scale is too large")
    
    print("\nTo properly export data:")
    print("1. In OpenBCI GUI → Settings → Data Format")
    print("2. Choose 'Volts' for properly scaled data")
    print("3. Or note that 'Counts' requires manual conversion")

if __name__ == "__main__":
    check_units()