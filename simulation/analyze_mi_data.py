import pandas as pd
import numpy as np
import os
from pathlib import Path

# Define the path to MI data
mi_path = Path("/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI/EEG data")

# Analyze T-005 Session 1 data
session1_path = mi_path / "T-005" / "Session 1"
csv_files = list(session1_path.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files in T-005 Session 1")

# Read the first CSV file
if csv_files:
    df = pd.read_csv(csv_files[0])
    print(f"\nAnalyzing: {csv_files[0].name}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Check sampling rate from the Time column
    if 'Time:512Hz' in df.columns:
        time_col = df['Time:512Hz']
        print(f"\nSampling rate confirmed: 512 Hz")
        print(f"Time range: {time_col.min():.4f} to {time_col.max():.4f} seconds")
        print(f"Duration: {time_col.max() - time_col.min():.2f} seconds")
    
    # Count channels
    channel_cols = [col for col in df.columns if 'Channel' in col]
    print(f"\nNumber of channels: {len(channel_cols)}")
    
    # Analyze Event IDs
    if 'Event Id' in df.columns:
        events = df['Event Id'].dropna()
        print(f"\nNumber of events: {len(events)}")
        print(f"Unique event IDs: {sorted(events.unique().astype(int))}")
        
        # Count occurrences of each event
        event_counts = events.value_counts().sort_index()
        print("\nEvent counts:")
        for event_id, count in event_counts.items():
            print(f"  Event {int(event_id)}: {count} occurrences")
    
    # Analyze epochs
    if 'Epoch' in df.columns:
        epochs = df['Epoch']
        print(f"\nEpoch range: {epochs.min()} to {epochs.max()}")
        epoch_changes = df[epochs.diff() != 0].index.tolist()
        print(f"Number of epoch changes: {len(epoch_changes)}")

# Analyze multiple sessions
print("\n" + "="*60)
print("ANALYZING ALL PARTICIPANTS AND SESSIONS")
print("="*60)

participants = ['T-005', 'T-008']
for participant in participants:
    part_path = mi_path / participant
    if part_path.exists():
        print(f"\n{participant}:")
        
        # Check sessions
        for session_type in ['Session 1', 'Session 2', 'Test']:
            session_path = part_path / session_type
            if session_path.exists():
                csv_files = list(session_path.glob("*.csv"))
                print(f"  {session_type}: {len(csv_files)} files")
                
                # Check report files
                report_files = list(session_path.glob("*report*.csv"))
                if report_files:
                    print(f"    Report files: {[f.name for f in report_files]}")

# Analyze report file structure
print("\n" + "="*60)
print("ANALYZING REPORT FILE STRUCTURE")
print("="*60)

report_path = mi_path / "T-005" / "Test" / "T-005_2D_High_report_file.csv"
if report_path.exists():
    report_df = pd.read_csv(report_path)
    print(f"Report shape: {report_df.shape}")
    print(f"Report columns: {list(report_df.columns)}")
    print(f"\nFirst few rows:")
    print(report_df.head(10))
    
    # Analyze the values
    print(f"\nUnique Achieved Values: {sorted(report_df['Achived Value'].unique())}")
    print(f"Unique Aiming Values: {sorted(report_df['Aiming Value'].unique())}")