import csv
import os
from pathlib import Path
from collections import Counter

# Define the path to MI data
mi_path = Path("/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI/EEG data")

# Analyze T-005 Session 1 data
session1_path = mi_path / "T-005" / "Session 1"
csv_files = list(session1_path.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files in T-005 Session 1")

# Read the first CSV file
if csv_files:
    with open(csv_files[0], 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        print(f"\nAnalyzing: {csv_files[0].name}")
        print(f"\nColumns ({len(header)}): {header}")
        
        # Count channels
        channel_cols = [col for col in header if 'Channel' in col]
        print(f"\nNumber of channels: {len(channel_cols)}")
        
        # Find Event Id column index
        event_id_idx = header.index('Event Id') if 'Event Id' in header else None
        epoch_idx = header.index('Epoch') if 'Epoch' in header else None
        time_idx = 0  # First column is time
        
        # Read data
        events = []
        epochs = []
        row_count = 0
        time_values = []
        
        for row in reader:
            row_count += 1
            if row_count <= 1000:  # Read first 1000 rows for analysis
                time_values.append(float(row[time_idx]))
                
                if event_id_idx is not None and row[event_id_idx]:
                    events.append(int(float(row[event_id_idx])))
                    
                if epoch_idx is not None:
                    epochs.append(int(row[epoch_idx]))
        
        print(f"\nRows analyzed: {row_count}")
        
        # Analyze time
        if time_values:
            print(f"\nTime analysis (first 1000 rows):")
            print(f"  Start time: {time_values[0]:.4f} seconds")
            print(f"  End time: {time_values[-1]:.4f} seconds")
            if len(time_values) > 1:
                time_diff = time_values[1] - time_values[0]
                print(f"  Time step: {time_diff:.6f} seconds")
                print(f"  Calculated sampling rate: {1/time_diff:.1f} Hz")
        
        # Analyze events
        if events:
            event_counts = Counter(events)
            print(f"\nEvent analysis:")
            print(f"  Total events found: {len(events)}")
            print(f"  Unique event IDs: {sorted(event_counts.keys())}")
            print(f"  Event counts:")
            for event_id, count in sorted(event_counts.items()):
                print(f"    Event {event_id}: {count} occurrences")
        
        # Analyze epochs
        if epochs:
            unique_epochs = sorted(set(epochs))
            print(f"\nEpoch analysis:")
            print(f"  Unique epochs: {unique_epochs[:10]}...")  # First 10

# Analyze all participants and sessions
print("\n" + "="*60)
print("ANALYZING ALL PARTICIPANTS AND SESSIONS")
print("="*60)

participants = ['T-005', 'T-008']
for participant in participants:
    part_path = mi_path / participant
    if part_path.exists():
        print(f"\n{participant}:")
        
        # Check Info.txt
        info_path = part_path / "Info.txt"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = f.read().strip()
                print(f"  Info: {info}")
        
        # Check sessions
        for session_type in ['Session 1', 'Session 2', 'Test']:
            session_path = part_path / session_type
            if session_path.exists():
                csv_files = list(session_path.glob("*.csv"))
                print(f"  {session_type}: {len(csv_files)} files")
                
                # List file names
                for csv_file in csv_files[:3]:  # First 3 files
                    print(f"    - {csv_file.name}")
                
                # Check report files
                report_files = list(session_path.glob("*report*.csv"))
                if report_files:
                    print(f"    Report files:")
                    for rf in report_files:
                        print(f"      - {rf.name}")

# Analyze report file
print("\n" + "="*60)
print("ANALYZING REPORT FILE STRUCTURE")
print("="*60)

report_path = mi_path / "T-005" / "Test" / "T-005_2D_High_report_file.csv"
if report_path.exists():
    with open(report_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        print(f"Report rows: {len(rows)}")
        print(f"Report columns: {list(rows[0].keys())}")
        
        # Analyze values
        achieved_values = [float(row['Achived Value']) for row in rows]
        aiming_values = [float(row['Aiming Value']) for row in rows]
        
        print(f"\nUnique Achieved Values: {sorted(set(achieved_values))}")
        print(f"Unique Aiming Values: {sorted(set(aiming_values))}")
        
        # Count trials per class
        aiming_counts = Counter(aiming_values)
        print(f"\nTrials per aiming value:")
        for val, count in sorted(aiming_counts.items()):
            print(f"  Class {int(val)}: {count} trials")
        
        print(f"\nFirst 10 rows:")
        for i, row in enumerate(rows[:10]):
            print(f"  Row {i+1}: Achieved={row['Achived Value']}, "
                  f"Aiming={row['Aiming Value']}, "
                  f"Classification={row['Classification percentage']}")

# Check OpenVibe script
print("\n" + "="*60)
print("CHECKING OPENVIBE SCRIPT")
print("="*60)

openvibe_path = mi_path.parent / "OpenVibe_marked_CSV.py"
if openvibe_path.exists():
    print(f"OpenVibe script found at: {openvibe_path}")
    print("Key observations from the script:")
    print("- Default epoch_length: 1 second")
    print("- Default overlap: 0.5 (50%)")
    print("- Default frequency: 512 Hz")
    print("- Extracts epochs between event markers")
    print("- Groups data by Event ID (class labels)")
else:
    print("OpenVibe script not found")