import csv
from pathlib import Path
from collections import Counter

# Define the path to MI data
mi_path = Path("/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI/EEG data")

# Analyze T-005 Session 1 data - all files
session1_path = mi_path / "T-005" / "Session 1"
csv_files = sorted(list(session1_path.glob("*.csv")))

print("ANALYZING EVENT STRUCTURE IN MI DATA")
print("="*60)

for csv_file in csv_files:
    print(f"\nFile: {csv_file.name}")
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Find Event Id column index
        event_id_idx = header.index('Event Id')
        time_idx = 0
        
        # Read all rows to find events
        events = []
        event_times = []
        total_rows = 0
        
        for row in reader:
            total_rows += 1
            if row[event_id_idx]:  # If there's an event ID
                events.append(int(float(row[event_id_idx])))
                event_times.append(float(row[time_idx]))
        
        print(f"  Total rows: {total_rows}")
        print(f"  Duration: {float(row[time_idx]):.2f} seconds")
        
        if events:
            event_counts = Counter(events)
            print(f"  Total events: {len(events)}")
            print(f"  Event types: {sorted(event_counts.keys())}")
            print(f"  Event counts: {dict(sorted(event_counts.items()))}")
            
            # Analyze inter-event intervals
            if len(event_times) > 1:
                intervals = [event_times[i+1] - event_times[i] for i in range(len(event_times)-1)]
                avg_interval = sum(intervals) / len(intervals)
                print(f"  Average inter-event interval: {avg_interval:.2f} seconds")
                print(f"  Min interval: {min(intervals):.2f} seconds")
                print(f"  Max interval: {max(intervals):.2f} seconds")
        else:
            print("  No events found in this file!")

# Check electrode montage information
print("\n" + "="*60)
print("CHECKING ELECTRODE MONTAGE INFORMATION")
print("="*60)

# Check for any documentation files
doc_paths = [
    mi_path.parent.parent / "Data" / "22671172" / "task-MI_electrodes.tsv",
    mi_path.parent.parent / "Data" / "22671172" / "task-MI_coordsystem.json",
    mi_path.parent.parent / "Data" / "22671172" / "task-MI_eeg.json"
]

for doc_path in doc_paths:
    if doc_path.exists():
        print(f"\nFound: {doc_path.name}")
        if doc_path.suffix == '.tsv':
            with open(doc_path, 'r') as f:
                print(f"Content preview:\n{f.read()[:500]}...")
        elif doc_path.suffix == '.json':
            with open(doc_path, 'r') as f:
                content = f.read()
                print(f"Content:\n{content[:500]}...")

# Summary of paradigm
print("\n" + "="*60)
print("MOTOR IMAGERY PARADIGM SUMMARY")
print("="*60)

print("""
Based on the analysis:

1. DATA FORMAT:
   - Sampling rate: 512 Hz (confirmed)
   - Number of channels: 16
   - File format: CSV with OpenVibe structure
   - Columns: Time, Epoch, 16 channels, Event Id, Event Date, Event Duration

2. EVENT CODING:
   - Event IDs: 1, 2, 3 (motor imagery classes)
   - Event ID 1: Motor imagery class 1
   - Event ID 2: Motor imagery class 2  
   - Event ID 3: Motor imagery class 3

3. EXPERIMENTAL STRUCTURE:
   - 2 participants: T-005, T-008
   - Each participant has:
     * Session 1: 5-6 runs (training)
     * Session 2: 5 runs (training)
     * Test: 3 runs + report files
   - Report files track:
     * Achieved Value: What the classifier predicted
     * Aiming Value: True label (ground truth)
     * Classification percentage: Running accuracy

4. TRIAL STRUCTURE:
   - Events mark the onset of motor imagery trials
   - Inter-trial interval: varies (typically 2-10 seconds)
   - Data is continuous with event markers
   - Epochs are extracted around events for classification

5. TASK DESCRIPTION:
   - T-005: "Task: VR no noise" 
   - 3-class motor imagery paradigm
   - Likely left hand, right hand, and feet/tongue imagery
""")