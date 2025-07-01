import csv
from pathlib import Path

# Check what Event ID 0 represents
mi_path = Path("/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI/EEG data")
csv_file = mi_path / "T-005" / "Session 1" / "test-[2023.03.15-15.39.03].csv"

print("ANALYZING EVENT ID 0")
print("="*60)

with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    
    event_id_idx = header.index('Event Id')
    time_idx = 0
    epoch_idx = header.index('Epoch')
    
    # Find sequences of events
    event_sequence = []
    prev_epoch = None
    
    for row in reader:
        if row[event_id_idx]:  # If there's an event
            event_id = int(float(row[event_id_idx]))
            time = float(row[time_idx])
            epoch = int(row[epoch_idx])
            
            event_sequence.append({
                'id': event_id,
                'time': time,
                'epoch': epoch
            })
            
            if len(event_sequence) <= 20:  # Print first 20 events
                print(f"Event {event_id} at time {time:.2f}s, epoch {epoch}")

print("\n" + "="*60)
print("EVENT PATTERN ANALYSIS")
print("="*60)

# Analyze the pattern
print("\nEvent sequence pattern:")
prev_id = None
for i, event in enumerate(event_sequence):
    if i < 30:  # First 30 events
        if prev_id is not None and event['id'] != 0:
            print(f"  {prev_id} -> {event['id']} (at {event['time']:.2f}s)")
        prev_id = event['id']

# Count transitions
transitions = {}
for i in range(len(event_sequence)-1):
    curr = event_sequence[i]['id']
    next_ev = event_sequence[i+1]['id']
    key = f"{curr}->{next_ev}"
    transitions[key] = transitions.get(key, 0) + 1

print("\nEvent transitions:")
for trans, count in sorted(transitions.items()):
    print(f"  {trans}: {count} times")

print("\n" + "="*60)
print("INTERPRETATION")
print("="*60)
print("""
Event ID 0 appears to be a REST or PREPARATION marker that occurs between motor imagery trials.

The typical sequence is:
1. Event 0 (Rest/Preparation period)
2. Event 1, 2, or 3 (Motor imagery trial)
3. Back to Event 0 (Rest)

This creates a block structure where:
- Event 0: Inter-trial interval / Rest period
- Events 1-3: Active motor imagery tasks
  * Event 1: Likely left hand imagery
  * Event 2: Likely right hand imagery  
  * Event 3: Likely feet or tongue imagery

The paradigm alternates between rest (0) and motor imagery (1-3) trials.
""")