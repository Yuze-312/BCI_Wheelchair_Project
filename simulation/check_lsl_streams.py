#!/usr/bin/env python
"""Check available LSL streams"""

from pylsl import resolve_streams
import time

print("Searching for LSL streams...")
print("="*60)

# Search for streams
streams = resolve_streams(wait_time=2.0)

if not streams:
    print("No LSL streams found!")
    print("\nTroubleshooting:")
    print("1. Is OpenBCI GUI running?")
    print("2. Is LSL streaming enabled in OpenBCI?")
    print("3. Check firewall settings")
else:
    print(f"Found {len(streams)} streams:\n")
    
    for i, stream in enumerate(streams):
        print(f"Stream {i+1}:")
        print(f"  Name: {stream.name()}")
        print(f"  Type: {stream.type()}")
        print(f"  Channels: {stream.channel_count()}")
        print(f"  Sample Rate: {stream.nominal_srate()} Hz")
        print(f"  Source ID: {stream.source_id()}")
        print()

# Check specifically for marker streams
marker_streams = [s for s in streams if s.type() == 'Markers']
print("\nMarker Streams:")
if marker_streams:
    for stream in marker_streams:
        print(f"  ✓ {stream.name()}")
else:
    print("  ✗ No marker streams found!")
    
# Check for the specific game marker stream
game_markers = [s for s in streams if 'SubwaySurfers_ErrP_Markers' in s.name()]
if game_markers:
    print("\n✓ Game marker stream found!")
else:
    print("\n✗ Game marker stream NOT found!")
    print("  Looking for stream containing: 'SubwaySurfers_ErrP_Markers'")

# Keep checking
print("\nContinuously monitoring (Ctrl+C to stop)...")
while True:
    streams = resolve_streams(wait_time=0.5)
    eeg_streams = [s for s in streams if s.type() == 'EEG' or 'eeg' in s.name().lower()]
    marker_streams = [s for s in streams if s.type() == 'Markers']
    print(f"\r[{time.strftime('%H:%M:%S')}] Streams: {len(streams)} total, {len(eeg_streams)} EEG, {len(marker_streams)} Markers", end='', flush=True)
    time.sleep(1)