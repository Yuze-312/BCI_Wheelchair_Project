"""
Stream Manager - Handles LSL stream connections and data buffering
"""

import numpy as np
from pylsl import StreamInlet, resolve_streams
import time
from threading import Lock


class StreamManager:
    """Manages EEG and marker streams with thread-safe buffering"""
    
    def __init__(self, phase='real'):
        """Initialize stream manager
        
        Args:
            phase: 'real' for actual EEG, 'phase1' for fake classifier
        """
        self.phase = phase
        self.eeg_inlet = None
        self.marker_inlet = None
        self.n_channels = None
        self.srate = None
        self.has_markers = False
        
        # Buffer for EEG data
        self.buffer = []
        self.buffer_lock = Lock()
        self.buffer_size = None  # Will be set based on sampling rate
        
        # Game state
        self.game_active = False
        self.data_collection_active = False
        
    def connect_to_streams(self):
        """Connect to EEG and marker streams"""
        print("Connecting to streams...")
        streams = resolve_streams()
        
        # Phase1 needs BOTH EEG and markers for data collection
        if self.phase == 'phase1':
            # Still need to verify EEG is recording!
            print("\n[PHASE1] Checking for EEG stream (OpenViBE must be recording)...")
            eeg_found = self._connect_eeg_stream(streams)
            if not eeg_found:
                print("\n" + "="*60)
                print("ERROR: No EEG stream found!")
                print("Please ensure OpenViBE is:")
                print("  1. Running and connected to the EEG device")
                print("  2. Streaming data via LSL")
                print("  3. Recording data to disk")
                print("="*60 + "\n")
                raise RuntimeError("Cannot start phase1 without EEG recording")
            self._connect_marker_stream(streams)
            return
        
        # Find EEG stream for real mode
        self._connect_eeg_stream(streams)
        self._connect_marker_stream(streams)
        
        # Initialize buffer size (5 seconds)
        if self.srate:
            self.buffer_size = int(self.srate * 5)
    
    def _connect_eeg_stream(self, streams):
        """Connect to EEG stream
        
        Returns:
            bool: True if EEG stream found and connected, False otherwise
        """
        eeg_streams = [s for s in streams if s.type() == 'EEG' or 'eeg' in s.name().lower()]
        if not eeg_streams:
            if self.phase == 'phase1':
                return False  # Return False for phase1 instead of raising
            raise RuntimeError("No EEG streams found!")
        
        # Look for preferred stream
        selected_eeg = None
        for stream in eeg_streams:
            if stream.name() == 'obci_eeg1':
                selected_eeg = stream
                print(f"Found preferred stream: {stream.name()}")
                break
        
        if not selected_eeg:
            # Test which stream has data
            print("Testing EEG streams for data...")
            for stream_info in eeg_streams:
                print(f"  Testing {stream_info.name()}...", end='', flush=True)
                test_inlet = StreamInlet(stream_info)
                chunk, _ = test_inlet.pull_chunk(timeout=0.5, max_samples=10)
                if chunk:
                    print(f" Has data! ({len(chunk)} samples)")
                    selected_eeg = stream_info
                    break
                else:
                    print(" No data")
            
            if not selected_eeg:
                # Use first 16-channel stream as fallback
                streams_16ch = [s for s in eeg_streams if s.channel_count() == 16]
                selected_eeg = streams_16ch[0] if streams_16ch else eeg_streams[0]
                print(f"\nWarning: No streams have data! Using {selected_eeg.name()}")
        
        print(f"Stream info: {selected_eeg.channel_count()} channels at {selected_eeg.nominal_srate()}Hz")
        
        self.eeg_inlet = StreamInlet(selected_eeg)
        self.n_channels = selected_eeg.channel_count()
        self.srate = selected_eeg.nominal_srate()
        
        print(f"Connected to EEG: {selected_eeg.name()}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.srate}Hz")
        return True  # Successfully connected
    
    def _connect_marker_stream(self, streams):
        """Connect to marker stream"""
        marker_streams = [s for s in streams if s.type() == 'Markers' and 'Outlet_Info' in s.name()]
        if marker_streams:
            self.marker_inlet = StreamInlet(marker_streams[0])
            print(f"Connected to game markers: {marker_streams[0].name()}")
            self.has_markers = True
        else:
            print("WARNING: No game marker stream found")
            print("   Waiting for game to start...")
            self.marker_inlet = None
            self.has_markers = False
    
    def pull_data(self):
        """Pull latest data from EEG stream"""
        if self.phase == 'phase1' or not self.eeg_inlet:
            return [], []
        
        chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.0)
        return chunk, timestamps
    
    def pull_markers(self):
        """Pull latest markers"""
        if not self.marker_inlet:
            return [], []
        
        markers, timestamps = self.marker_inlet.pull_chunk(timeout=0.0)
        return markers, timestamps
    
    def add_to_buffer(self, chunk):
        """Add data to buffer (thread-safe)"""
        if not chunk or not self.data_collection_active:
            return
        
        with self.buffer_lock:
            self.buffer.extend(chunk)
            # Keep buffer size limited
            if self.buffer_size and len(self.buffer) > self.buffer_size:
                self.buffer = self.buffer[-self.buffer_size:]
    
    def clear_buffer(self):
        """Clear the buffer"""
        with self.buffer_lock:
            self.buffer = []
    
    def get_buffer_data(self, samples=None):
        """Get data from buffer (thread-safe)
        
        Args:
            samples: Number of samples to get. If None, get all.
            
        Returns:
            numpy array of buffer data
        """
        with self.buffer_lock:
            if samples is None:
                return np.array(self.buffer) if self.buffer else np.array([])
            else:
                if len(self.buffer) >= samples:
                    return np.array(self.buffer[-samples:])
                else:
                    return np.array(self.buffer) if self.buffer else np.array([])
    
    def get_buffer_size(self):
        """Get current buffer size in samples"""
        with self.buffer_lock:
            return len(self.buffer)
    
    def try_reconnect_markers(self):
        """Try to find marker stream if not connected"""
        if self.has_markers:
            return False
        
        try:
            streams = resolve_streams()
            marker_streams = [s for s in streams if s.type() == 'Markers' and 'Outlet_Info' in s.name()]
            if marker_streams:
                self.marker_inlet = StreamInlet(marker_streams[0])
                print(f"\nGame started! Connected to markers: {marker_streams[0].name()}")
                self.has_markers = True
                return True
        except:
            pass
        
        return False