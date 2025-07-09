"""
ErrP Data Loader Module

Handles loading EEG data and event logs from the BCI wheelchair experiments.
Supports both OpenBCI and BrainFlow data formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPDataLoader:
    """Load and parse EEG data with associated event logs for ErrP analysis"""
    
    def __init__(self, sampling_rate: int = 512):
        """
        Initialize the ErrP data loader
        
        Args:
            sampling_rate: EEG sampling frequency in Hz (default: 512)
        """
        self.sampling_rate = sampling_rate
        self.channel_names = [f"Channel_{i}" for i in range(1, 17)]  # 16 channels
        
    def load_session(self, session_path: str) -> Dict:
        """
        Load a complete session including EEG data and event logs
        
        Args:
            session_path: Path to the session directory
            
        Returns:
            Dictionary containing:
                - eeg_data: DataFrame with EEG data
                - events: DataFrame with event markers
                - metadata: Session information
        """
        session_path = Path(session_path)
        
        if not session_path.exists():
            raise ValueError(f"Session path not found: {session_path}")
            
        logger.info(f"Loading session from: {session_path}")
        
        # Find EEG data file
        eeg_file = self._find_eeg_file(session_path)
        if not eeg_file:
            raise ValueError(f"No EEG data file found in: {session_path}")
            
        # Find event log file
        event_file = self._find_event_file(session_path)
        if not event_file:
            raise ValueError(f"No event log file found in: {session_path}")
            
        # Load EEG data
        eeg_data = self._load_eeg_data(eeg_file)
        
        # Load event log
        events = self._load_event_log(event_file)
        
        # Align timestamps
        aligned_data = self._align_timestamps(eeg_data, events)
        
        return {
            'eeg_data': aligned_data['eeg_data'],
            'events': aligned_data['events'],
            'metadata': {
                'session_path': str(session_path),
                'eeg_file': str(eeg_file),
                'event_file': str(event_file),
                'sampling_rate': self.sampling_rate,
                'duration': len(eeg_data) / self.sampling_rate,
                'n_channels': len(self.channel_names)
            }
        }
    
    def _find_eeg_file(self, session_path: Path) -> Optional[Path]:
        """Find EEG data file in session directory"""
        # Check for different file patterns in order of preference
        patterns = [
            "data_1.csv",       # Phase 1 data file
            "data_*.csv",       # Other data files
            "OpenBCI*.txt",     # OpenBCI format
            "BrainFlow*.csv",   # BrainFlow format
            "*.csv"             # Generic CSV (last resort)
        ]
        
        for pattern in patterns:
            files = list(session_path.glob(pattern))
            # Filter out event logs and other non-EEG files
            files = [f for f in files if 'event' not in f.name.lower() 
                    and 'errp' not in f.name.lower()
                    and 'phase1' not in f.name.lower()
                    and 'subway' not in f.name.lower()]
            if files:
                # If multiple files, prefer data_1.csv
                for f in files:
                    if f.name == 'data_1.csv':
                        return f
                return files[0]  # Return first matching file
                
        return None
    
    def _find_event_file(self, session_path: Path) -> Optional[Path]:
        """Find event log file in session directory"""
        # Check for different event log patterns
        patterns = [
            "phase1_events_*.csv",
            "subway_errp_*.csv",
            "*_events_*.csv",
            "*event*.csv"
        ]
        
        for pattern in patterns:
            files = list(session_path.glob(pattern))
            if files:
                return files[0]  # Return first matching file
                
        return None
    
    def _load_eeg_data(self, file_path: Path) -> pd.DataFrame:
        """Load EEG data from file"""
        logger.info(f"Loading EEG data from: {file_path}")
        
        # Detect file format
        if 'OpenBCI' in file_path.name:
            return self._load_openbci_data(file_path)
        elif 'BrainFlow' in file_path.name or file_path.suffix == '.csv':
            return self._load_csv_data(file_path)
        else:
            raise ValueError(f"Unknown file format: {file_path}")
    
    def _load_csv_data(self, file_path: Path) -> pd.DataFrame:
        """Load CSV format EEG data"""
        df = pd.read_csv(file_path)
        
        # Check if it's OpenViBE format with Oscillator columns
        if 'Oscillator 1' in df.columns:
            oscillator_cols = [col for col in df.columns if col.startswith('Oscillator')]
            channel_mapping = {f'Oscillator {i}': f'Channel_{i}' 
                             for i in range(1, len(oscillator_cols)+1)}
            df = df.rename(columns=channel_mapping)
            self.channel_names = [f"Channel_{i}" for i in range(1, len(oscillator_cols)+1)]
        
        # Add timestamp column if not present
        if 'timestamp' not in df.columns:
            # Create timestamps based on sampling rate
            df['timestamp'] = np.arange(len(df)) / self.sampling_rate
            
        return df
    
    def _load_openbci_data(self, file_path: Path) -> pd.DataFrame:
        """Load OpenBCI format data"""
        # OpenBCI format has header lines starting with %
        data_lines = []
        with open(file_path, 'r') as f:
            for line in f:
                if not line.startswith('%'):
                    data_lines.append(line.strip())
        
        # Parse data
        # Typical OpenBCI format: sample_index, channel_data..., timestamp, other
        # We'll need to adapt based on actual format
        raise NotImplementedError("OpenBCI format loader not yet implemented")
    
    def _load_event_log(self, file_path: Path) -> pd.DataFrame:
        """Load event log file"""
        logger.info(f"Loading event log from: {file_path}")
        
        events = pd.read_csv(file_path)
        
        # Log the columns found
        logger.info(f"Event log columns: {list(events.columns)}")
        
        # Check for different possible event column names
        event_col = None
        for possible_name in ['event', 'Event', 'event_type', 'marker', 'Marker']:
            if possible_name in events.columns:
                event_col = possible_name
                break
        
        if event_col is None:
            logger.error(f"No event column found. Available columns: {list(events.columns)}")
            raise ValueError(f"Event log missing event column. Found columns: {list(events.columns)}")
        
        # Rename to standard 'event' column
        if event_col != 'event':
            events = events.rename(columns={event_col: 'event'})
        
        # Ensure timestamp column exists
        if 'timestamp' not in events.columns:
            # Try other common names
            for possible_ts in ['Timestamp', 'time', 'Time']:
                if possible_ts in events.columns:
                    events = events.rename(columns={possible_ts: 'timestamp'})
                    break
        
        required_cols = ['timestamp', 'event']
        for col in required_cols:
            if col not in events.columns:
                raise ValueError(f"Event log missing required column: {col}")
        
        # Log unique event types
        unique_events = events['event'].unique()
        logger.info(f"Unique event types found: {sorted(unique_events)}")
        logger.info(f"Event counts: {events['event'].value_counts().to_dict()}")
                
        return events
    
    def _align_timestamps(self, eeg_data: pd.DataFrame, events: pd.DataFrame) -> Dict:
        """
        Align EEG data timestamps with event log timestamps
        
        Returns:
            Dictionary with aligned eeg_data and events
        """
        # Get time ranges
        eeg_start = eeg_data['timestamp'].min()
        eeg_end = eeg_data['timestamp'].max()
        event_start = events['timestamp'].min()
        event_end = events['timestamp'].max()
        
        logger.info(f"EEG time range: {eeg_start:.2f} - {eeg_end:.2f}")
        logger.info(f"Event time range: {event_start:.2f} - {event_end:.2f}")
        
        # Calculate offset if needed
        # Assuming events timestamp are absolute and EEG starts at 0
        time_offset = event_start - eeg_start
        
        # Add sample indices to events
        # For Phase 1 data, timestamps are relative to the start of recording
        if event_start == 0:
            # Timestamps are already relative to recording start
            events['sample_idx'] = (events['timestamp'] * self.sampling_rate).astype(int)
        else:
            # Adjust for any offset
            events['sample_idx'] = ((events['timestamp'] - event_start) * self.sampling_rate).astype(int)
        
        # Validate sample indices
        valid_mask = (events['sample_idx'] >= 0) & (events['sample_idx'] < len(eeg_data))
        if not valid_mask.all():
            logger.warning(f"Dropping {(~valid_mask).sum()} events outside EEG recording")
            events = events[valid_mask].copy()
        
        return {
            'eeg_data': eeg_data,
            'events': events
        }
    
    def get_error_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract error events from the event log
        
        Args:
            events: Event log DataFrame
            
        Returns:
            DataFrame containing only error events
        """
        # Based on simple_markers.py:
        # MARKER_RESPONSE_ERROR = 6
        # MARKER_NATURAL_ERROR = 12 (classifier wrong, kept as error)
        # MARKER_FORCED_ERROR = 13 (classifier correct, forced to error)
        
        error_events = events[events['event'].isin([6, 12, 13])].copy()
        
        logger.info(f"Found {len(error_events)} error events")
        if len(error_events) > 0:
            error_breakdown = events['event'].value_counts()
            for event_type in [6, 12, 13]:
                if event_type in error_breakdown.index:
                    logger.info(f"  Event {event_type}: {error_breakdown[event_type]} occurrences")
        
        return error_events
    
    def get_correct_events(self, events: pd.DataFrame) -> pd.DataFrame:
        """
        Extract correct feedback events for comparison
        
        Args:
            events: Event log DataFrame
            
        Returns:
            DataFrame containing only correct feedback events
        """
        # Based on simple_markers.py:
        # MARKER_RESPONSE_CORRECT = 5
        # MARKER_NATURAL_CORRECT = 10 (classifier correct, kept as is)
        # MARKER_FORCED_CORRECT = 11 (classifier wrong, forced to GT)
        
        correct_events = events[events['event'].isin([5, 10, 11])].copy()
        
        logger.info(f"Found {len(correct_events)} correct events")
        if len(correct_events) > 0:
            correct_breakdown = events['event'].value_counts()
            for event_type in [5, 10, 11]:
                if event_type in correct_breakdown.index:
                    logger.info(f"  Event {event_type}: {correct_breakdown[event_type]} occurrences")
        
        return correct_events