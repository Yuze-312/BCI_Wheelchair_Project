"""
ErrP Data Loader Module

Handles loading EEG data and event logs from the BCI wheelchair experiments.
Supports both OpenBCI and BrainFlow data formats.
"""
# Fix Qt platform plugin issue
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

# Set matplotlib backend to avoid Qt issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Now your regular imports
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt  # Import after setting backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy import signal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPDataLoader:
    """Load and parse EEG data with associated event logs for ErrP analysis"""
    
    def __init__(self, sampling_rate: int = 512, 
                artifact_params: Optional[Dict] = None):
        self.sampling_rate = sampling_rate
        self.channel_names = [f"Channel_{i}" for i in range(1, 17)]
        
        # Default artifact rejection parameters
        self.artifact_params = artifact_params or {
            'amplitude_threshold': 100,  # μV
            'gradient_threshold': 50,    # μV/sample
            'min_activity': 0.5,         # μV
            'eog_channels': ['Channel_1', 'Channel_2'],  # Fp1, Fp2
        }
        
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
    
    def _validate_eeg_data(self, eeg_data: pd.DataFrame) -> None:
        """Validate EEG data quality and format"""
        # Check for required columns
        if 'timestamp' not in eeg_data.columns:
            raise ValueError("EEG data missing timestamp column")
        
        # Check channel data
        missing_channels = [ch for ch in self.channel_names 
                        if ch not in eeg_data.columns]
        if missing_channels:
            logger.warning(f"Missing channels: {missing_channels}")
        
        # Check for NaN values
        nan_counts = eeg_data[self.channel_names].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values found: {nan_counts[nan_counts > 0]}")
    
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
        
        # Check for "Channel X" format (with spaces)
        elif 'Channel 1' in df.columns:
            # Find all channel columns
            channel_cols = [col for col in df.columns if col.startswith('Channel ') and col.split()[-1].isdigit()]
            # Create mapping from "Channel X" to "Channel_X"
            channel_mapping = {f'Channel {i}': f'Channel_{i}' for i in range(1, 17)}
            df = df.rename(columns=channel_mapping)
            
            # Update channel names based on what's actually in the data
            available_channels = [f'Channel_{i}' for i in range(1, 17) if f'Channel_{i}' in df.columns]
            self.channel_names = available_channels
        
        # Handle timestamp column - your data has "Time:512Hz"
        if 'Time:512Hz' in df.columns and 'timestamp' not in df.columns:
            df = df.rename(columns={'Time:512Hz': 'timestamp'})
        elif 'timestamp' not in df.columns:
            # Create timestamps based on sampling rate
            df['timestamp'] = np.arange(len(df)) / self.sampling_rate
        
        # Apply filtering to EEG channels
        eeg_columns = [col for col in self.channel_names if col in df.columns]
        if eeg_columns:
            # Convert to numpy array for filtering
            eeg_data = df[eeg_columns].values
            
            # Apply filters with default parameters
            filtered_data = self.filter_data(eeg_data)
            
            # Put filtered data back
            df[eeg_columns] = filtered_data
            
            # Use the default values from filter_data method
            logger.info("Applied 1.0-30Hz bandpass and 60Hz notch filters")
        
        self._validate_eeg_data(df)
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
        """Align EEG data timestamps with event log timestamps"""
        
        # Check if timestamps are Unix timestamps or relative
        if events['timestamp'].min() > 1e9:  # Likely Unix timestamp
            # Convert to relative time
            events['timestamp'] = events['timestamp'] - events['timestamp'].min()
        
        # Ensure EEG starts at 0
        if eeg_data['timestamp'].min() != 0:
            eeg_data['timestamp'] = eeg_data['timestamp'] - eeg_data['timestamp'].min()
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


    def get_comparison_epochs(self, session_data: Dict,
                            pre_time: float = 0.2, post_time: float = 0.8,
                            baseline: Tuple[float, float] = (-0.2, 0),
                            reject_artifacts: bool = True,
                            amplitude_threshold: float = 75.0) -> Dict:
        """
        Create epochs for both error and correct events for comparison
        """
        error_events = self.get_error_events(session_data['events'])
        correct_events = self.get_correct_events(session_data['events'])
        
        error_epochs = self.create_epochs(
            session_data['eeg_data'], error_events, 
            pre_time=pre_time, post_time=post_time, baseline=baseline,
            reject_artifacts=reject_artifacts, amplitude_threshold=amplitude_threshold
        )
        
        correct_epochs = self.create_epochs(
            session_data['eeg_data'], correct_events,
            pre_time=pre_time, post_time=post_time, baseline=baseline,
            reject_artifacts=reject_artifacts, amplitude_threshold=amplitude_threshold
        )
        
        return {
            'error': error_epochs,
            'correct': correct_epochs
        }
        
    def filter_data(self, data: np.ndarray, lowcut: float = 1.0, highcut: float = 30.0, 
                    notch_freq: float = 60.0) -> np.ndarray:
        """Apply bandpass and notch filtering"""
        nyquist = self.sampling_rate / 2
        
        # Check if data is long enough for the filter
        if len(data) < self.sampling_rate * 3:  # Less than 3 seconds
            logger.warning("Data too short for aggressive filtering, using gentler parameters")
            lowcut = 2.0  # Use higher cutoff for short data
        
        # Bandpass filter
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Use lower order for stability
        order = 2 if lowcut < 1.0 else 3
        
        try:
            b_band, a_band = signal.butter(order, [low, high], btype='band')
            
            # Check filter stability
            if not np.all(np.abs(np.roots(a_band)) < 1):
                logger.warning("Filter unstable, using SOS format")
                sos = signal.butter(order, [low, high], btype='band', output='sos')
                filtered = signal.sosfiltfilt(sos, data, axis=0)
            else:
                filtered = signal.filtfilt(b_band, a_band, data, axis=0)
            
            # Apply notch filter
            b_notch, a_notch = signal.iirnotch(notch_freq, 30, self.sampling_rate)
            filtered = signal.filtfilt(b_notch, a_notch, filtered, axis=0)
            
            # Check for NaN values
            if np.any(np.isnan(filtered)):
                logger.warning(f"Filter produced {np.sum(np.isnan(filtered))} NaN values")
                # Replace NaN with original data
                filtered[np.isnan(filtered)] = data[np.isnan(filtered)]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Filtering failed: {e}. Returning original data.")
            return data


    def create_epochs(self, eeg_data: pd.DataFrame, events: pd.DataFrame,
                    event_types: List[int] = None,
                    pre_time: float = 0.2, post_time: float = 0.8,
                    baseline: Tuple[float, float] = (-0.2, 0),
                    reject_artifacts: bool = True,
                    amplitude_threshold: float = 75.0) -> Dict:
        """
        Create epochs (EEG segments) around events
        """
        # Get available channels FIRST
        available_channels = [ch for ch in self.channel_names if ch in eeg_data.columns]
        if len(available_channels) < len(self.channel_names):
            logger.warning(f"Using {len(available_channels)} of {len(self.channel_names)} channels")
        
        # Filter events if specific types requested
        if event_types is not None:
            events = events[events['event'].isin(event_types)].copy()
        
        # Check if we have any events
        if len(events) == 0:
            logger.warning("No events found for epoching")
            return {
                'data': np.array([]),
                'events': pd.DataFrame(),
                'times': np.array([]),
                'sfreq': self.sampling_rate,
                'pre_time': pre_time,
                'post_time': post_time
            }
        
        pre_samples = int(pre_time * self.sampling_rate)
        post_samples = int(post_time * self.sampling_rate)
        
        # Time vector for epochs
        times = np.arange(-pre_samples, post_samples) / self.sampling_rate
        
        epochs = []
        valid_events = []
        
        for idx, event in events.iterrows():
            start_idx = int(event['sample_idx']) - pre_samples
            end_idx = int(event['sample_idx']) + post_samples
            
            # Check if epoch is within bounds
            if start_idx >= 0 and end_idx < len(eeg_data):
                # Extract epoch using available_channels
                epoch_data = eeg_data.iloc[start_idx:end_idx][available_channels].values

                if np.any(np.isnan(epoch_data)):
                    logger.warning(f"Epoch {idx} contains NaN values, skipping")
                    continue
                
                # Baseline correction if requested
                if baseline is not None:
                    baseline_start = int((baseline[0] - (-pre_time)) * self.sampling_rate)
                    baseline_end = int((baseline[1] - (-pre_time)) * self.sampling_rate)
                    baseline_mean = epoch_data[baseline_start:baseline_end].mean(axis=0)
                    epoch_data = epoch_data - baseline_mean
                
                # Check artifact rejection BEFORE appending
                if reject_artifacts and np.any(np.abs(epoch_data) > amplitude_threshold):
                    continue  # Skip this epoch
                
                epochs.append(epoch_data)
                valid_events.append(event)
        
        # Check if any epochs remain after rejection
        if len(epochs) == 0:
            logger.warning(f"No valid epochs created (all rejected or outside bounds)")
            return {
                'data': np.array([]),
                'events': pd.DataFrame(),
                'times': times,
                'sfreq': self.sampling_rate,
                'pre_time': pre_time,
                'post_time': post_time
            }
        
        epochs_array = np.array(epochs)  # Shape: (n_epochs, n_samples, n_channels)
        
        if reject_artifacts:
            n_rejected = len(events) - len(epochs)
            logger.info(f"Created {len(epochs)} epochs of shape {epochs_array.shape} ({n_rejected} rejected)")
        else:
            logger.info(f"Created {len(epochs)} epochs of shape {epochs_array.shape}")
        
        return {
            'data': epochs_array,
            'events': pd.DataFrame(valid_events).reset_index(drop=True),
            'times': times,
            'sfreq': self.sampling_rate,
            'pre_time': pre_time,
            'post_time': post_time
        }
    
    def explore_event_codes(self, session_data: Dict) -> None:
        """Explore event codes to understand the paradigm"""
        events = session_data['events']
        
        print("\n=== Event Code Analysis ===")
        print(f"Total events: {len(events)}")
        print(f"\nEvent code distribution:")
        print(events['event'].value_counts().sort_index())
        
        # If we have additional columns, show them
        if 'classifier_out' in events.columns and 'gt' in events.columns:
            print("\n=== Event Details ===")
            for event_type in sorted(events['event'].unique()):
                subset = events[events['event'] == event_type]
                print(f"\nEvent {event_type} (n={len(subset)}):")
                if 'classifier_out' in subset.columns:
                    print(f"  Classifier outputs: {subset['classifier_out'].value_counts().to_dict()}")
                if 'gt' in subset.columns:
                    print(f"  Ground truth: {subset['gt'].value_counts().to_dict()}")
                if 'confidence' in subset.columns:
                    print(f"  Confidence: mean={subset['confidence'].mean():.3f}, std={subset['confidence'].std():.3f}")

    def summarize_session(self, session_data: Dict) -> None:
        """Print summary statistics for loaded session"""
        events = session_data['events']
        
        print(f"\nSession Summary:")
        print(f"Duration: {session_data['metadata']['duration']:.2f} seconds")
        print(f"Sampling rate: {session_data['metadata']['sampling_rate']} Hz")
        print(f"Channels: {session_data['metadata']['n_channels']}")
        
        print(f"\nEvent Summary:")
        print(events['event'].value_counts().sort_index())
        
        # Error rate calculation
        error_events = self.get_error_events(events)
        correct_events = self.get_correct_events(events)
        total_feedback = len(error_events) + len(correct_events)
        
        if total_feedback > 0:
            error_rate = len(error_events) / total_feedback * 100
            print(f"\nError rate: {error_rate:.1f}% ({len(error_events)}/{total_feedback})")


def main():
    """Command line interface for ErrP data processing"""
    import argparse
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Process ErrP data from BCI wheelchair experiments')
    parser.add_argument('--data-dir', type=str, default='C:/Users/yuzeb/BCI_Final/BCI_Wheelchair_Project/EEG_data',
                        help='Base directory containing participant data')
    parser.add_argument('--participants', nargs='*', type=str,
                        help='Participant IDs to process (e.g., T-001 T-002). If not specified, process all.')
    parser.add_argument('--sessions', nargs='*', type=str,
                        help='Session names to process (e.g., "Session 1" "Session 2"). If not specified, process all.')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots for each session')
    parser.add_argument('--output-dir', type=str, default='./processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--threshold-test', action='store_true',
                        help='Test different artifact rejection thresholds')
    
    args = parser.parse_args()
    
    # Initialize data loader
    loader = ErrPDataLoader()
    
    # Find participants to process
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    if args.participants:
        participant_dirs = [data_dir / p for p in args.participants if (data_dir / p).exists()]
    else:
        participant_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('T-')])
    
    if not participant_dirs:
        print("No participant directories found!")
        return

    # Process each participant
    all_results = {}
    
    for participant_dir in participant_dirs:
        participant_id = participant_dir.name
        print(f"\n{'='*60}")
        print(f"Processing participant: {participant_id}")
        print(f"{'='*60}")
        
        # Find sessions to process
        if args.sessions:
            session_dirs = [participant_dir / s for s in args.sessions if (participant_dir / s).exists()]
        else:
            session_dirs = sorted([d for d in participant_dir.iterdir() if d.is_dir() and 'Session' in d.name])
        
        participant_results = {}
        
        for session_dir in session_dirs:
            session_name = session_dir.name
            print(f"\n{'-'*40}")
            print(f"Processing: {participant_id} - {session_name}")
            print(f"{'-'*40}")
            
            try:
                # Load session data
                session_data = loader.load_session(str(session_dir))
                
                # Explore event codes to understand the paradigm
                loader.explore_event_codes(session_data)
                
                # Print summary
                loader.summarize_session(session_data)
                
                # Create epochs
                epochs = loader.get_comparison_epochs(session_data, amplitude_threshold=100.0)
                
                # Store results
                participant_results[session_name] = {
                    'session_data': session_data,
                    'epochs': epochs
                }
                
                # Test artifact thresholds if requested
                if args.threshold_test:
                    print("\nArtifact rejection threshold analysis:")
                    test_artifact_thresholds(epochs)
                
                # Generate plots if requested
                if args.plot:
                    plot_session_overview(participant_id, session_name, epochs, loader.sampling_rate)
                
            except Exception as e:
                print(f"Error processing {participant_id} - {session_name}: {str(e)}")
                continue
        
        all_results[participant_id] = participant_results
    
    # Save processed data if output directory specified
    if args.output_dir:
        save_processed_data(all_results, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Processed {len(all_results)} participants")
    total_sessions = sum(len(sessions) for sessions in all_results.values())
    print(f"Total sessions processed: {total_sessions}")
    
    return all_results


def test_artifact_thresholds(epochs: Dict):
    """Test different artifact rejection thresholds"""
    error_data = epochs['error']['data']
    correct_data = epochs['correct']['data']
    
    # Check if we have data
    if error_data.size == 0 or correct_data.size == 0:
        print("No epochs available for threshold testing")
        return
    
    thresholds = [50, 75, 100, 125, 150, 200]
    
    print(f"{'Threshold':>10} | {'Error Rejected':>15} | {'Correct Rejected':>17}")
    print("-" * 50)
    
    for threshold in thresholds:
        # Check how many epochs exceed threshold
        if error_data.size > 0:
            bad_error = np.any(np.abs(error_data) > threshold, axis=(1,2)).sum()
            error_pct = bad_error/len(error_data)*100
        else:
            bad_error = 0
            error_pct = 0
            
        if correct_data.size > 0:
            bad_correct = np.any(np.abs(correct_data) > threshold, axis=(1,2)).sum()
            correct_pct = bad_correct/len(correct_data)*100
        else:
            bad_correct = 0
            correct_pct = 0
        
        print(f"+/-{threshold:>3}uV    | {bad_error:>3}/{len(error_data):<3} ({error_pct:>5.1f}%) | "
              f"{bad_correct:>3}/{len(correct_data):<3} ({correct_pct:>5.1f}%)")


def plot_session_overview(participant_id: str, session_name: str, epochs: Dict, sampling_rate: int):
    """Generate overview plots for a session"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{participant_id} - {session_name}', fontsize=16)
    
    error_data = epochs['error']['data']
    correct_data = epochs['correct']['data']
    times = epochs['error']['times']
    
    # 1. Average ERPs for multiple channels
    ax = axes[0, 0]
    channels_to_plot = [3, 5, 7]  # Adjust based on your montage
    for ch_idx in channels_to_plot:
        if ch_idx < error_data.shape[2]:
            error_avg = error_data[:, :, ch_idx].mean(axis=0)
            correct_avg = correct_data[:, :, ch_idx].mean(axis=0)
            ax.plot(times, error_avg, 'r-', alpha=0.7, label=f'Error Ch{ch_idx+1}')
            ax.plot(times, correct_avg, 'b-', alpha=0.7, label=f'Correct Ch{ch_idx+1}')
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title('Average ERPs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Voltage distribution
    ax = axes[0, 1]
    all_data = np.concatenate([error_data.flatten(), correct_data.flatten()])
    ax.hist(all_data, bins=100, alpha=0.7, color='gray', edgecolor='black')
    
    # Add threshold lines
    for threshold in [75, 100, 150]:
        ax.axvline(-threshold, color='r', linestyle='--', alpha=0.5)
        ax.axvline(threshold, color='r', linestyle='--', alpha=0.5, label=f'±{threshold}uV')
    
    ax.set_xlabel('Amplitude (uV)') 
    ax.set_ylabel('Count')
    ax.set_title('Voltage Distribution')
    ax.set_xlim(-200, 200)
    ax.legend()
    
    # 3. Single trial examples
    ax = axes[1, 0]
    # Plot first 5 error trials
    for i in range(min(5, len(error_data))):
        ax.plot(times, error_data[i, :, 3], 'r-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (uV)')
    ax.set_title('Example Error Trials (Ch4)')
    ax.grid(True, alpha=0.3)
    
    # 4. ERP image (all trials as heatmap)
    ax = axes[1, 1]
    # Concatenate error and correct trials
    all_trials = np.vstack([error_data[:, :, 3], correct_data[:, :, 3]])
    im = ax.imshow(all_trials, aspect='auto', cmap='RdBu_r', 
                   extent=[times[0], times[-1], len(all_trials), 0],
                   vmin=-50, vmax=50)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(len(error_data), color='g', linestyle='-', alpha=0.5, label='Error/Correct boundary')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Trial number')
    ax.set_title('ERP Image (Ch4)')
    plt.colorbar(im, ax=ax, label='uV')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(f'./figures/{participant_id}')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{session_name}_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {output_dir / f'{session_name}_overview.png'}")


def save_processed_data(all_results: Dict, output_dir: str):
    """Save processed epochs to disk"""
    import pickle
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for participant_id, sessions in all_results.items():
        participant_path = output_path / participant_id
        participant_path.mkdir(exist_ok=True)
        
        for session_name, data in sessions.items():
            session_file = participant_path / f'{session_name}_epochs.pkl'
            
            # Save only the epochs, not the raw data to save space
            save_data = {
                'epochs': data['epochs'],
                'metadata': data['session_data']['metadata'],
                'event_summary': data['session_data']['events']['event'].value_counts().to_dict()
            }
            
            with open(session_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            print(f"Saved: {session_file}")


if __name__ == "__main__":
    main()