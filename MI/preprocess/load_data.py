"""
Motor Imagery Data Loading Module
Handles loading and basic parsing of OpenViBE CSV files
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class MIDataLoader:
    def __init__(self, base_path: str, sampling_rate: int = 512):
        """
        Initialize MI data loader
        
        Args:
            base_path: Path to MI folder
            sampling_rate: Sampling frequency (default 512Hz)
        """
        self.base_path = Path(base_path)
        self.sampling_rate = sampling_rate
        self.channel_names = [f"Channel {i}" for i in range(1, 17)]
        
    def load_session(self, participant: str, session: str) -> Dict[str, pd.DataFrame]:
        """
        Load all runs from a specific session
        
        Args:
            participant: Participant ID (e.g., "T-005")
            session: Session name (e.g., "Session 1", "Test")
            
        Returns:
            Dictionary mapping filenames to DataFrames
        """
        session_path = self.base_path / participant / session
        
        if not session_path.exists():
            raise ValueError(f"Session path not found: {session_path}")
            
        data_dict = {}
        
        # Load all CSV files in the session
        # First try test-*.csv pattern (original format)
        csv_files = list(session_path.glob("test-*.csv"))
        
        # If no test-*.csv files, try data_*.csv pattern (OpenViBE export)
        if not csv_files:
            csv_files = list(session_path.glob("data_*.csv"))
            
        # If still no files, try any CSV file
        if not csv_files:
            csv_files = list(session_path.glob("*.csv"))
            # Exclude event log files
            csv_files = [f for f in csv_files if 'event' not in f.name.lower() and 'errp' not in f.name.lower()]
        
        for csv_file in sorted(csv_files):
            print(f"Loading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            
            # Check if this is OpenViBE format with Oscillator columns
            if 'Oscillator 1' in df.columns:
                # Rename columns to match expected format
                oscillator_cols = [col for col in df.columns if col.startswith('Oscillator')]
                channel_mapping = {f'Oscillator {i}': f'Channel {i}' for i in range(1, len(oscillator_cols)+1)}
                df = df.rename(columns=channel_mapping)
                self.channel_names = [f"Channel {i}" for i in range(1, len(oscillator_cols)+1)]
            
            # Look for separate event log file
            event_log = None
            possible_event_files = [
                session_path / f"subway_errp_*.csv",
                session_path / f"phase1_events_*.csv",
                session_path / f"event_log.csv"
            ]
            
            for pattern in possible_event_files:
                if isinstance(pattern, Path) and pattern.exists():
                    event_log = pattern
                    break
                else:
                    matching_files = list(session_path.glob(pattern.name))
                    if matching_files:
                        event_log = matching_files[0]
                        break
            
            if event_log and event_log.exists():
                print(f"  Found event log: {event_log.name}")
                df = self._merge_events(df, event_log)
            
            data_dict[csv_file.stem] = df
            
        return data_dict
    
    def _merge_events(self, eeg_df: pd.DataFrame, event_log_path: Path) -> pd.DataFrame:
        """
        Merge event markers from separate log file into EEG data
        
        Args:
            eeg_df: DataFrame with EEG data
            event_log_path: Path to event log CSV file
            
        Returns:
            DataFrame with Event Id column populated
        """
        # Load event log
        event_df = pd.read_csv(event_log_path)
        
        # Ensure Event Id column exists
        if 'Event Id' not in eeg_df.columns:
            eeg_df['Event Id'] = 0
        else:
            # Clear existing events
            eeg_df['Event Id'] = 0
            
        # Convert event timestamps to sample indices
        for _, event in event_df.iterrows():
            # Get event info
            timestamp = event['timestamp']
            event_type = event['event']
            
            # Skip non-cue events for MI analysis (we only need cues)
            # 2 = LEFT cue, 3 = RIGHT cue
            if event_type not in [2, 3]:
                continue
                
            # Find closest sample to timestamp
            # Assuming Time column exists or calculate from index
            if 'Time:512Hz' in eeg_df.columns:
                # OpenViBE format with time column
                time_col = 'Time:512Hz'
                sample_idx = (eeg_df[time_col] - timestamp).abs().idxmin()
            else:
                # Calculate from sampling rate
                sample_idx = int(timestamp * self.sampling_rate)
                if sample_idx >= len(eeg_df):
                    print(f"  Warning: Event at {timestamp}s is beyond data length")
                    continue
            
            # Set event marker
            eeg_df.loc[sample_idx, 'Event Id'] = event_type
            
            # Also get ground truth if available
            if 'gt' in event.index and pd.notna(event['gt']):
                # Map gt (0=left, 1=right) to event type (2=left, 3=right)
                gt_value = int(event['gt'])
                expected_event = 2 if gt_value == 0 else 3
                if event_type != expected_event:
                    print(f"  Warning: Event type {event_type} doesn't match GT {gt_value}")
        
        # Report merged events
        merged_events = eeg_df[eeg_df['Event Id'] != 0]
        print(f"  Merged {len(merged_events)} MI cue events into EEG data")
        
        return eeg_df
    
    def extract_trials(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract individual trials from continuous data
        
        Args:
            df: DataFrame with continuous EEG data
            
        Returns:
            List of trial dictionaries
        """
        trials = []
        
        # Find all non-zero events (MI trials)
        events = df[df['Event Id'].notna()].copy()
        events = events[events['Event Id'] != 0]  # Exclude rest markers
        
        # For phase1 data, we use markers 2 (LEFT) and 3 (RIGHT)
        # Map them to labels: LEFT=1, RIGHT=2 (matching the original label scheme)
        event_label_map = {2: 1, 3: 2}  # 2->1 (LEFT), 3->2 (RIGHT)
        
        # Extract trials around each event
        for idx, event in events.iterrows():
            event_id = int(event['Event Id'])
            
            # Map event ID to label if needed
            if event_id in event_label_map:
                label = event_label_map[event_id]
            else:
                # Keep original label for backward compatibility
                label = event_id
            
            # Define trial window: -1s to +4s around event
            pre_samples = int(1 * self.sampling_rate)  # 1 second before
            post_samples = int(4 * self.sampling_rate)  # 4 seconds after
            
            start_idx = max(0, idx - pre_samples)
            end_idx = min(len(df), idx + post_samples)
            
            # Extract EEG data
            trial_data = df.iloc[start_idx:end_idx][self.channel_names].values
            
            # Store trial info
            trial = {
                'data': trial_data,
                'label': label,  # Use mapped label
                'event_sample': idx,
                'start_sample': start_idx,
                'end_sample': end_idx,
                'duration_samples': end_idx - start_idx
            }
            
            trials.append(trial)
            
        return trials
    
    def get_participant_data(self, participant: str) -> Dict:
        """
        Load all data for a participant
        
        Args:
            participant: Participant ID
            
        Returns:
            Dictionary with all sessions and trials
        """
        participant_data = {
            'info': {},
            'sessions': {}
        }
        
        # Load info file if exists
        info_path = self.base_path / participant / "Info.txt"
        if info_path.exists():
            with open(info_path, 'r') as f:
                participant_data['info']['task'] = f.read().strip()
        
        # Load all sessions
        sessions = ["Session 1", "Session 2", "Test"]
        
        for session in sessions:
            try:
                session_data = self.load_session(participant, session)
                
                # Extract trials from each run
                session_trials = {}
                for run_name, df in session_data.items():
                    trials = self.extract_trials(df)
                    session_trials[run_name] = trials
                    print(f"  {run_name}: {len(trials)} trials extracted")
                
                participant_data['sessions'][session] = session_trials
                
            except ValueError as e:
                print(f"Skipping {session}: {e}")
                
        return participant_data
    
    def get_summary_statistics(self, participant_data: Dict) -> Dict:
        """
        Calculate summary statistics for loaded data
        """
        stats = {
            'total_trials': 0,
            'trials_per_class': {1: 0, 2: 0, 3: 0},
            'sessions': {}
        }
        
        for session_name, session_data in participant_data['sessions'].items():
            session_stats = {
                'runs': len(session_data),
                'trials': 0,
                'class_distribution': {1: 0, 2: 0, 3: 0}
            }
            
            for run_name, trials in session_data.items():
                session_stats['trials'] += len(trials)
                
                for trial in trials:
                    label = trial['label']
                    session_stats['class_distribution'][label] += 1
                    stats['trials_per_class'][label] += 1
                    stats['total_trials'] += 1
            
            stats['sessions'][session_name] = session_stats
            
        return stats
    
    def plot_trial_example(self, trial: Dict, channels: List[int] = None):
        """
        Plot example trial data
        
        Args:
            trial: Trial dictionary
            channels: List of channel indices to plot (default: first 4)
        """
        if channels is None:
            channels = [0, 1, 2, 3]
            
        data = trial['data']
        time = np.arange(data.shape[0]) / self.sampling_rate - 1.0  # Start at -1s
        
        fig, axes = plt.subplots(len(channels), 1, figsize=(12, 8), sharex=True)
        if len(channels) == 1:
            axes = [axes]
            
        for i, ch in enumerate(channels):
            axes[i].plot(time, data[:, ch])
            axes[i].set_ylabel(f'Ch {ch+1} (μV)')
            axes[i].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Event onset')
            axes[i].grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Time (s)')
        axes[0].set_title(f'MI Trial - Class {trial["label"]}')
        axes[0].legend()
        
        plt.tight_layout()
        return fig


def main():
    """Example usage"""
    # Initialize loader
    loader = MIDataLoader("/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI")
    
    # Load data for participant T-005
    print("Loading data for T-005...")
    t005_data = loader.get_participant_data("T-005")
    
    # Get statistics
    stats = loader.get_summary_statistics(t005_data)
    
    print("\n=== Data Summary for T-005 ===")
    print(f"Total trials: {stats['total_trials']}")
    print(f"Class distribution: {stats['trials_per_class']}")
    
    for session_name, session_stats in stats['sessions'].items():
        print(f"\n{session_name}:")
        print(f"  Runs: {session_stats['runs']}")
        print(f"  Trials: {session_stats['trials']}")
        print(f"  Distribution: {session_stats['class_distribution']}")
    
    # Plot example trial
    if t005_data['sessions']:
        first_session = list(t005_data['sessions'].values())[0]
        if first_session:
            first_run = list(first_session.values())[0]
            if first_run:
                print("\nPlotting example trial...")
                fig = loader.plot_trial_example(first_run[0])
                plt.show()


if __name__ == "__main__":
    main()