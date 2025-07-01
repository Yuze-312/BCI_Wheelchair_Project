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
        session_path = self.base_path / "EEG data" / participant / session
        
        if not session_path.exists():
            raise ValueError(f"Session path not found: {session_path}")
            
        data_dict = {}
        
        # Load all CSV files in the session
        for csv_file in sorted(session_path.glob("test-*.csv")):
            print(f"Loading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            data_dict[csv_file.stem] = df
            
        return data_dict
    
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
        
        # Extract trials around each event
        for idx, event in events.iterrows():
            event_id = int(event['Event Id'])
            
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
                'label': event_id,
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
        info_path = self.base_path / "EEG data" / participant / "Info.txt"
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