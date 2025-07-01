"""
Motor Imagery Epoch Extraction Module
Handles trial segmentation and epoch extraction for MI analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats
from collections import Counter

class EpochExtractor:
    def __init__(self, sampling_rate: int = 512):
        """
        Initialize epoch extractor
        
        Args:
            sampling_rate: Sampling frequency (default 512Hz)
        """
        self.sampling_rate = sampling_rate
        
    def extract_mi_epoch(self, trial: Dict, 
                        start_time: float = 0.5, 
                        end_time: float = 3.5) -> np.ndarray:
        """
        Extract motor imagery epoch from trial
        
        Args:
            trial: Preprocessed trial dictionary
            start_time: Start time relative to cue onset (seconds)
            end_time: End time relative to cue onset (seconds)
            
        Returns:
            Epoch data (samples x channels) or None if invalid
        """
        # Calculate sample indices
        # Note: Trial starts at -1s, so cue onset is at 1s
        cue_onset_sample = int(1.0 * self.sampling_rate)
        start_sample = cue_onset_sample + int(start_time * self.sampling_rate)
        end_sample = cue_onset_sample + int(end_time * self.sampling_rate)
        
        # Check if we have enough data
        if end_sample > trial['data'].shape[0]:
            # Trial is too short - skip it
            return None
            
        # Ensure indices are within bounds
        start_sample = max(0, start_sample)
        end_sample = min(trial['data'].shape[0], end_sample)
        
        # Calculate expected length
        expected_length = int((end_time - start_time) * self.sampling_rate)
        actual_length = end_sample - start_sample
        
        # Only accept epochs with the full expected length
        if actual_length < expected_length:
            return None
        
        # Extract epoch
        epoch = trial['data'][start_sample:end_sample, :]
        
        # Apply artifact mask if available
        if 'artifact_mask' in trial:
            mask = trial['artifact_mask'][start_sample:end_sample]
            if np.mean(mask) < 0.8:  # More than 20% artifacts
                return None
                
        return epoch
    
    def extract_baseline_epoch(self, trial: Dict,
                             duration: float = 0.5) -> np.ndarray:
        """
        Extract baseline epoch before cue onset
        
        Args:
            trial: Preprocessed trial dictionary
            duration: Baseline duration in seconds
            
        Returns:
            Baseline epoch data
        """
        # Baseline ends at cue onset (1s into trial)
        cue_onset_sample = int(1.0 * self.sampling_rate)
        start_sample = cue_onset_sample - int(duration * self.sampling_rate)
        
        start_sample = max(0, start_sample)
        
        return trial['data'][start_sample:cue_onset_sample, :]
    
    def calculate_erd_ers(self, trial: Dict,
                         baseline_duration: float = 0.5,
                         analysis_window: float = 0.5) -> Dict:
        """
        Calculate Event-Related Desynchronization/Synchronization
        
        Args:
            trial: Preprocessed trial dictionary
            baseline_duration: Duration of baseline period
            analysis_window: Window size for ERD/ERS calculation
            
        Returns:
            Dictionary with ERD/ERS values over time
        """
        # Extract baseline
        baseline = self.extract_baseline_epoch(trial, baseline_duration)
        baseline_power = np.mean(baseline**2, axis=0)  # Power per channel
        
        # Calculate ERD/ERS over time windows
        window_samples = int(analysis_window * self.sampling_rate)
        hop_samples = int(0.1 * self.sampling_rate)  # 100ms hop
        
        cue_onset_sample = int(1.0 * self.sampling_rate)
        n_samples = trial['data'].shape[0]
        
        times = []
        erd_values = []
        
        # Slide window across MI period
        for start in range(cue_onset_sample, n_samples - window_samples, hop_samples):
            end = start + window_samples
            
            # Calculate power in window
            window_data = trial['data'][start:end, :]
            window_power = np.mean(window_data**2, axis=0)
            
            # ERD = (baseline - window) / baseline * 100
            erd = (baseline_power - window_power) / baseline_power * 100
            
            # Store results
            time = (start - cue_onset_sample) / self.sampling_rate
            times.append(time)
            erd_values.append(erd)
            
        return {
            'times': np.array(times),
            'erd_values': np.array(erd_values),  # (time_points x channels)
            'baseline_power': baseline_power
        }
    
    def segment_continuous_epochs(self, trial: Dict,
                                 epoch_duration: float = 1.0,
                                 overlap: float = 0.5) -> List[np.ndarray]:
        """
        Segment trial into overlapping epochs
        
        Args:
            trial: Preprocessed trial dictionary
            epoch_duration: Duration of each epoch in seconds
            overlap: Overlap between epochs (0-1)
            
        Returns:
            List of epoch arrays
        """
        epochs = []
        
        # Calculate parameters
        epoch_samples = int(epoch_duration * self.sampling_rate)
        hop_samples = int(epoch_samples * (1 - overlap))
        
        # Start from cue onset
        cue_onset_sample = int(1.0 * self.sampling_rate)
        data = trial['data']
        
        # Extract epochs
        start = cue_onset_sample
        while start + epoch_samples <= data.shape[0]:
            epoch = data[start:start + epoch_samples, :]
            
            # Check artifact mask if available
            if 'artifact_mask' in trial:
                mask = trial['artifact_mask'][start:start + epoch_samples]
                if np.mean(mask) > 0.8:  # Less than 20% artifacts
                    epochs.append(epoch)
            else:
                epochs.append(epoch)
                
            start += hop_samples
            
        return epochs
    
    def create_feature_windows(self, participant_data: Dict,
                             window_config: Dict = None) -> Dict:
        """
        Create feature extraction windows for all trials
        
        Args:
            participant_data: Preprocessed participant data
            window_config: Configuration for window extraction
            
        Returns:
            Dictionary with organized epochs
        """
        if window_config is None:
            window_config = {
                'mi_start': 0.5,      # Start 0.5s after cue
                'mi_end': 3.5,        # End 3.5s after cue
                'baseline_duration': 0.5,
                'epoch_duration': 2.0,
                'epoch_overlap': 0.5
            }
        
        feature_data = {
            'epochs': [],
            'labels': [],
            'metadata': [],
            'erd_ers': []
        }
        
        # Process each session
        for session_name, session_data in participant_data['sessions'].items():
            for run_name, trials in session_data.items():
                for trial_idx, trial in enumerate(trials):
                    if not trial.get('is_valid', True):
                        continue
                    
                    # Extract MI epoch
                    mi_epoch = self.extract_mi_epoch(
                        trial,
                        window_config['mi_start'],
                        window_config['mi_end']
                    )
                    
                    if mi_epoch is not None:
                        # Double-check epoch shape
                        expected_samples = int((window_config['mi_end'] - window_config['mi_start']) * self.sampling_rate)
                        if mi_epoch.shape[0] != expected_samples:
                            print(f"Warning: Epoch shape mismatch in {session_name}/{run_name} trial {trial_idx}")
                            print(f"  Expected: {expected_samples} samples, got: {mi_epoch.shape[0]}")
                            continue
                            
                        # Calculate ERD/ERS
                        try:
                            erd_ers = self.calculate_erd_ers(
                                trial,
                                window_config['baseline_duration']
                            )
                        except Exception as e:
                            print(f"Warning: ERD/ERS calculation failed for {session_name}/{run_name} trial {trial_idx}: {e}")
                            continue
                        
                        # Store data
                        feature_data['epochs'].append(mi_epoch)
                        feature_data['labels'].append(trial['label'])
                        feature_data['metadata'].append({
                            'session': session_name,
                            'run': run_name,
                            'trial_idx': trial_idx
                        })
                        feature_data['erd_ers'].append(erd_ers)
        
        # Convert to arrays - handle potential shape mismatch
        if len(feature_data['epochs']) > 0:
            # Check if all epochs have the same shape
            shapes = [epoch.shape for epoch in feature_data['epochs']]
            if len(set(shapes)) == 1:
                # All same shape - can convert to array
                feature_data['epochs'] = np.array(feature_data['epochs'])
            else:
                # Different shapes - keep as list and warn
                print(f"Warning: Epochs have different shapes: {set(shapes)}")
                print("Keeping epochs as list. Consider adjusting epoch parameters.")
                
        feature_data['labels'] = np.array(feature_data['labels'])
        
        # Add shape info for debugging
        if len(feature_data['epochs']) > 0:
            if isinstance(feature_data['epochs'], np.ndarray):
                feature_data['epoch_shape'] = feature_data['epochs'][0].shape
            else:
                feature_data['epoch_shape'] = 'Variable (see individual epochs)'
        
        return feature_data
    
    def plot_erd_ers_topography(self, erd_ers_data: Dict, 
                               time_point: float = 1.0,
                               channel_positions: Optional[np.ndarray] = None):
        """
        Plot ERD/ERS topography at specific time point
        
        Args:
            erd_ers_data: ERD/ERS data from calculate_erd_ers
            time_point: Time point to visualize (seconds after cue)
            channel_positions: 2D positions of channels for topography
        """
        # Find closest time point
        time_idx = np.argmin(np.abs(erd_ers_data['times'] - time_point))
        erd_values = erd_ers_data['erd_values'][time_idx, :]
        
        if channel_positions is None:
            # Simple grid layout for 16 channels
            positions = []
            for i in range(4):
                for j in range(4):
                    positions.append([j, 3-i])
            channel_positions = np.array(positions)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot channels
        scatter = ax.scatter(channel_positions[:, 0], 
                           channel_positions[:, 1],
                           c=erd_values,
                           s=500,
                           cmap='RdBu_r',
                           vmin=-50,
                           vmax=50,
                           edgecolors='black',
                           linewidth=2)
        
        # Add channel numbers
        for i, pos in enumerate(channel_positions):
            ax.text(pos[0], pos[1], str(i+1), 
                   ha='center', va='center', fontsize=10)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('ERD/ERS (%)', fontsize=12)
        
        # Labels
        ax.set_title(f'ERD/ERS Topography at t={time_point:.1f}s', fontsize=14)
        ax.set_xlabel('Lateral position')
        ax.set_ylabel('Anterior-Posterior position')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def check_trial_durations(self, participant_data: Dict) -> Dict:
        """
        Check the duration of all trials to diagnose length issues
        
        Args:
            participant_data: Preprocessed participant data
            
        Returns:
            Dictionary with duration statistics
        """
        durations = []
        short_trials = []
        
        for session_name, session_data in participant_data['sessions'].items():
            for run_name, trials in session_data.items():
                for trial_idx, trial in enumerate(trials):
                    duration_samples = trial['data'].shape[0]
                    duration_seconds = duration_samples / self.sampling_rate
                    durations.append(duration_seconds)
                    
                    # Check if trial is too short for standard epoch extraction
                    # Standard: -1s to +3.5s = 4.5s total
                    if duration_seconds < 4.5:
                        short_trials.append({
                            'session': session_name,
                            'run': run_name,
                            'trial': trial_idx,
                            'duration': duration_seconds,
                            'label': trial['label']
                        })
        
        duration_stats = {
            'min': np.min(durations),
            'max': np.max(durations),
            'mean': np.mean(durations),
            'std': np.std(durations),
            'n_trials': len(durations),
            'n_short': len(short_trials),
            'short_trials': short_trials
        }
        
        print("\n=== Trial Duration Analysis ===")
        print(f"Total trials: {duration_stats['n_trials']}")
        print(f"Duration range: {duration_stats['min']:.2f}s - {duration_stats['max']:.2f}s")
        print(f"Mean duration: {duration_stats['mean']:.2f}s ± {duration_stats['std']:.2f}s")
        print(f"Short trials (<4.5s): {duration_stats['n_short']} ({100*duration_stats['n_short']/duration_stats['n_trials']:.1f}%)")
        
        if short_trials:
            print("\nShort trial details:")
            for st in short_trials[:5]:  # Show first 5
                print(f"  {st['session']}/{st['run']} trial {st['trial']}: {st['duration']:.2f}s (class {st['label']})")
            if len(short_trials) > 5:
                print(f"  ... and {len(short_trials)-5} more")
        
        return duration_stats
    
    def plot_class_comparison(self, feature_data: Dict, 
                            channel: int = 7):  # C3/C4 typically channel 7/8
        """
        Plot average ERD/ERS for each MI class
        
        Args:
            feature_data: Feature data dictionary
            channel: Channel index to plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # Group by class
        for class_idx, class_label in enumerate([1, 2, 3]):
            # Get trials for this class
            class_mask = feature_data['labels'] == class_label
            class_erd_ers = [feature_data['erd_ers'][i] 
                           for i in range(len(feature_data['labels'])) 
                           if class_mask[i]]
            
            if not class_erd_ers:
                continue
            
            # Average ERD/ERS
            times = class_erd_ers[0]['times']
            erd_values = np.array([erd['erd_values'][:, channel] 
                                 for erd in class_erd_ers])
            
            mean_erd = np.mean(erd_values, axis=0)
            std_erd = np.std(erd_values, axis=0)
            
            # Plot
            ax = axes[class_idx]
            ax.plot(times, mean_erd, 'b', linewidth=2)
            ax.fill_between(times, 
                          mean_erd - std_erd,
                          mean_erd + std_erd,
                          alpha=0.3)
            
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            
            ax.set_ylabel('ERD/ERS (%)')
            ax.set_title(f'Class {class_label} (n={len(class_erd_ers)})')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-50, 50)
            
        axes[-1].set_xlabel('Time after cue (s)')
        plt.tight_layout()
        
        return fig


def main():
    """Example usage"""
    try:
        from .load_data import MIDataLoader
        from .preprocess import MIPreprocessor
    except ImportError:
        from load_data import MIDataLoader
        from preprocess import MIPreprocessor
    
    # Load and preprocess data
    loader = MIDataLoader("/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI")
    preprocessor = MIPreprocessor(sampling_rate=512)
    extractor = EpochExtractor(sampling_rate=512)
    
    # Load T-005 data
    print("Loading data...")
    t005_data = loader.get_participant_data("T-005")
    
    # Preprocess first session
    if t005_data['sessions']:
        first_session_name = list(t005_data['sessions'].keys())[0]
        first_session = t005_data['sessions'][first_session_name]
        
        print(f"\nPreprocessing {first_session_name}...")
        preprocessed = preprocessor.preprocess_session(first_session)
        
        # Update the data
        t005_data['sessions'][first_session_name] = preprocessed
        
        # Check trial durations first
        print("\nChecking trial durations...")
        duration_stats = extractor.check_trial_durations(t005_data)
        
        # Extract features
        print("\nExtracting features...")
        feature_data = extractor.create_feature_windows(t005_data)
        
        print(f"\nExtracted features:")
        print(f"  Total epochs: {len(feature_data['epochs'])}")
        if len(feature_data['epochs']) > 0:
            print(f"  Epoch shape: {feature_data.get('epoch_shape', 'Variable')}")
            print(f"  Class distribution: {np.bincount(feature_data['labels'])[1:]}")
        
        # Plot class comparison only if we have valid epochs
        if len(feature_data['epochs']) > 0:
            print("\nPlotting class comparison...")
            fig = extractor.plot_class_comparison(feature_data)
            plt.show()
        else:
            print("\nNo valid epochs to plot!")
        
        # Plot ERD/ERS topography
        if feature_data['erd_ers']:
            print("\nPlotting ERD/ERS topography...")
            fig = extractor.plot_erd_ers_topography(feature_data['erd_ers'][0])
            plt.show()


if __name__ == "__main__":
    main()