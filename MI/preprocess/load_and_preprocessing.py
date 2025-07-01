"""
Complete MI Data Processing Pipeline
Integrates loading, preprocessing, and epoch extraction
"""

import numpy as np
import os
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from .load_data import MIDataLoader
from .preprocess import MIPreprocessor
from .epoch_extraction import EpochExtractor

class MIProcessingPipeline:
    def __init__(self, base_path: str, sampling_rate: int = 512):
        """
        Initialize complete processing pipeline
        
        Args:
            base_path: Path to MI data folder
            sampling_rate: Sampling frequency
        """
        self.base_path = Path(base_path)
        self.sampling_rate = sampling_rate
        
        # Initialize components
        self.loader = MIDataLoader(base_path, sampling_rate)
        self.preprocessor = MIPreprocessor(sampling_rate)
        self.extractor = EpochExtractor(sampling_rate)
        
        # Default preprocessing parameters
        self.preprocess_params = {
            'spatial_filter': 'car',     # 'car', 'laplacian', or None
            'remove_powerline': True,
            'bandpass_low': 8.0,
            'bandpass_high': 30.0,
            'artifact_threshold': 100.0
        }
        
        # Default epoch extraction parameters
        self.epoch_params = {
            'mi_start': 0.5,         # Start 0.5s after cue
            'mi_end': 3.5,           # End 3.5s after cue
            'baseline_duration': 0.5,
            'epoch_duration': 2.0,
            'epoch_overlap': 0.5
        }
        
    def process_participant(self, participant_id: str, 
                          sessions: Optional[List[str]] = None) -> Dict:
        """
        Process all data for a participant
        
        Args:
            participant_id: Participant ID (e.g., "T-005")
            sessions: List of sessions to process (default: all)
            
        Returns:
            Processed data dictionary
        """
        print(f"\n{'='*60}")
        print(f"Processing participant: {participant_id}")
        print(f"{'='*60}")
        
        # Load raw data
        print("\n1. Loading raw data...")
        raw_data = self.loader.get_participant_data(participant_id)
        stats = self.loader.get_summary_statistics(raw_data)
        print(f"   Total trials loaded: {stats['total_trials']}")
        print(f"   Class distribution: {stats['trials_per_class']}")
        
        # Select sessions to process
        if sessions is None:
            sessions = list(raw_data['sessions'].keys())
        
        # Preprocess each session
        print("\n2. Preprocessing data...")
        for session_name in sessions:
            if session_name in raw_data['sessions']:
                print(f"\n   Processing {session_name}...")
                preprocessed = self.preprocessor.preprocess_session(
                    raw_data['sessions'][session_name],
                    spatial_filter=self.preprocess_params['spatial_filter'],
                    remove_powerline=self.preprocess_params['remove_powerline']
                )
                raw_data['sessions'][session_name] = preprocessed
        
        # Extract features
        print("\n3. Extracting epochs and features...")
        feature_data = self.extractor.create_feature_windows(
            raw_data, 
            self.epoch_params
        )
        
        # Add metadata
        feature_data['participant_id'] = participant_id
        feature_data['sampling_rate'] = self.sampling_rate
        feature_data['preprocess_params'] = self.preprocess_params
        feature_data['epoch_params'] = self.epoch_params
        
        # Print summary
        print(f"\n4. Processing complete!")
        print(f"   Valid epochs: {len(feature_data['epochs'])}")
        if len(feature_data['epochs']) > 0:
            print(f"   Epoch shape: {feature_data['epochs'][0].shape}")
            print(f"   Final class distribution: {np.bincount(feature_data['labels'])[1:]}")
        
        return feature_data
    
    def process_all_participants(self) -> Dict[str, Dict]:
        """
        Process all participants in the dataset
        
        Returns:
            Dictionary mapping participant IDs to processed data
        """
        all_data = {}
        
        # Find all participants
        eeg_data_path = self.base_path / "EEG data"
        participants = [d.name for d in eeg_data_path.iterdir() if d.is_dir()]
        
        print(f"Found {len(participants)} participants: {participants}")
        
        # Process each participant
        for participant in participants:
            try:
                feature_data = self.process_participant(participant)
                all_data[participant] = feature_data
            except Exception as e:
                print(f"Error processing {participant}: {e}")
                
        return all_data
    
    def save_processed_data(self, data: Dict, output_path: str):
        """
        Save processed data to file
        
        Args:
            data: Processed data dictionary
            output_path: Path to save file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"Data saved to: {output_path}")
        
    def load_processed_data(self, file_path: str) -> Dict:
        """
        Load previously processed data
        
        Args:
            file_path: Path to saved data file
            
        Returns:
            Processed data dictionary
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        return data
    
    def generate_summary_report(self, all_data: Dict[str, Dict]):
        """
        Generate summary report and visualizations
        
        Args:
            all_data: Dictionary with all participants' data
        """
        print("\n" + "="*60)
        print("DATASET SUMMARY REPORT")
        print("="*60)
        
        total_epochs = 0
        all_labels = []
        
        for participant_id, data in all_data.items():
            n_epochs = len(data['epochs'])
            total_epochs += n_epochs
            
            print(f"\n{participant_id}:")
            print(f"  Epochs: {n_epochs}")
            
            if n_epochs > 0:
                labels = data['labels']
                all_labels.extend(labels)
                class_dist = np.bincount(labels)[1:]
                print(f"  Classes: {class_dist} (total: {sum(class_dist)})")
                print(f"  Balance: {class_dist / sum(class_dist) * 100}")
        
        print(f"\nTotal epochs across all participants: {total_epochs}")
        
        if all_labels:
            overall_dist = np.bincount(all_labels)[1:]
            print(f"Overall class distribution: {overall_dist}")
            print(f"Overall balance: {overall_dist / sum(overall_dist) * 100}%")
        
        # Create visualization
        self._plot_dataset_summary(all_data)
        
    def _plot_dataset_summary(self, all_data: Dict[str, Dict]):
        """Create summary visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Class distribution per participant
        ax = axes[0, 0]
        participants = []
        class_counts = {1: [], 2: [], 3: []}
        
        for pid, data in all_data.items():
            if len(data['epochs']) > 0:
                participants.append(pid)
                labels = data['labels']
                counts = np.bincount(labels, minlength=4)[1:]
                for i, c in enumerate(counts):
                    class_counts[i+1].append(c)
        
        x = np.arange(len(participants))
        width = 0.25
        
        for i, (class_id, counts) in enumerate(class_counts.items()):
            ax.bar(x + i*width, counts, width, label=f'Class {class_id}')
        
        ax.set_xlabel('Participant')
        ax.set_ylabel('Number of epochs')
        ax.set_title('Class Distribution by Participant')
        ax.set_xticks(x + width)
        ax.set_xticklabels(participants)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Overall class distribution pie chart
        ax = axes[0, 1]
        all_labels = []
        for data in all_data.values():
            if len(data['epochs']) > 0:
                all_labels.extend(data['labels'])
        
        if all_labels:
            class_counts = np.bincount(all_labels)[1:]
            ax.pie(class_counts, labels=['Class 1', 'Class 2', 'Class 3'],
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Overall Class Distribution')
        
        # 3. Valid trials per session
        ax = axes[1, 0]
        session_counts = {'Session 1': 0, 'Session 2': 0, 'Test': 0}
        
        for data in all_data.values():
            for meta in data.get('metadata', []):
                session = meta['session']
                if session in session_counts:
                    session_counts[session] += 1
        
        ax.bar(session_counts.keys(), session_counts.values())
        ax.set_xlabel('Session')
        ax.set_ylabel('Total epochs')
        ax.set_title('Epochs per Session Type')
        ax.grid(True, alpha=0.3)
        
        # 4. Data quality metrics
        ax = axes[1, 1]
        # This would show artifact rejection rates, SNR, etc.
        ax.text(0.5, 0.5, 'Data Quality Metrics\n(To be implemented)', 
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Processing Quality')
        
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of complete pipeline"""
    # Initialize pipeline
    # Use relative path for cross-platform compatibility
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "MI")
    pipeline = MIProcessingPipeline(base_path)
    
    # Option 1: Process single participant
    print("Processing T-005...")
    t005_data = pipeline.process_participant("T-005", sessions=["Session 1"])
    
    # Save processed data
    pipeline.save_processed_data(
        t005_data, 
        os.path.join(base_path, "processed_data", "T-005_processed.pkl")
    )
    
    # Option 2: Process all participants
    print("\n\nProcessing all participants...")
    all_data = pipeline.process_all_participants()
    
    # Generate summary report
    pipeline.generate_summary_report(all_data)
    
    # Save all data
    pipeline.save_processed_data(
        all_data,
        os.path.join(base_path, "processed_data", "all_participants.pkl")
    )
    
    print("\n✅ Pipeline complete!")
    print("Next steps:")
    print("1. Run feature_extraction.py to extract ML features")
    print("2. Run classical_ml.py to train classifiers")
    print("3. Integrate trained model with ErrP simulator")


if __name__ == "__main__":
    main()