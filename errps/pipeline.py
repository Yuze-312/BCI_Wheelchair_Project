"""
ErrP Processing Pipeline

Complete pipeline for processing EEG data to detect error-related potentials.
Integrates all components: loading, preprocessing, epoch extraction, and analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib
# Use non-interactive backend to avoid Qt issues on Windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

from .data_loader import ErrPDataLoader
from .preprocessor import ErrPPreprocessor
from .epoch_extractor import ErrPEpochExtractor
from .analyzer import ErrPAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPPipeline:
    """Complete ErrP processing pipeline"""
    
    def __init__(self, sampling_rate: int = 512):
        """
        Initialize the ErrP processing pipeline
        
        Args:
            sampling_rate: EEG sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        
        # Initialize components
        self.loader = ErrPDataLoader(sampling_rate)
        self.preprocessor = ErrPPreprocessor(sampling_rate)
        self.extractor = ErrPEpochExtractor(sampling_rate)
        self.analyzer = ErrPAnalyzer(sampling_rate)
        
        # Pipeline parameters
        self.params = {
            'preprocess': {
                'spatial_filter': 'car',
                'remove_powerline': True,
                'bandpass_low': 1.0,
                'bandpass_high': 10.0
            },
            'epochs': {
                'epoch_start': -0.2,
                'epoch_end': 0.8,
                'baseline_correction': True
            },
            'analysis': {
                'classifier_type': 'lda',
                'cross_validation_folds': 5
            }
        }
        
    def process_session(self, session_path: str, 
                       save_intermediate: bool = False) -> Dict:
        """
        Process a complete session from raw data to ErrP analysis
        
        Args:
            session_path: Path to session directory
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Dictionary with all processing results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing ErrP session: {session_path}")
        logger.info(f"{'='*60}")
        
        results = {}
        
        # 1. Load data
        logger.info("\n1. Loading session data...")
        try:
            session_data = self.loader.load_session(session_path)
            results['raw_data'] = session_data
            logger.info(f"   Loaded {len(session_data['eeg_data'])} samples")
            logger.info(f"   Found {len(session_data['events'])} events")
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return {'error': str(e)}
        
        # 2. Preprocess EEG data
        logger.info("\n2. Preprocessing EEG data...")
        preprocessed = self.preprocessor.preprocess(
            session_data['eeg_data'],
            apply_spatial_filter=self.params['preprocess']['spatial_filter'],
            remove_powerline=self.params['preprocess']['remove_powerline']
        )
        results['preprocessed_data'] = preprocessed
        
        # Update preprocessor parameters
        self.preprocessor.params['bandpass_low'] = self.params['preprocess']['bandpass_low']
        self.preprocessor.params['bandpass_high'] = self.params['preprocess']['bandpass_high']
        
        # 3. Extract epochs
        logger.info("\n3. Extracting epochs...")
        
        # Extract both error and correct epochs
        comparison_epochs = self.extractor.extract_comparison_epochs(
            preprocessed,
            session_data['events']
        )
        results['epochs'] = comparison_epochs
        
        # Report epoch counts
        n_error = len(comparison_epochs['error']['epochs'])
        n_correct = len(comparison_epochs['correct']['epochs'])
        logger.info(f"   Extracted {n_error} error epochs")
        logger.info(f"   Extracted {n_correct} correct epochs")
        
        # 4. Analyze ERPs
        logger.info("\n4. Analyzing ERPs...")
        erp_analysis = self.analyzer.analyze_erp_components(
            comparison_epochs['error'],
            comparison_epochs['correct']
        )
        results['erp_analysis'] = erp_analysis
        
        # 5. Extract features and train classifier
        if n_error >= 10 and n_correct >= 10:
            logger.info("\n5. Training ErrP classifier...")
            
            # Extract features from all epochs
            all_epochs = np.concatenate([
                comparison_epochs['error']['epochs'],
                comparison_epochs['correct']['epochs']
            ])
            
            all_features = self.extractor.extract_features(
                all_epochs,
                comparison_epochs['error']['times']
            )
            
            # Create labels (1 for error, 0 for correct)
            labels = np.concatenate([
                np.ones(n_error),
                np.zeros(n_correct)
            ])
            
            # Train classifier
            classifier_results = self.analyzer.train_classifier(
                all_features, labels,
                classifier_type=self.params['analysis']['classifier_type']
            )
            results['classifier'] = classifier_results
        else:
            logger.warning("Not enough epochs for classifier training")
            results['classifier'] = {'status': 'insufficient_data'}
        
        # 6. Save intermediate results if requested
        if save_intermediate:
            self._save_intermediate_results(results, session_path)
        
        # Add metadata
        results['metadata'] = {
            'session_path': str(session_path),
            'processing_date': datetime.now().isoformat(),
            'sampling_rate': self.sampling_rate,
            'parameters': self.params
        }
        
        logger.info("\n✅ Session processing complete!")
        
        return results
    
    def process_multiple_sessions(self, session_paths: List[str]) -> Dict:
        """
        Process multiple sessions and combine results
        
        Args:
            session_paths: List of session directory paths
            
        Returns:
            Combined results from all sessions
        """
        all_results = {}
        all_error_epochs = []
        all_correct_epochs = []
        
        # Process each session
        for session_path in session_paths:
            session_results = self.process_session(session_path)
            
            if 'error' not in session_results:
                session_name = Path(session_path).name
                all_results[session_name] = session_results
                
                # Collect epochs for combined analysis
                if 'epochs' in session_results:
                    all_error_epochs.extend(session_results['epochs']['error']['epochs'])
                    all_correct_epochs.extend(session_results['epochs']['correct']['epochs'])
        
        # Perform combined analysis if we have data from multiple sessions
        if len(all_error_epochs) > 0 and len(all_correct_epochs) > 0:
            logger.info(f"\nCombining data from {len(all_results)} sessions...")
            logger.info(f"Total error epochs: {len(all_error_epochs)}")
            logger.info(f"Total correct epochs: {len(all_correct_epochs)}")
            
            # Create combined epoch dictionaries
            combined_error = {
                'epochs': np.array(all_error_epochs),
                'labels': np.ones(len(all_error_epochs)) * 9,  # Error label
                'times': self.extractor.epoch_params['epoch_start'] + \
                        np.arange(all_error_epochs[0].shape[0]) / self.sampling_rate
            }
            
            combined_correct = {
                'epochs': np.array(all_correct_epochs),
                'labels': np.ones(len(all_correct_epochs)) * 8,  # Correct label
                'times': combined_error['times']
            }
            
            # Analyze combined data
            combined_analysis = self.analyzer.analyze_erp_components(
                combined_error, combined_correct
            )
            
            all_results['combined_analysis'] = combined_analysis
        
        return all_results
    
    def generate_report(self, results: Dict, output_path: Optional[str] = None):
        """
        Generate comprehensive report with visualizations
        
        Args:
            results: Processing results
            output_path: Path to save report
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        
        # If this is a single session result
        if 'erp_analysis' in results:
            self._plot_session_results(fig, results)
        # If this contains multiple sessions
        else:
            self._plot_multi_session_results(fig, results)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Report saved to {output_path}")
        
        plt.show()
    
    def _plot_session_results(self, fig: plt.Figure, results: Dict):
        """Plot results from a single session"""
        
        # 1. Raw vs preprocessed data sample
        if 'raw_data' in results and 'preprocessed_data' in results:
            ax = plt.subplot(3, 3, 1)
            
            # Plot 2 seconds of data
            sample_length = 2 * self.sampling_rate
            raw_sample = results['raw_data']['eeg_data'].iloc[:sample_length]
            prep_sample = results['preprocessed_data'].iloc[:sample_length]
            
            # Plot first channel
            time = np.arange(sample_length) / self.sampling_rate
            ax.plot(time, raw_sample.iloc[:, 0], 'b-', alpha=0.5, label='Raw')
            ax.plot(time, prep_sample.iloc[:, 0], 'r-', label='Preprocessed')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (μV)')
            ax.set_title('Raw vs Preprocessed (Ch1)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Event distribution
        if 'raw_data' in results:
            ax = plt.subplot(3, 3, 2)
            events = results['raw_data']['events']
            event_counts = events['event'].value_counts()
            
            ax.bar(event_counts.index.astype(str), event_counts.values)
            ax.set_xlabel('Event Type')
            ax.set_ylabel('Count')
            ax.set_title('Event Distribution')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Epoch counts
        if 'epochs' in results:
            ax = plt.subplot(3, 3, 3)
            epoch_counts = {
                'Error': len(results['epochs']['error']['epochs']),
                'Correct': len(results['epochs']['correct']['epochs'])
            }
            
            ax.bar(epoch_counts.keys(), epoch_counts.values())
            ax.set_ylabel('Number of Epochs')
            ax.set_title('Extracted Epochs')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 4-6. ERP analysis (if available)
        if 'erp_analysis' in results and 'error_erp' in results['erp_analysis']:
            analysis = results['erp_analysis']
            times = analysis['times'] * 1000  # Convert to ms
            
            # Select 3 channels to display
            n_channels = analysis['error_erp'].shape[1]
            channels_to_plot = [0, n_channels//2, n_channels-1] if n_channels > 3 else range(n_channels)
            
            for idx, ch in enumerate(channels_to_plot[:3]):
                ax = plt.subplot(3, 3, 4 + idx)
                
                # Plot ERPs
                ax.plot(times, analysis['error_erp'][:, ch], 'r-', label='Error', linewidth=2)
                ax.plot(times, analysis['correct_erp'][:, ch], 'b-', label='Correct', linewidth=2)
                
                # Highlight significant differences
                sig_mask = analysis['significant_mask'][:, ch]
                if sig_mask.any():
                    ax.fill_between(times, 
                                  analysis['error_erp'][:, ch],
                                  analysis['correct_erp'][:, ch],
                                  where=sig_mask, alpha=0.3, color='gray')
                
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Amplitude (μV)')
                ax.set_title(f'Channel {ch+1} ERPs')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 7. Classifier performance
        if 'classifier' in results and 'accuracy' in results['classifier']:
            ax = plt.subplot(3, 3, 7)
            clf_results = results['classifier']
            
            # Plot cross-validation scores
            cv_scores = clf_results['cv_scores']
            ax.bar(range(len(cv_scores)), cv_scores)
            ax.axhline(y=clf_results['cv_mean'], color='r', linestyle='--', 
                      label=f'Mean: {clf_results["cv_mean"]:.3f}')
            ax.set_xlabel('CV Fold')
            ax.set_ylabel('Accuracy')
            ax.set_title('Cross-Validation Performance')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 8. Confusion matrix
        if 'classifier' in results and 'confusion_matrix' in results['classifier']:
            ax = plt.subplot(3, 3, 8)
            cm = results['classifier']['confusion_matrix']
            
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Correct', 'Error'])
            ax.set_yticklabels(['Correct', 'Error'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center')
            
            plt.colorbar(im, ax=ax)
        
        # 9. Summary text
        ax = plt.subplot(3, 3, 9)
        ax.axis('off')
        
        summary_text = "Processing Summary\n" + "="*30 + "\n"
        
        if 'metadata' in results:
            summary_text += f"Session: {Path(results['metadata']['session_path']).name}\n"
            summary_text += f"Sampling Rate: {results['metadata']['sampling_rate']} Hz\n"
        
        if 'epochs' in results:
            n_error = len(results['epochs']['error']['epochs'])
            n_correct = len(results['epochs']['correct']['epochs'])
            summary_text += f"\nEpochs:\n"
            summary_text += f"  Error: {n_error}\n"
            summary_text += f"  Correct: {n_correct}\n"
        
        if 'classifier' in results and 'accuracy' in results['classifier']:
            summary_text += f"\nClassifier Performance:\n"
            summary_text += f"  Type: {results['classifier']['classifier_type']}\n"
            summary_text += f"  Accuracy: {results['classifier']['accuracy']:.3f}\n"
            summary_text += f"  CV Score: {results['classifier']['cv_mean']:.3f}±{results['classifier']['cv_std']:.3f}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    def _plot_multi_session_results(self, fig: plt.Figure, results: Dict):
        """Plot results from multiple sessions"""
        # Implementation for multiple session visualization
        # This would show combined analysis results
        pass
    
    def _save_intermediate_results(self, results: Dict, session_path: str):
        """Save intermediate processing results"""
        session_path = Path(session_path)
        output_dir = session_path / 'errp_processing'
        output_dir.mkdir(exist_ok=True)
        
        # Save preprocessed data
        if 'preprocessed_data' in results:
            results['preprocessed_data'].to_csv(
                output_dir / 'preprocessed_eeg.csv', index=False
            )
        
        # Save epochs
        if 'epochs' in results:
            np.save(output_dir / 'error_epochs.npy', 
                   results['epochs']['error']['epochs'])
            np.save(output_dir / 'correct_epochs.npy', 
                   results['epochs']['correct']['epochs'])
        
        # Save analysis results
        if 'erp_analysis' in results:
            with open(output_dir / 'erp_analysis.pkl', 'wb') as f:
                pickle.dump(results['erp_analysis'], f)
        
        logger.info(f"Intermediate results saved to {output_dir}")
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save complete processing results
        
        Args:
            results: Processing results
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to {output_path}")
    
    def load_results(self, file_path: str) -> Dict:
        """
        Load previously saved results
        
        Args:
            file_path: Path to saved results
            
        Returns:
            Processing results
        """
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
        
        return results


def main():
    """Example usage of the ErrP pipeline"""
    # Initialize pipeline
    pipeline = ErrPPipeline(sampling_rate=512)
    
    # Example: Process a single session
    session_path = "C:/Users/yuzeb/EEG_data/T-001/Session 3"
    
    # Process session
    results = pipeline.process_session(session_path, save_intermediate=True)
    
    # Generate report
    pipeline.generate_report(results, output_path="errp_analysis_report.png")
    
    # Save complete results
    pipeline.save_results(results, "errp_processing_results.pkl")
    
    # If classifier was trained, save the model
    if pipeline.analyzer.is_trained:
        pipeline.analyzer.save_model("errp_model")
    
    print("\n✅ ErrP processing pipeline complete!")
    print("Next steps:")
    print("1. Review the generated report")
    print("2. Use the trained model for real-time ErrP detection")
    print("3. Integrate with the BCI wheelchair control system")


if __name__ == "__main__":
    main()