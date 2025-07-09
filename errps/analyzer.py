"""
ErrP Analyzer Module

Analyzes extracted epochs to detect and characterize error-related potentials.
Includes:
- Statistical analysis of ErrP components
- Machine learning classification
- Visualization of results
- Real-time detection capabilities
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
# Use non-interactive backend to avoid Qt issues on Windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPAnalyzer:
    """Analyze EEG epochs to detect and characterize error-related potentials"""
    
    def __init__(self, sampling_rate: int = 512):
        """
        Initialize the ErrP analyzer
        
        Args:
            sampling_rate: EEG sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        
        # Analysis parameters
        self.analysis_params = {
            'significance_level': 0.05,
            'min_epochs_for_analysis': 10,
            'classifier_type': 'lda',  # 'lda', 'svm', or 'rf'
            'cross_validation_folds': 5
        }
        
    def analyze_erp_components(self, error_epochs: Dict, correct_epochs: Dict) -> Dict:
        """
        Perform statistical analysis of ERP components
        
        Args:
            error_epochs: Dictionary from epoch extractor with error epochs
            correct_epochs: Dictionary from epoch extractor with correct epochs
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        # Check if we have enough epochs
        n_error = len(error_epochs['epochs'])
        n_correct = len(correct_epochs['epochs'])
        
        logger.info(f"Analyzing {n_error} error epochs vs {n_correct} correct epochs")
        
        if n_error < self.analysis_params['min_epochs_for_analysis'] or \
           n_correct < self.analysis_params['min_epochs_for_analysis']:
            logger.warning("Not enough epochs for statistical analysis")
            return {'status': 'insufficient_data'}
        
        # Compute ERPs
        error_erp = np.mean(error_epochs['epochs'], axis=0)
        correct_erp = np.mean(correct_epochs['epochs'], axis=0)
        
        # Compute difference wave
        difference_wave = error_erp - correct_erp
        
        # Statistical comparison at each time point
        times = error_epochs['times']
        n_channels = error_erp.shape[1]
        p_values = np.zeros((len(times), n_channels))
        
        for t_idx in range(len(times)):
            for ch_idx in range(n_channels):
                # Perform t-test at this time point
                error_values = error_epochs['epochs'][:, t_idx, ch_idx]
                correct_values = correct_epochs['epochs'][:, t_idx, ch_idx]
                
                _, p_value = stats.ttest_ind(error_values, correct_values)
                p_values[t_idx, ch_idx] = p_value
        
        # Find significant time windows
        significant_mask = p_values < self.analysis_params['significance_level']
        
        # Identify ErrP components
        components = self._identify_components(difference_wave, times, significant_mask)
        
        results = {
            'error_erp': error_erp,
            'correct_erp': correct_erp,
            'difference_wave': difference_wave,
            'p_values': p_values,
            'significant_mask': significant_mask,
            'components': components,
            'n_error': n_error,
            'n_correct': n_correct,
            'times': times
        }
        
        return results
    
    def _identify_components(self, difference_wave: np.ndarray, 
                           times: np.ndarray, 
                           significant_mask: np.ndarray) -> Dict:
        """
        Identify ErrP components in the difference wave
        
        Args:
            difference_wave: Error minus correct ERP
            times: Time vector
            significant_mask: Boolean mask of significant differences
            
        Returns:
            Dictionary describing detected components
        """
        components = {}
        
        # Define expected component windows
        component_windows = {
            'ERN': (0.0, 0.15),    # Error-Related Negativity
            'Pe': (0.2, 0.5),      # Error Positivity
            'late': (0.5, 0.8)     # Late positive component
        }
        
        for comp_name, (start_time, end_time) in component_windows.items():
            # Get time window
            time_mask = (times >= start_time) & (times <= end_time)
            
            if not time_mask.any():
                continue
            
            # Analyze each channel
            channel_results = []
            
            for ch_idx in range(difference_wave.shape[1]):
                window_data = difference_wave[time_mask, ch_idx]
                window_sig = significant_mask[time_mask, ch_idx]
                
                if window_sig.any():
                    # Find peak
                    if comp_name == 'ERN':
                        # ERN is negative
                        peak_idx = np.argmin(window_data)
                    else:
                        # Pe and late are positive
                        peak_idx = np.argmax(window_data)
                    
                    peak_amplitude = window_data[peak_idx]
                    peak_latency = times[time_mask][peak_idx]
                    
                    channel_results.append({
                        'channel': ch_idx,
                        'peak_amplitude': peak_amplitude,
                        'peak_latency': peak_latency,
                        'is_significant': True
                    })
            
            if channel_results:
                components[comp_name] = channel_results
        
        return components
    
    def train_classifier(self, features: np.ndarray, labels: np.ndarray,
                        classifier_type: Optional[str] = None) -> Dict:
        """
        Train a classifier to detect ErrPs
        
        Args:
            features: Feature array (n_epochs x n_features)
            labels: Binary labels (0=correct, 1=error)
            classifier_type: Type of classifier ('lda', 'svm', or 'rf')
            
        Returns:
            Dictionary with training results
        """
        if classifier_type is None:
            classifier_type = self.analysis_params['classifier_type']
        
        logger.info(f"Training {classifier_type} classifier with {len(features)} samples")
        
        # Initialize scaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Initialize classifier
        if classifier_type == 'lda':
            self.classifier = LinearDiscriminantAnalysis()
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', probability=True)
        elif classifier_type == 'rf':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        # Perform cross-validation
        cv = StratifiedKFold(n_splits=self.analysis_params['cross_validation_folds'], 
                           shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.classifier, features_scaled, labels, cv=cv)
        
        # Train on full dataset
        self.classifier.fit(features_scaled, labels)
        self.is_trained = True
        
        # Get predictions for training set
        predictions = self.classifier.predict(features_scaled)
        
        # Calculate metrics
        accuracy = np.mean(predictions == labels)
        report = classification_report(labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(labels, predictions)
        
        results = {
            'classifier_type': classifier_type,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'n_features': features.shape[1],
            'n_samples': len(features),
            'class_balance': np.bincount(labels.astype(int))
        }
        
        logger.info(f"Training complete: Accuracy={accuracy:.3f}, "
                   f"CV={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")
        
        return results
    
    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict ErrP presence in new epochs
        
        Args:
            features: Feature array (n_epochs x n_features)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def visualize_results(self, analysis_results: Dict, 
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive visualization of ErrP analysis
        
        Args:
            analysis_results: Results from analyze_erp_components
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(15, 10))
        
        # Check if we have data
        if 'status' in analysis_results and analysis_results['status'] == 'insufficient_data':
            plt.text(0.5, 0.5, 'Insufficient data for analysis', 
                    ha='center', va='center', fontsize=16)
            return fig
        
        times = analysis_results['times'] * 1000  # Convert to ms
        n_channels = analysis_results['error_erp'].shape[1]
        
        # Select channels to plot (max 4)
        channels_to_plot = min(n_channels, 4)
        
        # 1. ERP comparison for selected channels
        for ch in range(channels_to_plot):
            ax = plt.subplot(channels_to_plot, 3, ch*3 + 1)
            
            # Plot ERPs
            ax.plot(times, analysis_results['error_erp'][:, ch], 
                   'r-', label='Error', linewidth=2)
            ax.plot(times, analysis_results['correct_erp'][:, ch], 
                   'b-', label='Correct', linewidth=2)
            
            # Mark significant regions
            sig_mask = analysis_results['significant_mask'][:, ch]
            if sig_mask.any():
                ax.fill_between(times, 
                              analysis_results['error_erp'][:, ch],
                              analysis_results['correct_erp'][:, ch],
                              where=sig_mask, alpha=0.3, color='gray',
                              label='p<0.05')
            
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Amplitude (μV)')
            ax.set_title(f'Channel {ch+1} - ERPs')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. Difference wave
            ax = plt.subplot(channels_to_plot, 3, ch*3 + 2)
            ax.plot(times, analysis_results['difference_wave'][:, ch], 
                   'g-', linewidth=2)
            ax.fill_between(times, 0, analysis_results['difference_wave'][:, ch],
                          where=sig_mask, alpha=0.3, color='green')
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('Difference (μV)')
            ax.set_title(f'Channel {ch+1} - Difference Wave')
            ax.grid(True, alpha=0.3)
            
            # 3. P-values
            ax = plt.subplot(channels_to_plot, 3, ch*3 + 3)
            ax.semilogy(times, analysis_results['p_values'][:, ch], 'k-')
            ax.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
            ax.set_xlabel('Time (ms)')
            ax.set_ylabel('p-value')
            ax.set_title(f'Channel {ch+1} - Statistical Significance')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def save_model(self, path: str):
        """
        Save trained classifier and scaler
        
        Args:
            path: Path to save model files
        """
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        classifier_path = path / 'errp_classifier.pkl'
        joblib.dump(self.classifier, classifier_path)
        
        # Save scaler
        scaler_path = path / 'errp_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        
        # Save parameters
        params_path = path / 'errp_params.pkl'
        params = {
            'sampling_rate': self.sampling_rate,
            'analysis_params': self.analysis_params,
            'classifier_type': self.analysis_params['classifier_type']
        }
        joblib.dump(params, params_path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load trained classifier and scaler
        
        Args:
            path: Path to model files
        """
        path = Path(path)
        
        # Load classifier
        classifier_path = path / 'errp_classifier.pkl'
        self.classifier = joblib.load(classifier_path)
        
        # Load scaler
        scaler_path = path / 'errp_scaler.pkl'
        self.scaler = joblib.load(scaler_path)
        
        # Load parameters
        params_path = path / 'errp_params.pkl'
        params = joblib.load(params_path)
        self.sampling_rate = params['sampling_rate']
        self.analysis_params = params['analysis_params']
        
        self.is_trained = True
        logger.info(f"Model loaded from {path}")