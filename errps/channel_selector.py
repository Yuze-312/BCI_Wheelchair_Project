"""
Channel selection for ErrP classification.
Implements methods to select optimal channels based on the exo approach.
"""

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChannelSelector:
    """
    Select optimal channels for ErrP classification.
    """
    
    # Standard 10-20 channel mapping to adapt exo channels to available channels
    CHANNEL_MAPPING = {
        # Frontal channels
        'FZ': ['Fz', 'FZ', 'F0', 'Fp1', 'Fp2', 'Ch1'],
        'F1': ['F1', 'F3', 'Ch2'],
        'FC1': ['FC1', 'F3', 'Ch3'],
        'FC2': ['FC2', 'F4', 'Ch4'],
        'FCz': ['FCz', 'FCZ', 'FC0', 'Fp1', 'Ch5'],
        
        # Central channels
        'CZ': ['Cz', 'CZ', 'C0', 'C3', 'C4', 'Ch6'],
        'C1': ['C1', 'C3', 'Ch7'],
        
        # Temporal-Parietal channels
        'TP9': ['TP9', 'T7', 'T3', 'T5', 'Ch8'],
        
        # Parietal-Occipital channels
        'PO9': ['PO9', 'P7', 'P3', 'O1', 'Ch9'],
        'PO10': ['PO10', 'P8', 'P4', 'O2', 'Ch10']
    }
    
    # Best channels from exo implementation
    EXO_BEST_CHANNELS = ['FZ', 'FC1', 'FC2', 'CZ', 'TP9', 'PO9', 'PO10', 'F1', 'C1', 'FCz']
    
    def __init__(self, channel_names: List[str]):
        """
        Initialize channel selector.
        
        Args:
            channel_names: List of available channel names
        """
        self.channel_names = channel_names
        self.selected_channels = None
        self.channel_scores = None
        
    def select_best_channels_sequentially(self, n_channels: int = 10) -> List[str]:
        """
        Select best channels sequentially based on numeric order.
        This is a simple fallback when performance-based selection is not available.
        
        Args:
            n_channels: Number of channels to select
            
        Returns:
            List of selected channel names
        """
        # For generic channel names (Ch1, Ch2, etc), prioritize lower numbers
        # which often correspond to more central/frontal locations
        selected = []
        
        # First try to get channels with lower numbers (often more important)
        for i in range(1, min(n_channels + 1, len(self.channel_names) + 1)):
            ch_name = f"Ch{i}"
            if ch_name in self.channel_names:
                selected.append(ch_name)
                if len(selected) >= n_channels:
                    break
        
        # If not enough, add any remaining channels
        for ch in self.channel_names:
            if ch not in selected:
                selected.append(ch)
                if len(selected) >= n_channels:
                    break
                    
        return selected[:n_channels]
    
    def select_channels_by_performance(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        n_channels: int = 10,
        method: str = 'individual'
    ) -> List[str]:
        """
        Select channels based on classification performance.
        
        Args:
            epochs: EEG epochs of shape (n_epochs, n_channels, n_samples)
            labels: Class labels
            n_channels: Number of channels to select
            method: Selection method ('individual' or 'forward')
            
        Returns:
            List of selected channel names
        """
        n_total_channels = epochs.shape[1]
        n_classes = len(np.unique(labels))
        
        # Choose appropriate scoring based on number of classes
        if n_classes == 2:
            scoring = 'roc_auc'
        else:
            scoring = 'accuracy'
        logger.info(f"Using {scoring} scoring for channel selection (n_classes={n_classes})")
        
        if method == 'individual':
            # Score each channel individually
            channel_scores = []
            
            for ch_idx in range(n_total_channels):
                # Extract single channel data
                ch_data = epochs[:, ch_idx, :].reshape(epochs.shape[0], -1)
                
                # Standardize
                scaler = StandardScaler()
                ch_data_scaled = scaler.fit_transform(ch_data)
                
                # Evaluate with cross-validation
                lda = LinearDiscriminantAnalysis()
                try:
                    scores = cross_val_score(lda, ch_data_scaled, labels, cv=5, scoring=scoring)
                    channel_scores.append(np.mean(scores))
                except Exception as e:
                    logger.warning(f"Channel {ch_idx} scoring failed: {e}. Using zero score.")
                    channel_scores.append(0.0)
            
            # Select top channels
            self.channel_scores = np.array(channel_scores)
            top_indices = np.argsort(channel_scores)[-n_channels:][::-1]
            self.selected_channels = [self.channel_names[i] for i in top_indices]
            
        elif method == 'forward':
            # Forward selection
            selected_indices = []
            remaining_indices = list(range(n_total_channels))
            
            for _ in range(n_channels):
                best_score = -np.inf
                best_idx = None
                
                for idx in remaining_indices:
                    # Try adding this channel
                    trial_indices = selected_indices + [idx]
                    trial_data = epochs[:, trial_indices, :].reshape(epochs.shape[0], -1)
                    
                    # Standardize
                    scaler = StandardScaler()
                    trial_data_scaled = scaler.fit_transform(trial_data)
                    
                    # Evaluate
                    lda = LinearDiscriminantAnalysis()
                    try:
                        scores = cross_val_score(lda, trial_data_scaled, labels, cv=5, scoring=scoring)
                        score = np.mean(scores)
                    except Exception as e:
                        logger.warning(f"Scoring failed for channel combination: {e}")
                        score = -np.inf
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    
            self.selected_channels = [self.channel_names[i] for i in selected_indices]
        
        else:
            raise ValueError(f"Unknown method: {method}")
            
        logger.info(f"Selected {len(self.selected_channels)} channels: {self.selected_channels}")
        return self.selected_channels
    
    def get_channel_indices(self, channels: List[str]) -> List[int]:
        """Get indices of specified channels."""
        return [self.channel_names.index(ch) for ch in channels if ch in self.channel_names]
    
    def apply_selection(self, epochs: np.ndarray) -> np.ndarray:
        """Apply channel selection to epochs."""
        if self.selected_channels is None:
            raise ValueError("No channels selected yet. Run selection first.")
            
        indices = self.get_channel_indices(self.selected_channels)
        return epochs[:, indices, :]