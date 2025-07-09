"""
Advanced feature extraction for ErrP classification.
Based on the hybrid approach from exo implementation:
- Feature A: Tangent-space covariance features using Riemannian geometry
- Feature B: N and Pe component mean amplitudes
"""

import numpy as np
from scipy import signal
from sklearn.covariance import shrunk_covariance
try:
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.utils.covariance import covariances
    PYRIEMANN_AVAILABLE = True
except ImportError:
    PYRIEMANN_AVAILABLE = False
    import warnings
    warnings.warn("pyriemann not installed. Using simplified covariance features instead of full tangent space.")
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """
    Extracts advanced features for ErrP classification using hybrid approach.
    """
    
    def __init__(
        self,
        sampling_rate: int = 512,
        n_channels: int = 16,
        channel_names: Optional[List[str]] = None,
        selected_channels: Optional[List[str]] = None
    ):
        """
        Initialize the advanced feature extractor.
        
        Args:
            sampling_rate: Sampling frequency in Hz
            n_channels: Number of EEG channels
            channel_names: List of channel names
            selected_channels: Subset of channels to use (if None, use all)
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.channel_names = channel_names or [f"Ch{i+1}" for i in range(n_channels)]
        
        # Best channels from exo implementation (adapt to available channels)
        self.exo_best_channels = ['FZ', 'FC1', 'FC2', 'CZ', 'TP9', 'PO9', 'PO10', 'F1', 'C1', 'FCz']
        self.selected_channels = selected_channels
        
        # Initialize tangent space transformer (will be fitted on training data)
        self.tangent_space = None
        
        # Time windows for features (in seconds)
        self.tangent_window = (0.15, 0.45)  # For covariance features
        self.n_window = (0.2, 0.3)          # N component window
        self.pe_window = (0.3, 0.4)         # Pe component window
        
    def get_channel_indices(self, channels: List[str]) -> List[int]:
        """Get indices of specified channels."""
        indices = []
        for ch in channels:
            if ch in self.channel_names:
                indices.append(self.channel_names.index(ch))
        return indices
    
    def extract_tangent_features(self, epochs: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Extract tangent space features from covariance matrices.
        
        Args:
            epochs: EEG epochs of shape (n_epochs, n_channels, n_samples)
            fit: Whether to fit the tangent space transformer (True for training)
            
        Returns:
            Tangent space features of shape (n_epochs, n_features)
        """
        if not PYRIEMANN_AVAILABLE:
            # Fallback to simplified covariance features
            return self._extract_simple_covariance_features(epochs, fit)
            
        # Convert time window to samples
        start_idx = int(self.tangent_window[0] * self.sampling_rate)
        end_idx = int(self.tangent_window[1] * self.sampling_rate)
        
        # Extract time window
        windowed_epochs = epochs[:, :, start_idx:end_idx]
        
        # Compute covariance matrices with shrinkage
        try:
            covs = covariances(windowed_epochs, estimator='scm')
        except:
            # Fallback if pyriemann covariances fails
            covs = np.array([np.cov(windowed_epochs[i]) for i in range(windowed_epochs.shape[0])])
        
        # Apply shrinkage for regularization
        for i in range(covs.shape[0]):
            try:
                # Check for NaN/Inf before shrinkage
                if np.any(np.isnan(covs[i])) or np.any(np.isinf(covs[i])):
                    # Replace with identity matrix if covariance is invalid
                    covs[i] = np.eye(covs[i].shape[0])
                else:
                    covs[i] = shrunk_covariance(covs[i])
            except:
                # If shrinkage fails, use original or identity
                if np.any(np.isnan(covs[i])) or np.any(np.isinf(covs[i])):
                    covs[i] = np.eye(covs[i].shape[0])
        
        # Initialize or transform using tangent space
        if fit or self.tangent_space is None:
            self.tangent_space = TangentSpace(metric='riemann')
            tangent_features = self.tangent_space.fit_transform(covs)
        else:
            tangent_features = self.tangent_space.transform(covs)
            
        return tangent_features
    
    def _extract_simple_covariance_features(self, epochs: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Extract simplified covariance-based features when pyriemann is not available.
        """
        # Convert time window to samples
        start_idx = int(self.tangent_window[0] * self.sampling_rate)
        end_idx = int(self.tangent_window[1] * self.sampling_rate)
        
        # Extract time window
        windowed_epochs = epochs[:, :, start_idx:end_idx]
        
        n_epochs, n_channels, n_samples = windowed_epochs.shape
        features = []
        
        for i in range(n_epochs):
            # Compute covariance matrix with shrinkage
            epoch_data = windowed_epochs[i]
            
            # Check if epoch has valid data
            if np.any(np.isnan(epoch_data)) or np.any(np.isinf(epoch_data)):
                # Use identity matrix for invalid epochs
                cov = np.eye(n_channels)
            else:
                try:
                    cov = np.cov(epoch_data)
                    # Check covariance validity
                    if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
                        cov = np.eye(n_channels)
                    else:
                        cov = shrunk_covariance(cov, shrinkage=0.1)
                except:
                    cov = np.eye(n_channels)
            
            # Extract upper triangular (including diagonal)
            upper_tri = cov[np.triu_indices(n_channels)]
            features.append(upper_tri)
        
        features = np.array(features)
        
        # Normalize features
        if fit:
            self.mean_features = np.mean(features, axis=0)
            self.std_features = np.std(features, axis=0) + 1e-6
        
        if hasattr(self, 'mean_features'):
            features = (features - self.mean_features) / self.std_features
            
        return features
    
    def extract_amplitude_features(self, epochs: np.ndarray) -> np.ndarray:
        """
        Extract N and Pe component mean amplitudes.
        
        Args:
            epochs: EEG epochs of shape (n_epochs, n_channels, n_samples)
            
        Returns:
            Amplitude features of shape (n_epochs, n_channels * 2)
        """
        n_epochs = epochs.shape[0]
        n_channels = epochs.shape[1]
        
        # Convert time windows to samples
        n_start = int(self.n_window[0] * self.sampling_rate)
        n_end = int(self.n_window[1] * self.sampling_rate)
        pe_start = int(self.pe_window[0] * self.sampling_rate)
        pe_end = int(self.pe_window[1] * self.sampling_rate)
        
        # Extract mean amplitudes for each window
        n_amplitudes = np.mean(epochs[:, :, n_start:n_end], axis=2)
        pe_amplitudes = np.mean(epochs[:, :, pe_start:pe_end], axis=2)
        
        # Concatenate features
        amplitude_features = np.hstack([n_amplitudes, pe_amplitudes])
        
        return amplitude_features
    
    def extract_features(self, epochs: np.ndarray, fit: bool = False) -> Dict[str, np.ndarray]:
        """
        Extract all features (tangent space + amplitudes).
        
        Args:
            epochs: EEG epochs of shape (n_epochs, n_channels, n_samples)
            fit: Whether to fit the tangent space transformer (True for training)
            
        Returns:
            Dictionary containing:
                - 'tangent': Tangent space features
                - 'amplitude': Amplitude features
                - 'hybrid': Combined features
        """
        # Apply channel selection if specified
        if self.selected_channels:
            channel_indices = self.get_channel_indices(self.selected_channels)
            if channel_indices:
                epochs = epochs[:, channel_indices, :]
                logger.info(f"Using {len(channel_indices)} selected channels")
        
        # Extract tangent space features
        tangent_features = self.extract_tangent_features(epochs, fit=fit)
        
        # Extract amplitude features
        amplitude_features = self.extract_amplitude_features(epochs)
        
        # Combine features
        hybrid_features = np.hstack([tangent_features, amplitude_features])
        
        return {
            'tangent': tangent_features,
            'amplitude': amplitude_features,
            'hybrid': hybrid_features
        }
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names for each feature type."""
        n_channels = len(self.selected_channels) if self.selected_channels else self.n_channels
        
        # Tangent space features (upper triangular of covariance matrix)
        n_tangent = n_channels * (n_channels + 1) // 2
        tangent_names = [f"tangent_{i}" for i in range(n_tangent)]
        
        # Amplitude features
        amplitude_names = []
        channels = self.selected_channels if self.selected_channels else self.channel_names
        for ch in channels:
            amplitude_names.extend([f"{ch}_N_amp", f"{ch}_Pe_amp"])
        
        return {
            'tangent': tangent_names,
            'amplitude': amplitude_names,
            'hybrid': tangent_names + amplitude_names
        }