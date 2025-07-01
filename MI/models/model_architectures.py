#!/usr/bin/env python
"""
Model Architectures for Motor Imagery Classification

Implements multiple architectures:
1. Original: CSP + LDA (current implementation)
2. Wheelchair-Gaussian: Laplacian + PSD + CVA + Gaussian Classifier
3. VR-SVM: Laplacian + PSD + SVM with RBF kernel
"""

import numpy as np
from scipy import signal
from scipy.spatial.distance import cdist
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
import pickle
from abc import ABC, abstractmethod


class BaseModelArchitecture(ABC):
    """Base class for all MI classification architectures"""
    
    @abstractmethod
    def preprocess(self, raw_data, srate):
        """Preprocess raw EEG data"""
        pass
    
    @abstractmethod
    def extract_features(self, preprocessed_data):
        """Extract features from preprocessed data"""
        pass
    
    @abstractmethod
    def train_classifier(self, features, labels):
        """Train the classifier"""
        pass
    
    @abstractmethod
    def predict(self, features):
        """Make predictions"""
        pass


class OriginalCSPLDA(BaseModelArchitecture):
    """Original CSP + LDA architecture (current implementation)"""
    
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.csp = None
        self.classifier = None
        self.scaler = StandardScaler()
        
    def preprocess(self, raw_data, srate):
        """Bandpass filter 8-30 Hz"""
        nyquist = srate / 2
        if nyquist > 30:
            b, a = signal.butter(5, [8/nyquist, 30/nyquist], btype='band')
            return signal.filtfilt(b, a, raw_data, axis=0)
        return raw_data
    
    def extract_features(self, preprocessed_data):
        """Apply CSP"""
        if self.csp is None:
            raise ValueError("CSP not fitted yet")
        return self.csp.transform(preprocessed_data)
    
    def train_classifier(self, features, labels):
        """Train LDA classifier"""
        features_scaled = self.scaler.fit_transform(features)
        self.classifier = LinearDiscriminantAnalysis()
        self.classifier.fit(features_scaled, labels)
        
    def predict(self, features):
        """Predict with LDA"""
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict(features_scaled)
    
    def predict_proba(self, features):
        """Get prediction probabilities"""
        features_scaled = self.scaler.transform(features)
        return self.classifier.predict_proba(features_scaled)


class WheelchairGaussian(BaseModelArchitecture):
    """Brain-Controlled Wheelchairs architecture
    Laplacian + PSD + CVA + Gaussian Classifier
    """
    
    def __init__(self, electrode_positions=None):
        self.electrode_positions = electrode_positions
        self.laplacian_matrix = None
        self.cva = None
        self.gaussian_classifier = None
        self.selected_features = None
        self.rejection_threshold = 0.3
        self.evidence_alpha = 0.3  # Exponential smoothing parameter
        self.accumulated_evidence = None
        
    def compute_laplacian_matrix(self, n_channels=16):
        """Compute Laplacian spatial filter matrix"""
        # Simple implementation - each channel minus average of neighbors
        # In practice, would use actual electrode positions
        laplacian = np.eye(n_channels)
        
        # Define neighbor relationships (simplified)
        # Would use actual 10-20 montage positions
        for i in range(n_channels):
            laplacian[i, i] = 1
            # Subtract average of neighbors (simplified to adjacent channels)
            if i > 0:
                laplacian[i, i-1] = -0.25
            if i < n_channels - 1:
                laplacian[i, i+1] = -0.25
                
        return laplacian
    
    def preprocess(self, raw_data, srate):
        """Apply Laplacian spatial filter"""
        n_samples, n_channels = raw_data.shape
        
        if self.laplacian_matrix is None:
            self.laplacian_matrix = self.compute_laplacian_matrix(n_channels)
            
        # Apply Laplacian filter
        filtered_data = raw_data @ self.laplacian_matrix.T
        return filtered_data
    
    def extract_features(self, preprocessed_data, srate=512):
        """Extract PSD features using Welch method
        - 4-48 Hz with 2 Hz resolution
        - 500ms windows with 25% overlap
        - Computed every 62.5ms (16 times/s)
        """
        n_samples, n_channels = preprocessed_data.shape
        
        # Welch parameters
        nperseg = int(0.5 * srate)  # 500ms windows
        noverlap = int(0.125 * srate)  # 25% overlap
        nfft = int(srate / 2)  # For 2 Hz resolution
        
        all_features = []
        
        # Slide window every 62.5ms
        window_step = int(0.0625 * srate)  # 62.5ms
        
        for start_idx in range(0, n_samples - nperseg, window_step):
            window_data = preprocessed_data[start_idx:start_idx + nperseg, :]
            
            channel_features = []
            for ch in range(n_channels):
                # Compute PSD using Welch method
                freqs, psd = signal.welch(window_data[:, ch], 
                                        fs=srate,
                                        nperseg=nperseg,
                                        noverlap=noverlap,
                                        nfft=nfft)
                
                # Extract 4-48 Hz range
                freq_mask = (freqs >= 4) & (freqs <= 48)
                psd_features = psd[freq_mask]
                
                # Also extract mu band (8-13 Hz) power specifically
                mu_mask = (freqs >= 8) & (freqs <= 13)
                mu_power = np.mean(psd[mu_mask])
                
                channel_features.extend(psd_features)
                channel_features.append(mu_power)
                
            all_features.append(channel_features)
            
        return np.array(all_features)
    
    def apply_cva(self, features, labels=None):
        """Apply Canonical Variate Analysis for feature selection"""
        if labels is not None:
            # Training mode - fit CVA
            from sklearn.cross_decomposition import CCA
            
            # Create binary matrix for labels
            unique_labels = np.unique(labels)
            Y = np.zeros((len(labels), len(unique_labels)))
            for i, label in enumerate(labels):
                Y[i, label] = 1
                
            # Fit CCA (similar to CVA)
            # For binary classification, max components = min(n_classes, n_features)
            n_classes = Y.shape[1]
            max_components = min(n_classes, features.shape[1], 10)
            self.cva = CCA(n_components=max_components)
            self.cva.fit(features, Y)
            
            # Select most discriminative features
            scores = np.abs(self.cva.x_weights_).mean(axis=1)
            n_features = int(features.shape[1] * 0.3)  # Keep top 30%
            self.selected_features = np.argsort(scores)[-n_features:]
            
        # Apply feature selection
        if self.selected_features is not None:
            return features[:, self.selected_features]
        return features
    
    def train_classifier(self, features, labels):
        """Train Gaussian classifier"""
        # Apply CVA feature selection
        selected_features = self.apply_cva(features, labels)
        
        # Train Gaussian classifier
        self.gaussian_classifier = GaussianClassifier()
        self.gaussian_classifier.fit(selected_features, labels)
        
    def predict(self, features):
        """Predict with rejection and evidence accumulation"""
        # Apply feature selection
        selected_features = self.apply_cva(features)
        
        # Get probabilities from Gaussian classifier
        probas = self.gaussian_classifier.predict_proba(selected_features)
        
        # Apply rejection threshold
        max_proba = np.max(probas, axis=1)
        predictions = np.argmax(probas, axis=1)
        
        # Reject low confidence predictions
        predictions[max_proba < self.rejection_threshold] = -1  # -1 for rejected
        
        # Evidence accumulation
        if self.accumulated_evidence is None or self.accumulated_evidence.shape != probas.shape:
            self.accumulated_evidence = probas
        else:
            # Exponential smoothing
            self.accumulated_evidence = (self.evidence_alpha * probas + 
                                       (1 - self.evidence_alpha) * self.accumulated_evidence)
            
        # Make decision based on accumulated evidence
        final_predictions = np.argmax(self.accumulated_evidence, axis=1)
        
        return final_predictions
    
    def predict_proba(self, features):
        """Get prediction probabilities"""
        selected_features = self.apply_cva(features)
        return self.gaussian_classifier.predict_proba(selected_features)


class VRSVM(BaseModelArchitecture):
    """High stimuli VR training architecture
    Small Laplacian + PSD (alpha band) + SVM with RBF
    """
    
    def __init__(self):
        self.small_laplacian_matrix = None
        self.svm_classifier = None
        self.scaler = StandardScaler()
        
    def compute_small_laplacian(self, n_channels=16):
        """Compute small Laplacian filter
        Each channel minus average of immediate neighbors only
        """
        # This is a simplified version
        # Real implementation would use actual electrode positions
        laplacian = np.eye(n_channels)
        
        # Simple nearest neighbor structure
        for i in range(n_channels):
            n_neighbors = 0
            if i > 0:
                laplacian[i, i-1] = -1
                n_neighbors += 1
            if i < n_channels - 1:
                laplacian[i, i+1] = -1
                n_neighbors += 1
                
            if n_neighbors > 0:
                laplacian[i, :] /= n_neighbors
                laplacian[i, i] = 1
                
        return laplacian
    
    def preprocess(self, raw_data, srate):
        """Apply small Laplacian and bandpass filter"""
        n_samples, n_channels = raw_data.shape
        
        # Apply small Laplacian
        if self.small_laplacian_matrix is None:
            self.small_laplacian_matrix = self.compute_small_laplacian(n_channels)
            
        laplacian_filtered = raw_data @ self.small_laplacian_matrix.T
        
        # Apply 6-40 Hz bandpass filter (3rd order Butterworth)
        nyquist = srate / 2
        b, a = signal.butter(3, [6/nyquist, 40/nyquist], btype='band')
        filtered_data = signal.filtfilt(b, a, laplacian_filtered, axis=0)
        
        return filtered_data
    
    def extract_features(self, preprocessed_data, srate=512):
        """Extract alpha band (8-13 Hz) power features
        Uses 1-second windows with 0.5s overlap
        """
        n_samples, n_channels = preprocessed_data.shape
        
        # Window parameters
        window_size = int(1.0 * srate)  # 1 second
        overlap = int(0.5 * srate)  # 0.5 second overlap
        step_size = window_size - overlap
        
        all_features = []
        
        for start_idx in range(0, n_samples - window_size + 1, step_size):
            window_data = preprocessed_data[start_idx:start_idx + window_size, :]
            
            channel_features = []
            for ch in range(n_channels):
                # Compute FFT
                fft_result = np.fft.rfft(window_data[:, ch])
                freqs = np.fft.rfftfreq(window_size, 1/srate)
                
                # Square to get PSD
                psd = np.abs(fft_result) ** 2
                
                # Extract alpha band (8-13 Hz) power
                alpha_mask = (freqs >= 8) & (freqs <= 13)
                alpha_power = np.mean(psd[alpha_mask])
                
                channel_features.append(alpha_power)
                
            all_features.append(channel_features)
            
        return np.array(all_features)
    
    def train_classifier(self, features, labels):
        """Train SVM with RBF kernel"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train SVM with exponential RBF kernel
        self.svm_classifier = SVC(
            kernel='rbf',
            gamma='scale',  # Uses 1 / (n_features * X.var())
            C=1.0,
            probability=True,  # Enable probability estimates
            class_weight='balanced'  # Handle class imbalance
        )
        
        self.svm_classifier.fit(features_scaled, labels)
        
    def predict(self, features):
        """Predict with SVM"""
        features_scaled = self.scaler.transform(features)
        return self.svm_classifier.predict(features_scaled)
    
    def predict_proba(self, features):
        """Get prediction probabilities"""
        features_scaled = self.scaler.transform(features)
        return self.svm_classifier.predict_proba(features_scaled)


class GaussianClassifier(BaseEstimator, ClassifierMixin):
    """Gaussian classifier for wheelchair architecture"""
    
    def __init__(self):
        self.class_means = {}
        self.class_covs = {}
        self.class_priors = {}
        self.classes = None
        
    def fit(self, X, y):
        """Fit Gaussian parameters for each class"""
        self.classes = np.unique(y)
        n_samples = len(y)
        
        for class_label in self.classes:
            # Get samples for this class
            class_mask = y == class_label
            class_samples = X[class_mask]
            
            # Compute mean and covariance
            self.class_means[class_label] = np.mean(class_samples, axis=0)
            self.class_covs[class_label] = np.cov(class_samples.T)
            
            # Add regularization to avoid singular covariance
            self.class_covs[class_label] += np.eye(X.shape[1]) * 1e-3
            
            # Compute prior
            self.class_priors[class_label] = np.sum(class_mask) / n_samples
            
        return self
    
    def predict_proba(self, X):
        """Compute class probabilities using Gaussian distributions"""
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        probas = np.zeros((n_samples, n_classes))
        
        for i, class_label in enumerate(self.classes):
            # Compute Gaussian probability
            mean = self.class_means[class_label]
            cov = self.class_covs[class_label]
            prior = self.class_priors[class_label]
            
            # Multivariate Gaussian PDF
            diff = X - mean
            inv_cov = np.linalg.inv(cov)
            
            # Compute log probability to avoid numerical issues
            log_prob = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
            log_prob -= 0.5 * np.log(np.linalg.det(cov))
            log_prob -= 0.5 * X.shape[1] * np.log(2 * np.pi)
            log_prob += np.log(prior)
            
            probas[:, i] = np.exp(log_prob)
            
        # Normalize to get probabilities
        # Handle cases where all probabilities are 0 or NaN
        prob_sums = np.sum(probas, axis=1, keepdims=True)
        prob_sums[prob_sums == 0] = 1  # Avoid division by zero
        probas = probas / prob_sums
        
        # Replace NaN with uniform probabilities
        nan_rows = np.any(np.isnan(probas), axis=1)
        probas[nan_rows] = 1.0 / n_classes
        
        return probas
    
    def predict(self, X):
        """Predict class labels"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]


# Model factory function
def create_model(architecture='original', **kwargs):
    """Factory function to create model architectures
    
    Args:
        architecture: 'original', 'wheelchair', or 'vr_svm'
        **kwargs: Architecture-specific parameters
        
    Returns:
        Model instance
    """
    if architecture == 'original':
        return OriginalCSPLDA(n_components=kwargs.get('n_components', 4))
    elif architecture == 'wheelchair':
        return WheelchairGaussian(electrode_positions=kwargs.get('electrode_positions'))
    elif architecture == 'vr_svm':
        return VRSVM()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    print("MI Model Architectures")
    print("=" * 50)
    print("\nAvailable architectures:")
    print("1. 'original' - CSP + LDA (current implementation)")
    print("2. 'wheelchair' - Laplacian + PSD + CVA + Gaussian")
    print("3. 'vr_svm' - Small Laplacian + Alpha Power + SVM")
    print("\nUse create_model(architecture_name) to instantiate")