#!/usr/bin/env python
"""
ERP Hybrid Classifier - Adapted for BCI Wheelchair Project
Based on the exo implementation with tangent space features and LDA.
Supports both within-subject k-fold CV and LOSO (if multiple subjects available).
"""

import numpy as np
import pandas as pd
import warnings
import os
import pickle
import argparse
from datetime import datetime
import logging

warnings.filterwarnings("ignore", category=RuntimeWarning)

from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# We now use participant-specific channel selection instead of fixed channels


def load_participant_data(participant: str, data_dir: str = "../errp_results"):
    """Load epochs from merged data file."""
    npz_file = os.path.join(data_dir, f"{participant}_merged_errp_data.npz")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"No data found at {npz_file}. Run process_errp_data.py first.")
    
    logger.info(f"Loading data from {npz_file}")
    data = np.load(npz_file, allow_pickle=True)
    
    error_epochs = data['error_epochs']
    correct_epochs = data['correct_epochs']
    
    # Check shape and transpose if needed
    if error_epochs.ndim == 3 and error_epochs.shape[2] == 16 and error_epochs.shape[1] > 16:
        logger.info("Transposing epochs to correct format")
        error_epochs = error_epochs.transpose(0, 2, 1)
        if len(correct_epochs) > 0:
            correct_epochs = correct_epochs.transpose(0, 2, 1)
    
    # Combine epochs and create labels
    if len(correct_epochs) > 0:
        all_epochs = np.concatenate([correct_epochs, error_epochs], axis=0)
        labels = np.concatenate([
            np.zeros(len(correct_epochs), dtype=int),
            np.ones(len(error_epochs), dtype=int)
        ])
    else:
        all_epochs = error_epochs
        labels = np.ones(len(error_epochs), dtype=int)
    
    return {
        'epochs': all_epochs,  # (n_epochs, n_channels, n_samples)
        'labels': labels,
        'sampling_rate': int(data['sampling_rate']),
        'n_error': len(error_epochs),
        'n_correct': len(correct_epochs) if len(correct_epochs) > 0 else 0
    }


def select_best_channels(epochs, labels, n_channels=5):
    """
    Dynamically select best channels for this participant based on performance.
    
    Args:
        epochs: EEG epochs (n_epochs, n_channels, n_samples)
        labels: Class labels
        n_channels: Number of channels to select
        
    Returns:
        indices: Selected channel indices
        selected_names: Selected channel names
    """
    from channel_selector import ChannelSelector
    
    # Generate channel names
    channel_names = [f"Ch{i+1}" for i in range(epochs.shape[1])]
    
    # Create selector and find best channels
    selector = ChannelSelector(channel_names)
    logger.info("Performing participant-specific channel selection...")
    
    # Select channels based on individual performance
    selected_channels = selector.select_channels_by_performance(
        epochs, labels, 
        n_channels=n_channels,
        method='individual'  # Score each channel independently
    )
    
    # Get indices
    indices = selector.get_channel_indices(selected_channels)
    
    return indices, selected_channels


def extract_hybrid_features(epochs, sampling_rate=512):
    """
    Extract hybrid features: tangent space + amplitude features.
    
    Time windows:
    - Tangent space: 0.0-0.45s (covariance features)
    - N component: 0.0-0.30s (mean amplitude)
    - Pe component: 0.30-0.50s (mean amplitude)
    
    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        X: Feature matrix (n_epochs, n_features)
    """
    n_epochs, n_channels, n_samples = epochs.shape
    
    # Create time vector
    epoch_duration = n_samples / sampling_rate
    t = np.linspace(-0.2, epoch_duration - 0.2, n_samples, endpoint=False)
    
    # 1. Tangent space features (0.0-0.45s window)
    win = np.where((t >= 0.0) & (t <= 0.45))[0]
    if len(win) == 0:
        # Fallback if window is outside epoch range
        win = np.arange(int(0.0 * sampling_rate), int(0.45 * sampling_rate))
        win = win[win < n_samples]
    
    epochs_win = epochs[:, :, win]
    
    # Compute covariance matrices
    def cov_meanfree(X, ridge=1e-4):
        X = X - X.mean(axis=1, keepdims=True)
        return np.cov(X) + ridge * np.eye(X.shape[0])
    
    covs = np.array([cov_meanfree(ep) for ep in epochs_win])
    
    # Transform to tangent space
    ts = TangentSpace(metric='riemann')
    ts_feat = ts.fit_transform(covs)  # (n_epochs, n_features)
    
    # 2. Amplitude features
    # N component (0-300ms)
    idx_N = np.where((t >= 0.0) & (t <= 0.30))[0]
    if len(idx_N) == 0:
        idx_N = np.arange(int(0.0 * sampling_rate), int(0.30 * sampling_rate))
        idx_N = idx_N[idx_N < n_samples]
    
    # Pe component (300-500ms)
    idx_P = np.where((t >= 0.30) & (t <= 0.50))[0]
    if len(idx_P) == 0:
        idx_P = np.arange(int(0.30 * sampling_rate), int(0.50 * sampling_rate))
        idx_P = idx_P[idx_P < n_samples]
    
    amp_feat = np.hstack([
        epochs[:, :, idx_N].mean(axis=2),  # Mean amplitude in N window
        epochs[:, :, idx_P].mean(axis=2)   # Mean amplitude in Pe window
    ])
    
    # Combine features
    X = np.hstack([ts_feat, amp_feat])
    
    return X, {'n_tangent': ts_feat.shape[1], 'n_amplitude': amp_feat.shape[1]}


def train_hybrid_classifier(participant: str, data_dir: str, output_dir: str, 
                          mode: str = 'kfold', n_folds: int = 5):
    """
    Train the hybrid classifier with specified cross-validation mode.
    
    Args:
        participant: Participant ID
        data_dir: Directory with merged .npz files
        output_dir: Output directory for results
        mode: 'kfold' for k-fold CV or 'loso' if multiple subjects
        n_folds: Number of folds for k-fold CV
    """
    # Load data
    data = load_participant_data(participant, data_dir)
    epochs = data['epochs']
    labels = data['labels']
    
    logger.info(f"Data shape: {epochs.shape}")
    logger.info(f"Class distribution: {data['n_correct']} correct, {data['n_error']} error")
    
    # Select channels dynamically based on participant's data
    channel_indices, selected_names = select_best_channels(epochs, labels, n_channels=5)
    logger.info(f"Selected channels for {participant}: {selected_names}")
    
    epochs_selected = epochs[:, channel_indices, :]
    
    # Extract features
    logger.info("Extracting hybrid features...")
    X, feature_info = extract_hybrid_features(epochs_selected, data['sampling_rate'])
    logger.info(f"Feature dimensions: {X.shape[1]} ({feature_info['n_tangent']} tangent + {feature_info['n_amplitude']} amplitude)")
    
    # Create pipeline
    pipe = make_pipeline(StandardScaler(), LDA())
    
    # Cross-validation
    if mode == 'kfold':
        logger.info(f"\nPerforming {n_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        accs, aucs = [], []
        all_predictions = []
        all_labels = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, labels), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            accs.append(acc)
            aucs.append(auc)
            all_predictions.extend(y_pred)
            all_labels.extend(y_test)
            
            logger.info(f"Fold {fold}: Accuracy={acc:.3f}, AUC={auc:.3f}")
        
        # Overall results
        overall_acc = np.mean(accs)
        overall_auc = np.mean(aucs)
        std_acc = np.std(accs)
        std_auc = np.std(aucs)
        
        logger.info(f"\n{n_folds}-Fold CV Results:")
        logger.info(f"Accuracy: {overall_acc:.3f} ± {std_acc:.3f}")
        logger.info(f"AUC: {overall_auc:.3f} ± {std_auc:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"FN={cm[1,0]}, TP={cm[1,1]}")
    
    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    pipe.fit(X, labels)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'participant': participant,
        'mode': mode,
        'selected_channels': selected_names,
        'feature_info': feature_info,
        'cv_results': {
            'accuracies': accs if mode == 'kfold' else [],
            'aucs': aucs if mode == 'kfold' else [],
            'mean_accuracy': overall_acc if mode == 'kfold' else 0,
            'mean_auc': overall_auc if mode == 'kfold' else 0,
            'std_accuracy': std_acc if mode == 'kfold' else 0,
            'std_auc': std_auc if mode == 'kfold' else 0
        },
        'pipeline': pipe,
        'timestamp': timestamp
    }
    
    # Save model
    model_file = os.path.join(output_dir, f"{participant}_hybrid_model_{timestamp}.pkl")
    with open(model_file, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\nModel saved to: {model_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{participant}_hybrid_results_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Hybrid ERP Classifier Results\n")
        f.write(f"============================\n")
        f.write(f"Participant: {participant}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Selected channels: {selected_names}\n")
        f.write(f"Features: {X.shape[1]} dims ({feature_info['n_tangent']} tangent + {feature_info['n_amplitude']} amplitude)\n")
        f.write(f"\nCross-validation: {mode}\n")
        if mode == 'kfold':
            f.write(f"Folds: {n_folds}\n")
            f.write(f"Accuracy: {overall_acc:.3f} ± {std_acc:.3f}\n")
            f.write(f"AUC: {overall_auc:.3f} ± {std_auc:.3f}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train hybrid ERP classifier')
    parser.add_argument('--participant', required=True, help='Participant ID (e.g., T-001)')
    parser.add_argument('--data-dir', default='errp_results', help='Directory with merged .npz files')
    parser.add_argument('--output-dir', default='hybrid_results', help='Output directory')
    parser.add_argument('--mode', default='kfold', choices=['kfold', 'loso'], help='CV mode')
    parser.add_argument('--n-folds', type=int, default=5, help='Number of folds for k-fold CV')
    args = parser.parse_args()
    
    results = train_hybrid_classifier(
        args.participant,
        args.data_dir,
        args.output_dir,
        args.mode,
        args.n_folds
    )
    
    print("\nDone ✓")


if __name__ == "__main__":
    main()