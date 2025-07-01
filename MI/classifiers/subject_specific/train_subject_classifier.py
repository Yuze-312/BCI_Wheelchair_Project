#!/usr/bin/env python
"""
Subject-Specific MI Classifier Training
Trains a personalized model for a single participant
"""

import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from mne.decoding import CSP
import argparse
import sys
import warnings

# Suppress CCA warnings
warnings.filterwarnings('ignore', message='y residual is constant')

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

# Add path for model architectures
sys.path.append(os.path.join(os.path.dirname(__file__), '../../models'))
from model_architectures import create_model

def load_subject_data(participant_id, data_dir='MI/processed_data'):
    """Load data for a specific participant only"""
    
    print(f"Loading MI data for participant: {participant_id}")
    
    # Load processed data
    data_file = f"{data_dir}/{participant_id}_processed.pkl"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"No data found for participant {participant_id}")
        
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    epochs = data['epochs']
    labels = data['labels']
    
    print(f"  Raw epochs shape: {epochs.shape if hasattr(epochs, 'shape') else 'unknown'}")
    print(f"  Raw labels shape: {labels.shape if hasattr(labels, 'shape') else 'unknown'}")
    print(f"  Unique labels: {np.unique(labels)}")
    
    # Binary classification only (remove rest class)
    mask = labels != 3
    print(f"  Mask sum (non-rest trials): {np.sum(mask)}")
    
    # Ensure numpy arrays
    epochs = np.array(epochs)
    labels = np.array(labels)
    
    epochs = epochs[mask]
    labels = labels[mask]
    
    print(f"  After filtering - epochs shape: {epochs.shape}")
    print(f"  After filtering - labels shape: {labels.shape}")
    
    if len(labels) == 0:
        raise ValueError("No trials left after filtering! Check if data contains only rest class.")
    
    # Convert to binary: Left=0, Right=1
    labels = (labels == 2).astype(int)
    
    print(f"  Loaded {len(labels)} trials")
    print(f"  Left: {np.sum(labels==0)}, Right: {np.sum(labels==1)}")
    print(f"  Data shape: {epochs.shape}")
    
    return epochs, labels

def extract_personalized_features(X_epochs, y_labels, n_components=4):
    """Extract features optimized for individual"""
    
    print("\nExtracting personalized CSP features...")
    print(f"  Input shape: {X_epochs.shape}")
    
    # CSP expects (n_epochs, n_channels, n_times)
    # Ensure double precision to avoid sklearn warnings
    X_epochs = X_epochs.astype(np.float64)
    
    # Check if we need to transpose
    if len(X_epochs.shape) == 3:
        X_csp_format = np.transpose(X_epochs, (0, 2, 1))
        print(f"  Transposed to: {X_csp_format.shape}")
    else:
        print(f"  ERROR: Expected 3D array but got shape: {X_epochs.shape}")
        raise ValueError(f"Expected 3D array but got shape: {X_epochs.shape}")
    
    # Use fewer components for subject-specific (less overfitting)
    # Small regularization helps with limited training data
    csp = CSP(n_components=n_components, reg=0.01, log=True, norm_trace=False)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_csp_format, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Fit CSP on training data only
    csp.fit(X_train, y_train)
    
    # Transform all data
    X_train_csp = csp.transform(X_train)
    X_test_csp = csp.transform(X_test)
    
    print(f"  CSP features shape: {X_train_csp.shape}")
    print(f"  Feature range: [{np.min(X_train_csp):.2f}, {np.max(X_train_csp):.2f}]")
    
    return X_train_csp, X_test_csp, y_train, y_test, csp

def train_subject_specific_models(X_train, X_test, y_train, y_test, X_train_epochs=None, X_test_epochs=None, srate=125):
    """Train multiple models optimized for individual including new architectures"""
    
    print("\n" + "="*60)
    print("SUBJECT-SPECIFIC CLASSIFIERS")
    print("="*60)
    
    results = {}
    models = {}
    architectures = {}
    
    # 1. LDA (often best for small sample sizes)
    print("\n1. Linear Discriminant Analysis:")
    # Ensure data is C-contiguous and double precision
    X_train = np.ascontiguousarray(X_train, dtype=np.float64)
    X_test = np.ascontiguousarray(X_test, dtype=np.float64)
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_train, y_train)
    acc_lda = lda.score(X_test, y_test)
    results['LDA'] = acc_lda
    models['LDA'] = lda
    print(f"  Accuracy: {acc_lda:.1%}")
    
    # Cross-validation for robustness
    cv_scores = cross_val_score(lda, X_train, y_train, cv=5)
    print(f"  CV Mean: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
    
    # 2. Random Forest (can capture non-linear patterns)
    print("\n2. Random Forest:")
    rf = RandomForestClassifier(
        n_estimators=100,  # Fewer trees for subject-specific
        max_depth=5,       # Shallower to prevent overfitting
        min_samples_split=10,
        random_state=42
    )
    rf.fit(X_train, y_train)
    acc_rf = rf.score(X_test, y_test)
    results['RF'] = acc_rf
    models['RF'] = rf
    print(f"  Accuracy: {acc_rf:.1%}")
    
    # 3. Wheelchair Architecture (if epochs provided)
    if X_train_epochs is not None and X_test_epochs is not None:
        print("\n3. Wheelchair Architecture (Laplacian + PSD + Gaussian):")
        wheelchair_model = create_model('wheelchair')
        
        # Extract features for training
        train_features = []
        for epoch in X_train_epochs:
            preprocessed = wheelchair_model.preprocess(epoch, srate)
            features = wheelchair_model.extract_features(preprocessed, srate)
            avg_features = np.mean(features, axis=0)
            train_features.append(avg_features)
        
        train_features = np.array(train_features)
        
        # Train model
        wheelchair_model.train_classifier(train_features, y_train)
        
        # Extract test features
        test_features = []
        for epoch in X_test_epochs:
            preprocessed = wheelchair_model.preprocess(epoch, srate)
            features = wheelchair_model.extract_features(preprocessed, srate)
            avg_features = np.mean(features, axis=0)
            test_features.append(avg_features)
        
        test_features = np.array(test_features)
        
        # Evaluate
        y_pred = wheelchair_model.predict(test_features)
        valid_mask = y_pred != -1
        if np.sum(valid_mask) > 0:
            acc_wheelchair = accuracy_score(y_test[valid_mask], y_pred[valid_mask])
            rejection_rate = 1 - np.mean(valid_mask)
            print(f"  Accuracy: {acc_wheelchair:.1%} (Rejection rate: {rejection_rate:.1%})")
            results['Wheelchair'] = acc_wheelchair
            models['Wheelchair'] = wheelchair_model
            architectures['wheelchair'] = wheelchair_model
        else:
            print("  All predictions rejected!")
    
    # 4. VR-SVM Architecture (if epochs provided)
    if X_train_epochs is not None and X_test_epochs is not None:
        print("\n4. VR-SVM Architecture (Small Laplacian + Alpha + SVM):")
        vr_svm_model = create_model('vr_svm')
        
        # Extract features for training
        train_features = []
        for epoch in X_train_epochs:
            preprocessed = vr_svm_model.preprocess(epoch, srate)
            features = vr_svm_model.extract_features(preprocessed, srate)
            avg_features = np.mean(features, axis=0)
            train_features.append(avg_features)
        
        train_features = np.array(train_features)
        
        # Train model
        vr_svm_model.train_classifier(train_features, y_train)
        
        # Extract test features
        test_features = []
        for epoch in X_test_epochs:
            preprocessed = vr_svm_model.preprocess(epoch, srate)
            features = vr_svm_model.extract_features(preprocessed, srate)
            avg_features = np.mean(features, axis=0)
            test_features.append(avg_features)
        
        test_features = np.array(test_features)
        
        # Evaluate
        y_pred = vr_svm_model.predict(test_features)
        acc_vr_svm = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc_vr_svm:.1%}")
        results['VR-SVM'] = acc_vr_svm
        models['VR-SVM'] = vr_svm_model
        architectures['vr_svm'] = vr_svm_model
    
    # Find best model
    best_name = max(results.keys(), key=lambda k: results[k])
    best_acc = results[best_name]
    
    if best_name in ['Wheelchair', 'VR-SVM']:
        best_model = models[best_name]
    else:
        best_model = models[best_name]
    
    print(f"\n🏆 Best model: {best_name} ({best_acc:.1%})")
    
    return best_model, best_name, results, architectures

def save_subject_model(participant_id, model, csp, scaler, results, architectures=None, best_name=None, save_dir='models/trained_models'):
    """Save subject-specific model with metadata"""
    
    # Create filename with participant ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{save_dir}/subject_{participant_id}_{timestamp}.pkl"
    
    # Prepare model data based on architecture type
    if best_name in ['Wheelchair', 'VR-SVM']:
        arch_key = 'wheelchair' if best_name == 'Wheelchair' else 'vr_svm'
        model_data = {
            'participant_id': participant_id,
            'architecture': arch_key,
            'model': architectures[arch_key] if architectures else model,
            'results': results,
            'timestamp': timestamp,
            'model_type': 'subject_specific',
            'srate': 125  # Default sampling rate
        }
    else:
        model_data = {
            'participant_id': participant_id,
            'model': model,
            'csp': csp,
            'scaler': scaler,
            'results': results,
            'timestamp': timestamp,
            'model_type': 'subject_specific',
            'fs': 512,  # Sampling rate
            'n_channels': 16,  # Expected channels
            'training_trials': len(y_train) if 'y_train' in globals() else None
        }
    
    # Save
    os.makedirs(save_dir, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model saved to: {filename}")
    
    # Also save as "current" for easy loading
    current_file = f"{save_dir}/subject_{participant_id}_current.pkl"
    with open(current_file, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✓ Also saved as: {current_file}")
    
    return filename

def discover_participants(data_dir='processed_data'):
    """Automatically discover all participants from data files"""
    participants = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('_processed.pkl') and not file.startswith('all_'):
                participant_id = file.replace('_processed.pkl', '')
                participants.append(participant_id)
    
    return sorted(participants)

def train_all_participants(data_dir='processed_data'):
    """Train subject-specific models for all discovered participants"""
    participants = discover_participants(data_dir)
    
    if not participants:
        print("No participants found in processed_data/")
        return
    
    print("\n" + "="*60)
    print("SUBJECT-SPECIFIC CLASSIFIER TRAINING")
    print("="*60)
    print(f"\nFound {len(participants)} participants: {', '.join(participants)}")
    
    all_results = {}
    
    for participant in participants:
        print(f"\n\n{'='*60}")
        print(f"Training for: {participant}")
        print('='*60)
        
        try:
            # Load subject data
            X, y = load_subject_data(participant, data_dir)
            
            # Extract features
            X_train, X_test, y_train, y_test, csp = extract_personalized_features(X, y)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # For architectures that need raw epochs, pass None
            # (The extract_personalized_features already splits the data)
            X_train_epochs = None
            X_test_epochs = None
            
            # Train models
            best_model, model_name, results, architectures = train_subject_specific_models(
                X_train_scaled, X_test_scaled, y_train, y_test,
                X_train_epochs, X_test_epochs, srate=125
            )
            
            # Save model
            save_subject_model(participant, best_model, csp, scaler, results, architectures, model_name)
            
            all_results[participant] = {
                'best_model': model_name,
                'accuracy': results[model_name],
                'all_results': results
            }
            
        except Exception as e:
            print(f"\nError training {participant}: {e}")
            all_results[participant] = {'error': str(e)}
    
    # Summary report
    print("\n\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    for participant, result in all_results.items():
        if 'error' in result:
            print(f"\n{participant}: Failed - {result['error']}")
        else:
            print(f"\n{participant}:")
            print(f"  Best model: {result['best_model']}")
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  All models: {', '.join([f'{k}={v:.1%}' for k,v in result['all_results'].items()])}")
    
    print("\n" + "="*60)
    print("✓ Training complete for all participants!")
    print("\nModels saved in: models/trained_models/subject_<ID>_current.pkl")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Train subject-specific MI classifier')
    parser.add_argument('--participant', type=str, default=None,
                       help='Specific participant ID (optional, trains all if not specified)')
    parser.add_argument('--n_components', type=int, default=4, 
                       help='Number of CSP components (default: 4)')
    parser.add_argument('--data_dir', type=str, default='processed_data',
                       help='Directory with processed data')
    args = parser.parse_args()
    
    if args.participant:
        # Train single participant
        try:
            # Load subject data
            X, y = load_subject_data(args.participant, args.data_dir)
            
            # Extract features
            X_train, X_test, y_train, y_test, csp = extract_personalized_features(
                X, y, n_components=args.n_components
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # For architectures that need raw epochs, pass None
            # (The extract_personalized_features already splits the data)
            X_train_epochs = None
            X_test_epochs = None
            
            # Train models
            best_model, model_name, results, architectures = train_subject_specific_models(
                X_train_scaled, X_test_scaled, y_train, y_test,
                X_train_epochs, X_test_epochs, srate=125
            )
            
            # Save model
            save_subject_model(args.participant, best_model, csp, scaler, results, architectures, model_name)
            
            print("\n" + "="*60)
            print("TRAINING COMPLETE!")
            print("="*60)
            print(f"\nSubject: {args.participant}")
            print(f"Best accuracy: {results[model_name]:.1%}")
            
        except Exception as e:
            print(f"\n Error: {e}")
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
    else:
        # Train all participants
        train_all_participants(args.data_dir)

if __name__ == "__main__":
    main()