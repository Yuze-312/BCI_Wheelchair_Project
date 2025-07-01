"""
Improved MI Classifier - Based on Diagnosis Results
Now includes Filter Bank CSP (FBCSP) for better performance
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from mne.decoding import CSP
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import os


def discover_participants(data_dir='MI/processed_data'):
    """Automatically discover all participants from data files"""
    participants = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('_processed.pkl') and not file.startswith('all_'):
                participant_id = file.replace('_processed.pkl', '')
                participants.append(participant_id)
    
    return sorted(participants)

def load_and_preprocess_data(data_dir: str):
    """Load data from all discovered participants for universal model"""
    
    participants = discover_participants(data_dir)
    
    if not participants:
        raise ValueError("No participants found in processed_data/")
    
    print("\nLoading MI data for universal classification...")
    print(f"Found participants: {', '.join(participants)}")
    
    all_epochs = []
    all_labels = []
    participant_info = {}
    
    for participant in participants:
        try:
            with open(f"{data_dir}/{participant}_processed.pkl", 'rb') as f:
                data = pickle.load(f)
            
            epochs = data['epochs']
            labels = data['labels']
            
            # Binary classification only
            mask = labels != 3
            epochs = epochs[mask]
            labels = labels[mask]
            
            # Convert to binary: Left=0, Right=1
            labels = (labels == 2).astype(int)
            
            all_epochs.append(epochs)
            all_labels.append(labels)
            
            participant_info[participant] = {
                'n_trials': len(labels),
                'n_left': np.sum(labels==0),
                'n_right': np.sum(labels==1)
            }
            
            print(f"  {participant}: {len(labels)} trials (Left={np.sum(labels==0)}, Right={np.sum(labels==1)})")
            
        except Exception as e:
            print(f"  ⚠️  Error loading {participant}: {e}")
    
    if not all_epochs:
        raise ValueError("No data could be loaded")
    
    X = np.vstack(all_epochs)
    y = np.hstack(all_labels)
    
    print(f"\nCombined: {X.shape[0]} trials, shape: {X.shape}")
    print(f"Class balance: Left={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%), Right={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    return X, y


def extract_csp_features(X_epochs, y_labels, n_components=6):
    """Extract CSP features for better MI discrimination"""
    
    print(f"Extracting CSP features (n_components={n_components})...")
    
    # CSP expects (n_epochs, n_channels, n_times)
    X_csp_format = np.transpose(X_epochs, (0, 2, 1))
    
    # Initialize and fit CSP
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    
    # Split data to fit CSP only on training data
    X_train, X_test, y_train, y_test = train_test_split(
        X_csp_format, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )
    
    # Fit CSP on training data
    csp.fit(X_train, y_train)
    
    # Transform both training and test data
    X_train_csp = csp.transform(X_train)
    X_test_csp = csp.transform(X_test)
    
    print(f"CSP features shape: {X_train_csp.shape}")
    
    return X_train_csp, X_test_csp, y_train, y_test, csp


def bandpass_filter(data, low_freq, high_freq, sfreq=512, order=5):
    """Apply bandpass filter to the data"""
    nyquist = sfreq / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    # Design butterworth filter
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data


def extract_fbcsp_features(X_epochs, y_labels, n_components=4, n_features=20):
    """
    Extract Filter Bank CSP features
    Uses multiple frequency bands to capture different MI rhythms
    """
    
    print("Extracting Filter Bank CSP features...")
    
    # Define frequency bands for motor imagery
    # Alpha (8-12 Hz), low beta (12-20 Hz), high beta (20-30 Hz)
    frequency_bands = [
        (4, 8),    # Theta
        (8, 12),   # Alpha
        (12, 16),  # Low beta 1
        (16, 20),  # Low beta 2
        (20, 24),  # Mid beta
        (24, 30),  # High beta
        (8, 30),   # Wide band (alpha + beta)
    ]
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(
        X_epochs, y_labels, test_size=0.3, random_state=42, stratify=y_labels
    )
    
    # Extract CSP features for each frequency band
    all_train_features = []
    all_test_features = []
    
    for i, (low_freq, high_freq) in enumerate(frequency_bands):
        print(f"  Band {i+1}: {low_freq}-{high_freq} Hz")
        
        # Filter the data
        X_train_filt = bandpass_filter(X_train, low_freq, high_freq)
        X_test_filt = bandpass_filter(X_test, low_freq, high_freq)
        
        # Convert to CSP format (n_epochs, n_channels, n_times)
        X_train_csp = np.transpose(X_train_filt, (0, 2, 1))
        X_test_csp = np.transpose(X_test_filt, (0, 2, 1))
        
        # Fit CSP for this band
        csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        csp.fit(X_train_csp, y_train)
        
        # Transform data
        train_features = csp.transform(X_train_csp)
        test_features = csp.transform(X_test_csp)
        
        all_train_features.append(train_features)
        all_test_features.append(test_features)
    
    # Concatenate features from all bands
    X_train_fbcsp = np.hstack(all_train_features)
    X_test_fbcsp = np.hstack(all_test_features)
    
    print(f"FBCSP features shape before selection: {X_train_fbcsp.shape}")
    
    # Feature selection using mutual information
    selector = SelectKBest(mutual_info_classif, k=min(n_features, X_train_fbcsp.shape[1]))
    selector.fit(X_train_fbcsp, y_train)
    
    X_train_selected = selector.transform(X_train_fbcsp)
    X_test_selected = selector.transform(X_test_fbcsp)
    
    print(f"FBCSP features shape after selection: {X_train_selected.shape}")
    
    return X_train_selected, X_test_selected, y_train, y_test, selector


def extract_traditional_features(X_epochs):
    """Extract traditional MI features"""
    
    print("Extracting traditional features...")
    
    features_list = []
    
    for epoch in X_epochs:
        epoch_features = []
        
        # 1. Channel means (temporal averaging)
        channel_means = np.mean(epoch, axis=0)
        epoch_features.extend(channel_means)
        
        # 2. Channel variances 
        channel_vars = np.var(epoch, axis=0)
        epoch_features.extend(channel_vars)
        
        # 3. Mean power in MI window (0.5-2.5s, assuming 512Hz)
        mi_window = epoch[256:1280, :]  # 0.5-2.5s
        mi_power = np.mean(mi_window**2, axis=0)
        epoch_features.extend(mi_power)
        
        features_list.append(epoch_features)
    
    X_features = np.array(features_list)
    print(f"Traditional features shape: {X_features.shape}")
    
    return X_features


def train_improved_classifiers(X_csp, X_fbcsp, X_traditional, y, y_fbcsp, test_size=0.3):
    """Train improved classifiers with different feature sets including FBCSP"""
    
    print("\n" + "="*60)
    print("IMPROVED MI CLASSIFIERS")
    print("="*60)
    
    results = {}
    models = {}
    
    # 1. Random Forest with CSP features
    print("\n1. Random Forest + CSP Features:")
    X_train, X_test, y_train, y_test = train_test_split(
        X_csp, y, test_size=test_size, random_state=42, stratify=y
    )
    
    rf_csp = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    rf_csp.fit(X_train, y_train)
    acc_rf_csp = rf_csp.score(X_test, y_test)
    results['RF + CSP'] = acc_rf_csp
    models['RF + CSP'] = (rf_csp, X_test, y_test)
    
    print(f"  Accuracy: {acc_rf_csp:.1%}")
    
    # 2. LDA with CSP features
    print("\n2. LDA + CSP Features:")
    lda_csp = LinearDiscriminantAnalysis()
    lda_csp.fit(X_train, y_train)
    acc_lda_csp = lda_csp.score(X_test, y_test)
    results['LDA + CSP'] = acc_lda_csp
    models['LDA + CSP'] = (lda_csp, X_test, y_test)
    
    print(f"  Accuracy: {acc_lda_csp:.1%}")
    
    # 3. LDA with FBCSP features (NEW!)
    print("\n3. LDA + Filter Bank CSP Features:")
    if X_fbcsp is not None:
        # FBCSP already has train/test split
        X_train_fbcsp, X_test_fbcsp = X_fbcsp[:2]
        y_train_fbcsp, y_test_fbcsp = y_fbcsp
        
        lda_fbcsp = LinearDiscriminantAnalysis()
        lda_fbcsp.fit(X_train_fbcsp, y_train_fbcsp)
        acc_lda_fbcsp = lda_fbcsp.score(X_test_fbcsp, y_test_fbcsp)
        results['LDA + FBCSP'] = acc_lda_fbcsp
        models['LDA + FBCSP'] = (lda_fbcsp, X_test_fbcsp, y_test_fbcsp)
        
        print(f"  Accuracy: {acc_lda_fbcsp:.1%}")
    
    # 4. Random Forest with FBCSP features (NEW!)
    print("\n4. Random Forest + Filter Bank CSP Features:")
    if X_fbcsp is not None:
        rf_fbcsp = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        rf_fbcsp.fit(X_train_fbcsp, y_train_fbcsp)
        acc_rf_fbcsp = rf_fbcsp.score(X_test_fbcsp, y_test_fbcsp)
        results['RF + FBCSP'] = acc_rf_fbcsp
        models['RF + FBCSP'] = (rf_fbcsp, X_test_fbcsp, y_test_fbcsp)
        
        print(f"  Accuracy: {acc_rf_fbcsp:.1%}")
    
    # 5. Random Forest with traditional features
    print("\n5. Random Forest + Traditional Features:")
    X_train_trad, X_test_trad, y_train_trad, y_test_trad = train_test_split(
        X_traditional, y, test_size=test_size, random_state=42, stratify=y
    )
    
    rf_trad = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=3,
        random_state=42
    )
    
    rf_trad.fit(X_train_trad, y_train_trad)
    acc_rf_trad = rf_trad.score(X_test_trad, y_test_trad)
    results['RF + Traditional'] = acc_rf_trad
    models['RF + Traditional'] = (rf_trad, X_test_trad, y_test_trad)
    
    print(f"  Accuracy: {acc_rf_trad:.1%}")
    
    # Find best model
    best_approach = max(results.keys(), key=lambda k: results[k])
    best_acc = results[best_approach]
    
    print(f"\n🏆 Best approach: {best_approach} ({best_acc:.1%})")
    
    # Return best model and its test data
    best_model, X_test_best, y_test_best = models[best_approach]
    return best_model, X_test_best, y_test_best, results


def evaluate_final_model(model, X_test, y_test):
    """Detailed evaluation of the final model"""
    
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Left', 'Right']))
    
    # Feature importance (if Random Forest)
    if hasattr(model, 'feature_importances_'):
        print(f"\nTop 5 Important Features:")
        feature_importance = model.feature_importances_
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.3f}")


def main():
    """Main improved MI classification pipeline with FBCSP"""
    
    print("="*60)
    print("IMPROVED MI CLASSIFICATION WITH FILTER BANK CSP")
    print("Enhanced multi-band frequency analysis")
    print("="*60)
    
    # Load data
    data_dir = "MI/processed_data"
    X_epochs, y = load_and_preprocess_data(data_dir)
    
    # Extract regular CSP features
    print("\n" + "-"*60)
    print("EXTRACTING STANDARD CSP FEATURES")
    print("-"*60)
    X_train_csp, X_test_csp, y_train_csp, y_test_csp, csp = extract_csp_features(X_epochs, y)
    
    # Extract Filter Bank CSP features (NEW!)
    print("\n" + "-"*60)
    print("EXTRACTING FILTER BANK CSP FEATURES")
    print("-"*60)
    X_train_fbcsp, X_test_fbcsp, y_train_fbcsp, y_test_fbcsp, fbcsp_selector = extract_fbcsp_features(X_epochs, y)
    
    # Extract traditional features for comparison
    X_traditional = extract_traditional_features(X_epochs)
    
    # Train improved classifiers
    best_model, X_test_best, y_test_best, all_results = train_improved_classifiers(
        np.vstack([X_train_csp, X_test_csp]), 
        (X_train_fbcsp, X_test_fbcsp),
        X_traditional, 
        np.hstack([y_train_csp, y_test_csp]),
        (y_train_fbcsp, y_test_fbcsp)
    )
    
    # Detailed evaluation
    evaluate_final_model(best_model, X_test_best, y_test_best)
    
    # Save best model
    best_approach = max(all_results.keys(), key=lambda k: all_results[k])
    best_acc = all_results[best_approach]
    
    model_data = {
        'model': best_model,
        'csp': csp if 'CSP' in best_approach and 'FB' not in best_approach else None,
        'fbcsp_selector': fbcsp_selector if 'FBCSP' in best_approach else None,
        'approach': best_approach,
        'test_accuracy': best_acc,
        'all_results': all_results
    }
    
    output_dir = Path("MI/models")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "mi_improved_classifier.pkl"
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModel saved to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("IMPROVEMENT SUMMARY")
    print("="*60)
    print("Previous best: LDA + CSP 68%")
    print(f"Current best: {best_approach} {best_acc:.1%}")
    
    improvement = best_acc - 0.68
    print(f"Improvement from previous: {improvement:+.1%}")
    
    if best_acc > 0.75:
        print("🌟 OUTSTANDING: Achieved >75% accuracy!")
        print("🌟 Excellent for real-time BCI implementation")
    elif best_acc > 0.70:
        print("✅ EXCELLENT: Achieved >70% accuracy!")
        print("✅ Very good for BCI implementation")
    elif best_acc > 0.65:
        print("✅ GOOD: Maintained good performance")
        print("✅ Ready for real-time BCI")
    else:
        print("⚠  MODERATE: Performance needs improvement")
    
    print(f"\nAll approaches tested:")
    for approach, acc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {approach}: {acc:.1%}")


if __name__ == "__main__":
    main()