#!/usr/bin/env python
"""
Compare Universal vs Subject-Specific MI Classifiers
Shows the performance difference between approaches
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from universal.train_universal_classifier import load_and_preprocess_data as load_universal
from subject_specific.train_subject_classifier import (
    discover_participants, load_subject_data, extract_personalized_features, 
    train_subject_specific_models
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

def compare_approaches(data_dir='../../processed_data'):
    """Compare universal vs subject-specific performance"""
    
    print("\n" + "="*70)
    print("UNIVERSAL vs SUBJECT-SPECIFIC CLASSIFIER COMPARISON")
    print("="*70)
    
    # Discover participants
    participants = discover_participants(data_dir)
    print(f"\nFound {len(participants)} participants: {', '.join(participants)}")
    
    # 1. Train and evaluate universal model
    print("\n" + "-"*70)
    print("1. UNIVERSAL CLASSIFIER (All participants combined)")
    print("-"*70)
    
    try:
        # Load all data
        X_all, y_all = load_universal(data_dir)
        
        # Train universal model (simplified for comparison)
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import train_test_split
        
        # CSP + LDA pipeline
        X_all_csp = np.transpose(X_all, (0, 2, 1))  # Convert to CSP format
        X_train, X_test, y_train, y_test = train_test_split(
            X_all_csp, y_all, test_size=0.3, random_state=42, stratify=y_all
        )
        
        csp = CSP(n_components=6, reg=None, log=True)
        csp.fit(X_train, y_train)
        
        X_train_csp = csp.transform(X_train)
        X_test_csp = csp.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_csp)
        X_test_scaled = scaler.transform(X_test_csp)
        
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_train_scaled, y_train)
        
        universal_accuracy = lda.score(X_test_scaled, y_test)
        print(f"\nUniversal model accuracy: {universal_accuracy:.1%}")
        
        # Cross-validation
        cv_scores = cross_val_score(lda, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation: {cv_scores.mean():.1%} (±{cv_scores.std():.1%})")
        
    except Exception as e:
        print(f"Error training universal model: {e}")
        universal_accuracy = 0
    
    # 2. Train and evaluate subject-specific models
    print("\n" + "-"*70)
    print("2. SUBJECT-SPECIFIC CLASSIFIERS")
    print("-"*70)
    
    subject_results = {}
    
    for participant in participants:
        print(f"\n{participant}:")
        
        try:
            # Load individual data
            X_subj, y_subj = load_subject_data(participant, data_dir)
            
            # Extract features
            X_train, X_test, y_train, y_test, csp = extract_personalized_features(
                X_subj, y_subj, n_components=4  # Fewer components for subject-specific
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            best_model, model_name, results = train_subject_specific_models(
                X_train_scaled, X_test_scaled, y_train, y_test
            )
            
            subject_results[participant] = results[model_name]
            print(f"  Best accuracy: {results[model_name]:.1%} ({model_name})")
            
        except Exception as e:
            print(f"  Error: {e}")
            subject_results[participant] = 0
    
    # 3. Summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    print(f"\nUniversal Classifier:")
    print(f"  - Single model for all participants")
    print(f"  - Accuracy: {universal_accuracy:.1%}")
    print(f"  - Pros: No calibration needed")
    print(f"  - Cons: Lower accuracy, may show bias")
    
    if subject_results:
        avg_subject_acc = np.mean(list(subject_results.values()))
        std_subject_acc = np.std(list(subject_results.values()))
        
        print(f"\nSubject-Specific Classifiers:")
        print(f"  - Individual model per participant")
        print(f"  - Average accuracy: {avg_subject_acc:.1%} (±{std_subject_acc:.1%})")
        print(f"  - Range: {min(subject_results.values()):.1%} - {max(subject_results.values()):.1%}")
        print(f"  - Pros: Higher accuracy, personalized")
        print(f"  - Cons: Requires calibration session")
        
        print(f"\nPerformance gain: {(avg_subject_acc - universal_accuracy)*100:.1f} percentage points")
        
        # Individual comparison
        print("\nPer-participant comparison:")
        for participant, subj_acc in subject_results.items():
            gain = (subj_acc - universal_accuracy) * 100
            symbol = "↑" if gain > 0 else "↓" if gain < 0 else "="
            print(f"  {participant}: {subj_acc:.1%} ({symbol}{abs(gain):.1f}pp vs universal)")
    
    print("\n" + "="*70)
    print("RECOMMENDATION:")
    if subject_results and avg_subject_acc > universal_accuracy + 0.1:
        print("✅ Subject-specific models show significant improvement")
        print("   → Use calibration for serious BCI applications")
    else:
        print("⚠️  Limited improvement from subject-specific models")
        print("   → May need more training data or better features")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare MI classifier approaches')
    parser.add_argument('--data_dir', type=str, default='../../processed_data',
                       help='Directory with processed data')
    args = parser.parse_args()
    
    compare_approaches(args.data_dir)