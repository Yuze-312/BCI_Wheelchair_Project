# run_offline_classifier.py
import argparse
from classifier import OfflineErrPClassifier
import pickle
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', message='Pandas requires version')

def load_participant_data(participant_id, data_dir='processed_data'):
    """Load all processed epochs for a participant"""
    participant_path = Path(data_dir) / participant_id
    
    all_error_epochs = []
    all_correct_epochs = []
    loaded_sessions = {}
    
    # Get all session files except Session 4
    session_files = sorted(participant_path.glob('*.pkl'))
    
    for session_file in session_files:
        # Skip Session 4
        if 'Session_4' in session_file.stem:
            print(f"  Skipping {session_file.stem} (as requested)")
            continue
            
        with open(session_file, 'rb') as f:
            data = pickle.load(f)
        
        session_name = session_file.stem.replace('_epochs', '')
        
        # Extract epochs
        error_epochs = data['epochs']['error']['data']
        correct_epochs = data['epochs']['correct']['data']
        
        if error_epochs.size > 0:
            all_error_epochs.append(error_epochs)
            print(f"  Loaded {session_name}: {len(error_epochs)} error epochs")
        
        if correct_epochs.size > 0:
            all_correct_epochs.append(correct_epochs)
            print(f"  Loaded {session_name}: {len(correct_epochs)} correct epochs")
        
        loaded_sessions[session_name] = {
            'error': error_epochs,
            'correct': correct_epochs
        }
    
    # Concatenate all epochs
    if all_error_epochs:
        error_epochs = np.vstack(all_error_epochs)
    else:
        error_epochs = np.array([])
    
    if all_correct_epochs:
        correct_epochs = np.vstack(all_correct_epochs)
    else:
        correct_epochs = np.array([])
    
    return error_epochs, correct_epochs, loaded_sessions

def main():
    participant_id = 'T-002'
    
    print("="*60)
    print(f"PROCESSING: {participant_id}")
    print("="*60)
    
    # Load data
    error_epochs, correct_epochs, loaded_sessions = load_participant_data(participant_id)
    
    print(f"Loaded {participant_id} - Sessions: {', '.join(loaded_sessions.keys())}")
    print(f"  Error epochs: {len(error_epochs)}")
    print(f"  Correct epochs: {len(correct_epochs)}")
    
    if len(error_epochs) < 10:
        print("ERROR: Not enough error epochs for training!")
        return
    
    # Initialize classifier
    classifier = OfflineErrPClassifier()
    
    # 1. Debug features to understand what's happening
    print("\n" + "="*60)
    print("RUNNING FEATURE ANALYSIS")
    print("="*60)
    X_error_debug, X_correct_debug = classifier.debug_features(error_epochs, correct_epochs)
    
    # 2. Evaluate different feature types
    print("\nEvaluating feature extraction methods...")
    feature_results = classifier.evaluate_features(error_epochs, correct_epochs)
    
    # Find best feature type
    best_feature_type = max(feature_results.items(), 
                           key=lambda x: x[1]['mean'])[0]
    print(f"\nBest feature type: {best_feature_type}")
    
    # 3. Test with feature selection
    print("\n" + "="*60)
    print("TESTING FEATURE SELECTION")
    print("="*60)
    for k in [5, 10, 15, 20]:
        print(f"\nWith {k} features:")
        classifier.evaluate_with_feature_selection(
            error_epochs, correct_epochs, 
            feature_type=best_feature_type, 
            k_features=k
        )
    
    # 4. Analyze feature importance
    print("\n" + "="*60)
    print("ANALYZING FEATURE IMPORTANCE")
    print("="*60)
    classifier.analyze_feature_importance(
        error_epochs, correct_epochs, 
        feature_type=best_feature_type
    )
    
    # Also analyze targeted Pe features
    print("\n" + "="*60)
    print("ANALYZING TARGETED Pe FEATURES")
    print("="*60)
    classifier.analyze_feature_importance(
        error_epochs, correct_epochs, 
        feature_type='targeted_pe'
    )
    
    # 5. Train final classifier with best approach
    print("\n" + "="*60)
    print("TRAINING FINAL CLASSIFIER")
    print("="*60)
    
    # Compare regular vs targeted Pe features
    print("\nTraining with best feature type:", best_feature_type)
    results_regular = classifier.detailed_evaluation(
        error_epochs, correct_epochs, 
        feature_type=best_feature_type
    )
    
    # Reset classifier for new training
    classifier_pe = OfflineErrPClassifier()
    print("\nTraining with targeted Pe features:")
    results_pe = classifier_pe.detailed_evaluation(
        error_epochs, correct_epochs, 
        use_targeted_pe=True
    )
    
    # Save best results
    if results_pe['roc_auc'] > results_regular['roc_auc']:
        print("\nTargeted Pe features performed better!")
        classifier_pe.save_results(results_pe, participant_id)
        best_results = results_pe
    else:
        print(f"\n{best_feature_type} features performed better!")
        classifier.save_results(results_regular, participant_id)
        best_results = results_regular
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Results for {participant_id}:")
    print(f"  Best classifier: {best_results['classifier']['classifier']}")
    print(f"  Feature type: {best_results['feature_type']}")
    print(f"  ROC AUC: {best_results['roc_auc']:.3f}")
    
    report = best_results['classification_report']
    print(f"  Error detection rate: {report['1.0']['recall']:.3f}")
    print(f"  False alarm rate: {1 - report['0.0']['recall']:.3f}")


def train_classifier(error_epochs, correct_epochs, participant_id, args, output_path):
    """Train and evaluate classifier for given data"""
    
    # Initialize classifier
    classifier = OfflineErrPClassifier()
    
    # Set custom channels if specified
    if args.channels:
        classifier.selected_channels = args.channels
        print(f"Using channels: {args.channels}")
    
    # Create participant output directory
    participant_output = output_path / participant_id
    participant_output.mkdir(exist_ok=True)
    
    # Step 1: Feature evaluation (if auto mode)
    if args.feature_type == 'auto':
        print("\nEvaluating feature extraction methods...")
        feature_results = classifier.evaluate_features(error_epochs, correct_epochs)
        
        # Find best feature type
        best_feature = max(feature_results.items(), key=lambda x: x[1]['mean'])[0]
        print(f"\nBest feature type: {best_feature}")
        
        # Save feature comparison
        with open(participant_output / 'feature_comparison.txt', 'w') as f:
            f.write(f"Feature Evaluation Results for {participant_id}\n")
            f.write("="*50 + "\n\n")
            for feat_type, results in feature_results.items():
                f.write(f"{feat_type}:\n")
                f.write(f"  Mean AUC: {results['mean']:.3f} (+/- {results['std']:.3f})\n")
                f.write(f"  Features: {results['n_features']}\n\n")
    else:
        best_feature = args.feature_type
        print(f"Using specified feature type: {best_feature}")
    
    # Step 2: Detailed evaluation
    print("\nTraining classifiers...")
    results = classifier.detailed_evaluation(error_epochs, correct_epochs, 
                                           feature_type=best_feature)
    
    # Save figure with participant name
    plt.suptitle(f'Classifier Evaluation - {participant_id}', fontsize=16)
    plt.tight_layout()
    plt.savefig(participant_output / 'classifier_evaluation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed results
    classifier.save_results(results, participant_output)
    
    # Save summary
    with open(participant_output / 'classifier_summary.txt', 'w') as f:
        f.write(f"Classifier Summary for {participant_id}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Feature type: {best_feature}\n")
        f.write(f"Best classifier: {results['classifier']['classifier']}\n")
        f.write(f"Parameters: {results['classifier']['params']}\n")
        f.write(f"Cross-validation AUC: {results['classifier']['score']:.3f}\n")
        f.write(f"Final ROC AUC: {results['roc_auc']:.3f}\n\n")
        
        f.write("Classification Report:\n")
        f.write("-"*30 + "\n")
        report = results['classification_report']
        f.write(f"Error detection rate: {report['1.0']['recall']:.3f}\n")
        f.write(f"Error precision: {report['1.0']['precision']:.3f}\n")
        f.write(f"False alarm rate: {1 - report['0.0']['recall']:.3f}\n")
        f.write(f"Overall accuracy: {report['accuracy']:.3f}\n")
    
    # Print summary
    print(f"\nResults for {participant_id}:")
    print(f"  Best classifier: {results['classifier']['classifier']}")
    print(f"  ROC AUC: {results['roc_auc']:.3f}")
    print(f"  Error detection rate: {results['classification_report']['1.0']['recall']:.3f}")
    print(f"  False alarm rate: {1 - results['classification_report']['0.0']['recall']:.3f}")

if __name__ == "__main__":
    main()