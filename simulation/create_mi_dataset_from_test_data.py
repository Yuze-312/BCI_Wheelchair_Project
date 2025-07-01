#!/usr/bin/env python
"""
Create MI dataset from test_data in T-005 format and test with 78% accuracy model
"""

import numpy as np
import pandas as pd
import os
import glob
from datetime import datetime
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_test_data():
    """Load EEG and event data from test_data folder"""
    test_data_dir = "test_data"
    
    # Load EEG data
    brainflow_files = glob.glob(os.path.join(test_data_dir, "BrainFlow-RAW*.csv"))
    if not brainflow_files:
        raise FileNotFoundError("No BrainFlow EEG data found")
    
    brainflow_file = sorted(brainflow_files)[-1]
    print(f"Loading EEG data from: {brainflow_file}")
    
    # Load EEG data (tab-separated)
    eeg_df = pd.read_csv(brainflow_file, sep='\t', header=None)
    
    # Extract channels 1-16
    eeg_data = eeg_df.iloc[:, 1:17].values
    
    # Apply gain correction
    eeg_data = eeg_data / 24  # Convert to microvolts
    
    # Create timestamps based on sampling rate (250 Hz)
    # The first column might be sample numbers, not timestamps
    n_samples = len(eeg_data)
    fs = 250  # Hz
    timestamps = np.arange(n_samples) / fs  # Convert to seconds
    
    # Load event logger
    event_files = glob.glob(os.path.join(test_data_dir, "subway_errp*.csv"))
    if not event_files:
        raise FileNotFoundError("No event logger data found")
    
    event_file = sorted(event_files)[-1]
    print(f"Loading events from: {event_file}")
    
    events_df = pd.read_csv(event_file)
    
    return eeg_data, timestamps, events_df

def extract_mi_epochs(eeg_data, timestamps, events_df, fs=250):
    """Extract motor imagery epochs from continuous data"""
    
    # Filter for cue events
    cue_events = events_df[events_df['event_type'].isin(['cue_left', 'cue_right'])].copy()
    
    # Create epochs list
    epochs = []
    labels = []
    epoch_info = []
    
    # Define epoch parameters
    pre_cue = 0.5  # 500ms before cue
    post_cue = 4.0  # 4s after cue (MI period)
    epoch_length = int((pre_cue + post_cue) * fs)
    pre_samples = int(pre_cue * fs)
    
    print(f"\nExtracting epochs:")
    print(f"  Pre-cue: {pre_cue}s")
    print(f"  Post-cue: {post_cue}s")
    print(f"  Epoch length: {epoch_length} samples")
    print(f"  EEG timestamp range: {timestamps[0]:.3f}s - {timestamps[-1]:.3f}s")
    print(f"  Event timestamp range: {events_df['timestamp'].min():.3f}s - {events_df['timestamp'].max():.3f}s")
    
    for idx, event in cue_events.iterrows():
        # Find closest timestamp in EEG data
        event_time = event['timestamp']
        
        # Convert event timestamp to sample index
        # Assuming timestamps start from 0
        time_diff = np.abs(timestamps - event_time)
        event_sample = np.argmin(time_diff)
        
        # Check if we found a close enough match
        if time_diff[event_sample] > 1.0:  # More than 1s difference
            print(f"Warning: Large time difference {time_diff[event_sample]:.3f}s for event at {event_time:.3f}s")
            continue
        
        # Extract epoch
        start_idx = event_sample - pre_samples
        end_idx = start_idx + epoch_length
        
        # Check bounds
        if start_idx < 0 or end_idx > len(eeg_data):
            print(f"Skipping epoch at {event_time:.3f}s (out of bounds)")
            continue
        
        epoch = eeg_data[start_idx:end_idx, :]
        
        # Get label (0 for left, 1 for right)
        label = 0 if event['cue_class'] == 'left' else 1
        
        epochs.append(epoch)
        labels.append(label)
        
        # Store epoch info
        epoch_info.append({
            'timestamp': event_time,
            'cue_class': event['cue_class'],
            'trial_id': event['trial_id'],
            'label': label
        })
    
    print(f"\nExtracted {len(epochs)} epochs")
    print(f"  Left: {labels.count(0)}")
    print(f"  Right: {labels.count(1)}")
    
    return np.array(epochs), np.array(labels), epoch_info

def create_mi_dataset(epochs, labels, epoch_info, output_dir='test_data/mi_dataset'):
    """Create dataset in T-005 MI format"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create session directory
    session_dir = os.path.join(output_dir, 'Session_test')
    os.makedirs(session_dir, exist_ok=True)
    
    # Save each epoch as a separate CSV file (T-005 format)
    print(f"\nSaving dataset to: {session_dir}")
    
    # Also create a combined file for easier loading
    all_epochs_data = []
    
    fs = 250  # Sampling rate
    
    for i, (epoch, label, info) in enumerate(zip(epochs, labels, epoch_info)):
        # Create time vector
        time_vec = np.arange(epoch.shape[0]) / fs
        
        # Create epoch number column
        epoch_nums = np.full(epoch.shape[0], i)
        
        # Create event columns (empty for most samples)
        event_id = [''] * epoch.shape[0]
        event_date = [''] * epoch.shape[0]
        event_duration = [''] * epoch.shape[0]
        
        # Add event marker at cue onset (0.5s into epoch)
        cue_idx = int(0.5 * fs)
        event_id[cue_idx] = str(label + 1)  # 1 for left, 2 for right
        event_date[cue_idx] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        event_duration[cue_idx] = '4.0'  # MI duration
        
        # Create dataframe
        epoch_df = pd.DataFrame({
            f'Time:{fs}Hz': time_vec,
            'Epoch': epoch_nums
        })
        
        # Add channel data
        for ch in range(16):
            epoch_df[f'Channel {ch+1}'] = epoch[:, ch]
        
        # Add event columns
        epoch_df['Event Id'] = event_id
        epoch_df['Event Date'] = event_date
        epoch_df['Event Duration'] = event_duration
        
        # Save individual epoch file
        filename = f"test-epoch_{i:03d}_class_{label}.csv"
        filepath = os.path.join(session_dir, filename)
        epoch_df.to_csv(filepath, index=False)
        
        # Add to combined data
        all_epochs_data.append(epoch_df)
    
    # Save combined file
    combined_df = pd.concat(all_epochs_data, ignore_index=True)
    combined_file = os.path.join(session_dir, "test-all_epochs.csv")
    combined_df.to_csv(combined_file, index=False)
    
    # Save labels and info
    labels_file = os.path.join(session_dir, "labels.npy")
    np.save(labels_file, labels)
    
    info_file = os.path.join(session_dir, "epoch_info.csv")
    pd.DataFrame(epoch_info).to_csv(info_file, index=False)
    
    print(f"Dataset saved successfully!")
    print(f"  Combined file: {combined_file}")
    print(f"  Labels file: {labels_file}")
    print(f"  Info file: {info_file}")
    
    return session_dir

def prepare_features_for_model(epochs, model_dict):
    """Prepare features for the model using CSP"""
    
    print(f"\nPreparing features for model:")
    print(f"  Input shape: {epochs.shape}")
    
    # Check if model has CSP transformer
    if isinstance(model_dict, dict) and 'csp' in model_dict:
        print("  Using CSP features from model")
        csp = model_dict['csp']
        
        # CSP expects (n_epochs, n_channels, n_times)
        epochs_csp = np.transpose(epochs, (0, 2, 1))
        
        # Transform using pre-fitted CSP
        features = csp.transform(epochs_csp)
        print(f"  CSP features shape: {features.shape}")
        
    else:
        # Try basic feature extraction
        print("  Using basic feature extraction")
        
        # Extract simple features: mean, variance for each channel
        features = []
        for epoch in epochs:
            epoch_features = []
            for ch in range(epoch.shape[1]):
                ch_data = epoch[:, ch]
                # Basic features
                epoch_features.extend([
                    np.mean(ch_data),
                    np.var(ch_data),
                ])
            features.append(epoch_features)
        
        features = np.array(features)
        print(f"  Basic features shape: {features.shape}")
        
        # If still too many features, select first 6
        if features.shape[1] > 6:
            print(f"  Selecting first 6 features from {features.shape[1]}")
            features = features[:, :6]
    
    print(f"  Final feature shape: {features.shape}")
    
    return features

def test_with_model(features, labels, model_path):
    """Test the data with the 78% accuracy model"""
    
    print(f"\nLoading model from: {model_path}")
    
    # Try different loading methods
    try:
        # Try pickle first
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"  Pickle load failed: {e}")
        # Try joblib
        try:
            import joblib
            model = joblib.load(model_path)
        except Exception as e2:
            print(f"  Joblib load failed: {e2}")
            raise
    
    print(f"Model loaded: {type(model).__name__}")
    
    # Handle different model formats
    if isinstance(model, dict):
        # Model might be stored in a dict with metadata
        if 'model' in model:
            actual_model = model['model']
        elif 'classifier' in model:
            actual_model = model['classifier']
        elif 'pipeline' in model:
            actual_model = model['pipeline']
        else:
            # Try to find the actual model object
            for key, value in model.items():
                if hasattr(value, 'predict'):
                    actual_model = value
                    break
            else:
                print(f"Model dict keys: {list(model.keys())}")
                raise ValueError("Could not find model in dict")
        model = actual_model
    
    print(f"Actual model type: {type(model).__name__}")
    
    # Make predictions
    predictions = model.predict(features)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  Confusion Matrix:")
    print(f"    Predicted")
    print(f"    L    R")
    print(f"L  {cm[0,0]:3d}  {cm[0,1]:3d}")
    print(f"R  {cm[1,0]:3d}  {cm[1,1]:3d}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Left', 'Right'],
                yticklabels=['Left', 'Right'])
    plt.title(f'Confusion Matrix - Test Data\nAccuracy: {accuracy:.2%}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('test_data/model_performance_test_data.png', dpi=150)
    plt.close()
    
    # Plot prediction distribution
    plt.figure(figsize=(10, 6))
    
    # Get prediction probabilities if available
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(features)
        
        plt.subplot(1, 2, 1)
        plt.hist(probs[:, 0][labels == 0], bins=20, alpha=0.5, label='Left trials', color='blue')
        plt.hist(probs[:, 0][labels == 1], bins=20, alpha=0.5, label='Right trials', color='red')
        plt.xlabel('Left Class Probability')
        plt.ylabel('Count')
        plt.legend()
        plt.title('Probability Distribution')
        
        plt.subplot(1, 2, 2)
        # Plot confidence vs accuracy
        confidence = np.max(probs, axis=1)
        correct = predictions == labels
        
        plt.scatter(range(len(confidence)), confidence, c=correct, cmap='RdYlGn', alpha=0.6)
        plt.xlabel('Trial')
        plt.ylabel('Confidence')
        plt.title('Prediction Confidence')
        plt.colorbar(label='Correct')
    else:
        # Just plot predictions
        plt.scatter(range(len(predictions)), predictions, c=labels, cmap='coolwarm', alpha=0.6)
        plt.xlabel('Trial')
        plt.ylabel('Prediction')
        plt.title('Predictions vs True Labels')
        plt.colorbar(label='True Label')
    
    plt.tight_layout()
    plt.savefig('test_data/model_predictions_test_data.png', dpi=150)
    plt.close()
    
    print(f"\nPlots saved to test_data/")
    
    return predictions, accuracy, cm

def main():
    """Main processing pipeline"""
    
    print("="*60)
    print("Creating MI Dataset from Test Data")
    print("="*60)
    
    # Load test data
    eeg_data, timestamps, events_df = load_test_data()
    print(f"\nLoaded data:")
    print(f"  EEG shape: {eeg_data.shape}")
    print(f"  Duration: {timestamps[-1]:.1f}s")
    print(f"  Events: {len(events_df)}")
    
    # Extract MI epochs
    epochs, labels, epoch_info = extract_mi_epochs(eeg_data, timestamps, events_df)
    
    if len(epochs) == 0:
        print("\nNo valid epochs found!")
        return
    
    # Create dataset in T-005 format
    dataset_dir = create_mi_dataset(epochs, labels, epoch_info)
    
    # Find the best model (try different ones)
    model_paths = [
        "../MI/models/mi_improved_classifier.pkl",
        "../MI/models/mi_classifier_simple.pkl",
        "../MI/models/subject_T-005_current.pkl",
        "../MI/models/best_model_T-005.pkl",
        "../MI/models/mi_riemannian_final.pkl"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("\nError: Could not find model!")
        print("Searched paths:")
        for path in model_paths:
            print(f"  {path}")
        return
    
    # Load model to prepare features
    print(f"\nLoading model for feature extraction: {model_path}")
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Prepare features for model
    features = prepare_features_for_model(epochs, model_dict)
    
    # Test with model
    predictions, accuracy, cm = test_with_model(features, labels, model_path)
    
    print("\n" + "="*60)
    print("Processing Complete!")
    print("="*60)

if __name__ == "__main__":
    main()