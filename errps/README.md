# ErrP (Error-Related Potential) Processing Module

This module provides tools for processing EEG data to detect and analyze error-related potentials from BCI experiments using advanced hybrid classification approaches.

## Overview

The ErrP module implements a state-of-the-art pipeline that:
- Loads EEG data and event logs from BCI wheelchair experiments
- Preprocesses EEG signals with optimized filtering (1-40 Hz bandpass)
- Extracts epochs around error and correct feedback events
- Performs advanced feature extraction using Riemannian geometry and amplitude features
- Implements participant-specific channel selection
- Trains hybrid classifiers combining tangent space and ERP features
- Generates comprehensive analysis reports and visualizations

## Installation

The module is already integrated into the BCI Wheelchair Project. Required dependencies:
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- joblib
- pyriemann (optional, for tangent space features)

## Usage

### Command Line Interface

#### 1. Process EEG Data
```bash
# Process a single session
python process_errp_data.py --participant T-001 --session "Session 3"

# Process multiple sessions
python process_errp_data.py --participant T-001 --sessions "Session 3" "Session 4" "Session 5"

# Process all sessions for a participant
python process_errp_data.py --participant T-001 --all-sessions

# Process all participants
python process_errp_data.py --all-participants
```

#### 2. Train Hybrid Classifier
```bash
# Train classifier with default settings (5-fold CV, LDA)
python train_errp_classifier.py --participant T-001

# Specify custom parameters
python train_errp_classifier.py --participant T-001 --n-folds 10 --output-dir custom_results
```

#### 3. Analyze Results
```bash
# Generate comprehensive ERP analysis
python errp_analysis.py --participant T-001

# Analyze specific sessions
python analyze_errp_data.py --participant T-001 --sessions "Session 3" "Session 4"
```

### Python API

```python
# Load processed data
import numpy as np
data = np.load('errp_results/T-001_merged_errp_data.npz', allow_pickle=True)

# Access epoch data
error_epochs = data['error_epochs']    # Shape: (n_error, n_timepoints, n_channels)
correct_epochs = data['correct_epochs'] # Shape: (n_correct, n_timepoints, n_channels)

# Load trained model
import pickle
with open('hybrid_results/T-001_hybrid_model_*.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
pipeline = model_data['pipeline']
selected_channels = model_data['selected_channels']
```

## Data Structure

### Input Data Organization
```
EEG_data/
├── T-001/
│   ├── Session 1/
│   │   ├── data_1.csv         # BrainFlow EEG data
│   │   └── phase1_events_*.csv # Event log
│   ├── Session 2/
│   └── Session 3/
└── T-002/
    └── ...
```

### Processed Data Format
```python
# Merged epoch data (*.npz files)
{
    'error_epochs': array,      # Shape: (n_error, n_timepoints, n_channels)
    'correct_epochs': array,    # Shape: (n_correct, n_timepoints, n_channels)
    'times': array,            # Time vector: -0.2 to 0.8 seconds
    'sampling_rate': 512,      # Hz
    'channel_names': list,     # 16 EEG channels
    'sessions': list          # Session metadata
}
```

## Event Markers

The module recognizes these event markers from the BCI simulator:
- `8` - MARKER_FEEDBACK_CORRECT
- `9` - MARKER_FEEDBACK_ERROR
- `10` - MARKER_PRIMARY_ERRP

## Output Files

Processing generates several output files:
- `errp_results/*_merged_errp_data.npz` - Processed epoch data
- `hybrid_results/*_hybrid_model_*.pkl` - Trained classifier and metadata
- `errp_analysis/*_analysis.png` - ERP visualizations
- `errp_analysis/*_analysis_results.npz` - Statistical analysis

## Processing Pipeline

### 1. Data Loading & Preprocessing
- **Load**: BrainFlow CSV files (16 channels @ 512 Hz)
- **Filter**: Bandpass 1-40 Hz (Butterworth, 4th order)
- **Spatial Filter**: Common Average Reference (CAR)
- **Artifact Removal**: Amplitude-based rejection

### 2. Epoch Extraction
- **Window**: -200ms to +800ms around feedback
- **Baseline**: -200ms to 0ms correction
- **Separation**: Error vs Correct feedback epochs

### 3. Feature Extraction (Hybrid Approach)
- **Tangent Space Features** (0-450ms):
  - Covariance matrix computation
  - Riemannian tangent space projection
- **Amplitude Features**:
  - N component: Mean amplitude 0-300ms
  - Pe component: Mean amplitude 300-500ms

### 4. Channel Selection
- **Method**: Participant-specific selection
- **Process**: Score channels individually via 5-fold CV
- **Output**: Top 5 channels per participant

### 5. Classification
- **Pipeline**: StandardScaler → LDA
- **Validation**: 5-fold stratified cross-validation
- **Metrics**: Accuracy, AUC, confusion matrix

## Key Features

### Hybrid Classification Approach
- **Tangent Space Features**: Leverages Riemannian geometry for robust covariance-based features
- **Amplitude Features**: Captures traditional ERP components (N200, Pe)
- **Participant-Specific Optimization**: Dynamic channel selection per individual

### Performance Characteristics
- Typical accuracy: 70-85% (participant-dependent)
- Real-time capable: ~50ms processing per epoch
- Robust to session variability

## Example Results

The analysis generates comprehensive visualizations including:
- **ERP Comparisons**: Error vs correct waveforms with confidence intervals
- **Difference Waves**: Highlighting N200 and Pe components
- **Statistical Maps**: Time-point significance testing
- **Channel Analysis**: All 16 channels with amplitude differences
- **Component Summary**: Peak amplitudes and latencies

## Integration with BCI System

The trained ErrP detector can be integrated into the real-time BCI control system to:
- Detect errors in motor imagery classification
- Provide error feedback for adaptation
- Improve overall BCI performance through error-aware control

## Troubleshooting

Common issues:
- **No data files found**: Check file naming (should be `data_*.csv`)
- **No events found**: Ensure event log files are present (`phase1_events_*.csv`)
- **Insufficient epochs**: Need minimum 10 error and 10 correct epochs
- **Shape mismatch**: Data is automatically transposed if needed
- **Missing pyriemann**: Falls back to simplified covariance features

## Module Structure

```
errps/
├── process_errp_data.py      # Main processing script
├── train_errp_classifier.py  # Hybrid classifier training
├── errp_analysis.py         # Visualization and analysis
├── data_loader.py           # BrainFlow CSV loader
├── preprocessor.py          # Signal preprocessing
├── epoch_extractor.py       # Event-based epoching
├── advanced_feature_extractor.py  # Hybrid feature extraction
├── channel_selector.py      # Dynamic channel selection
└── hybrid_classifier.py     # Classification pipeline
```

## References

This module implements methods based on:
- Barachant et al. (2012) - Riemannian geometry applied to BCI
- Chavarriaga et al. (2014) - Error-related potentials in brain-computer interfaces
- Lotte & Congedo (2016) - A review of classification algorithms for EEG-based BCI