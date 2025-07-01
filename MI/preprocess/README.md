# Preprocessing Module

This folder contains all preprocessing functions for EEG/MI data processing.

## Files

### `load_data.py`
Functions for loading raw EEG data from CSV files:
- Load participant data from CSV format
- Handle multiple sessions
- Extract trial information and metadata

### `load_and_preprocessing.py`
Complete data processing pipeline:
- Combines loading, preprocessing, and feature extraction
- Main class: `MIProcessingPipeline`
- Used by `process_new_participant.py` script

### `preprocess.py`
Main preprocessing functions including:
- **Temporal Filtering**
  - `bandpass_filter()`: Extract mu (8-13Hz) and beta (13-30Hz) rhythms
  - `notch_filter()`: Remove power line interference (50/60 Hz)

- **Spatial Filtering**
  - `apply_car()`: Common Average Reference
  - `apply_laplacian()`: Laplacian spatial filter

- **Artifact Handling**
  - `detect_artifacts()`: Amplitude-based artifact detection
  - `baseline_correction()`: Remove DC offset

### `epoch_extraction.py`
Functions for extracting epochs from continuous data:
- Extract motor imagery periods
- Create overlapping windows
- Handle cue-based segmentation

### `full_preprocessing_pipeline.py`
Complete pipeline combining all preprocessing steps:
1. Temporal filtering (notch + bandpass)
2. Spatial filtering (CAR/Laplacian)
3. Artifact detection
4. Baseline correction
5. CSP feature extraction

## Usage

### Basic Preprocessing
```python
from MI.preprocess import MIPreprocessor

# Initialize preprocessor
preprocessor = MIPreprocessor(sampling_rate=512)

# Apply bandpass filter
filtered_data = preprocessor.bandpass_filter(raw_data, low_freq=8, high_freq=30)

# Apply CAR
car_data = preprocessor.apply_car(filtered_data)

# Detect artifacts
artifact_mask = preprocessor.detect_artifacts(car_data, threshold=100)
```

### Full Pipeline
```python
from MI.preprocess.full_preprocessing_pipeline import FullPreprocessingPipeline

# Create pipeline
pipeline = FullPreprocessingPipeline(sampling_rate=512)

# Process raw data to features
features, csp, info = pipeline.process_raw_to_features(raw_data, labels)
```

### Processing New Participant
```bash
# From project root
python MI/process_new_participant.py --participant T-009
```

## Preprocessing Parameters

### Default Settings
- **Sampling Rate**: 512 Hz
- **Bandpass**: 8-30 Hz (motor imagery frequencies)
- **Notch**: 50 Hz (adjust to 60 Hz for US)
- **Spatial Filter**: CAR (Common Average Reference)
- **Artifact Threshold**: 100 μV
- **CSP Components**: 4 (2 per class)

### Customization
```python
preprocess_params = {
    'bandpass_low': 8.0,
    'bandpass_high': 30.0,
    'notch_freq': 50.0,
    'spatial_filter': 'car',  # or 'laplacian'
    'artifact_threshold': 100.0
}
```

## Data Flow

```
Raw EEG Data (CSV)
    ↓
Temporal Filtering (Notch + Bandpass)
    ↓
Spatial Filtering (CAR/Laplacian)
    ↓
Artifact Detection & Rejection
    ↓
Baseline Correction
    ↓
Epoch Extraction
    ↓
CSP Feature Extraction
    ↓
Features for Classification
```

## Notes

- All filters use zero-phase filtering (`filtfilt`) to avoid phase distortion
- Artifact rejection extends ±100ms around detected artifacts
- CSP is fitted only on training data to avoid overfitting
- Preprocessing parameters should be consistent between training and testing