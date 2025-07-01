# Subject-Specific MI Classifier

This folder contains tools for training personalized MI classifiers for individual participants.

## Overview
- Trains on single participant data only
- Optimized for individual brain patterns
- Typically achieves 70-90% accuracy
- Requires calibration session (~10-20 minutes)

## Key Differences from Universal
- **Fewer CSP components** (4 vs 6) - prevents overfitting
- **Smaller models** - less complex due to limited data
- **Auto regularization** - handles small sample sizes
- **Cross-validation** - ensures robustness

## Files
- `train_subject_classifier.py` - Main training script
- `collect_calibration_data.py` - Data collection for new users
- `adaptive_classifier.py` - Online adaptation with ErrP feedback

## Usage

### 1. Train for existing participant
```bash
# Train model for participant T-005
python train_subject_classifier.py T-005

# Train with custom settings
python train_subject_classifier.py T-005 --n_components 6
```

### 2. Collect data for new participant
```bash
# Run calibration session
python collect_calibration_data.py --participant YOUR_ID --trials 100

# This will:
# - Show LEFT/RIGHT cues
# - Record EEG data
# - Save to processed_data/YOUR_ID_processed.pkl
```

### 3. Use trained model
```python
# Load subject-specific model
import pickle

with open('models/subject_YOUR_ID_current.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
model = model_data['model']
csp = model_data['csp']
scaler = model_data['scaler']
```

## Model Files
Models are saved with timestamps and participant IDs:
- `subject_T-005_20231215_143022.pkl` - Timestamped version
- `subject_T-005_current.pkl` - Latest version for easy loading

## Advantages
✅ Higher accuracy (70-90% typical)
✅ Captures individual patterns
✅ Works with personal electrode placement
✅ Can adapt over time

## Disadvantages
❌ Requires calibration session
❌ Model only works for that person
❌ Needs retraining if cap placement changes significantly

## Best Practices
1. **Calibration session**: Collect 80-150 trials
2. **Good signal quality**: Check impedances first
3. **Consistent setup**: Same electrode positions
4. **Regular updates**: Retrain if performance drops
5. **Start simple**: Use LDA before trying complex models