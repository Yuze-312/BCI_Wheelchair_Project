# Universal MI Classifier

This folder contains the cross-subject (universal) MI classifier that combines data from multiple participants.

## Overview
- Combines data from all available subjects (T-005, T-008, etc.)
- Trains a single model to work across different people
- Typically achieves 50-70% accuracy due to inter-subject variability

## Files
- `train_universal_classifier.py` - Main training script
- Pre-trained models in `MI/models/`:
  - `mi_improved_classifier.pkl` - Best universal model
  - `best_model_combined.pkl` - Combined dataset model

## Usage
```bash
# Train universal classifier
python train_universal_classifier.py

# The model will be saved to MI/models/
```

## Limitations
- Lower accuracy than subject-specific models
- May show bias (e.g., always predicting RIGHT)
- Doesn't account for individual brain differences
- Electrode placement variations affect performance

## When to Use
- Initial testing without calibration
- Baseline comparison
- When no subject-specific data available
- Research on transfer learning approaches