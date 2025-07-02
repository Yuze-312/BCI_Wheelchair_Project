# Motor Imagery Classification - Riemannian Approach

## Overview
Clean implementation of Motor Imagery (MI) classification using Riemannian geometry, which works directly with EEG covariance matrices instead of hand-crafted features.

## Why Riemannian Geometry?
- **No feature engineering**: Works directly with spatial covariance patterns
- **Natural for EEG**: Respects the geometry of covariance matrices (SPD manifold)
- **Better performance**: Often 5-10% improvement over Euclidean approaches
- **Robust**: Handles artifacts and non-stationarity well

## Quick Start

### 1. Process Raw Data
```bash
python process_mi_data.py
```
Preprocesses raw EEG data (8-30 Hz bandpass, CAR/Laplacian spatial filter)

### 2. Train Riemannian Classifier
```bash
python train_riemannian_mi.py
```
Trains and evaluates different Riemannian classifiers:
- MDM (Minimum Distance to Mean)
- Tangent Space + LDA
- Tangent Space + SVM

### 3. Train Traditional Classifier (Optional)
```bash
python train_binary_mi.py
```
For comparison with traditional feature-based approach

## File Structure
```
MI/
├── process_mi_data.py          # Data preprocessing
├── train_riemannian_mi.py      # Riemannian classification (NEW)
├── train_binary_mi.py          # Traditional binary classification
├── train_mi_simple.py          # Multi-class classification
│
├── data_processing/            # Core processing modules
├── feature_extraction/         # Feature extraction (for traditional approach)
├── models/                     # Saved models
│   └── mi_riemannian_classifier.pkl  # Trained Riemannian model
│
└── processed_data/            # Preprocessed epochs
    ├── T-005_processed.pkl
    └── T-008_processed.pkl
```

## Performance
- **Binary (Left vs Right)**: ~65-70% accuracy
- **Approach**: Tangent Space + LDA typically performs best
- **Dataset**: 165 trials from 2 subjects

## Implementation Details

The Riemannian pipeline:
1. **Covariance estimation**: Ledoit-Wolf regularized (`lwf`)
2. **Classification**:
   - Direct: MDM (Riemannian distance)
   - Projected: Tangent space + LDA/SVM
3. **No feature extraction needed!**

## Dependencies
```bash
pip install numpy scipy scikit-learn matplotlib pyriemann
```

## Key Advantages
- Works with small datasets
- No hyperparameter tuning for features
- Transfer learning ready
- Online adaptation possible