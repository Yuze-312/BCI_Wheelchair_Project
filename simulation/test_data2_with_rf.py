#!/usr/bin/env python
"""
Test test_data2 with Random Forest model
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the extracted epochs and labels from test_data2
print("Loading test_data2 epochs...")

# We'll need to re-extract since we don't have them saved
import glob
import os

# Load EEG data
brainflow_file = "test_data2/BrainFlow-RAW_2025-06-27_21-38-48_2.csv"
eeg_df = pd.read_csv(brainflow_file, sep='\t', header=None)
eeg_data = eeg_df.iloc[:, 1:17].values / 24  # Apply gain correction
n_samples = len(eeg_data)
fs = 250
timestamps = np.arange(n_samples) / fs

# Load events
events_df = pd.read_csv("test_data2/subway_errp_1751056870.csv")

# Extract epochs
cue_events = events_df[events_df['event_type'].isin(['cue_left', 'cue_right'])]
epochs = []
labels = []

pre_cue = 0.5
post_cue = 4.0
epoch_length = int((pre_cue + post_cue) * fs)
pre_samples = int(pre_cue * fs)

for idx, event in cue_events.iterrows():
    event_time = event['timestamp']
    time_diff = np.abs(timestamps - event_time)
    event_sample = np.argmin(time_diff)
    
    if time_diff[event_sample] > 1.0:
        continue
    
    start_idx = event_sample - pre_samples
    end_idx = start_idx + epoch_length
    
    if start_idx < 0 or end_idx > len(eeg_data):
        continue
    
    epoch = eeg_data[start_idx:end_idx, :]
    label = 0 if event['cue_class'] == 'left' else 1
    
    epochs.append(epoch)
    labels.append(label)

epochs = np.array(epochs)
labels = np.array(labels)

print(f"Epochs shape: {epochs.shape}")
print(f"Labels: {labels}")
print(f"Class distribution: Left={np.sum(labels==0)}, Right={np.sum(labels==1)}")

# Extract features for RF
print("\nExtracting features for Random Forest...")

# Use multiple feature types
features_list = []

for epoch in epochs:
    epoch_features = []
    
    # For each channel
    for ch in range(epoch.shape[1]):
        ch_data = epoch[:, ch]
        
        # Time domain features
        epoch_features.extend([
            np.mean(ch_data),
            np.std(ch_data),
            np.var(ch_data),
            np.max(ch_data) - np.min(ch_data),  # Range
            np.percentile(ch_data, 25),  # Q1
            np.percentile(ch_data, 75),  # Q3
        ])
    
    features_list.append(epoch_features)

features = np.array(features_list)
print(f"Features shape: {features.shape}")

# Train Random Forest with cross-validation
print("\n" + "="*60)
print("RANDOM FOREST TRAINING")
print("="*60)

# Initialize RF with balanced class weights
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,  # Limit depth to prevent overfitting
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

# Cross-validation (leave-one-out for small dataset)
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
cv_scores = []
predictions_cv = np.zeros(len(labels))

for train_idx, test_idx in loo.split(features):
    X_train, X_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    predictions_cv[test_idx] = pred
    cv_scores.append(pred == y_test)

cv_accuracy = np.mean(cv_scores)
print(f"Leave-One-Out CV Accuracy: {cv_accuracy:.2%}")

# Train on full dataset for final model
rf.fit(features, labels)
predictions_full = rf.predict(features)
train_accuracy = accuracy_score(labels, predictions_full)

print(f"Training Accuracy: {train_accuracy:.2%}")

# Confusion matrices
cm_cv = confusion_matrix(labels, predictions_cv)
cm_train = confusion_matrix(labels, predictions_full)

print("\nCross-Validation Confusion Matrix:")
print(f"       Predicted")
print(f"       L    R")
print(f"True L {cm_cv[0,0]:2d}  {cm_cv[0,1]:2d}")
print(f"     R {cm_cv[1,0]:2d}  {cm_cv[1,1]:2d}")

print("\nTraining Confusion Matrix:")
print(f"       Predicted")
print(f"       L    R")
print(f"True L {cm_train[0,0]:2d}  {cm_train[0,1]:2d}")
print(f"     R {cm_train[1,0]:2d}  {cm_train[1,1]:2d}")

# Feature importance
feature_names = []
for ch in range(16):
    feature_names.extend([f'Ch{ch+1}_mean', f'Ch{ch+1}_std', f'Ch{ch+1}_var', 
                         f'Ch{ch+1}_range', f'Ch{ch+1}_Q1', f'Ch{ch+1}_Q3'])

# Get top features
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:20]  # Top 20 features

print("\nTop 20 Most Important Features:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. CV Confusion Matrix
ax = axes[0, 0]
sns.heatmap(cm_cv, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'])
ax.set_title(f'Cross-Validation Confusion Matrix\nAccuracy: {cv_accuracy:.2%}')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# 2. Feature Importance
ax = axes[0, 1]
top_features = [feature_names[idx] for idx in indices[:10]]
top_importances = [importances[idx] for idx in indices[:10]]
ax.barh(range(10), top_importances)
ax.set_yticks(range(10))
ax.set_yticklabels(top_features)
ax.set_xlabel('Importance')
ax.set_title('Top 10 Feature Importances')
ax.invert_yaxis()

# 3. Prediction probabilities
ax = axes[1, 0]
probs = rf.predict_proba(features)
left_probs = probs[:, 0]
right_probs = probs[:, 1]

# Scatter plot with true labels
colors = ['blue' if l == 0 else 'red' for l in labels]
ax.scatter(range(len(labels)), right_probs, c=colors, alpha=0.6)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Sample')
ax.set_ylabel('Right Class Probability')
ax.set_title('Prediction Probabilities\n(Blue=Left, Red=Right)')
ax.set_ylim([0, 1])

# 4. Comparison with LDA
ax = axes[1, 1]

# Load and test with LDA
with open("../MI/models/mi_improved_classifier.pkl", 'rb') as f:
    lda_data = pickle.load(f)

if 'csp' in lda_data:
    csp = lda_data['csp']
    lda = lda_data['model']
    epochs_csp = np.transpose(epochs, (0, 2, 1))
    features_csp = csp.transform(epochs_csp)
    lda_pred = lda.predict(features_csp)
    lda_acc = accuracy_score(labels, lda_pred)
else:
    lda_acc = 0.5  # Placeholder

model_names = ['Random Forest\n(LOO-CV)', 'Random Forest\n(Training)', 'LDA+CSP']
accuracies = [cv_accuracy, train_accuracy, lda_acc]
colors = ['green', 'lightgreen', 'orange']

bars = ax.bar(model_names, accuracies, color=colors, alpha=0.7)
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison')
ax.set_ylim([0, 1])

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.1%}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('test_data2/rf_analysis.png', dpi=150)
plt.close()

# Check for bias
unique_cv_preds, cv_pred_counts = np.unique(predictions_cv, return_counts=True)
print(f"\nCV Prediction distribution:")
for pred, count in zip(unique_cv_preds, cv_pred_counts):
    class_name = "Left" if pred == 0 else "Right"
    print(f"  {class_name}: {count} ({count/len(predictions_cv)*100:.1f}%)")

if len(unique_cv_preds) == 1:
    print("  WARNING: Model shows prediction bias!")
else:
    print("  Model shows reasonable prediction diversity")

print("\nAnalysis complete! Check test_data2/rf_analysis.png for visualizations.")