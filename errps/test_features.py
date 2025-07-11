# test_features.py (fixed version)
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load data
participant_path = Path('processed_data/T-002')

# List all files to see the naming pattern
print("Available files:")
for f in participant_path.glob('*.pkl'):
    print(f"  {f.name}")

# Get the first session file (adjust pattern as needed)
session_files = list(participant_path.glob('*.pkl'))
if not session_files:
    print("No pickle files found!")
    exit()

# Use Session 2 if available, otherwise use first file
session_file = None
for f in session_files:
    if 'Session_2' in f.name or 'Session 2' in f.name:
        session_file = f
        break
if session_file is None:
    session_file = session_files[0]

print(f"\nUsing file: {session_file.name}")

with open(session_file, 'rb') as f:
    data = pickle.load(f)

error_epochs = data['epochs']['error']['data']
correct_epochs = data['epochs']['correct']['data']

print(f"Error epochs shape: {error_epochs.shape}")
print(f"Correct epochs shape: {correct_epochs.shape}")

# Check if we have times array
if 'times' in data['epochs']['error']:
    times = data['epochs']['error']['times']
else:
    # Generate times based on epoch length
    n_samples = error_epochs.shape[1]
    times = np.linspace(-0.2, 0.8, n_samples)

print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} seconds")

# Plot the grand averages for all channels
n_channels = min(16, error_epochs.shape[2])
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()

for ch in range(n_channels):
    ax = axes[ch]
    
    error_mean = error_epochs[:, :, ch].mean(axis=0)
    correct_mean = correct_epochs[:, :, ch].mean(axis=0)
    
    ax.plot(times, error_mean, 'r-', label='Error', linewidth=2)
    ax.plot(times, correct_mean, 'b-', label='Correct', linewidth=2)
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title(f'Channel {ch+1}')
    ax.set_xlim(-0.2, 0.8)
    
    # Mark the feature windows
    ax.axvspan(0.15, 0.20, alpha=0.2, color='yellow', label='Early')
    ax.axvspan(0.25, 0.35, alpha=0.2, color='green', label='Pe')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    if ch == 0:
        ax.legend()
        
# Hide unused subplots
for i in range(n_channels, 16):
    axes[i].set_visible(False)
        
plt.tight_layout()
plt.savefig('all_channels_erp.png', dpi=150)
print("\nSaved: all_channels_erp.png")
plt.close()

# Find channel with biggest difference
print("\nAnalyzing channel differences...")
max_diff = 0
best_channel = 0
best_time_idx = 0

for ch in range(n_channels):
    error_mean = error_epochs[:, :, ch].mean(axis=0)
    correct_mean = correct_epochs[:, :, ch].mean(axis=0)
    
    diff = np.abs(error_mean - correct_mean)
    max_diff_ch = np.max(diff)
    max_idx = np.argmax(diff)
    
    if max_diff_ch > max_diff:
        max_diff = max_diff_ch
        best_channel = ch
        best_time_idx = max_idx

print(f"Best channel: {best_channel+1} with max difference {max_diff:.2f}μV at {times[best_time_idx]:.3f}s")

# Test feature extraction around the peak difference
window_size = 25  # ~50ms window
peak_start = max(0, best_time_idx - window_size)
peak_end = min(len(times), best_time_idx + window_size)

print(f"\nExtracting features from {times[peak_start]:.3f} to {times[peak_end]:.3f}s")

features_error = []
features_correct = []

# Extract mean amplitude in peak window for best channel
for epoch in error_epochs:
    peak_amp = epoch[peak_start:peak_end, best_channel].mean()
    features_error.append([peak_amp])
    
for epoch in correct_epochs:
    peak_amp = epoch[peak_start:peak_end, best_channel].mean()
    features_correct.append([peak_amp])

X = np.vstack([features_error, features_correct])
y = np.hstack([np.ones(len(features_error)), np.zeros(len(features_correct))])

# Test simple classification
lda = LinearDiscriminantAnalysis()
scores = cross_val_score(lda, X, y, cv=5, scoring='balanced_accuracy')
print(f"\nSimple peak amplitude classification (best channel): {scores.mean():.3f} (+/- {scores.std():.3f})")

# Fix the histogram plotting section (replace lines around 141)
# Convert to numpy arrays and flatten
features_error_array = np.array(features_error).flatten()
features_correct_array = np.array(features_correct).flatten()

# Plot feature distributions
plt.figure(figsize=(10, 6))
plt.hist(features_error_array, bins=20, alpha=0.6, label='Error', color='red', density=True)
plt.hist(features_correct_array, bins=20, alpha=0.6, label='Correct', color='blue', density=True)
plt.xlabel('Peak Amplitude (μV)')
plt.ylabel('Density')
plt.title(f'Distribution of Peak Amplitudes (Channel {best_channel+1})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('feature_distributions.png', dpi=150)
print("Saved: feature_distributions.png")

# Test multiple window features
print("\nTesting different time windows...")
window_results = []

for center_time in np.arange(0.0, 0.6, 0.05):
    center_idx = np.argmin(np.abs(times - center_time))
    start_idx = max(0, center_idx - 25)
    end_idx = min(len(times), center_idx + 25)
    
    X_window = []
    for epoch in np.vstack([error_epochs, correct_epochs]):
        feat = epoch[start_idx:end_idx, best_channel].mean()
        X_window.append([feat])
    
    X_window = np.array(X_window)
    
    scores = cross_val_score(lda, X_window, y, cv=3, scoring='balanced_accuracy')
    window_results.append((center_time, scores.mean()))
    
# Plot results
plt.figure(figsize=(10, 6))
window_times, window_scores = zip(*window_results)
plt.plot(window_times, window_scores, 'o-')
plt.xlabel('Window Center Time (s)')
plt.ylabel('Balanced Accuracy')
plt.title('Classification Performance vs Time Window')
plt.grid(True, alpha=0.3)
plt.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Chance')
plt.savefig('window_performance.png', dpi=150)
print("Saved: window_performance.png")

# Print best window
best_window_idx = np.argmax([w[1] for w in window_results])
best_time, best_score = window_results[best_window_idx]
print(f"\nBest window centered at {best_time:.3f}s with accuracy {best_score:.3f}")