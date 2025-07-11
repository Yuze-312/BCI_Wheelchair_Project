# errp_offline_classifier.py (complete updated version)
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
import matplotlib
matplotlib.use('Agg')

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='Pandas requires version')
warnings.filterwarnings('ignore', module='pandas')

# Alternative: Set environment variable to suppress at system level
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'

class OfflineErrPClassifier:
    def __init__(self, sampling_rate=512):
        self.sampling_rate = sampling_rate
        self.feature_windows = [
            (0.15, 0.20),  # Early negativity
            (0.25, 0.35),  # Pe component
        ]
        self.selected_channels = [0, 1, 2, 3]  # Ch1, Ch4, Ch6, Ch8
        
        # Updated classifiers with class balancing
        self.classifiers = {
            'LDA': LinearDiscriminantAnalysis(),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'LR_L1': LogisticRegression(penalty='l1', solver='liblinear', 
                                       class_weight='balanced', C=0.1, random_state=42)
        }
        
        self.best_classifier = None
        self.best_params = None
        self.scaler = StandardScaler()
        

    def extract_features(self, epochs, feature_type='mean'):
        """
        Extract features from epochs with proper implementation of different feature types
        
        Args:
            epochs: numpy array (n_epochs, n_samples, n_channels)
            feature_type: Type of features to extract
        """
        times = np.linspace(-0.2, 0.8, epochs.shape[1])
        
        # Define time windows based on ERP components visible in plots
        windows = {
            'n200': (0.15, 0.25),   # N200 component
            'pe': (0.45, 0.55),     # Pe component (CORRECTED to match your plots!)
            'late': (0.60, 0.70)    # Late positivity
        }
        
        features = []
        
        # Extract features for each window and channel
        for window_name, (start, end) in windows.items():
            start_idx = np.argmin(np.abs(times - start))
            end_idx = np.argmin(np.abs(times - end))
            
            for ch in self.selected_channels:
                window_data = epochs[:, start_idx:end_idx, ch]
                
                if feature_type == 'mean':
                    # Mean amplitude in window
                    features.append(np.mean(window_data, axis=1))
                    
                elif feature_type == 'peak':
                    # Peak amplitude in window
                    features.append(np.max(window_data, axis=1))
                    
                elif feature_type == 'both':
                    # Both mean and peak
                    features.append(np.mean(window_data, axis=1))
                    features.append(np.max(window_data, axis=1))
                    
                elif feature_type == 'temporal':
                    # Time-based features
                    peak_idx = np.argmax(window_data, axis=1)
                    features.append(np.max(window_data, axis=1))  # Peak amplitude
                    features.append(peak_idx / window_data.shape[1])  # Normalized latency
                    
                elif feature_type == 'peak_to_peak':
                    # Peak-to-peak amplitude
                    features.append(np.ptp(window_data, axis=1))
                    
                elif feature_type == 'slope':
                    # Average slope in window
                    slopes = np.diff(window_data, axis=1).mean(axis=1)
                    features.append(slopes)
                    
                elif feature_type == 'area':
                    # Area under curve (absolute values)
                    features.append(np.trapz(np.abs(window_data), axis=1))
                    
                elif feature_type == 'comprehensive':
                    # Multiple features per window
                    features.append(np.mean(window_data, axis=1))          # Mean
                    features.append(np.max(window_data, axis=1))           # Peak
                    features.append(np.std(window_data, axis=1))           # Variability
                    features.append(np.ptp(window_data, axis=1))           # Range
                    
                    # Peak latency (normalized to window)
                    peak_idx = np.argmax(window_data, axis=1)
                    features.append(peak_idx / window_data.shape[1])
        
        return np.column_stack(features)


    def extract_targeted_pe_features(self, epochs):
        """
        Extract features specifically targeting the Pe component differences
        Based on the clear amplitude gap visible in your plots
        """
        times = np.linspace(-0.2, 0.8, epochs.shape[1])
        
        # Focus on Pe window where you see the clearest difference
        pe_start = np.argmin(np.abs(times - 0.45))
        pe_end = np.argmin(np.abs(times - 0.55))
        
        # Also get baseline for normalization
        baseline_end = np.argmin(np.abs(times - 0.0))
        
        features = []
        
        for ch in self.selected_channels:
            # Pe window data
            pe_data = epochs[:, pe_start:pe_end, ch]
            
            # Baseline data
            baseline_data = epochs[:, :baseline_end, ch]
            baseline_mean = np.mean(baseline_data, axis=1)
            baseline_std = np.std(baseline_data, axis=1)
            
            # 1. Raw Pe features
            pe_peak = np.max(pe_data, axis=1)
            pe_mean = np.mean(pe_data, axis=1)
            
            # 2. Baseline-corrected features (should enhance differences)
            pe_peak_corrected = pe_peak - baseline_mean
            pe_mean_corrected = pe_mean - baseline_mean
            
            # 3. Z-scored features (normalized by baseline variability)
            pe_peak_zscore = (pe_peak - baseline_mean) / (baseline_std + 1e-6)
            
            # 4. Peak latency within Pe window
            peak_idx = np.argmax(pe_data, axis=1)
            peak_latency = peak_idx / pe_data.shape[1]
            
            # Add all features
            features.extend([
                pe_peak,
                pe_mean,
                pe_peak_corrected,
                pe_mean_corrected,
                pe_peak_zscore,
                peak_latency
            ])
        
        return np.column_stack(features)


    def debug_features(self, error_epochs, correct_epochs):
        """
        Debug function to verify features are capturing the differences
        """
        from scipy import stats
        
        # Extract features using the targeted Pe approach
        X_error = self.extract_targeted_pe_features(error_epochs)
        X_correct = self.extract_targeted_pe_features(correct_epochs)
        
        print("\n" + "="*60)
        print("FEATURE DEBUGGING")
        print("="*60)
        
        feature_names = []
        for ch in self.selected_channels:
            feature_names.extend([
                f'Ch{ch+1}_pe_peak',
                f'Ch{ch+1}_pe_mean', 
                f'Ch{ch+1}_pe_peak_corrected',
                f'Ch{ch+1}_pe_mean_corrected',
                f'Ch{ch+1}_pe_peak_zscore',
                f'Ch{ch+1}_peak_latency'
            ])
        
        print("\nFeature statistics:")
        print(f"{'Feature':<30} {'Error Mean':>12} {'Correct Mean':>12} {'Diff':>10} {'p-value':>10}")
        print("-" * 75)
        
        significant_features = []
        
        for i, name in enumerate(feature_names):
            error_mean = X_error[:, i].mean()
            correct_mean = X_correct[:, i].mean()
            diff = error_mean - correct_mean
            _, p_value = stats.ttest_ind(X_error[:, i], X_correct[:, i])
            
            print(f"{name:<30} {error_mean:>12.3f} {correct_mean:>12.3f} {diff:>10.3f} {p_value:>10.4f}")
            
            if p_value < 0.05:
                significant_features.append((name, p_value))
        
        print(f"\nSignificant features (p < 0.05): {len(significant_features)}")
        for feat, p in significant_features:
            print(f"  - {feat}: p = {p:.4f}")
        
        # Test performance with targeted features
        print("\nTesting targeted Pe features performance...")
        
        # Balance dataset
        n_error = len(X_error)
        n_correct = len(X_correct)
        if n_correct > 3 * n_error:
            indices = np.random.choice(n_correct, size=3*n_error, replace=False)
            X_correct_balanced = X_correct[indices]
        else:
            X_correct_balanced = X_correct
        
        X = np.vstack([X_error, X_correct_balanced])
        y = np.hstack([np.ones(len(X_error)), np.zeros(len(X_correct_balanced))])
        
        # Scale and test
        X_scaled = StandardScaler().fit_transform(X)
        svm = SVC(kernel='rbf', class_weight='balanced', random_state=42)
        scores = cross_val_score(svm, X_scaled, y, cv=3, scoring='balanced_accuracy')
        
        print(f"\nTargeted Pe features performance:")
        print(f"  Balanced Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
        print(f"  Number of features: {X.shape[1]}")
        
        # Visualize best features
        if significant_features:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Plot top 4 most significant features
            for idx, (feat_name, _) in enumerate(significant_features[:4]):
                if idx >= 4:
                    break
                feat_idx = feature_names.index(feat_name)
                ax = axes[idx]
                
                # Plot distributions
                ax.hist(X_error[:, feat_idx], bins=15, alpha=0.5, color='red', 
                    label='Error', density=True)
                ax.hist(X_correct[:, feat_idx], bins=15, alpha=0.5, color='blue', 
                    label='Correct', density=True)
                
                ax.set_xlabel('Feature Value')
                ax.set_ylabel('Density')
                ax.set_title(feat_name)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add means
                ax.axvline(X_error[:, feat_idx].mean(), color='darkred', 
                        linestyle='--', linewidth=2)
                ax.axvline(X_correct[:, feat_idx].mean(), color='darkblue', 
                        linestyle='--', linewidth=2)
            
            plt.tight_layout()
            plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        return X_error, X_correct
        
    def evaluate_features(self, error_epochs, correct_epochs):
        """Evaluate different feature extraction methods"""
        results = {}
        
        # Balance dataset for evaluation
        n_error = len(error_epochs)
        n_correct = len(correct_epochs)
        
        # Undersample correct trials for balanced evaluation
        if n_correct > 3 * n_error:
            indices = np.random.choice(n_correct, size=3*n_error, replace=False)
            correct_epochs_balanced = correct_epochs[indices]
        else:
            correct_epochs_balanced = correct_epochs
        
        # Test all feature types including new ones
        feature_types = ['mean', 'peak', 'both', 'temporal', 'peak_to_peak', 
                        'slope', 'area', 'comprehensive']
        
        for feature_type in feature_types:
            print(f"\nEvaluating {feature_type} features...")
            
            try:
                # Extract features
                X_error = self.extract_features(error_epochs, feature_type)
                X_correct = self.extract_features(correct_epochs_balanced, feature_type)
                
                X = np.vstack([X_error, X_correct])
                y = np.hstack([np.ones(len(X_error)), np.zeros(len(X_correct))])
                
                # Scale features
                X_scaled = StandardScaler().fit_transform(X)
                
                # Quick LDA test with balanced accuracy
                lda = LinearDiscriminantAnalysis()
                scores = cross_val_score(lda, X_scaled, y, cv=3, 
                                    scoring='balanced_accuracy')
                
                results[feature_type] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'n_features': X.shape[1]
                }
                
                print(f"  Balanced Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
                print(f"  Number of features: {X.shape[1]}")
                
            except Exception as e:
                print(f"  Error with {feature_type}: {str(e)}")
                results[feature_type] = {
                    'scores': [0],
                    'mean': 0,
                    'std': 0,
                    'n_features': 0
                }
        
        # Also test targeted Pe features
        print(f"\nEvaluating targeted_pe features...")
        try:
            X_error = self.extract_targeted_pe_features(error_epochs)
            X_correct = self.extract_targeted_pe_features(correct_epochs_balanced)
            
            X = np.vstack([X_error, X_correct])
            y = np.hstack([np.ones(len(X_error)), np.zeros(len(X_correct))])
            
            X_scaled = StandardScaler().fit_transform(X)
            lda = LinearDiscriminantAnalysis()
            scores = cross_val_score(lda, X_scaled, y, cv=3, scoring='balanced_accuracy')
            
            results['targeted_pe'] = {
                'scores': scores,
                'mean': scores.mean(),
                'std': scores.std(),
                'n_features': X.shape[1]
            }
            
            print(f"  Balanced Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
            print(f"  Number of features: {X.shape[1]}")
            
        except Exception as e:
            print(f"  Error with targeted_pe: {str(e)}")
        
        return results

    def analyze_feature_importance(self, error_epochs, correct_epochs, feature_type='comprehensive'):
        """Analyze which features are most important for classification"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        
        # Extract features
        if feature_type == 'targeted_pe':
            X_error = self.extract_targeted_pe_features(error_epochs)
            X_correct = self.extract_targeted_pe_features(correct_epochs)
        else:
            X_error = self.extract_features(error_epochs, feature_type)
            X_correct = self.extract_features(correct_epochs, feature_type)
        
        # Balance dataset
        n_error = len(X_error)
        n_correct = len(X_correct)
        if n_correct > 3 * n_error:
            indices = np.random.choice(n_correct, size=3*n_error, replace=False)
            X_correct = X_correct[indices]
        
        X = np.vstack([X_error, X_correct])
        y = np.hstack([np.ones(len(X_error)), np.zeros(len(X_correct))])
        
        # Scale
        X_scaled = StandardScaler().fit_transform(X)
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_scaled, y)
        
        # Get feature names based on feature type
        if feature_type == 'targeted_pe':
            feature_names = []
            for ch in self.selected_channels:
                feature_names.extend([
                    f'Ch{ch+1}_pe_peak',
                    f'Ch{ch+1}_pe_mean',
                    f'Ch{ch+1}_pe_peak_corrected',
                    f'Ch{ch+1}_pe_mean_corrected',
                    f'Ch{ch+1}_pe_peak_zscore',
                    f'Ch{ch+1}_peak_latency'
                ])
        else:
            # Generate feature names for other types
            feature_names = []
            windows = {'n200': (0.15, 0.25), 'pe': (0.45, 0.55), 'late': (0.60, 0.70)}
            
            for window_name, (start, end) in windows.items():
                window_label = f"{window_name}({int(start*1000)}-{int(end*1000)}ms)"
                
                if feature_type == 'comprehensive':
                    for ch in self.selected_channels:
                        feature_names.extend([
                            f"{window_label}_Ch{ch+1}_mean",
                            f"{window_label}_Ch{ch+1}_peak",
                            f"{window_label}_Ch{ch+1}_std",
                            f"{window_label}_Ch{ch+1}_p2p",
                            f"{window_label}_Ch{ch+1}_latency"
                        ])
                else:
                    for ch in self.selected_channels:
                        feature_names.append(f"{window_label}_Ch{ch+1}")
        
        # Get importances
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot top features
        plt.figure(figsize=(10, 6))
        top_n = min(20, len(feature_names))
        plt.bar(range(top_n), importances[indices[:top_n]])
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title(f'Top {top_n} Most Important Features ({feature_type})')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print top features
        print(f"\nTop 10 most important features ({feature_type}):")
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
        
        return feature_names, importances
    
    def optimize_classifier(self, X, y):
        """Find best classifier and parameters"""
        print("\nOptimizing classifiers...")
        
        # Compute class weights for balanced training
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weight_dict = dict(zip(classes, class_weights))
        print(f"Class weights: {class_weight_dict}")
        
        # Updated parameter grids
        param_grids = {
            'LDA': {
                'solver': ['svd', 'lsqr'],
                'shrinkage': [None]  # shrinkage only works with specific solvers
            },
            'SVM': {
                'C': [0.01, 0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            },
            'RF': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'LR_L1': {
                'C': [0.01, 0.1, 1, 10]
            }
        }
        
        best_score = 0
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for clf_name, clf in self.classifiers.items():
            print(f"\n{clf_name}:")
            
            if clf_name in param_grids:
                grid = GridSearchCV(clf, param_grids[clf_name], cv=cv, 
                                   scoring='balanced_accuracy', n_jobs=-1)
                grid.fit(X, y)
                
                print(f"  Best parameters: {grid.best_params_}")
                print(f"  Best balanced accuracy: {grid.best_score_:.3f}")
                
                if grid.best_score_ > best_score:
                    best_score = grid.best_score_
                    self.best_classifier = grid.best_estimator_
                    self.best_params = {
                        'classifier': clf_name,
                        'params': grid.best_params_,
                        'score': grid.best_score_
                    }
        
        return self.best_classifier
    
    def evaluate_with_feature_selection(self, error_epochs, correct_epochs, feature_type='comprehensive', k_features=10):
        """Evaluate performance with feature selection"""
        print(f"\nEvaluating with feature selection (k={k_features})...")
        
        # Extract features
        if feature_type == 'targeted_pe':
            X_error = self.extract_targeted_pe_features(error_epochs)
            X_correct = self.extract_targeted_pe_features(correct_epochs)
        else:
            X_error = self.extract_features(error_epochs, feature_type)
            X_correct = self.extract_features(correct_epochs, feature_type)
        
        # Balance dataset
        n_error = len(X_error)
        n_correct = len(X_correct)
        if n_correct > 3 * n_error:
            indices = np.random.choice(n_correct, size=3*n_error, replace=False)
            X_correct = X_correct[indices]
        
        X = np.vstack([X_error, X_correct])
        y = np.hstack([np.ones(len(X_error)), np.zeros(len(X_correct))])
        
        # Scale
        X_scaled = StandardScaler().fit_transform(X)
        
        # Feature selection
        selector = SelectKBest(f_classif, k=min(k_features, X.shape[1]))
        X_selected = selector.fit_transform(X_scaled, y)
        
        print(f"Selected {X_selected.shape[1]} features from {X_scaled.shape[1]}")
        
        # Test different classifiers
        for clf_name, clf in self.classifiers.items():
            scores = cross_val_score(clf, X_selected, y, cv=3, scoring='balanced_accuracy')
            print(f"{clf_name}: {scores.mean():.3f} (+/- {scores.std():.3f})")
        
        return selector
    
    def detailed_evaluation(self, error_epochs, correct_epochs, feature_type='comprehensive', use_targeted_pe=False):
        """Perform detailed evaluation with plots"""
        # Extract features
        if use_targeted_pe:
            X_error = self.extract_targeted_pe_features(error_epochs)
            X_correct = self.extract_targeted_pe_features(correct_epochs)
            feature_type = 'targeted_pe'
        else:
            X_error = self.extract_features(error_epochs, feature_type)
            X_correct = self.extract_features(correct_epochs, feature_type)
        
        # Balance the dataset by undersampling majority class
        n_error = len(X_error)
        n_correct = len(X_correct)
        
        print(f"\nOriginal class distribution:")
        print(f"  Error trials: {n_error}")
        print(f"  Correct trials: {n_correct}")
        
        # Balance by undersampling
        if n_correct > 3 * n_error:
            print(f"Balancing dataset by undersampling correct trials to 3:1 ratio")
            indices = np.random.choice(n_correct, size=3*n_error, replace=False)
            X_correct = X_correct[indices]
        
        X = np.vstack([X_error, X_correct])
        y = np.hstack([np.ones(len(X_error)), np.zeros(len(X_correct))])
        
        print(f"\nBalanced class distribution:")
        print(f"  Error trials: {sum(y==1)}")
        print(f"  Correct trials: {sum(y==0)}")
        
        # Scale
        X_scaled = self.scaler.fit_transform(X)
        
        # Optimize classifier
        self.optimize_classifier(X_scaled, y)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_results = {
            'y_true': [],
            'y_pred': [],
            'y_prob': []
        }
        
        print(f"\nDetailed evaluation using {self.best_params['classifier']}:")
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train
            self.best_classifier.fit(X_train, y_train)
            
            # Predict
            y_pred = self.best_classifier.predict(X_test)
            y_prob = self.best_classifier.predict_proba(X_test)[:, 1]
            
            fold_results['y_true'].extend(y_test)
            fold_results['y_pred'].extend(y_pred)
            fold_results['y_prob'].extend(y_prob)
            
            # Fold metrics
            fold_bacc = balanced_accuracy_score(y_test, y_pred)
            error_recall = np.mean(y_pred[y_test==1] == 1) if sum(y_test==1) > 0 else 0
            print(f"  Fold {fold+1}: Balanced Accuracy = {fold_bacc:.3f}, Error Recall = {error_recall:.3f}")
        
        # Overall results
        y_true = np.array(fold_results['y_true'])
        y_pred = np.array(fold_results['y_pred'])
        y_prob = np.array(fold_results['y_prob'])
        
        # Print classification report
        print("\nOverall Classification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['Correct', 'Error'],
                                  digits=3))
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('True')
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xticklabels(['Correct', 'Error'])
        axes[0,0].set_yticklabels(['Correct', 'Error'])
        
        # 2. ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[0,1].plot(fpr, tpr, 'b-', label=f'AUC = {roc_auc:.3f}')
        axes[0,1].plot([0, 1], [0, 1], 'k--')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curve')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Probability Distribution
        axes[1,0].hist(y_prob[y_true==0], bins=20, alpha=0.5, 
                      label='Correct', color='blue', density=True)
        axes[1,0].hist(y_prob[y_true==1], bins=20, alpha=0.5, 
                      label='Error', color='red', density=True)
        axes[1,0].set_xlabel('Predicted Probability of Error')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Probability Distributions')
        axes[1,0].legend()
        axes[1,0].axvline(0.5, color='k', linestyle='--', alpha=0.5)
        
        # 4. Feature Importance (if available)
        if hasattr(self.best_classifier, 'feature_importances_'):
            importances = self.best_classifier.feature_importances_
            axes[1,1].bar(range(len(importances)), importances)
            axes[1,1].set_xlabel('Feature Index')
            axes[1,1].set_ylabel('Importance')
            axes[1,1].set_title('Feature Importance')
        else:
            axes[1,1].text(0.5, 0.5, 'Feature importance\nnot available\nfor this classifier',
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        
        return {
            'classifier': self.best_params,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'feature_type': feature_type
        }
    
    def save_results(self, results, participant_id, output_dir='classifier_results'):
        """Save classifier and results"""
        output_path = Path(output_dir) / participant_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_data = {
            'classifier': self.best_classifier,
            'scaler': self.scaler,
            'feature_windows': self.feature_windows,
            'selected_channels': self.selected_channels,
            'best_params': self.best_params,
            'results': results
        }
        
        with open(output_path / 'errp_classifier.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save plots
        plt.savefig(output_path / 'classifier_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nResults saved to {output_path}")