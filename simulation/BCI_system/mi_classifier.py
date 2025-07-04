"""
MI Classifier - Handles model loading and classification
"""

import numpy as np
from scipy import signal
import pickle
import os


class MIClassifier:
    """Motor Imagery classifier using CSP + LDA"""
    
    def __init__(self, debug=False):
        """Initialize classifier
        
        Args:
            debug: Enable debug output
        """
        self.debug_mode = debug
        self.classifier = None
        self.csp = None
        self.scaler = None
        self.fs = 512  # Default sampling rate
        self.selected_channels = None
        
        # Processing parameters
        self.epoch_length = 2.0  # seconds
        self.confidence_threshold = 0.3
        
        # Bandpass filter coefficients
        self.b = None
        self.a = None
        
    def load_model(self, model_path=None, participant_id=None):
        """Load pretrained MI classifier
        
        Args:
            model_path: Path to model file. If None, searches for participant model.
            participant_id: Participant ID to search for (e.g., 'T-005')
        """
        if model_path is None:
            # Navigate up from BCI_system to simulation, then to project root
            simulation_dir = os.path.dirname(os.path.dirname(__file__))
            project_root = os.path.dirname(simulation_dir)
            models_dir = os.path.join(project_root, 'MI/models/trained_models')
            
            # Try to find participant-specific model
            if participant_id:
                # Try current model first
                participant_model = os.path.join(models_dir, f'subject_{participant_id}_current.pkl')
                if os.path.exists(participant_model):
                    model_path = participant_model
                    print(f"Loading {participant_id}'s model from {model_path}...")
                else:
                    # Look for any model for this participant
                    import glob
                    participant_models = glob.glob(os.path.join(models_dir, f'subject_{participant_id}_*.pkl'))
                    if participant_models:
                        # Use the most recent one
                        model_path = max(participant_models, key=os.path.getmtime)
                        print(f"Loading {participant_id}'s model from {model_path}...")
                    else:
                        print(f"No model found for participant {participant_id}")
            
            # Fallback options
            if model_path is None:
                # Try default T-005 model
                default_model = os.path.join(models_dir, 'subject_T-005_current.pkl')
                if os.path.exists(default_model):
                    model_path = default_model
                    print(f"Using default T-005 model (no model for {participant_id if participant_id else 'unknown'})")
                else:
                    # Last resort: universal model
                    fallback_path = os.path.join(project_root, 'MI/models/mi_improved_classifier.pkl')
                    if os.path.exists(fallback_path):
                        model_path = fallback_path
                        print(f"Using universal model (no participant-specific model found)")
                    else:
                        raise FileNotFoundError(f"No models found in {models_dir}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract components
        self.classifier = model_data['model']  # LDA classifier
        self.csp = model_data['csp']          # CSP spatial filters
        self.scaler = model_data.get('scaler', None)  # Feature scaler if exists
        
        # Model info
        self.fs = model_data.get('fs', 512)    # Sampling rate used for training
        self.selected_channels = model_data.get('selected_channels', None)
        
        # Get accuracy info
        accuracy_info = "Unknown"
        if 'results' in model_data and isinstance(model_data['results'], dict):
            if 'LDA' in model_data['results']:
                accuracy_info = f"{model_data['results']['LDA']:.1%}"
        elif 'test_accuracy' in model_data:
            accuracy_info = f"{model_data['test_accuracy']:.1%}"
        
        print("Loaded pretrained MI model")
        print(f"  Model type: {type(self.classifier).__name__}")
        print(f"  CSP components: {self.csp.n_components}")
        print(f"  Training accuracy: {accuracy_info}")
        
        if 'participant_id' in model_data:
            print(f"  Trained on: {model_data['participant_id']}")
    
    def setup_filter(self, srate):
        """Setup bandpass filter for MI band (8-30 Hz)
        
        Args:
            srate: Sampling rate
        """
        nyquist = srate / 2
        if nyquist > 30:
            self.b, self.a = signal.butter(5, [8/nyquist, 30/nyquist], btype='band')
        else:
            self.b, self.a = None, None
    
    def preprocess_epoch(self, epoch_data, srate):
        """Preprocess data to match training format
        
        Args:
            epoch_data: Raw EEG data (samples, channels)
            srate: Current sampling rate
            
        Returns:
            Preprocessed data ready for CSP (1, channels, samples)
        """
        # 1. Select channels if specified
        if self.selected_channels is not None:
            available_channels = min(epoch_data.shape[1], len(self.selected_channels))
            epoch_data = epoch_data[:, :available_channels]
        
        # 2. Resample if necessary
        if srate != self.fs:
            n_samples_target = int(self.epoch_length * self.fs)
            n_samples_current = epoch_data.shape[0]
            
            resampled = np.zeros((n_samples_target, epoch_data.shape[1]))
            for ch in range(epoch_data.shape[1]):
                resampled[:, ch] = signal.resample(epoch_data[:, ch], n_samples_target)
            epoch_data = resampled
        
        # 3. Bandpass filter
        if self.b is not None:
            filtered = signal.filtfilt(self.b, self.a, epoch_data, axis=0)
        else:
            filtered = epoch_data
        
        # 4. Format for CSP (trials, channels, samples)
        formatted = filtered.T[np.newaxis, :, :]  # (1, channels, samples)
        
        # 5. Pad channels if needed
        if hasattr(self.csp, 'patterns_') and self.csp.patterns_.shape[0] > formatted.shape[1]:
            expected_channels = self.csp.patterns_.shape[0]
            current_channels = formatted.shape[1]
            if current_channels < expected_channels:
                padding = np.zeros((formatted.shape[0], expected_channels - current_channels, formatted.shape[2]))
                formatted = np.concatenate([formatted, padding], axis=1)
                if self.debug_mode:
                    print(f"[DEBUG] Padded from {current_channels} to {expected_channels} channels")
        
        return formatted
    
    def check_signal_quality(self, epoch_data):
        """Check if signal quality is acceptable
        
        Returns:
            tuple: (is_good, quality_info)
        """
        # Check for flat channels
        channel_vars = np.var(epoch_data, axis=0)
        flat_channels = np.sum(channel_vars < 0.1)
        
        # Check for excessive amplitude
        max_amp = np.max(np.abs(epoch_data))
        
        # Check power in EEG bands
        if self.b is not None:
            filtered = signal.filtfilt(self.b, self.a, epoch_data, axis=0)
            signal_power = np.mean(np.square(filtered))
        else:
            signal_power = np.mean(np.square(epoch_data))
        
        quality_info = {
            'flat_channels': flat_channels,
            'max_amplitude': max_amp,
            'signal_power': signal_power,
            'channel_variances': channel_vars
        }
        
        # Quality thresholds
        n_channels = epoch_data.shape[1] if epoch_data.ndim > 1 else 1
        is_good = (flat_channels < n_channels * 0.3 and  # Less than 30% flat
                  max_amp < 200 and  # Not saturated
                  signal_power > 0.5)  # Has some signal
        
        return is_good, quality_info
    
    def classify(self, epoch_data, srate):
        """Classify MI from epoch data
        
        Args:
            epoch_data: EEG data (samples, channels)
            srate: Sampling rate
            
        Returns:
            tuple: (mi_class, confidence) where mi_class is 'left', 'right', or None
        """
        if self.classifier is None:
            raise RuntimeError("Model not loaded! Call load_model() first.")
        
        try:
            # Check signal quality
            quality_ok, quality_info = self.check_signal_quality(epoch_data)
            
            if not quality_ok and self.debug_mode:
                print(f"\nWARNING: Poor signal quality detected:")
                print(f"   Flat channels: {quality_info['flat_channels']}")
                print(f"   Max amplitude: {quality_info['max_amplitude']:.1f}μV")
                print(f"   Signal power: {quality_info['signal_power']:.2f}")
            
            # Preprocess
            processed = self.preprocess_epoch(epoch_data, srate)
            if self.debug_mode:
                print(f"[DEBUG] Epoch shape: {epoch_data.shape}, Processed shape: {processed.shape}")
            
            # Apply CSP
            csp_features = self.csp.transform(processed)
            if self.debug_mode:
                print(f"[DEBUG] CSP features shape: {csp_features.shape}")
            
            # Scale features if scaler exists
            if self.scaler is not None:
                csp_features = self.scaler.transform(csp_features)
            
            # Classify
            prediction = self.classifier.predict(csp_features)
            probabilities = self.classifier.predict_proba(csp_features)
            
            # Calculate confidence using probability margin only
            # This is more reliable than uncalibrated feature magnitude
            confidence = abs(probabilities[0][0] - probabilities[0][1])
            
            # Optional: Apply power transform to make confidence more discriminative
            # confidence = confidence ** 0.5  # Square root makes it less extreme
            
            if self.debug_mode:
                print(f"\nClassification:")
                print(f"   Prediction: {prediction[0]} ({'LEFT' if prediction[0] == 0 else 'RIGHT'})")
                print(f"   Probabilities: LEFT={probabilities[0][0]:.3f}, RIGHT={probabilities[0][1]:.3f}")
                print(f"   Confidence: {confidence:.3f} (threshold: {self.confidence_threshold})")
                print(f"   Margin-based confidence (no boost)")
            
            # Return result
            if confidence >= self.confidence_threshold:
                mi_class = 'left' if prediction[0] == 0 else 'right'
                return mi_class, confidence
            else:
                if self.debug_mode:
                    print(f"   Below threshold - no detection")
                return None, confidence
                
        except Exception as e:
            print(f"Classification error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0.0