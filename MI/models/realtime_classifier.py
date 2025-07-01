"""
Real-time Motor Imagery Classifier
Optimized for integration with ErrP simulator
"""

import numpy as np
import joblib
import time
from collections import deque
from typing import Dict, List, Tuple, Optional
import threading
import queue

class RealtimeMIClassifier:
    """
    Real-time motor imagery classifier with buffering and threading
    """
    
    def __init__(self, model_path: str, 
                 sampling_rate: int = 512,
                 window_length: float = 3.0,
                 update_interval: float = 0.1):
        """
        Initialize real-time classifier
        
        Args:
            model_path: Path to trained model
            sampling_rate: EEG sampling rate
            window_length: Classification window in seconds
            update_interval: How often to update predictions
        """
        self.sampling_rate = sampling_rate
        self.window_length = window_length
        self.update_interval = update_interval
        
        # Load model and preprocessors
        self.load_model(model_path)
        
        # Initialize buffers
        self.window_samples = int(window_length * sampling_rate)
        self.eeg_buffer = deque(maxlen=self.window_samples)
        
        # Threading components
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.classifier_thread = None
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.prediction_history = deque(maxlen=100)
        
        # Confidence thresholding and smoothing (IEEE approach)
        self.confidence_threshold = 0.7  # Reject predictions below this
        self.smoothing_alpha = 0.3  # Exponential smoothing factor
        self.probability_history = None  # For exponential smoothing
        self.min_evidence_samples = 3  # Minimum samples before decision
        
    def load_model(self, model_path: str):
        """Load trained model and associated components"""
        print(f"Loading model from {model_path}...")
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.classifier_name = model_data['classifier_name']
        
        # Load feature extractor if saved
        if 'feature_extractor' in model_data:
            self.feature_extractor = model_data['feature_extractor']
        else:
            # Use default feature extraction
            from ..feature_extraction.feature_extractor import MIFeatureExtractor
            self.feature_extractor = MIFeatureExtractor(self.sampling_rate)
            
        # Load CSP if available
        self.csp = model_data.get('csp', None)
        
        print(f"Model loaded: {self.classifier_name}")
        print(f"Expected accuracy: {model_data.get('accuracy', 'Unknown'):.3f}")
        
    def start(self):
        """Start real-time classification thread"""
        if not self.is_running:
            self.is_running = True
            self.classifier_thread = threading.Thread(target=self._classification_loop)
            self.classifier_thread.start()
            print("Real-time classifier started")
            
    def stop(self):
        """Stop real-time classification"""
        self.is_running = False
        if self.classifier_thread:
            self.classifier_thread.join()
        print("Real-time classifier stopped")
        
    def add_eeg_data(self, data: np.ndarray):
        """
        Add new EEG data to buffer
        
        Args:
            data: New EEG samples (samples x channels)
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        for sample in data:
            self.eeg_buffer.append(sample)
            
    def _classification_loop(self):
        """Main classification loop running in separate thread"""
        last_update = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_update >= self.update_interval:
                if len(self.eeg_buffer) >= self.window_samples:
                    # Get current window
                    eeg_window = np.array(self.eeg_buffer)
                    
                    # Classify
                    start_time = time.time()
                    prediction = self._classify_window(eeg_window)
                    processing_time = time.time() - start_time
                    
                    # Store results
                    self.processing_times.append(processing_time)
                    self.prediction_history.append(prediction)
                    
                    # Queue prediction
                    self.prediction_queue.put(prediction)
                    
                last_update = current_time
                
            # Small sleep to prevent CPU overload
            time.sleep(0.001)
            
    def _classify_window(self, eeg_window: np.ndarray) -> Dict:
        """
        Classify a single EEG window
        
        Args:
            eeg_window: EEG data (samples x channels)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(eeg_window)
            
            # Apply CSP if available
            if self.csp is not None:
                # CSP expects 3D input: (n_epochs, n_samples, n_channels)
                eeg_3d = eeg_window[np.newaxis, :, :]
                csp_features = self.csp.transform(eeg_3d)
                features = np.concatenate([csp_features.flatten(), features])
                
            # Reshape for classifier
            features = features.reshape(1, -1)
            
            # Predict
            probabilities = self.model.predict_proba(features)[0]
            
            # Apply confidence thresholding and smoothing
            result = self.get_prediction_with_confidence(probabilities)
            result['timestamp'] = time.time()
            result['raw_probabilities'] = probabilities
            
        except Exception as e:
            print(f"Classification error: {e}")
            result = {
                'prediction': None,
                'command': 'ERROR',
                'confidence': 0.0,
                'probabilities': None,
                'timestamp': time.time(),
                'error': str(e)
            }
            
        return result
    
    def apply_exponential_smoothing(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing to probability estimates (IEEE approach)
        Accumulates evidence over time to prevent accidental commands
        
        Args:
            probabilities: Current prediction probabilities
            
        Returns:
            Smoothed probability estimates
        """
        if self.probability_history is None:
            self.probability_history = probabilities.copy()
            self.evidence_count = 1
        else:
            # Exponential smoothing: new = α * current + (1 - α) * previous
            self.probability_history = (self.smoothing_alpha * probabilities + 
                                      (1 - self.smoothing_alpha) * self.probability_history)
            self.evidence_count += 1
            
        return self.probability_history
    
    def reset_smoothing(self):
        """Reset exponential smoothing state"""
        self.probability_history = None
        self.evidence_count = 0
    
    def get_prediction_with_confidence(self, probabilities: np.ndarray) -> Dict:
        """
        Get prediction with confidence thresholding (IEEE approach)
        
        Args:
            probabilities: Class probabilities
            
        Returns:
            Dictionary with prediction, confidence, and rejection status
        """
        # Apply exponential smoothing
        smoothed_probs = self.apply_exponential_smoothing(probabilities)
        
        # Get confidence
        confidence = np.max(smoothed_probs)
        prediction_class = np.argmax(smoothed_probs)
        
        # Check if we have enough evidence
        if self.evidence_count < self.min_evidence_samples:
            return {
                'prediction': None,
                'command': 'ACCUMULATING',
                'confidence': confidence,
                'rejected': True,
                'reason': 'insufficient_samples'
            }
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            return {
                'prediction': None,
                'command': 'REJECTED',
                'confidence': confidence,
                'rejected': True,
                'reason': 'low_confidence'
            }
        
        # Valid prediction
        mi_commands = {1: 'LEFT', 2: 'RIGHT', 3: 'REST'}
        return {
            'prediction': prediction_class,
            'command': mi_commands.get(prediction_class, 'UNKNOWN'),
            'confidence': confidence,
            'rejected': False,
            'smoothed_probabilities': smoothed_probs
        }
    
    def get_prediction(self, timeout: float = 0.1) -> Optional[Dict]:
        """
        Get latest prediction (non-blocking)
        
        Args:
            timeout: Maximum time to wait for prediction
            
        Returns:
            Prediction dictionary or None
        """
        try:
            return self.prediction_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_performance_stats(self) -> Dict:
        """Get classifier performance statistics"""
        if not self.processing_times:
            return {}
            
        processing_times = list(self.processing_times)
        predictions = list(self.prediction_history)
        
        # Calculate statistics
        stats = {
            'mean_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'predictions_per_second': 1.0 / self.update_interval,
            'buffer_fill': len(self.eeg_buffer) / self.window_samples,
            'total_predictions': len(predictions)
        }
        
        # Add prediction distribution if available
        if predictions:
            recent_predictions = [p['prediction'] for p in predictions[-20:] 
                                if p['prediction'] is not None]
            if recent_predictions:
                unique, counts = np.unique(recent_predictions, return_counts=True)
                stats['recent_distribution'] = dict(zip(unique, counts))
                
        return stats
    
    def simulate_eeg_stream(self, duration: float = 10.0):
        """
        Simulate EEG data stream for testing
        
        Args:
            duration: Simulation duration in seconds
        """
        print(f"Simulating EEG stream for {duration} seconds...")
        
        # Generate random EEG-like data
        n_channels = 16
        chunk_size = int(0.1 * self.sampling_rate)  # 100ms chunks
        
        start_time = time.time()
        while time.time() - start_time < duration:
            # Generate chunk
            chunk = np.random.randn(chunk_size, n_channels) * 10  # μV scale
            
            # Add to buffer
            self.add_eeg_data(chunk)
            
            # Wait to simulate real-time
            time.sleep(0.1)
            
            # Check for predictions
            prediction = self.get_prediction(timeout=0.01)
            if prediction:
                print(f"Prediction: {prediction['command']} "
                     f"(confidence: {prediction['confidence']:.2f})")
                
        print("Simulation complete")
        
        # Print statistics
        stats = self.get_performance_stats()
        print(f"\nPerformance Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


class MISimulatorInterface:
    """
    Interface between MI classifier and ErrP simulator
    """
    
    def __init__(self, classifier: RealtimeMIClassifier):
        """
        Initialize interface
        
        Args:
            classifier: Real-time MI classifier instance
        """
        self.classifier = classifier
        self.last_command = None
        self.command_buffer = deque(maxlen=5)
        
    def process_cue(self, cue_word: str) -> Tuple[str, float]:
        """
        Process MI cue and return command after classification
        
        Args:
            cue_word: 'LEFT' or 'RIGHT'
            
        Returns:
            Tuple of (command, confidence)
        """
        # Reset smoothing for new cue
        self.classifier.reset_smoothing()
        
        # Map cue to expected class
        cue_to_class = {'LEFT': 1, 'RIGHT': 2}
        expected_class = cue_to_class.get(cue_word)
        
        # Wait for classification
        max_wait = 3.0  # Maximum wait time
        start_time = time.time()
        
        valid_predictions = []
        rejected_count = 0
        
        while time.time() - start_time < max_wait:
            prediction = self.classifier.get_prediction(timeout=0.1)
            
            if prediction:
                if prediction.get('rejected', False):
                    rejected_count += 1
                    # Continue accumulating evidence
                    if prediction.get('reason') == 'insufficient_samples':
                        continue
                else:
                    # Valid prediction with sufficient confidence
                    valid_predictions.append(prediction)
                    
                    # Return first valid prediction (already smoothed)
                    return prediction['command'], prediction['confidence']
                    
        # Timeout - check if we have any predictions
        if rejected_count > 0:
            # Had predictions but all rejected for low confidence
            return 'REST', 0.0
        else:
            # No predictions at all
            return 'TIMEOUT', 0.0
            
    def inject_error(self, command: str, error_rate: float = 0.3) -> Tuple[str, bool]:
        """
        Inject errors to simulate MI decoder errors
        
        Args:
            command: Original command
            error_rate: Probability of error
            
        Returns:
            Tuple of (final_command, was_error_injected)
        """
        if np.random.random() < error_rate:
            # Flip command
            if command == 'LEFT':
                return 'RIGHT', True
            elif command == 'RIGHT':
                return 'LEFT', True
                
        return command, False


def main():
    """Example usage"""
    import os
    
    model_path = "/Users/flatbai/Desktop/BCI_Final_Year/ErrPs/MI/models/best_model_T005.pkl"
    
    if os.path.exists(model_path):
        # Initialize classifier
        classifier = RealtimeMIClassifier(
            model_path=model_path,
            sampling_rate=512,
            window_length=3.0,
            update_interval=0.1
        )
        
        # Start classifier
        classifier.start()
        
        # Simulate EEG stream
        classifier.simulate_eeg_stream(duration=10.0)
        
        # Stop classifier
        classifier.stop()
        
    else:
        print(f"Model not found at {model_path}")
        print("Please train a model first using train_classifier.py")


if __name__ == "__main__":
    main()