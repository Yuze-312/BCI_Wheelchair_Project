#!/usr/bin/env python
"""
EEG Model Integration - Main EEG processing with pretrained MI model
This is the MAIN file for real-time EEG processing during experiments

Supports two modes:
1. Continuous detection (default)
2. Cue-aware detection (only during game cues)
"""

import numpy as np
from pylsl import StreamInlet, resolve_streams
import time
from scipy import signal
import pickle
import os
import sys
from threading import Thread, Lock
from collections import deque
import random
import csv
from datetime import datetime

# Add path to MI module
sys.path.append(os.path.join(os.path.dirname(__file__), 'MI'))

class TrainedModelEEGProcessor:
    def _find_project_root(self):
        """Find project root by looking for marker files"""
        current = os.path.dirname(os.path.abspath(__file__))
        
        # Look for project markers going up the directory tree
        markers = ['README.md', '.git', 'requirements.txt', 'setup.py']
        
        while current != os.path.dirname(current):  # not at filesystem root
            # Check if this is the BCI_Wheelchair_Project directory
            if os.path.basename(current) == 'BCI_Wheelchair_Project':
                return current
                
            # Check for any marker files
            for marker in markers:
                if os.path.exists(os.path.join(current, marker)):
                    return current
                    
            # Go up one directory
            current = os.path.dirname(current)
            
        # If we can't find project root, use parent of current file's directory
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def _get_control_file_path(self):
        """Get the control file path relative to project root"""
        project_root = self._find_project_root()
        return os.path.join(project_root, 'simulation/eeg_cumulative_control.txt')
    
    def __init__(self, debug=False, phase='real', error_rate=0.2, wait_time=4.0):
        """Initialize with pretrained model
        
        Args:
            debug: If True, show detailed classification info
            phase: 'real' for continuous real-time classifier, 'phase1' for fake classifier with GT
            error_rate: Error injection rate for phase1 (default 20%)
            wait_time: Time to wait after cue for phase1 (default 4s)
        """
        # Continuous mode always responds to cues but also processes continuously
        # (cue_aware mode has been deprecated and merged into continuous)
        self.debug_mode = debug
        self.phase = phase
        self.error_rate = error_rate
        self.fake_wait_time = wait_time
        
        self.connect_to_streams()
        
        if self.phase == 'real':
            self.load_trained_model()
        else:
            print(f"Phase 1: Using fake classifier with {self.error_rate*100:.0f}% error rate")
            
        self.setup_processing()
        self.command_history = []
        
        # Phase 1 specific
        if self.phase == 'phase1':
            self.setup_event_logger()
            self.fake_stats = {
                'total_cues': 0,
                'correct_actions': 0,
                'errors_injected': 0,
                'cue_left': 0,
                'cue_right': 0,
                'action_left': 0,
                'action_right': 0
            }
        
        # Cue tracking (for cue-aware mode)
        self.cue_active = False
        self.cue_start_time = None
        self.cue_type = None  # 'left' or 'right'
        self.cue_window = 5.0  # seconds after cue to allow detection
        self.cue_lock = Lock()
        self.cue_processed = False  # For phase1 to ensure single processing
        
        # Game state
        self.game_active = False
        self.trial_active = False
        self.current_trial = 0
        
    def setup_event_logger(self):
        """Setup CSV event logger for phase1"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"logs/phase1_fake_classifier_{timestamp}.csv"
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # CSV headers
        self.csv_headers = [
            'timestamp', 'event_type', 'trial_id', 'cue_class', 
            'predicted_class', 'accuracy', 'error_injected', 
            'wait_time', 'confidence', 'details'
        ]
        
        # Create CSV file
        with open(self.log_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
            
        print(f"Event logger created: {self.log_filename}")
        
    def log_event(self, event_type, **kwargs):
        """Log an event to CSV"""
        if self.phase != 'phase1':
            return
            
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'trial_id': self.current_trial,
            'cue_class': kwargs.get('cue_class', ''),
            'predicted_class': kwargs.get('predicted_class', ''),
            'accuracy': kwargs.get('accuracy', ''),
            'error_injected': kwargs.get('error_injected', ''),
            'wait_time': kwargs.get('wait_time', ''),
            'confidence': kwargs.get('confidence', ''),
            'details': kwargs.get('details', '')
        }
        
        with open(self.log_filename, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writerow(event)
        
    def load_trained_model(self, custom_model_path=None):
        """Load the pretrained MI classifier
        
        Args:
            custom_model_path: Optional path to subject-specific model
        """
        if custom_model_path:
            model_path = custom_model_path
            print(f"Loading subject-specific model from {model_path}...")
        else:
            # Use T-005's model by default (best performance: 78.6%)
            model_path = 'MI/models/trained_models/subject_T-005_current.pkl'
            print(f"Loading T-005's model from {model_path}...")
            print("  Note: T-005 achieved 78.6% accuracy")
        
        if not os.path.exists(model_path):
            # Fallback to universal model if T-005's model not found
            fallback_path = 'MI/models/mi_improved_classifier.pkl'
            if os.path.exists(fallback_path):
                print(f"  T-005 model not found, using universal model instead")
                model_path = fallback_path
            else:
                raise FileNotFoundError(f"Model not found at {model_path} or {fallback_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            
        # Extract components (handle both model structures)
        self.classifier = model_data['model']  # LDA classifier
        self.csp = model_data['csp']          # CSP spatial filters
        self.scaler = model_data.get('scaler', None)  # Feature scaler if exists
        
        # Model info
        self.fs = model_data.get('fs', 512)    # Sampling rate used for training
        self.selected_channels = model_data.get('selected_channels', None)
        
        # Get accuracy from results if available
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
        
        # For T-005's model, show participant info
        if 'participant_id' in model_data:
            print(f"  Trained on: {model_data['participant_id']}")
        
    def connect_to_streams(self):
        """Connect to EEG and optionally marker streams"""
        print("Connecting to streams...")
        streams = resolve_streams()
        
        # Phase1 only needs markers, not EEG
        if self.phase == 'phase1':
            # Set dummy values for phase1 since we don't use EEG
            self.n_channels = 16  # Dummy value
            self.srate = 512  # Dummy sampling rate
            
            # Find marker stream only
            marker_streams = [s for s in streams if s.type() == 'Markers' and 'Outlet_Info' in s.name()]
            if marker_streams:
                self.marker_inlet = StreamInlet(marker_streams[0])
                print(f"Connected to game markers: {marker_streams[0].name()}")
                self.has_markers = True
            else:
                print("WARNING: No game marker stream found")
                print("   Waiting for game to start...")
                self.marker_inlet = None
                self.has_markers = False
            return
        
        # Find EEG stream for real mode
        eeg_streams = [s for s in streams if s.type() == 'EEG' or 'eeg' in s.name().lower()]
        if not eeg_streams:
            raise RuntimeError("No EEG streams found!")
        
        # Look for the working stream (obci_eeg1) first
        selected_eeg = None
        for stream in eeg_streams:
            if stream.name() == 'obci_eeg1':
                selected_eeg = stream
                print(f"Found preferred stream: {stream.name()}")
                break
        
        if not selected_eeg:
            # Test which stream has data
            print("Testing EEG streams for data...")
            working_stream = None
            
            for i, stream_info in enumerate(eeg_streams):
                print(f"  Testing {stream_info.name()}...", end='', flush=True)
                test_inlet = StreamInlet(stream_info)
                
                # Quick test pull
                chunk, _ = test_inlet.pull_chunk(timeout=0.5, max_samples=10)
                if chunk:
                    print(f" Has data! ({len(chunk)} samples)")
                    working_stream = stream_info
                    break
                else:
                    print(" No data")
            
            if working_stream:
                selected_eeg = working_stream
                print(f"\nUsing working stream: {selected_eeg.name()}")
            else:
                # Prefer 16-channel as fallback
                streams_16ch = [s for s in eeg_streams if s.channel_count() == 16]
                selected_eeg = streams_16ch[0] if streams_16ch else eeg_streams[0]
                print(f"\nWarning: No streams have data! Using {selected_eeg.name()} anyway")
        
        print(f"Stream info: {selected_eeg.channel_count()} channels at {selected_eeg.nominal_srate()}Hz")
        
        self.eeg_inlet = StreamInlet(selected_eeg)
        self.n_channels = selected_eeg.channel_count()
        self.srate = selected_eeg.nominal_srate()
        
        print(f"Connected to EEG: {selected_eeg.name()}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sampling rate: {self.srate}Hz")
        
        # Find marker stream for BOTH modes (needed for cue detection)
        marker_streams = [s for s in streams if s.type() == 'Markers' and 'Outlet_Info' in s.name()]
        if marker_streams:
            self.marker_inlet = StreamInlet(marker_streams[0])
            print(f"Connected to game markers: {marker_streams[0].name()}")
            self.has_markers = True
        else:
            print("  No game marker stream found")
            print("   Waiting for game to start...")
            self.marker_inlet = None
            self.has_markers = False
        
    def setup_processing(self):
        """Setup processing pipeline"""
        # MI frequency band (8-30 Hz)
        nyquist = self.srate / 2
        if nyquist > 30:
            self.b, self.a = signal.butter(5, [8/nyquist, 30/nyquist], btype='band')
        else:
            self.b, self.a = None, None
            
        # Phase1 doesn't need EEG buffers
        if self.phase == 'phase1':
            self.min_detection_interval = 0.5
            self.last_detection = 0
            self.confidence_threshold = 0.7  # For display consistency
        else:
            # Buffer for epoch extraction - Updated for 5s MI window
            self.epoch_length = 2.0  # seconds (CSP requirement)
            # 5 seconds buffer to match max cue duration
            self.buffer_size = int(self.srate * 5)  # 5 seconds total
            self.buffer = []
            
            # Detection parameters
            self.min_detection_interval = 1.0  # Process every 1s in continuous mode
            self.last_detection = 0
            self.confidence_threshold = 0.3  # Standard threshold
            
            # Sliding window parameters for real-time processing
            self.sliding_window_interval = 0.5  # Process every 500ms
            self.last_sliding_window_time = 0
            self.use_recent_data = True  # Use most recent 2s instead of oldest
        
        # MI-specific timing for 5s window (matches MAX_THINK_TIME in simulator)
        self.post_cue_delay = 0.0  # Start immediately
        self.mi_window_end = 5.0   # End at 5s to match max cue duration
        self.cue_detection_start = None
        
        # Stats
        self.detection_stats = {'left': 0, 'right': 0, 'cue_left': 0, 'cue_right': 0}
        
        # Data collection starts when game is detected
        self.data_collection_active = False  # Will be set True when game markers detected
        
        # Control file path - must match where control_interface_cumulative.py looks
        self.control_file_path = self._get_control_file_path()
        
        # Voting mode flag - initialize for ALL phases
        self.voting_in_progress = False
        self.command_sent_for_current_cue = False
        
    def preprocess_epoch(self, epoch_data):
        """Preprocess data to match training format"""
        # epoch_data shape: (samples, channels)
        
        # 1. Select channels if specified
        if self.selected_channels is not None:
            # Map channel indices
            available_channels = min(self.n_channels, len(self.selected_channels))
            epoch_data = epoch_data[:, :available_channels]
        
        # 2. Resample if necessary
        if self.srate != self.fs:
            # Resample to match training sample rate
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
        # We have single trial, so shape is (1, channels, samples)
        formatted = filtered.T[np.newaxis, :, :]  # (1, channels, samples)
        
        # 5. Pad channels if needed (CSP was trained with 16 channels)
        if hasattr(self.csp, 'patterns_') and self.csp.patterns_.shape[0] > formatted.shape[1]:
            expected_channels = self.csp.patterns_.shape[0]
            current_channels = formatted.shape[1]
            if current_channels < expected_channels:
                # Pad with zeros for missing channels
                padding = np.zeros((formatted.shape[0], expected_channels - current_channels, formatted.shape[2]))
                formatted = np.concatenate([formatted, padding], axis=1)
                print(f"[DEBUG] Padded from {current_channels} to {expected_channels} channels")
        
        return formatted
    
    def check_signal_quality(self, epoch_data):
        """Check if signal quality is acceptable"""
        # Check for flat channels (no signal)
        channel_vars = np.var(epoch_data, axis=0)
        flat_channels = np.sum(channel_vars < 0.1)
        
        # Check for excessive amplitude (poor contact or artifacts)
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
        is_good = (flat_channels < self.n_channels * 0.3 and  # Less than 30% flat
                  max_amp < 200 and  # Not saturated
                  signal_power > 0.5)  # Has some signal
                  
        return is_good, quality_info
    
    def make_fake_decision(self):
        """Make a decision based on GT with error injection for phase1"""
        # Get ground truth
        with self.cue_lock:
            if not self.cue_active or self.cue_processed:
                return None
            
            gt_class = self.cue_type
            self.cue_processed = True
        
        # Decide if we inject an error
        inject_error = random.random() < self.error_rate
        
        if inject_error:
            # Flip the decision
            predicted_class = 'right' if gt_class == 'left' else 'left'
            self.fake_stats['errors_injected'] += 1
            accuracy = False
        else:
            # Follow ground truth
            predicted_class = gt_class
            self.fake_stats['correct_actions'] += 1
            accuracy = True
        
        # Update action stats
        self.fake_stats[f'action_{predicted_class}'] += 1
        
        # Generate fake confidence
        if inject_error:
            # Lower confidence for errors
            confidence = random.uniform(0.55, 0.75)
        else:
            # Higher confidence for correct
            confidence = random.uniform(0.75, 0.95)
        
        # Log the decision
        self.log_event(
            'fake_classifier_decision',
            cue_class=gt_class,
            predicted_class=predicted_class,
            accuracy=accuracy,
            error_injected=inject_error,
            wait_time=self.fake_wait_time,
            confidence=confidence,
            details=f"GT: {gt_class}, Predicted: {predicted_class}, Error: {inject_error}"
        )
        
        # Log feedback event (for ErrP analysis)
        if inject_error:
            self.log_event('feedback_error', error_injected=True)
            # Log ErrP event 300ms later
            Thread(target=self._delayed_errp_log, args=(0.3,), daemon=True).start()
        else:
            self.log_event('feedback_correct')
        
        return predicted_class, confidence, inject_error
    
    def _delayed_errp_log(self, delay):
        """Log ErrP event after delay"""
        time.sleep(delay)
        self.log_event('primary_errp', details='Fake classifier error - ErrP expected')
    
    def process_voting_window(self):
        """Process 4-second window with two classifiers voting"""
        if self.phase != 'real':
            return
            
        cue_start = self.cue_start_time
        cue_type = self.cue_type
        
        print(f"[VOTING] Collecting 4s of data, will decide with 1s remaining...")
        
        # Wait for 4 seconds of data before making decision
        while time.time() - cue_start < 4.0:
            elapsed = time.time() - cue_start
            buffer_seconds = len(self.buffer) / self.srate if self.srate > 0 else 0
            time_remaining = 5.0 - elapsed  # 5s total window
            
            # Show progress every second
            if int(elapsed) > int(elapsed - 0.1):
                print(f"   [VOTING] {elapsed:.1f}s elapsed, {time_remaining:.1f}s remaining, buffer: {buffer_seconds:.1f}s")
            
            time.sleep(0.1)
        
        # At 4s mark, we have ~2048 samples (4s * 512Hz)
        # Ensure we have enough data
        available_samples = len(self.buffer)
        
        if available_samples < int(4 * self.srate):
            print(f"WARNING: [VOTING] Insufficient data: {available_samples} samples (need {int(4*self.srate)})")
            with self.cue_lock:
                self.voting_in_progress = False
                self.cue_active = False  # Clear cue state on error
            return
        
        # Extract two 2-second windows from the 4s of data
        window1_samples = self.buffer[:int(2 * self.srate)]  # 0-2s
        window2_samples = self.buffer[int(2 * self.srate):int(4 * self.srate)]  # 2-4s
        
        print(f"\n[VOTING] Processing two windows (1s remaining to act):")
        print(f"   Window 1: 0-2s ({len(window1_samples)} samples)")
        print(f"   Window 2: 2-4s ({len(window2_samples)} samples)")
        
        # Convert to numpy arrays
        window1 = np.array(window1_samples)
        window2 = np.array(window2_samples)
        
        # Get predictions from both windows
        class1, conf1 = self.classify_mi(window1)
        class2, conf2 = self.classify_mi(window2)
        
        # Print results in the requested format
        current_time = time.strftime('%H:%M:%S')
        print(f"\n[VOTING] Results:")
        if class1:
            print(f"[{current_time}] Classifier 1: {class1.upper()} (conf: {conf1:.2f})")
        else:
            print(f"[{current_time}] Classifier 1: None (conf: {conf1:.2f})")
        
        if class2:
            print(f"[{current_time}] Classifier 2: {class2.upper()} (conf: {conf2:.2f})")
        else:
            print(f"[{current_time}] Classifier 2: None (conf: {conf2:.2f})")
        
        # Voting logic
        final_class = None
        final_conf = 0
        
        if class1 and class2:
            if class1 == class2:
                # Agreement - average confidence
                final_class = class1
                final_conf = (conf1 + conf2) / 2
                print(f"   AGREEMENT: Both voted {final_class.upper()}")
            else:
                # Disagreement - use higher confidence
                if conf1 > conf2:
                    final_class = class1
                    final_conf = conf1
                    print(f"   DISAGREEMENT: Using Classifier 1 ({final_class.upper()}, conf: {final_conf:.2f})")
                else:
                    final_class = class2
                    final_conf = conf2
                    print(f"   DISAGREEMENT: Using Classifier 2 ({final_class.upper()}, conf: {final_conf:.2f})")
        elif class1:
            final_class = class1
            final_conf = conf1
            print(f"   WARNING: Only Classifier 1 detected: {final_class.upper()}")
        elif class2:
            final_class = class2
            final_conf = conf2
            print(f"   WARNING: Only Classifier 2 detected: {final_class.upper()}")
        else:
            print(f"   NO DETECTION from either classifier")
        
        # Send final decision
        if final_class:
            # Verify we've actually waited 4 seconds
            elapsed_since_cue = time.time() - cue_start
            if elapsed_since_cue < 3.9:  # Small buffer for timing precision
                print(f"[TIMING ERROR] Only {elapsed_since_cue:.1f}s elapsed, expected 4s!")
                time.sleep(4.0 - elapsed_since_cue)
            
            # Check if it matches the cue
            match_str = "MATCH" if final_class == cue_type else "MISMATCH"
            print(f"\n[FINAL DECISION] {final_class.upper()} (conf: {final_conf:.2f}) {match_str}")
            
            # Send ONE command per cue
            if not self.command_sent_for_current_cue:
                command = '1' if final_class == 'left' else '2'
                self.command_history.append(command)
                
                # Write to file
                try:
                    with open(self.control_file_path, 'w') as f:
                        f.write(', '.join(self.command_history))
                    print(f"   Command sent: {command} with 1s remaining (Total: {len(self.command_history)})")
                    self.command_sent_for_current_cue = True
                except Exception as e:
                    print(f"   ERROR: Failed to write command: {e}")
                
                # Update stats
                self.detection_stats[final_class] += 1
            else:
                print(f"   WARNING: Command already sent for this cue - ignoring")
        else:
            print(f"\n[NO DECISION] Confidence too low for both classifiers")
        
        # Reset voting flag after voting completes (at 4s mark)
        # Wait for the remaining 1s before resetting to ensure no interference
        time.sleep(1.0)  # Wait the final second
        
        # Reset both flags to ensure clean state
        with self.cue_lock:
            self.voting_in_progress = False
            self.cue_active = False  # Clear cue state after voting completes
        
        print(f"   [VOTING] Voting window complete - resuming continuous mode")
        print("─" * 60)
    
    def classify_mi(self, epoch_data):
        """Classify using pretrained model with debug info"""
        try:
            # Check signal quality first
            quality_ok, quality_info = self.check_signal_quality(epoch_data)
            
            if not quality_ok and hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"\nWARNING: Poor signal quality detected:")
                print(f"   Flat channels: {quality_info['flat_channels']}/{self.n_channels}")
                print(f"   Max amplitude: {quality_info['max_amplitude']:.1f}μV")
                print(f"   Signal power: {quality_info['signal_power']:.2f}")
            
            # Preprocess
            processed = self.preprocess_epoch(epoch_data)
            print(f"[DEBUG] Epoch shape: {epoch_data.shape}, Processed shape: {processed.shape}")
            
            # Apply CSP
            try:
                csp_features = self.csp.transform(processed)  # (1, n_components)
                print(f"[DEBUG] CSP features shape: {csp_features.shape}")
            except Exception as e:
                print(f"[DEBUG] CSP error: {e}")
                print(f"[DEBUG] CSP expects shape: (n_trials, n_channels, n_samples)")
                print(f"[DEBUG] CSP n_components: {self.csp.n_components}")
                raise
            
            # Debug: Show CSP features
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"\n[DEBUG] CSP Features: {csp_features[0]}")
                print(f"   Feature range: [{np.min(csp_features):.3f}, {np.max(csp_features):.3f}]")
            
            # Scale features if scaler exists
            if self.scaler is not None:
                csp_features_scaled = self.scaler.transform(csp_features)
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"   Scaled range: [{np.min(csp_features_scaled):.3f}, {np.max(csp_features_scaled):.3f}]")
                csp_features = csp_features_scaled
            
            # Classify
            prediction = self.classifier.predict(csp_features)
            probabilities = self.classifier.predict_proba(csp_features)
            
            # Get confidence - use margin between classes as confidence
            # For binary classification: how sure are we about the prediction?
            prob_diff = abs(probabilities[0][0] - probabilities[0][1])
            
            # Also check if the features look like ANY MI activity
            # Low CSP feature variance might indicate no MI
            feature_magnitude = np.linalg.norm(csp_features)
            typical_magnitude = 1.0  # This should be calibrated from training data
            
            # Combine margin and magnitude for better confidence
            confidence = prob_diff * min(1.0, feature_magnitude / typical_magnitude)
            
            # Boost confidence by adding 0.3 to compensate for conservative model
            confidence = min(1.0, confidence + 0.3)  # Cap at 1.0
            
            # Debug: Show classification details
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"\n Classification:")
                print(f"   Prediction: {prediction[0]} ({'LEFT' if prediction[0] == 0 else 'RIGHT'})")
                print(f"   Probabilities: LEFT={probabilities[0][0]:.3f}, RIGHT={probabilities[0][1]:.3f}")
                print(f"   Confidence: {confidence:.3f} (threshold: {self.confidence_threshold})")
            
            # Return result - use active threshold if available
            threshold = getattr(self, 'active_threshold', self.confidence_threshold)
            if confidence >= threshold:
                # Class 0 = Left, Class 1 = Right (based on typical MI paradigm)
                mi_class = 'left' if prediction[0] == 0 else 'right'
                return mi_class, confidence
            else:
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    print(f"    Below threshold - no detection")
                return None, confidence
                
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0
    
    def monitor_markers(self):
        """Monitor game markers in separate thread (cue-aware mode)"""
        if not self.has_markers:
            return
            
        print("Monitoring game markers...")
        print(f"   Marker inlet: {self.marker_inlet}")
        print(f"   Has markers: {self.has_markers}")
        marker_count = 0
        
        while True:
            try:
                # Pull markers
                markers, timestamps = self.marker_inlet.pull_chunk(timeout=0.0)
                
                for marker in markers:
                    marker_type = marker[0]
                    marker_count += 1
                    
                    # Debug: show all markers
                    print(f"  [MARKER DEBUG] Received: {marker_type} (count: {marker_count})")
                    
                    # Game state - any marker indicates game is active
                    if not self.game_active:
                        self.game_active = True
                        self.data_collection_active = True  # Enable continuous collection
                        print(f"Game activity detected! (marker: {marker_type})")
                        print("   Starting continuous data collection...")
                    
                    # Marker types: 1 = trial separator, 2/3 = cues, 5/6 = responses
                    if marker_type == 1:  # Trial separator
                        # Don't reset voting flag here - let the voting thread handle it
                        if self.phase == 'real' and hasattr(self, 'voting_in_progress') and self.voting_in_progress:
                            print("   [VOTING] Trial separator received - voting thread will complete soon")
                        
                    # Response markers - 5=CORRECT, 6=ERROR
                    elif marker_type == 5 or marker_type == 6:
                        # These indicate user responses
                        if self.phase == 'phase1':
                            with self.cue_lock:
                                if marker_type == 6:  # Error response
                                    self.fake_stats['primary_errors'] += 1
                                self.cue_active = False
                                self.cue_processed = True
                        
                    # Cue tracking (most important!) - numeric markers: 2=LEFT, 3=RIGHT
                    elif marker_type == 2 or marker_type == 3:
                        with self.cue_lock:
                            self.cue_active = True
                            self.cue_start_time = time.time()
                            self.cue_type = 'left' if marker_type == 2 else 'right'
                            self.cue_processed = False  # Reset for new cue
                            self.command_sent_for_current_cue = False  # Reset command flag
                            
                            if self.phase == 'phase1':
                                self.fake_stats['total_cues'] += 1
                                self.fake_stats[f'cue_{self.cue_type}'] += 1
                                self.log_event(f'cue_{self.cue_type}', cue_class=self.cue_type)
                            else:
                                self.detection_stats[f'cue_{self.cue_type}'] += 1
                                self.cue_detection_start = self.cue_start_time + self.post_cue_delay
                                # CRITICAL: Set voting flag FIRST to prevent race condition
                                self.voting_in_progress = True
                                # Clear buffer and start fresh data collection
                                self.buffer = []
                                self.data_collection_active = True
                                # Start voting window thread for real mode
                                vote_thread = Thread(target=self.process_voting_window, daemon=True)
                                vote_thread.start()
                                print(f"   [VOTING] Pausing continuous processing for 4s voting window...")
                            
                        if self.phase == 'phase1':
                            print(f"\n[PHASE 1 - FAKE] CUE: {self.cue_type.upper()} - Will respond in {self.fake_wait_time}s")
                        else:
                            print(f"\n[VOTING MODE] CUE: {self.cue_type.upper()} - Collecting 4s, deciding at t=4s (1s left to act)")
                        
                    # Cue end - marker 4 (but we don't use this anymore)
                    elif marker_type == 4:
                        with self.cue_lock:
                            if self.phase == 'phase1':
                                # Always clear for phase1 to avoid double processing
                                self.cue_active = False
                                self.cue_processed = True
                            else:
                                # Real mode: voting thread handles clearing cue_active
                                # Just stop data collection
                                self.data_collection_active = False
                        if self.phase == 'phase1':
                            print("  Phase1: Cue window CLOSED")
                        else:
                            print("  Cue end marker received - voting thread will handle cleanup")
                        
                    # Handle cue timeout for phase1
                    elif marker_type == 7 and self.phase == 'phase1':  # NO_RESPONSE/timeout
                        with self.cue_lock:
                            self.cue_active = False
                            self.cue_processed = True
                        print("  Phase1: Cue timed out - marked as processed")
                        
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Marker error: {e}")
                time.sleep(0.1)
    
    def print_phase1_stats(self):
        """Print session statistics for phase1"""
        if self.phase != 'phase1':
            return
            
        print("\nPhase 1 Session Statistics:")
        print(f"Total cues: {self.fake_stats['total_cues']}")
        print(f"  Left cues: {self.fake_stats['cue_left']}")
        print(f"  Right cues: {self.fake_stats['cue_right']}")
        print(f"Correct actions: {self.fake_stats['correct_actions']} ({self.fake_stats['correct_actions']/max(1,self.fake_stats['total_cues'])*100:.1f}%)")
        print(f"Errors injected: {self.fake_stats['errors_injected']} ({self.fake_stats['errors_injected']/max(1,self.fake_stats['total_cues'])*100:.1f}%)")
        print(f"Actions taken:")
        print(f"  Left: {self.fake_stats['action_left']}")
        print(f"  Right: {self.fake_stats['action_right']}")
    
    def run(self):
        """Main processing loop"""
        if self.phase == 'phase1':
            print(f"\nEEG Processing - PHASE 1: Fake Classifier")
            print(f"Error injection rate: {self.error_rate*100:.0f}%")
            print(f"Wait time after cue: {self.fake_wait_time}s")
            print(f"Following ground truth from game cues\n")
        else:
            print(f"\n🧠 EEG Processing with Pretrained Model [Continuous Mode]")
            print(f"Confidence threshold: {self.confidence_threshold}")
            print(f"Processing continuously, enhanced during cues")
        
        # Add note about expected error rate with T-005's model
        if hasattr(self, 'classifier'):
            participant_id = getattr(self, 'participant_id', 'Unknown')
            if participant_id == 'T-005' or 'T-005' in str(getattr(self, 'model_path', '')):
                print(f"Using T-005's model - expect ~20-35% natural error rate")
                print(f"   (Good for ErrP elicitation without artificial errors)")
        
        print("Press Ctrl+C to stop\n")
        
        # Start marker monitoring for BOTH modes (we need cues even in continuous mode)
        if self.has_markers:
            marker_thread = Thread(target=self.monitor_markers, daemon=True)
            marker_thread.start()
            print("Marker monitoring started")
        
        # Load existing commands from file to stay synchronized
        try:
            with open(self.control_file_path, 'r') as f:
                content = f.read().strip()
                if content:
                    self.command_history = content.split(', ')
                    print(f"Loaded {len(self.command_history)} existing commands")
                else:
                    self.command_history = []
        except:
            self.command_history = []
        
        # Option to clear history
        if input("\nClear command history? (y/N): ").lower() == 'y':
            self.command_history = []
            with open(self.control_file_path, 'w') as f:
                f.write('')
            print("History cleared")
        
        # Continuous mode status
        if self.has_markers:
            print("\nGame connected - Continuous detection active")
        else:
            print("\nWaiting for game to start...")
            print("   Will begin processing once game markers are detected")
        print()
        
        # Add buffer monitoring
        last_buffer_log = time.time()
        
        while True:
            try:
                # Check for game connection if no markers yet
                if not self.has_markers:
                    if not hasattr(self, '_last_reconnect_attempt'):
                        self._last_reconnect_attempt = 0
                    
                    if time.time() - self._last_reconnect_attempt > 2.0:  # Check every 2 seconds
                        self._last_reconnect_attempt = time.time()
                        
                        # Try to find marker stream
                        try:
                            streams = resolve_streams()
                            marker_streams = [s for s in streams if s.type() == 'Markers' and 'SubwaySurfers_ErrP' in s.name()]
                            if marker_streams:
                                self.marker_inlet = StreamInlet(marker_streams[0])
                                print(f"\nGame started! Connected to markers: {marker_streams[0].name()}")
                                self.has_markers = True
                                # Start marker monitoring thread
                                marker_thread = Thread(target=self.monitor_markers, daemon=True)
                                marker_thread.start()
                                print(" Monitoring game markers...")
                                print("Waiting for cues...\n")
                        except Exception as e:
                            pass  # Silent retry
                    
                    # Show waiting message periodically
                    if not hasattr(self, '_last_waiting_msg'):
                        self._last_waiting_msg = 0
                    if time.time() - self._last_waiting_msg > 10.0:
                        self._last_waiting_msg = time.time()
                        print(f"[{time.strftime('%H:%M:%S')}] Still waiting for game to start...")
                
                # Phase 1: Check if we need to make a fake decision
                if self.phase == 'phase1':
                    with self.cue_lock:
                        # Check if we should process (after wait time OR if cue is about to timeout)
                        if self.cue_active and not self.cue_processed and self.cue_start_time:
                            elapsed = time.time() - self.cue_start_time
                            # Process after wait time OR just before cue timeout (4.9s to avoid timeout)
                            should_process = elapsed >= self.fake_wait_time or elapsed >= 4.9
                        else:
                            should_process = False
                    
                    if should_process:
                        # Make fake decision
                        result = self.make_fake_decision()
                        
                        if result:
                            predicted_class, confidence, error_injected = result
                            
                            # Add command
                            command = '1' if predicted_class == 'left' else '2'
                            self.command_history.append(command)
                            
                            # Write to file
                            with open(self.control_file_path, 'w') as f:
                                f.write(', '.join(self.command_history))
                            
                            # Display
                            error_marker = "ERROR INJECTED" if error_injected else "OK"
                            print(f"[{time.strftime('%H:%M:%S')}] FAKE MI: {predicted_class.upper()} (conf: {confidence:.2f}) {error_marker}")
                            print(f"  Commands: {', '.join(self.command_history[-5:])}")
                            
                            if self.debug_mode:
                                print(f"  GT was: {self.cue_type.upper()}")
                                print(f"  Total errors: {self.fake_stats['errors_injected']}/{self.fake_stats['total_cues']} ({self.fake_stats['errors_injected']/max(1,self.fake_stats['total_cues'])*100:.1f}%)")
                            
                            self.last_detection = time.time()
                    
                    time.sleep(0.01)
                    continue
                
                # Real mode: Pull EEG data
                chunk, timestamps = self.eeg_inlet.pull_chunk(timeout=0.0)
                
                # Debug data reception (only in debug mode)
                if self.debug_mode:
                    if not hasattr(self, '_chunk_debug_time'):
                        self._chunk_debug_time = time.time()
                        self._chunk_count = 0
                        self._sample_count = 0
                    
                    if time.time() - self._chunk_debug_time > 1.0:
                        print(f"\n[Data Reception] Past 1s: {self._chunk_count} chunks, {self._sample_count} samples")
                        if self._chunk_count == 0:
                            print("   ERROR: NO DATA RECEIVED FROM EEG STREAM!")
                        self._chunk_debug_time = time.time()
                        self._chunk_count = 0
                        self._sample_count = 0
                
                if chunk:
                    if self.debug_mode:
                        self._chunk_count += 1
                        self._sample_count += len(chunk)
                    # Only add to buffer if data collection is active
                    if getattr(self, 'data_collection_active', False):
                        self.buffer.extend(chunk)
                        
                        # Mark first buffer fill
                        if not hasattr(self, '_first_buffer_fill'):
                            self._first_buffer_fill = True
                            if self.debug_mode:
                                print(f"\nFirst data added to buffer: {len(chunk)} samples")
                        
                        # Keep buffer size limited
                        if len(self.buffer) > self.buffer_size:
                            self.buffer = self.buffer[-self.buffer_size:]
                
                # Buffer status logging every 5 seconds (only in debug mode)
                current_time = time.time()
                if self.debug_mode and current_time - last_buffer_log > 5.0:
                    buffer_seconds = len(self.buffer) / self.srate if self.srate > 0 else 0
                    collection_active = getattr(self, 'data_collection_active', False)
                    
                    print(f"\n[Buffer Status] {time.strftime('%H:%M:%S')} - {len(self.buffer)} samples ({buffer_seconds:.1f}s), {(len(self.buffer) / self.buffer_size * 100):.0f}% full")
                    if not collection_active and (not self.has_markers or not self.game_active):
                        print("   Waiting for game...")
                    
                    last_buffer_log = current_time
                
                # Check if we should process (moved OUTSIDE the buffer logging block)
                # Continuous mode - process if game is connected and data collection is active
                # BUT skip if voting is in progress
                if self.voting_in_progress:
                    should_process = False  # Block all processing during voting
                    if self.debug_mode and hasattr(self, '_last_voting_block') and time.time() - self._last_voting_block > 1.0:
                        print(f"   [DEBUG] Blocking continuous processing - voting in progress")
                        self._last_voting_block = time.time()
                elif self.has_markers and self.game_active and self.data_collection_active:
                    should_process = True
                    
                    # Adjust confidence threshold based on cue state
                    if hasattr(self, 'cue_active') and self.cue_active:
                        # During cue: use normal threshold
                        self.active_threshold = self.confidence_threshold
                    else:
                        # No cue: use higher threshold to reduce false positives
                        self.active_threshold = min(0.8, self.confidence_threshold * 1.5)
                else:
                    # No game connection or collection not active - don't process
                    should_process = False
                
                # Sliding window approach: process every sliding_window_interval
                current_time = time.time()
                sliding_window_ready = current_time - self.last_sliding_window_time >= self.sliding_window_interval
                # Need at least 2 seconds for one epoch
                data_ok = len(self.buffer) >= int(self.srate * 2.0)
                
                # Debug: Show processing conditions
                if self.debug_mode and sliding_window_ready:
                    print(f"\n[PROCESS CHECK] should_process={should_process}, sliding_ready={sliding_window_ready}, data_ok={data_ok}")
                    print(f"   Buffer size: {len(self.buffer)} samples, need: {int(self.srate * 2.0)}")
                    print(f"   Markers: {self.has_markers}, Game: {self.game_active}, Collection: {self.data_collection_active}")
                
                # Skip processing if voting is in progress (already handled above)
                
                # Process using sliding window for continuous real-time detection
                if should_process and sliding_window_ready and data_ok:
                        # Update sliding window timer
                        self.last_sliding_window_time = current_time
                        
                        # Extract epoch samples
                        epoch_samples = int(self.srate * self.epoch_length)  # 2 seconds
                        
                        if self.use_recent_data and len(self.buffer) >= epoch_samples:
                            # Use most recent 2s of data for real-time responsiveness
                            epoch_data = np.array(self.buffer[-epoch_samples:])
                            
                            # Single window approach for sliding window
                            mi_class, confidence = self.classify_mi(epoch_data)
                            
                            # Log sliding window info
                            if self.debug_mode:
                                buffer_time = len(self.buffer) / self.srate
                                print(f"\n[Sliding Window Analysis] (buffer: {buffer_time:.1f}s)")
                                print(f"   Using most recent 2s of data")
                                print(f"   Result: {mi_class or 'None'} (conf: {confidence:.3f})")
                        else:
                            # Fallback: dual-window approach with overlap
                            if len(self.buffer) >= int(self.srate * 3.0):
                                offset_samples = int(self.srate * 1.0)  # 1s offset
                                
                                # For recent data: use the last 3s of buffer
                                if self.use_recent_data:
                                    # Late window: last 2s
                                    late_epoch = np.array(self.buffer[-epoch_samples:])
                                    # Early window: 2-4s from end
                                    early_epoch = np.array(self.buffer[-(epoch_samples + offset_samples):-offset_samples])
                                else:
                                    # Original approach: use oldest data
                                    early_epoch = np.array(self.buffer[:epoch_samples])
                                    late_epoch = np.array(self.buffer[offset_samples:offset_samples + epoch_samples])
                                
                                # Classify both windows
                                print(f"\n[CLASSIFICATION] Processing windows...")
                                print(f"   Early window shape: {early_epoch.shape}")
                                print(f"   Late window shape: {late_epoch.shape}")
                                early_class, early_conf = self.classify_mi(early_epoch)
                                late_class, late_conf = self.classify_mi(late_epoch)
                                print(f"   Results: early={early_class} ({early_conf:.3f}), late={late_class} ({late_conf:.3f})")
                                
                                # Use higher confidence result
                                if early_conf >= late_conf and early_class is not None:
                                    mi_class, confidence = early_class, early_conf
                                    window_used = "early"
                                elif late_class is not None:
                                    mi_class, confidence = late_class, late_conf
                                    window_used = "late"
                                else:
                                    mi_class = None
                                    confidence = max(early_conf, late_conf)
                                    window_used = "none"
                                    
                                if self.debug_mode:
                                    print(f"\n[Dual-window analysis] (buffer: {len(self.buffer)/self.srate:.1f}s)")
                                    print(f"   Early: {early_class or 'None'} (conf: {early_conf:.3f})")
                                    print(f"   Late: {late_class or 'None'} (conf: {late_conf:.3f})")
                                    print(f"   → Using {window_used.upper()} window")
                            else:
                                # Not enough data yet
                                continue
                        
                        
                        # Debug: Check for bias
                        if not hasattr(self, '_prediction_counts'):
                            self._prediction_counts = {'left': 0, 'right': 0, 'none': 0}
                        
                        if mi_class:
                            self._prediction_counts[mi_class] += 1
                            total = self._prediction_counts['left'] + self._prediction_counts['right']
                            if total > 0 and total % 10 == 0:  # Every 10 predictions
                                left_pct = self._prediction_counts['left'] / total * 100
                                right_pct = self._prediction_counts['right'] / total * 100
                                print(f"\n[Prediction bias check] (n={total}):")
                                print(f"   LEFT: {self._prediction_counts['left']} ({left_pct:.1f}%)")
                                print(f"   RIGHT: {self._prediction_counts['right']} ({right_pct:.1f}%)")
                                if left_pct > 80 or right_pct > 80:
                                    print("   WARNING: Strong bias detected!")
                        
                        if mi_class:
                            # CRITICAL: Never send commands during cue windows
                            if self.cue_active:
                                print(f"   [BLOCKED] MI detected during cue window - voting mode will handle this")
                                continue  # Skip entirely during cue windows
                            
                            # For sliding window: only send command if it's a new detection or high confidence
                            time_since_last = current_time - self.last_detection
                            
                            # Send command if: first detection OR enough time passed OR very high confidence
                            should_send_command = (
                                time_since_last >= self.min_detection_interval or
                                confidence >= 0.85  # High confidence override
                            )
                            
                            if should_send_command:
                                # FINAL CHECK: Absolutely no commands during voting
                                if self.voting_in_progress or self.cue_active:
                                    print(f"   [SAFETY] Command blocked - voting/cue active")
                                    continue
                                
                                # Add command
                                command = '1' if mi_class == 'left' else '2'
                                self.command_history.append(command)
                                
                                # Write to file
                                try:
                                    with open(self.control_file_path, 'w') as f:
                                        f.write(', '.join(self.command_history))
                                    # Show when command is written
                                    print(f"   [CONTINUOUS] Command written (no cue active)")
                                    if self.debug_mode:
                                        print(f"   [FILE] Written: {', '.join(self.command_history)}")
                                except Exception as e:
                                    print(f"   [ERROR] Failed to write control file: {e}")
                                
                                # Update stats
                                self.detection_stats[mi_class] += 1
                                
                                # Display with context
                                cue_info = ""
                                if self.cue_active:
                                    cue_elapsed = current_time - self.cue_start_time
                                    cue_info = f" (CUE: {self.cue_type.upper()}, t={cue_elapsed:.1f}s)"
                                    if mi_class != self.cue_type:
                                        cue_info += "  MISMATCH!"
                                
                                window_info = getattr(self, 'window_used', 'single')
                                print(f"[{time.strftime('%H:%M:%S')}] MI: {mi_class.upper()} (conf: {confidence:.2f}){cue_info}")
                                print(f"  Commands: {', '.join(self.command_history[-5:])}")
                                print(f"  Stats - L:{self.detection_stats['left']} R:{self.detection_stats['right']}")
                                
                                self.last_detection = current_time
                            else:
                                # Sliding window detected but waiting for min interval
                                if self.debug_mode:
                                    print(f"  [Sliding] {mi_class.upper()} detected (conf: {confidence:.2f}) - waiting {self.min_detection_interval - time_since_last:.1f}s")
                        else:
                            # Optional: show low confidence detections
                            if confidence > 0.4:
                                print(f"[{time.strftime('%H:%M:%S')}] Low confidence: {confidence:.2f}")
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
        
        print("\nStopped")
        if self.phase == 'phase1':
            self.print_phase1_stats()
            print(f"Event log saved to: {self.log_filename}")
        print(f"Total commands: {len(self.command_history)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG Model Integration')
    # Removed --continuous flag as it's now the default for real mode
    parser.add_argument('--debug', action='store_true',
                       help='Show detailed debug information')
    parser.add_argument('--phase1', action='store_true',
                       help='Use Phase 1 fake classifier with ground truth')
    parser.add_argument('--error-rate', type=float, default=0,
                       help='Error injection rate for phase1 (default: 0.2 = 20%)')
    parser.add_argument('--wait-time', type=float, default=4.0,
                       help='Time to wait after cue for phase1 (default: 4.0s)')
    args = parser.parse_args()
    
    try:
        # Create processor with specified mode
        phase = 'phase1' if args.phase1 else 'real'
        
        processor = TrainedModelEEGProcessor(
            debug=args.debug,
            phase=phase,
            error_rate=args.error_rate,
            wait_time=args.wait_time
        )
        
        print("\n" + "="*60)
        if phase == 'phase1':
            print("PHASE 1: FAKE CLASSIFIER WITH GROUND TRUTH")
        else:
            print("READY FOR TESTING!")
        print("="*60)
        
        if phase == 'phase1':
            print("\nPhase 1 Instructions:")
            print("1. Start the Subway Surfers game")
            print("2. The fake classifier will detect cues from the game")
            print(f"3. After {args.wait_time}s, it will make a decision based on GT")
            print(f"4. {args.error_rate*100:.0f}% of decisions will be errors (flipped)")
            print("5. All events are logged for ErrP analysis")
        else:
            print("\nCONTINUOUS MODE")
            print("Instructions:")
            print("1. The model detects MI continuously")
            print("2. Imagine LEFT or RIGHT hand movement anytime")
            print("3. No game synchronization required")
            
        print("\nNote: This model was trained on specific data.")
        print("It may need retraining for your specific setup.\n")
        
        processor.run()
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Check model file exists: MI/models/mi_improved_classifier.pkl")
        print("2. Ensure EEG is streaming")
        print("3. You may need to retrain the model for your specific EEG setup")