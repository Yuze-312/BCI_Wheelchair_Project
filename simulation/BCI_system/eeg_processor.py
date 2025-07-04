"""
EEG Processor - Main orchestrator for the BCI system
"""

import os
import time
import numpy as np
from threading import Thread

from .stream_manager import StreamManager
from .mi_classifier import MIClassifier
from .command_writer import CommandWriter
from .voting_controller import VotingController
from .phase1_event_logger import Phase1EventLogger


class EEGProcessor:
    """Main EEG processor that coordinates all components"""
    
    def __init__(self, debug=False, phase='real', participant_id='unknown', error_rate=0.0, manipulation_rate=0.0):
        """Initialize EEG processor
        
        Args:
            debug: Enable debug output
            phase: 'real' for actual EEG, 'phase1' for fake classifier
            participant_id: Participant ID for phase1 data collection
            error_rate: Error injection rate for phase1 (0.0 = 100% accuracy)
            manipulation_rate: For real mode - target success rate via manipulation (0.0 = no manipulation, 0.75 = 75% success)
        """
        self.debug_mode = debug
        self.phase = phase
        self.participant_id = participant_id
        self.manipulation_rate = manipulation_rate
        
        # Initialize components
        self.stream_manager = StreamManager(phase=phase)
        self.classifier = MIClassifier(debug=debug)
        # For phase1, always start with clean command history
        self.command_writer = CommandWriter(clear_on_init=(phase == 'phase1'))
        self.voting_controller = VotingController(
            self.classifier, 
            self.stream_manager, 
            self.command_writer,
            manipulation_rate=self.manipulation_rate if phase == 'real' else 0.0
        )
        
        # Continuous processing state
        self.min_detection_interval = 1.0  # seconds
        self.last_detection = 0
        self.sliding_window_interval = 0.5  # Process every 500ms
        self.last_sliding_window_time = 0
        
        # Statistics
        self._prediction_counts = {'left': 0, 'right': 0, 'none': 0}
        
        # For phase1 fake classifier
        if phase == 'phase1':
            self.error_rate = error_rate  # Use provided error rate
            self.fake_wait_time = 4.0
            self.fake_stats = {
                'total_cues': 0,
                'correct_actions': 0,
                'errors_injected': 0,
                'cue_left': 0,
                'cue_right': 0
            }
            # Initialize phase1 event logger
            self.phase1_logger = Phase1EventLogger(participant_id=participant_id)
    
    def initialize(self):
        """Initialize all components"""
        # Connect to streams
        self.stream_manager.connect_to_streams()
        
        # Load model for real mode
        if self.phase == 'real':
            self.classifier.load_model(participant_id=self.participant_id)
            self.classifier.setup_filter(self.stream_manager.srate)
        else:
            print(f"Phase 1: Using fake classifier with 100% accuracy (for data collection)")
        
        # Start marker monitoring
        if self.stream_manager.has_markers:
            marker_thread = Thread(target=self._monitor_markers, daemon=True)
            marker_thread.start()
            print("Marker monitoring started")
    
    def run(self):
        """Main processing loop"""
        if self.phase == 'phase1':
            print(f"\nEEG Processing - PHASE 1: Data Collection Mode")
            print(f"Error rate: {self.error_rate*100:.0f}% (Accuracy: {100-self.error_rate*100:.0f}%)")
            print(f"Wait time after cue: {self.fake_wait_time}s")
            print(f"Following ground truth from game cues\n")
        else:
            print(f"\nEEG Processing with Pretrained Model [Continuous Mode]")
            print(f"Confidence threshold: {self.classifier.confidence_threshold}")
            print(f"Processing continuously, enhanced during cues")
            
            # Show manipulation status
            if self.manipulation_rate > 0:
                print(f"\n{'='*60}")
                print(f"[ErrP ELICITATION MODE ACTIVE]")
                print(f"Manipulation Rate: {self.manipulation_rate:.0%} success rate")
                print(f"Expected Error Rate: {(1-self.manipulation_rate)*100:.0f}%")
                print(f"Purpose: Controlled ErrP elicitation for adaptive learning")
                print(f"{'='*60}\n")
            else:
                print(f"\n[Standard Mode] No manipulation - using raw classifier output\n")
        
        print("Press Ctrl+C to stop\n")
        
        # Option to clear history
        if input("\nClear command history? (y/N): ").lower() == 'y':
            self.command_writer.clear_history()
        
        # Wait for game if needed
        if self.stream_manager.has_markers:
            print("\nGame connected - Continuous detection active")
        else:
            print("\nWaiting for game to start...")
            print("   Will begin processing once game markers are detected")
        print()
        
        # Main loop
        last_buffer_log = time.time()
        self._last_reconnect_attempt = 0
        
        while True:
            try:
                # Try to connect to markers if not connected
                if not self.stream_manager.has_markers:
                    if time.time() - self._last_reconnect_attempt > 2.0:
                        self._last_reconnect_attempt = time.time()
                        if self.stream_manager.try_reconnect_markers():
                            marker_thread = Thread(target=self._monitor_markers, daemon=True)
                            marker_thread.start()
                            print("Monitoring game markers...")
                            print("Waiting for cues...\n")
                
                # Phase 1: Check for fake decisions
                if self.phase == 'phase1':
                    self._process_phase1()
                    time.sleep(0.01)
                    continue
                
                # Real mode: Process EEG data
                chunk, timestamps = self.stream_manager.pull_data()
                
                # Add to buffer
                if chunk:
                    self.stream_manager.add_to_buffer(chunk)
                
                # Buffer status logging
                current_time = time.time()
                if self.debug_mode and current_time - last_buffer_log > 5.0:
                    buffer_seconds = self.stream_manager.get_buffer_size() / self.stream_manager.srate if self.stream_manager.srate > 0 else 0
                    print(f"\n[Buffer Status] {time.strftime('%H:%M:%S')} - {self.stream_manager.get_buffer_size()} samples ({buffer_seconds:.1f}s)")
                    last_buffer_log = current_time
                
                # Check if we should process
                should_process = self._should_process_continuous()
                
                # Process using sliding window
                if should_process and self._is_sliding_window_ready():
                    self._process_continuous()
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
        
        print("\nStopped")
        
        # Print phase1 stats if applicable
        if self.phase == 'phase1':
            self._print_phase1_stats()
            
        print(f"Total commands: {self.command_writer.get_stats()['total']}")
    
    def _should_process_continuous(self):
        """Check if continuous processing should run"""
        # Block if voting is in progress
        if self.voting_controller.is_voting_active():
            return False
        
        # Check basic conditions
        return (self.stream_manager.has_markers and 
                self.stream_manager.game_active and 
                self.stream_manager.data_collection_active)
    
    def _is_sliding_window_ready(self):
        """Check if sliding window is ready to process"""
        current_time = time.time()
        if current_time - self.last_sliding_window_time >= self.sliding_window_interval:
            # Need at least 2 seconds of data
            if self.stream_manager.get_buffer_size() >= int(self.stream_manager.srate * 2.0):
                self.last_sliding_window_time = current_time
                return True
        return False
    
    def _process_continuous(self):
        """Process continuous MI detection"""
        # Get 2 seconds of data
        epoch_samples = int(self.stream_manager.srate * 2.0)
        epoch_data = self.stream_manager.get_buffer_data(epoch_samples)
        
        if len(epoch_data) < epoch_samples:
            return
        
        # Classify
        mi_class, confidence = self.classifier.classify(epoch_data, self.stream_manager.srate)
        
        # Update statistics
        if mi_class:
            self._prediction_counts[mi_class] += 1
            total = self._prediction_counts['left'] + self._prediction_counts['right']
            if total > 0 and total % 10 == 0:
                left_pct = self._prediction_counts['left'] / total * 100
                right_pct = self._prediction_counts['right'] / total * 100
                print(f"\n[Prediction bias check] (n={total}):")
                print(f"   LEFT: {self._prediction_counts['left']} ({left_pct:.1f}%)")
                print(f"   RIGHT: {self._prediction_counts['right']} ({right_pct:.1f}%)")
                if left_pct > 80 or right_pct > 80:
                    print("   WARNING: Strong bias detected!")
        
        if mi_class:
            # Never send commands during cue windows
            if self.voting_controller.is_cue_active():
                print(f"   [BLOCKED] MI detected during cue window - voting mode will handle this")
                return
            
            # Check if enough time passed since last detection
            current_time = time.time()
            time_since_last = current_time - self.last_detection
            
            should_send = (time_since_last >= self.min_detection_interval or
                          confidence >= 0.85)  # High confidence override
            
            if should_send:
                # Final safety check
                if self.voting_controller.is_voting_active() or self.voting_controller.is_cue_active():
                    print(f"   [SAFETY] Command blocked - voting/cue active")
                    return
                
                # During real-time continuous mode, we detect but don't send commands
                # Commands are only sent during voting windows
                if self.debug_mode:
                    print(f"[{time.strftime('%H:%M:%S')}] [CONTINUOUS] MI detected: {mi_class.upper()} (conf: {confidence:.2f})")
                    print(f"   Note: Commands only sent during voting windows")
                else:
                    # Minimal logging for continuous detections
                    print(f"[CONTINUOUS] {mi_class.upper()} detected (conf: {confidence:.2f})")
                
                self.last_detection = current_time
    
    def _process_phase1(self):
        """Process phase 1 fake classifier"""
        import random
        
        # Check if there's an active cue to process
        if not hasattr(self, '_phase1_cue_active'):
            self._phase1_cue_active = False
            self._phase1_cue_type = None
            self._phase1_cue_time = None
            
        if self._phase1_cue_active:
            # Check if enough time has passed
            elapsed = time.time() - self._phase1_cue_time
            if elapsed >= self.fake_wait_time:
                # Make fake decision based on ground truth
                correct_action = self._phase1_cue_type
                
                # Inject error based on error rate
                if random.random() < self.error_rate:
                    # Flip the decision
                    predicted_action = 'right' if correct_action == 'left' else 'left'
                    self.fake_stats['errors_injected'] += 1
                else:
                    predicted_action = correct_action
                    self.fake_stats['correct_actions'] += 1
                
                # Set confidence based on correctness
                if self.error_rate == 0:
                    # Perfect confidence when following ground truth
                    confidence = 0.95
                else:
                    # Variable confidence based on correctness
                    confidence = 0.8 if predicted_action == correct_action else 0.6
                
                # Log the response to phase1 logger
                self.phase1_logger.log_response(predicted_action, confidence)
                
                # Send command
                command = '1' if predicted_action == 'left' else '2'
                self.command_writer.write_command(predicted_action)
                
                # Print decision
                if self.error_rate == 0:
                    match_str = "FOLLOWING GT"
                else:
                    match_str = "CORRECT" if predicted_action == correct_action else "ERROR"
                print(f"\n[PHASE1] Decision: {predicted_action.upper()} ({match_str})")
                print(f"  Confidence: {confidence:.2f}")
                print(f"  Wait time: {self.fake_wait_time}s")
                
                # Small delay to ensure command is processed before cue ends
                time.sleep(0.1)
                
                # Reset cue state
                self._phase1_cue_active = False
                self._phase1_cue_type = None
    
    def _monitor_markers(self):
        """Monitor game markers in separate thread"""
        print("Monitoring game markers...")
        marker_count = 0
        
        while True:
            try:
                markers, timestamps = self.stream_manager.pull_markers()
                
                for marker in markers:
                    marker_type = marker[0]
                    marker_count += 1
                    
                    if self.debug_mode:
                        print(f"  [MARKER DEBUG] Received: {marker_type} (count: {marker_count})")
                    
                    # Game state
                    if not self.stream_manager.game_active:
                        self.stream_manager.game_active = True
                        self.stream_manager.data_collection_active = True
                        print(f"Game activity detected! (marker: {marker_type})")
                        print("   Starting continuous data collection...")
                    
                    # Cue detection - numeric markers: 2=LEFT, 3=RIGHT
                    if marker_type == 2 or marker_type == 3:
                        cue_type = 'left' if marker_type == 2 else 'right'
                        
                        if self.phase == 'phase1':
                            # Phase1: Log cue and prepare for fake decision
                            self.phase1_logger.log_cue(cue_type)
                            self._phase1_cue_active = True
                            self._phase1_cue_type = cue_type
                            self._phase1_cue_time = time.time()
                            self.fake_stats['total_cues'] += 1
                            self.fake_stats[f'cue_{cue_type}'] += 1
                            # Reset command writer for new cue
                            self.command_writer.reset_for_new_cue()
                            print(f"\n[PHASE1] CUE: {cue_type.upper()} - Will respond in {self.fake_wait_time}s")
                        else:
                            # Real mode: Start voting window
                            self.voting_controller.start_voting_window(cue_type)
                    
                    # Trial separator
                    elif marker_type == 1:
                        if self.phase == 'phase1':
                            # Log trial start for phase1
                            self.phase1_logger.log_trial_start()
                        if self.voting_controller.is_voting_active():
                            print("   [VOTING] Trial separator received - voting thread will complete soon")
                
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Marker error: {e}")
                time.sleep(0.1)
    
    def _print_phase1_stats(self):
        """Print phase1 session statistics"""
        print("\nPhase 1 Session Statistics:")
        print(f"Total cues: {self.fake_stats['total_cues']}")
        print(f"  Left cues: {self.fake_stats['cue_left']}")
        print(f"  Right cues: {self.fake_stats['cue_right']}")
        print(f"Correct actions: {self.fake_stats['correct_actions']} ({self.fake_stats['correct_actions']/max(1,self.fake_stats['total_cues'])*100:.1f}%)")
        print(f"Errors injected: {self.fake_stats['errors_injected']} ({self.fake_stats['errors_injected']/max(1,self.fake_stats['total_cues'])*100:.1f}%)")
        
        # Show log info
        log_info = self.phase1_logger.get_log_info()
        print(f"\nPhase1 event log saved to:")
        print(f"  {log_info['log_file']}")
        print(f"  Format: subway_errp compatible (for easy data processing)")