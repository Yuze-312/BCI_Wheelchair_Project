"""
Voting Controller - Handles the 4-second voting window logic
"""

import time
import numpy as np
from threading import Thread, Lock


class VotingController:
    """Manages voting window for MI detection"""
    
    def __init__(self, classifier, stream_manager, command_writer, manipulation_rate=0.0):
        """Initialize voting controller
        
        Args:
            classifier: MIClassifier instance
            stream_manager: StreamManager instance  
            command_writer: CommandWriter instance
            manipulation_rate: Target success rate via manipulation (0.0 = no manipulation)
        """
        self.classifier = classifier
        self.stream_manager = stream_manager
        self.command_writer = command_writer
        self.manipulation_rate = manipulation_rate
        
        # State
        self.voting_in_progress = False
        self.cue_active = False
        self.cue_start_time = None
        self.cue_type = None
        self.cue_lock = Lock()
        
        # Detection stats
        self.detection_stats = {
            'left': 0, 
            'right': 0, 
            'cue_left': 0, 
            'cue_right': 0
        }
    
    def start_voting_window(self, cue_type):
        """Start voting window in separate thread
        
        Args:
            cue_type: 'left' or 'right'
        """
        with self.cue_lock:
            self.cue_active = True
            self.cue_start_time = time.time()
            self.cue_type = cue_type
            self.voting_in_progress = True
            
            # Clear buffer and start fresh data collection
            self.stream_manager.clear_buffer()
            self.stream_manager.data_collection_active = True
            
            # Reset command writer for new cue
            self.command_writer.reset_for_new_cue()
            
            # Update stats
            self.detection_stats[f'cue_{cue_type}'] += 1
        
        # Start voting thread
        vote_thread = Thread(target=self._process_voting_window, daemon=True)
        vote_thread.start()
        
        print(f"[VOTING] Pausing continuous processing for 4s voting window...")
        print(f"\n[VOTING MODE] CUE: {cue_type.upper()} - Collecting 4s, deciding at t=4s (1s left to act)")
    
    def _process_voting_window(self):
        """Process 4-second voting window (runs in separate thread)"""
        cue_start = self.cue_start_time
        cue_type = self.cue_type
        
        print(f"[VOTING] Collecting 4s of data, will decide with 1s remaining...")
        
        # Wait for 4 seconds of data
        while time.time() - cue_start < 4.0:
            elapsed = time.time() - cue_start
            buffer_seconds = self.stream_manager.get_buffer_size() / self.stream_manager.srate if self.stream_manager.srate > 0 else 0
            time_remaining = 5.0 - elapsed  # 5s total window
            
            # Show progress every second
            if int(elapsed) > int(elapsed - 0.1):
                print(f"   [VOTING] {elapsed:.1f}s elapsed, {time_remaining:.1f}s remaining, buffer: {buffer_seconds:.1f}s")
            
            time.sleep(0.1)
        
        # Get buffer data
        buffer_size = self.stream_manager.get_buffer_size()
        required_samples = int(4 * self.stream_manager.srate)
        
        if buffer_size < required_samples:
            print(f"WARNING: [VOTING] Insufficient data: {buffer_size} samples (need {required_samples})")
            with self.cue_lock:
                self.voting_in_progress = False
                self.cue_active = False
            return
        
        # Extract two 2-second windows from the 4s of data
        all_data = self.stream_manager.get_buffer_data()
        window1_data = all_data[:int(2 * self.stream_manager.srate)]
        window2_data = all_data[int(2 * self.stream_manager.srate):int(4 * self.stream_manager.srate)]
        
        print(f"\n[VOTING] Processing two windows (1s remaining to act):")
        print(f"   Window 1: 0-2s ({len(window1_data)} samples)")
        print(f"   Window 2: 2-4s ({len(window2_data)} samples)")
        
        # Get predictions from both windows
        class1, conf1 = self.classifier.classify(window1_data, self.stream_manager.srate)
        class2, conf2 = self.classifier.classify(window2_data, self.stream_manager.srate)
        
        # Print results
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
        
        # Apply manipulation if enabled
        manipulation_type = None
        original_class = final_class
        original_conf = final_conf
        
        if self.manipulation_rate > 0 and cue_type:
            import random
            
            # Decide if we should manipulate to achieve target success rate
            if random.random() < self.manipulation_rate:
                # Force correct action
                if final_class != cue_type:
                    # Classifier was wrong, fix it
                    final_class = cue_type
                    manipulation_type = 11  # MARKER_FORCED_CORRECT
                else:
                    # Classifier was already correct
                    manipulation_type = 10  # MARKER_NATURAL_CORRECT
            else:
                # Force error (to maintain ~75% success rate)
                opposite = 'right' if cue_type == 'left' else 'left'
                if final_class == cue_type:
                    # Classifier was correct, break it
                    final_class = opposite
                    manipulation_type = 13  # MARKER_FORCED_ERROR
                else:
                    # Classifier was already wrong
                    manipulation_type = 12  # MARKER_NATURAL_ERROR
            
            # Log manipulation details
            print(f"\n[MANIPULATION] Target rate: {self.manipulation_rate:.0%}")
            print(f"   Original: {original_class or 'None'} (conf: {original_conf:.2f})")
            print(f"   Ground truth: {cue_type}")
            print(f"   Final action: {final_class}")
            print(f"   Type: {manipulation_type} ({['','','','','','','','','','','NATURAL_CORRECT','FORCED_CORRECT','NATURAL_ERROR','FORCED_ERROR'][manipulation_type-1] if manipulation_type else 'None'})")
        
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
            
            # Send manipulation marker if applicable
            if manipulation_type:
                # Send marker via stream manager
                if hasattr(self.stream_manager, 'send_marker'):
                    self.stream_manager.send_marker(manipulation_type)
            
            # Send command
            if self.command_writer.write_command(final_class):
                print(f"   Command sent with 1s remaining")
                self.detection_stats[final_class] += 1
        else:
            print(f"\n[NO DECISION] Confidence too low for both classifiers")
        
        # Wait for the remaining 1s
        time.sleep(1.0)
        
        # Reset flags
        with self.cue_lock:
            self.voting_in_progress = False
            self.cue_active = False
        
        print(f"   [VOTING] Voting window complete - resuming continuous mode")
        print("─" * 60)
    
    def is_voting_active(self):
        """Check if voting is currently in progress"""
        with self.cue_lock:
            return self.voting_in_progress
    
    def is_cue_active(self):
        """Check if cue is currently active"""
        with self.cue_lock:
            return self.cue_active