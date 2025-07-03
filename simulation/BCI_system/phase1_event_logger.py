"""
Phase1 Event Logger - Records events in the same format as simulator's subway_errp logs
"""

import os
import csv
import time
from datetime import datetime


class Phase1EventLogger:
    """Event logger for phase1 that matches simulator's subway_errp format exactly"""
    
    def __init__(self, participant_id='unknown'):
        """Initialize phase1 event logger
        
        Args:
            participant_id: Participant identifier (e.g., 'T-009')
        """
        self.participant_id = participant_id
        
        # Setup logging directory
        self._setup_directories()
        
        # Create CSV file with same headers as simulator
        self._create_log_file()
        
        # State tracking
        self.current_trial = 0
        self.cue_onset_time = None
        self.cue_type = None  # 'left' or 'right'
        self.cue_gt = None    # 0 for left, 1 for right
    
    def _setup_directories(self):
        """Create directory structure for pre-training event logs"""
        # Base directory for pre-training event logs
        self.base_dir = "pre_training_event_log"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Participant-specific directory
        self.participant_dir = os.path.join(self.base_dir, self.participant_id)
        os.makedirs(self.participant_dir, exist_ok=True)
        
        # Session timestamp
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_log_file(self):
        """Create the CSV log file with same format as simulator"""
        self.log_filename = os.path.join(self.participant_dir, f"phase1_events_{self.session_timestamp}.csv")
        
        with open(self.log_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Exact same headers as subway_errp logs
            writer.writerow(['timestamp', 'rel_time', 'trial', 'event', 'classifier_out', 'confidence', 'gt'])
        
        print(f"\n[PHASE1] Event logger initialized:")
        print(f"  Participant: {self.participant_id}")
        print(f"  Log file: {self.log_filename}")
    
    def log_trial_start(self):
        """Log trial start marker (event=1)"""
        self.current_trial += 1
        self._write_event(
            timestamp=time.time(),
            rel_time='',
            trial=self.current_trial,
            event=1,  # Trial start marker
            classifier_out='',
            confidence='',
            gt=''
        )
    
    def log_cue(self, cue_type):
        """Log cue presentation (event=2 for LEFT, event=3 for RIGHT)
        
        Args:
            cue_type: 'left' or 'right'
        """
        self.cue_onset_time = time.time()
        self.cue_type = cue_type
        self.cue_gt = 0 if cue_type == 'left' else 1
        
        # Event marker: 2=LEFT cue, 3=RIGHT cue
        event_marker = 2 if cue_type == 'left' else 3
        
        self._write_event(
            timestamp=self.cue_onset_time,
            rel_time=0.000,
            trial=self.current_trial,
            event=event_marker,
            classifier_out='',
            confidence='',
            gt=self.cue_gt
        )
    
    def log_response(self, predicted_class, confidence, manipulation_data=None):
        """Log classifier response with optional manipulation data
        
        Args:
            predicted_class: 'left' or 'right'
            confidence: Confidence value (0-1)
            manipulation_data: Dict with manipulation info {
                'original_class': str,
                'original_confidence': float,
                'manipulation_type': int,
                'executed_action': str
            }
        """
        if self.cue_onset_time is None:
            return
        
        # Calculate relative time from cue
        current_time = time.time()
        rel_time = current_time - self.cue_onset_time
        
        # Determine if response is correct
        predicted_gt = 0 if predicted_class == 'left' else 1
        is_correct = (predicted_gt == self.cue_gt)
        
        # Event marker: 5=correct response, 6=error response
        event_marker = 5 if is_correct else 6
        
        self._write_event(
            timestamp=current_time,
            rel_time=f"{rel_time:.3f}",
            trial=self.current_trial,
            event=event_marker,
            classifier_out=predicted_gt,
            confidence=confidence,
            gt=self.cue_gt
        )
        
        # Log manipulation marker if provided
        if manipulation_data and 'manipulation_type' in manipulation_data:
            self._write_event(
                timestamp=current_time + 0.001,  # Slight offset
                rel_time=f"{rel_time + 0.001:.3f}",
                trial=self.current_trial,
                event=manipulation_data['manipulation_type'],
                classifier_out=0 if manipulation_data.get('original_class') == 'left' else 1,
                confidence=manipulation_data.get('original_confidence', confidence),
                gt=self.cue_gt
            )
        
        # Reset cue state
        self.cue_onset_time = None
        self.cue_type = None
        self.cue_gt = None
    
    def _write_event(self, timestamp, rel_time, trial, event, classifier_out, confidence, gt):
        """Write event to CSV file"""
        with open(self.log_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{timestamp:.3f}",
                rel_time,
                trial,
                event,
                classifier_out,
                confidence,
                gt
            ])
    
    def get_log_info(self):
        """Get information about current log"""
        return {
            'participant_id': self.participant_id,
            'log_file': self.log_filename,
            'trials_logged': self.current_trial
        }