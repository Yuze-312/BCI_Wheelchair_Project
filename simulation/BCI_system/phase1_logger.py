"""
Phase1 Event Logger - Records event markers for pre-training data collection
"""

import os
import csv
import time
from datetime import datetime
from threading import Lock


class Phase1Logger:
    """Event logger for phase1 data collection"""
    
    def __init__(self, participant_id='unknown'):
        """Initialize phase1 event logger
        
        Args:
            participant_id: Participant identifier (e.g., 'T-009')
        """
        self.participant_id = participant_id
        self.log_lock = Lock()
        
        # Setup logging directory
        self._setup_directories()
        
        
        # CSV headers
        self.csv_headers = [
            'timestamp', 'event_type', 'trial_id', 'cue_class', 
            'predicted_class', 'accuracy', 'error_injected', 
            'wait_time', 'confidence', 'details'
        ]
        
        # Create CSV file
        self._create_log_file()
        
        # State tracking
        self.current_trial = 0
        self.cue_active = False
        self.cue_type = None
        self.cue_start_time = None
    
    def _setup_directories(self):
        """Create directory structure for pre-training datasets"""
        # Base directory for pre-training data
        self.base_dir = "pre_training_datasets"
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Participant-specific directory
        self.participant_dir = os.path.join(self.base_dir, self.participant_id)
        os.makedirs(self.participant_dir, exist_ok=True)
        
        # Session directory (timestamped)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.participant_dir, f"session_{self.session_timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
    
    def _create_log_file(self):
        """Create the CSV log file"""
        self.log_filename = os.path.join(self.session_dir, f"phase1_events_{self.session_timestamp}.csv")
        
        with open(self.log_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            writer.writeheader()
        
        print(f"Phase1 event logger created:")
        print(f"  Participant: {self.participant_id}")
        print(f"  Log file: {self.log_filename}")
        print(f"  Ready to sync with OpenViBE data")
    
    def log_event(self, event_type, **kwargs):
        """Log an event to CSV
        
        Args:
            event_type: Type of event (e.g., 'cue_left', 'response', etc.)
            **kwargs: Additional event parameters
        """
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
        
        with self.log_lock:
            with open(self.log_filename, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writerow(event)
    
    def log_cue(self, cue_type):
        """Log cue presentation
        
        Args:
            cue_type: 'left' or 'right'
        """
        self.cue_active = True
        self.cue_type = cue_type
        self.cue_start_time = time.time()
        self.current_trial += 1
        
        self.log_event(
            f'cue_{cue_type}',
            cue_class=cue_type,
            trial_id=self.current_trial
        )
    
    def log_response(self, predicted_class, confidence, error_injected=False, wait_time=4.0):
        """Log classifier response
        
        Args:
            predicted_class: 'left' or 'right'
            confidence: Confidence value (0-1)
            error_injected: Whether error was injected
            wait_time: Time waited before response
        """
        if not self.cue_active:
            return
        
        # Calculate accuracy
        accuracy = 1 if predicted_class == self.cue_type else 0
        
        self.log_event(
            'response',
            cue_class=self.cue_type,
            predicted_class=predicted_class,
            accuracy=accuracy,
            error_injected=int(error_injected),
            wait_time=wait_time,
            confidence=confidence
        )
        
        # Reset cue state
        self.cue_active = False
        self.cue_type = None
    
    def get_session_info(self):
        """Get information about current session"""
        return {
            'participant_id': self.participant_id,
            'session_dir': self.session_dir,
            'log_file': self.log_filename,
            'trials_logged': self.current_trial
        }
    
    def create_info_file(self):
        """Create an info file with session metadata"""
        info_file = os.path.join(self.session_dir, 'session_info.txt')
        
        with open(info_file, 'w') as f:
            f.write(f"Phase1 Data Collection Session\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Participant ID: {self.participant_id}\n")
            f.write(f"Session Time: {self.session_timestamp}\n")
            f.write(f"Log File: {os.path.basename(self.log_filename)}\n")
            f.write(f"\nInstructions for data processing:\n")
            f.write(f"1. Ensure OpenViBE data is saved in parallel\n")
            f.write(f"2. Use timestamps to synchronize events\n")
            f.write(f"3. Run process_new_participant.py to create training data\n")