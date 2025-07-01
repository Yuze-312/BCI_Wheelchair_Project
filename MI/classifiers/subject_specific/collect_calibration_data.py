#!/usr/bin/env python
"""
Calibration Data Collection for Subject-Specific MI Classifier
Presents cues and collects EEG data for training personalized model
"""

import numpy as np
import pygame
import time
import pickle
import os
from datetime import datetime
from pylsl import StreamInlet, StreamOutlet, StreamInfo, resolve_streams
import argparse
from collections import deque
import random

class CalibrationSession:
    def __init__(self, participant_id, n_trials=100):
        """Initialize calibration session"""
        self.participant_id = participant_id
        self.n_trials = n_trials
        self.trial_data = []
        
        # Timing parameters (matching training data)
        self.baseline_duration = 1.0  # Pre-cue baseline
        self.cue_duration = 1.0       # Cue display
        self.mi_duration = 3.0        # Motor imagery period
        self.feedback_duration = 1.0  # Feedback display
        self.rest_duration = 1.5      # Inter-trial interval
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption(f"MI Calibration - {participant_id}")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        
        # Fonts
        self.font_large = pygame.font.Font(None, 120)
        self.font_medium = pygame.font.Font(None, 60)
        self.font_small = pygame.font.Font(None, 40)
        
        # Connect to EEG
        self.connect_to_eeg()
        
        # Create marker stream
        self.setup_markers()
        
    def connect_to_eeg(self):
        """Connect to EEG stream"""
        print("Connecting to EEG stream...")
        streams = resolve_streams()
        
        eeg_streams = [s for s in streams if s.type() == 'EEG' or 'eeg' in s.name().lower()]
        if not eeg_streams:
            raise RuntimeError("No EEG streams found!")
        
        # Prefer 16-channel
        streams_16ch = [s for s in eeg_streams if s.channel_count() == 16]
        selected = streams_16ch[0] if streams_16ch else eeg_streams[0]
        
        self.inlet = StreamInlet(selected)
        self.n_channels = selected.channel_count()
        self.srate = selected.nominal_srate()
        
        print(f"✓ Connected to: {selected.name()}")
        print(f"  Channels: {self.n_channels}")
        print(f"  Sample rate: {self.srate}Hz")
        
        # Buffer for storing trial data
        self.buffer_size = int(self.srate * 10)  # 10 second buffer
        self.eeg_buffer = deque(maxlen=self.buffer_size)
        self.timestamp_buffer = deque(maxlen=self.buffer_size)
        
    def setup_markers(self):
        """Create marker stream for synchronization"""
        info = StreamInfo('MI_Calibration_Markers', 'Markers', 1, 0, 'string', 'mi_cal_001')
        self.marker_outlet = StreamOutlet(info)
        
    def send_marker(self, marker):
        """Send marker with timestamp"""
        self.marker_outlet.push_sample([marker])
        print(f"Marker: {marker}")
        
    def run_trial(self, trial_num, mi_class):
        """Run a single calibration trial"""
        
        # Clear screen
        self.screen.fill(self.BLACK)
        pygame.display.flip()
        
        # Record trial start time
        trial_start = time.time()
        trial_start_sample = len(self.eeg_buffer)
        
        # 1. Baseline period
        self.send_marker(f"BASELINE_START_{trial_num}")
        self.show_fixation()
        self.wait_and_collect(self.baseline_duration)
        self.send_marker(f"BASELINE_END_{trial_num}")
        
        # 2. Cue presentation
        cue_text = "LEFT" if mi_class == 0 else "RIGHT"
        self.send_marker(f"CUE_{cue_text}_{trial_num}")
        self.show_cue(cue_text)
        self.wait_and_collect(self.cue_duration)
        
        # 3. Motor imagery period
        self.send_marker(f"MI_START_{trial_num}")
        self.show_mi_prompt(cue_text)
        self.wait_and_collect(self.mi_duration)
        self.send_marker(f"MI_END_{trial_num}")
        
        # 4. Feedback (just acknowledgment for calibration)
        self.show_feedback()
        self.wait_and_collect(self.feedback_duration)
        
        # 5. Rest period
        self.screen.fill(self.BLACK)
        pygame.display.flip()
        self.wait_and_collect(self.rest_duration)
        
        # Extract trial data from buffer
        trial_end_sample = len(self.eeg_buffer)
        trial_data = {
            'trial_num': trial_num,
            'mi_class': mi_class,
            'label': cue_text,
            'start_sample': trial_start_sample,
            'end_sample': trial_end_sample,
            'trial_duration': time.time() - trial_start
        }
        
        return trial_data
        
    def show_fixation(self):
        """Show fixation cross"""
        self.screen.fill(self.BLACK)
        # Draw cross
        pygame.draw.line(self.screen, self.WHITE, (380, 300), (420, 300), 3)
        pygame.draw.line(self.screen, self.WHITE, (400, 280), (400, 320), 3)
        pygame.display.flip()
        
    def show_cue(self, direction):
        """Show directional cue"""
        self.screen.fill(self.BLACK)
        
        # Draw arrow
        if direction == "LEFT":
            points = [(320, 300), (380, 270), (380, 285), (480, 285), 
                     (480, 315), (380, 315), (380, 330)]
            color = (100, 100, 255)  # Blue for left
        else:
            points = [(480, 300), (420, 270), (420, 285), (320, 285), 
                     (320, 315), (420, 315), (420, 330)]
            color = (255, 100, 100)  # Red for right
            
        pygame.draw.polygon(self.screen, color, points)
        
        # Show text
        text = self.font_large.render(direction, True, self.WHITE)
        text_rect = text.get_rect(center=(400, 150))
        self.screen.blit(text, text_rect)
        
        pygame.display.flip()
        
    def show_mi_prompt(self, direction):
        """Show motor imagery prompt"""
        self.screen.fill(self.BLACK)
        
        # Main instruction
        text1 = self.font_medium.render("IMAGINE", True, self.WHITE)
        text1_rect = text1.get_rect(center=(400, 250))
        self.screen.blit(text1, text1_rect)
        
        text2 = self.font_large.render(f"{direction} HAND", True, self.GREEN)
        text2_rect = text2.get_rect(center=(400, 350))
        self.screen.blit(text2, text2_rect)
        
        pygame.display.flip()
        
    def show_feedback(self):
        """Show feedback (just positive for calibration)"""
        self.screen.fill(self.BLACK)
        text = self.font_medium.render("Good!", True, self.GREEN)
        text_rect = text.get_rect(center=(400, 300))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        
    def wait_and_collect(self, duration):
        """Wait while collecting EEG data"""
        start_time = time.time()
        while time.time() - start_time < duration:
            # Collect EEG data
            chunk, timestamps = self.inlet.pull_chunk(timeout=0.0)
            if chunk:
                self.eeg_buffer.extend(chunk)
                self.timestamp_buffer.extend(timestamps)
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
            
            self.clock.tick(60)
        return True
        
    def show_progress(self, current, total):
        """Show session progress"""
        self.screen.fill(self.BLACK)
        
        # Progress text
        text1 = self.font_medium.render(f"Trial {current}/{total}", True, self.WHITE)
        text1_rect = text1.get_rect(center=(400, 250))
        self.screen.blit(text1, text1_rect)
        
        # Progress bar
        bar_width = 400
        bar_height = 30
        bar_x = 200
        bar_y = 320
        
        # Background
        pygame.draw.rect(self.screen, self.GRAY, (bar_x, bar_y, bar_width, bar_height))
        
        # Progress
        progress = current / total
        pygame.draw.rect(self.screen, self.GREEN, 
                        (bar_x, bar_y, int(bar_width * progress), bar_height))
        
        # Instructions
        text2 = self.font_small.render("Press SPACE to continue", True, self.WHITE)
        text2_rect = text2.get_rect(center=(400, 400))
        self.screen.blit(text2, text2_rect)
        
        pygame.display.flip()
        
        # Wait for space
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        return False
            self.clock.tick(60)
        return True
        
    def run_session(self):
        """Run complete calibration session"""
        print(f"\nStarting calibration for participant: {self.participant_id}")
        print(f"Total trials: {self.n_trials}")
        
        # Generate balanced trial sequence
        n_per_class = self.n_trials // 2
        trials = [0] * n_per_class + [1] * n_per_class
        random.shuffle(trials)
        
        # Show instructions
        self.show_instructions()
        
        # Run trials
        for i, mi_class in enumerate(trials):
            # Show progress every 10 trials
            if i % 10 == 0:
                if not self.show_progress(i, self.n_trials):
                    break
            
            # Run trial
            trial_data = self.run_trial(i+1, mi_class)
            self.trial_data.append(trial_data)
            
        # Save data
        self.save_calibration_data()
        
        # Show completion
        self.show_completion()
        
    def show_instructions(self):
        """Show session instructions"""
        self.screen.fill(self.BLACK)
        
        instructions = [
            "MI Calibration Session",
            "",
            "You will see LEFT or RIGHT arrows",
            "When you see the arrow, imagine moving that hand",
            "Keep imagining until the screen changes",
            "",
            "Try to stay relaxed and avoid moving",
            "Blink only during rest periods",
            "",
            "Press SPACE to begin"
        ]
        
        y = 100
        for line in instructions:
            if line == "MI Calibration Session":
                text = self.font_medium.render(line, True, self.WHITE)
            else:
                text = self.font_small.render(line, True, self.WHITE)
            text_rect = text.get_rect(center=(400, y))
            self.screen.blit(text, text_rect)
            y += 40
            
        pygame.display.flip()
        
        # Wait for space
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
            self.clock.tick(60)
            
    def show_completion(self):
        """Show session completion"""
        self.screen.fill(self.BLACK)
        
        text1 = self.font_large.render("Complete!", True, self.GREEN)
        text1_rect = text1.get_rect(center=(400, 250))
        self.screen.blit(text1, text1_rect)
        
        text2 = self.font_medium.render(f"Collected {len(self.trial_data)} trials", True, self.WHITE)
        text2_rect = text2.get_rect(center=(400, 350))
        self.screen.blit(text2, text2_rect)
        
        pygame.display.flip()
        time.sleep(3)
        
    def save_calibration_data(self):
        """Save collected data for training"""
        # Convert buffer to numpy array
        eeg_data = np.array(self.eeg_buffer)
        timestamps = np.array(self.timestamp_buffer)
        
        # Prepare data structure
        calibration_data = {
            'participant_id': self.participant_id,
            'eeg_data': eeg_data,
            'timestamps': timestamps,
            'trials': self.trial_data,
            'srate': self.srate,
            'n_channels': self.n_channels,
            'session_date': datetime.now().isoformat()
        }
        
        # Save raw data
        filename = f"calibration_{self.participant_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = os.path.join('../../processed_data', filename)
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(calibration_data, f)
            
        print(f"\n✓ Calibration data saved to: {filepath}")
        print(f"  Next step: Process with data_processing pipeline")
        print(f"  Then train with: python train_subject_classifier.py {self.participant_id}")

def main():
    parser = argparse.ArgumentParser(description='Collect MI calibration data')
    parser.add_argument('--participant', type=str, required=True,
                       help='Participant ID (e.g., YOUR_NAME)')
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of trials to collect (default: 100)')
    args = parser.parse_args()
    
    try:
        session = CalibrationSession(args.participant, args.trials)
        session.run_session()
        pygame.quit()
        
    except Exception as e:
        pygame.quit()
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure EEG stream is running")
        print("2. Check pygame is installed")
        print("3. Ensure you have write permissions")

if __name__ == "__main__":
    main()