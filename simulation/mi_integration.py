"""
Motor Imagery Integration Module for Subway Surfers Simulator
Replaces keyboard input with real-time MI predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MI'))

import pygame
import numpy as np
import time
from collections import deque
# Import the real-time classifier
from models.realtime_classifier import RealtimeMIClassifier, MISimulatorInterface
from pylsl import StreamInlet, resolve_streams
try:
    from pylsl import local_clock
except ImportError:
    import time
    local_clock = time.time
import threading


class MockMIDecoder:
    """
    Mock MI decoder for testing without real EEG hardware
    Simulates MI classification with configurable accuracy
    """
    
    def __init__(self):
        self.is_running = False
        self.accuracy = 0.8
        
    def start(self):
        """Start the mock decoder"""
        self.is_running = True
        
    def stop(self):
        """Stop the mock decoder"""
        self.is_running = False
        
    def get_prediction(self):
        """Get mock prediction with configurable accuracy"""
        if not self.is_running:
            return None, 0.0
            
        # Simulate classification with configured accuracy
        if np.random.random() < self.accuracy:
            # Correct prediction (would match the cue)
            return 'correct', 0.7 + np.random.random() * 0.3
        else:
            # Incorrect prediction
            return 'incorrect', 0.4 + np.random.random() * 0.3
    
    def reset(self):
        """Reset decoder state"""
        pass
    
    def connect_lsl(self, stream_type="MockEEG"):
        """Mock LSL connection - always succeeds"""
        print(f"Mock decoder: Simulating connection to {stream_type} stream")
        self.is_running = True
        return True
    
    def predict(self):
        """Get prediction in the format expected by MIController"""
        if not self.is_running:
            return {'class': -1, 'label': 'Rest', 'confidence': 0.0}
        
        # Simulate some processing time
        time.sleep(0.05)
        
        # Random prediction with configured accuracy
        if np.random.random() < 0.7:  # 70% chance of valid prediction
            if np.random.random() < 0.5:
                return {'class': 0, 'label': 'Left', 'confidence': 0.7 + np.random.random() * 0.3}
            else:
                return {'class': 1, 'label': 'Right', 'confidence': 0.7 + np.random.random() * 0.3}
        else:
            # No clear prediction
            return {'class': -1, 'label': 'Rest', 'confidence': 0.3}


class MIController:
    """
    Controller that integrates MI decoder with game input
    Manages cue presentation and MI decoding
    """
    
    def __init__(self, decoder=None, cue_duration=3.0, mi_window=(0.5, 2.5)):
        """
        Initialize MI controller
        
        Args:
            decoder: MIRealtimeDecoder instance
            cue_duration: Total cue duration in seconds
            mi_window: Tuple of (start, end) for MI detection window within cue
        """
        # Initialize decoder if not provided
        if decoder is None:
            # For now, create a mock decoder since we need the actual model path
            self.decoder = MockMIDecoder()  # Will be defined below
        else:
            self.decoder = decoder
        
        # Timing parameters
        self.cue_duration = cue_duration
        self.mi_window_start = mi_window[0]
        self.mi_window_end = mi_window[1]
        
        # State management
        self.current_cue = None  # 'LEFT', 'RIGHT', or None
        self.cue_start_time = None
        self.in_cue = False
        self.mi_detected = False
        self.last_prediction = None
        
        # Visual feedback
        self.feedback_duration = 0.5
        self.feedback_start_time = None
        self.show_feedback = False
        
        # Performance tracking
        self.trial_history = deque(maxlen=100)
        self.reaction_times = deque(maxlen=100)
        
        # Threading for continuous MI monitoring
        self.monitoring = False
        self.monitor_thread = None
        
        # Simulated EEG mode (for testing without real EEG)
        self.simulation_mode = True
        self.simulation_accuracy = 0.7  # Simulate 70% accuracy
    
    def start_monitoring(self):
        """Start continuous MI monitoring in background thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print("MI monitoring started")
    
    def stop_monitoring(self):
        """Stop MI monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        print("MI monitoring stopped")
    
    def _monitoring_loop(self):
        """Background loop for continuous MI detection"""
        while self.monitoring:
            try:
                # Only process during MI window of active cue
                if self.in_cue and self.current_cue and not self.mi_detected:
                    current_time = time.time()
                    time_since_cue = current_time - self.cue_start_time
                    
                    # Check if we're in the MI detection window
                    if self.mi_window_start <= time_since_cue <= self.mi_window_end:
                        # Get MI prediction
                        if self.simulation_mode:
                            prediction = self._simulate_mi_prediction()
                        else:
                            prediction = self.decoder.predict()
                        
                        # Check if valid prediction
                        if prediction['class'] != -1 and prediction['confidence'] > 0.6:
                            self.last_prediction = prediction
                            self.mi_detected = True
                            
                            # Record reaction time
                            reaction_time = (current_time - self.cue_start_time) * 1000  # ms
                            self.reaction_times.append(reaction_time)
                            
                            print(f"MI detected: {prediction['label']} "
                                  f"(confidence: {prediction['confidence']:.2f}, "
                                  f"RT: {reaction_time:.0f}ms)")
                
                time.sleep(0.05)  # 20Hz monitoring rate
                
            except Exception as e:
                print(f"MI monitoring error: {e}")
                time.sleep(0.1)
    
    def _simulate_mi_prediction(self):
        """Simulate MI prediction for testing"""
        # Simulate MI detection with some delay
        time_since_cue = time.time() - self.cue_start_time
        
        if time_since_cue < 0.8:  # Too early
            return {'class': -1, 'label': 'Rest', 'confidence': 0.3}
        
        # Simulate prediction based on cue with configured accuracy
        if np.random.random() < self.simulation_accuracy:
            # Correct prediction
            if self.current_cue == 'LEFT':
                return {'class': 0, 'label': 'Left', 'confidence': 0.75 + np.random.random() * 0.2}
            else:
                return {'class': 1, 'label': 'Right', 'confidence': 0.75 + np.random.random() * 0.2}
        else:
            # Incorrect prediction
            if self.current_cue == 'LEFT':
                return {'class': 1, 'label': 'Right', 'confidence': 0.65 + np.random.random() * 0.2}
            else:
                return {'class': 0, 'label': 'Left', 'confidence': 0.65 + np.random.random() * 0.2}
    
    def start_cue(self, direction):
        """
        Start a new MI cue
        
        Args:
            direction: 'LEFT' or 'RIGHT'
        """
        self.current_cue = direction.upper()
        self.cue_start_time = time.time()
        self.in_cue = True
        self.mi_detected = False
        self.last_prediction = None
        
        print(f"MI Cue started: {self.current_cue}")
    
    def end_cue(self):
        """End the current MI cue"""
        if self.in_cue:
            self.in_cue = False
            
            # Record trial result
            if self.last_prediction:
                correct = (self.current_cue == self.last_prediction['label'].upper())
                self.trial_history.append({
                    'cue': self.current_cue,
                    'prediction': self.last_prediction['label'],
                    'correct': correct,
                    'confidence': self.last_prediction['confidence'],
                    'reaction_time': self.reaction_times[-1] if self.reaction_times else None
                })
            else:
                # No MI detected
                self.trial_history.append({
                    'cue': self.current_cue,
                    'prediction': None,
                    'correct': False,
                    'confidence': 0.0,
                    'reaction_time': None
                })
            
            self.current_cue = None
            print("MI Cue ended")
    
    def get_current_action(self):
        """
        Get the current MI action if available
        
        Returns:
            pygame key event (K_LEFT, K_RIGHT) or None
        """
        if self.mi_detected and self.last_prediction:
            if self.last_prediction['label'] == 'Left':
                return pygame.K_LEFT
            elif self.last_prediction['label'] == 'Right':
                return pygame.K_RIGHT
        
        return None
    
    def render_cue(self, screen, font=None):
        """
        Render the MI cue on screen
        
        Args:
            screen: pygame screen surface
            font: pygame font object
        """
        if not self.in_cue or not self.current_cue:
            return
        
        if font is None:
            font = pygame.font.Font(None, 72)
        
        # Calculate remaining time
        time_elapsed = time.time() - self.cue_start_time
        time_remaining = max(0, self.cue_duration - time_elapsed)
        
        # Cue text
        cue_text = f"Think {self.current_cue}"
        text_surface = font.render(cue_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(screen.get_width() // 2, 100))
        
        # Background for visibility
        bg_rect = text_rect.inflate(40, 20)
        pygame.draw.rect(screen, (0, 0, 0), bg_rect)
        pygame.draw.rect(screen, (255, 255, 0), bg_rect, 3)
        
        screen.blit(text_surface, text_rect)
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x = (screen.get_width() - bar_width) // 2
        bar_y = text_rect.bottom + 20
        
        # Background
        pygame.draw.rect(screen, (50, 50, 50), (bar_x, bar_y, bar_width, bar_height))
        
        # Progress
        progress = 1.0 - (time_remaining / self.cue_duration)
        fill_width = int(bar_width * progress)
        
        # Color based on MI detection window
        if self.mi_window_start <= time_elapsed <= self.mi_window_end:
            bar_color = (0, 255, 0) if self.mi_detected else (255, 255, 0)
        else:
            bar_color = (100, 100, 100)
        
        pygame.draw.rect(screen, bar_color, (bar_x, bar_y, fill_width, bar_height))
        
        # Border
        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 2)
        
        # MI detection indicator
        if self.mi_detected and self.last_prediction:
            detect_text = f"Detected: {self.last_prediction['label']} ({self.last_prediction['confidence']:.0%})"
            detect_surface = pygame.font.Font(None, 36).render(detect_text, True, (0, 255, 0))
            detect_rect = detect_surface.get_rect(center=(screen.get_width() // 2, bar_y + 40))
            screen.blit(detect_surface, detect_rect)
    
    def render_feedback(self, screen, font=None):
        """Render MI feedback after cue"""
        if not self.show_feedback:
            return
        
        if font is None:
            font = pygame.font.Font(None, 48)
        
        # Check if feedback period expired
        if time.time() - self.feedback_start_time > self.feedback_duration:
            self.show_feedback = False
            return
        
        # Render feedback based on last trial
        if self.trial_history:
            last_trial = self.trial_history[-1]
            if last_trial['prediction']:
                if last_trial['correct']:
                    text = "Correct!"
                    color = (0, 255, 0)
                else:
                    text = f"Error: Thought {last_trial['prediction']}, Cued {last_trial['cue']}"
                    color = (255, 0, 0)
            else:
                text = "No MI detected"
                color = (255, 255, 0)
            
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect(center=(screen.get_width() // 2, 200))
            
            # Background
            bg_rect = text_rect.inflate(40, 20)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect)
            screen.blit(text_surface, text_rect)
    
    def start_feedback(self):
        """Start showing feedback"""
        self.show_feedback = True
        self.feedback_start_time = time.time()
    
    def get_performance_stats(self):
        """Get MI performance statistics"""
        if not self.trial_history:
            return None
        
        total_trials = len(self.trial_history)
        correct_trials = sum(1 for t in self.trial_history if t['correct'])
        detected_trials = sum(1 for t in self.trial_history if t['prediction'] is not None)
        
        valid_rts = [t['reaction_time'] for t in self.trial_history 
                     if t['reaction_time'] is not None]
        
        stats = {
            'total_trials': total_trials,
            'accuracy': correct_trials / total_trials if total_trials > 0 else 0,
            'detection_rate': detected_trials / total_trials if total_trials > 0 else 0,
            'mean_reaction_time': np.mean(valid_rts) if valid_rts else 0,
            'std_reaction_time': np.std(valid_rts) if valid_rts else 0,
            'mean_confidence': np.mean([t['confidence'] for t in self.trial_history])
        }
        
        return stats
    
    def set_simulation_mode(self, enabled, accuracy=0.7):
        """Enable/disable simulation mode"""
        self.simulation_mode = enabled
        self.simulation_accuracy = accuracy
        print(f"MI simulation mode: {enabled} (accuracy: {accuracy:.0%})")


# Integration function to modify game input handling
def integrate_mi_control(game_instance, mi_controller):
    """
    Integrate MI control into the game
    This function should be called to replace keyboard input with MI
    
    Args:
        game_instance: The game object that needs MI control
        mi_controller: MIController instance
    """
    # Store original key handler
    original_handle_key = getattr(game_instance, 'handle_key_press', None)
    
    def mi_handle_key_press():
        """Replacement key handler that uses MI predictions"""
        # Check for MI action
        mi_action = mi_controller.get_current_action()
        
        if mi_action is not None:
            # Create fake key event
            fake_event = type('Event', (), {
                'type': pygame.KEYDOWN,
                'key': mi_action
            })()
            
            # Call original handler with MI action
            if original_handle_key:
                original_handle_key(fake_event)
            
            return True
        
        return False
    
    # Replace the key handler
    game_instance.handle_mi_input = mi_handle_key_press
    
    print("MI control integrated into game")


if __name__ == "__main__":
    # Test the MI controller
    print("Testing MI Controller...")
    
    # Initialize pygame for testing
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("MI Controller Test")
    clock = pygame.time.Clock()
    
    # Create MI controller
    mi_controller = MIController()
    mi_controller.set_simulation_mode(True, accuracy=0.8)
    mi_controller.start_monitoring()
    
    # Test sequence
    running = True
    test_cues = ['LEFT', 'RIGHT', 'LEFT', 'RIGHT']
    current_cue_index = 0
    cue_active = False
    last_cue_time = time.time()
    
    while running and current_cue_index < len(test_cues):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        current_time = time.time()
        
        # Start new cue every 5 seconds
        if not cue_active and current_time - last_cue_time > 2.0:
            mi_controller.start_cue(test_cues[current_cue_index])
            cue_active = True
            last_cue_time = current_time
        
        # End cue after duration
        if cue_active and current_time - last_cue_time > mi_controller.cue_duration:
            mi_controller.end_cue()
            mi_controller.show_feedback()
            cue_active = False
            current_cue_index += 1
            last_cue_time = current_time
        
        # Clear screen
        screen.fill((50, 50, 50))
        
        # Render MI interface
        mi_controller.render_cue(screen)
        mi_controller.render_feedback(screen)
        
        # Show stats
        stats = mi_controller.get_performance_stats()
        if stats and stats['total_trials'] > 0:
            stats_text = f"Accuracy: {stats['accuracy']:.0%} | Detection: {stats['detection_rate']:.0%}"
            text_surface = pygame.font.Font(None, 24).render(stats_text, True, (255, 255, 255))
            screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)
    
    # Cleanup
    mi_controller.stop_monitoring()
    pygame.quit()
    
    # Final stats
    print("\nFinal Performance:")
    stats = mi_controller.get_performance_stats()
    if stats:
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")