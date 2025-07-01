"""
Cumulative Real-time Control Interface for ErrP Simulator
Maintains history of all commands while processing real-time actions
"""

import pygame
import time
import os
from pylsl import StreamInlet, resolve_streams
import threading
from collections import deque


class ControlInterface:
    """Control interface with cumulative command tracking"""
    
    def __init__(self, mode='keyboard'):
        """
        Initialize control interface
        
        Args:
            mode: 'keyboard', 'mock_eeg', or 'real_eeg'
        """
        self.mode = mode
        self.running = True
        self.current_action = None
        self.action_consumed = False
        self.command_history = []
        self.last_command_count = 0
        
        # Mode-specific initialization
        if mode == 'mock_eeg':
            self._init_mock_eeg()
        elif mode == 'real_eeg':
            self._init_real_eeg()
        
        print(f"Control Interface initialized in {mode} mode")
    
    def _init_mock_eeg(self):
        """Initialize mock EEG with cumulative file control"""
        print("Initializing mock EEG with file control...")
        self.control_file = os.path.join(os.path.dirname(__file__), 'eeg_cumulative_control.txt')
        
        # Create control file if it doesn't exist
        if not os.path.exists(self.control_file):
            with open(self.control_file, 'w') as f:
                f.write("# Cumulative Real-time Control\n")
                f.write("# Each update shows full history: 1, 2, 2, 1\n")
                f.write("# The newest command is the last one\n")
                f.write("")
        
        # Start monitoring thread (same as real_eeg)
        self.monitor_thread = threading.Thread(target=self._monitor_control_file)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"✓ Mock EEG control file: {self.control_file}")
        print("  Each update shows: previous_commands, new_command")
        print("  Example: 1 → 1, 2 → 1, 2, 1 → etc.")
        return True
    
    def _init_real_eeg(self):
        """Initialize cumulative real-time monitoring"""
        self.control_file = os.path.join(os.path.dirname(__file__), 'eeg_cumulative_control.txt')
        
        # Create control file if it doesn't exist
        if not os.path.exists(self.control_file):
            with open(self.control_file, 'w') as f:
                f.write("# Cumulative Real-time Control\n")
                f.write("# Each update shows full history: 1, 2, 2, 1\n")
                f.write("# The newest command is the last one\n")
                f.write("")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_control_file)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"✓ Cumulative real-time control file: {self.control_file}")
        print("  Each update shows: previous_commands, new_command")
        print("  Example: 1 → 1, 2 → 1, 2, 1 → etc.")
    
    def _monitor_control_file(self):
        """Monitor control file for new cumulative updates"""
        last_mtime = 0
        last_content = ""
        
        while self.running:
            try:
                # Check file modification time
                current_mtime = os.path.getmtime(self.control_file)
                
                # Also read content to detect changes even without save
                with open(self.control_file, 'r') as f:
                    content = f.read()
                
                # Check if file changed (by time or content)
                if current_mtime != last_mtime or content != last_content:
                    last_mtime = current_mtime
                    last_content = content
                    
                    # Extract commands from content
                    lines = content.split('\n')
                    command_line = None
                    
                    # Find the last non-empty, non-comment line
                    for line in reversed(lines):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            command_line = line
                            break
                    
                    if command_line:
                        # Parse comma-separated values
                        parts = command_line.replace(',', ' ').split()
                        commands = [p.strip() for p in parts if p.strip() in ['1', '2']]
                        
                        # Check if we have new commands
                        if len(commands) > self.last_command_count:
                            # New command(s) detected
                            new_commands = commands[self.last_command_count:]
                            
                            for cmd in new_commands:
                                # Process the new command
                                if cmd == '1':
                                    self.current_action = pygame.K_LEFT
                                    action_name = "LEFT"
                                else:  # cmd == '2'
                                    self.current_action = pygame.K_RIGHT
                                    action_name = "RIGHT"
                                
                                self.action_consumed = False
                                
                                # Add to history
                                self.command_history.append({
                                    'command': cmd,
                                    'action': action_name,
                                    'timestamp': time.time(),
                                    'detected_at': time.strftime('%H:%M:%S.%f')[:-3],
                                    'cumulative_position': len(self.command_history) + 1
                                })
                                
                                print(f"[{time.strftime('%H:%M:%S')}] New command detected: {action_name}")
                                print(f"                     Cumulative history: {', '.join(commands)}")
                            
                            self.last_command_count = len(commands)
                
            except Exception as e:
                pass  # Silently handle file access errors
            
            time.sleep(0.01)  # Check 100 times per second for faster response
    
    
    def get_action(self, cue_active=False):
        """
        Get current control action
        
        Args:
            cue_active: Whether a cue is currently active
            
        Returns:
            pygame.K_LEFT, pygame.K_RIGHT, or None
        """
        if self.mode == 'keyboard':
            return None
        
        elif self.mode == 'mock_eeg':
            # Mock EEG uses same cumulative control as real_eeg
            if not cue_active:
                # Reset action when cue is not active
                self.current_action = None
                self.action_consumed = False
                return None
            
            # Return current action if not consumed
            if self.current_action and not self.action_consumed:
                self.action_consumed = True
                return self.current_action
            
            return None
        
        elif self.mode == 'real_eeg':
            # Real-time cumulative control
            if not cue_active:
                # Reset action when cue is not active
                self.current_action = None
                self.action_consumed = False
                return None
            
            # Return current action if not consumed
            if self.current_action and not self.action_consumed:
                self.action_consumed = True
                return self.current_action
            
            return None
        
        return None
    
    def reset_for_new_cue(self):
        """Reset action state for a new cue"""
        self.current_action = None
        self.action_consumed = False
    
    def process_keyboard_event(self, event):
        """Process keyboard events (only in keyboard mode)"""
        if self.mode != 'keyboard':
            return None
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                return pygame.K_LEFT
            elif event.key == pygame.K_RIGHT:
                return pygame.K_RIGHT
        
        return None
    
    def get_status(self):
        """Get current status of control interface"""
        status = {
            'mode': self.mode,
            'active': self.running
        }
        
        if self.mode == 'real_eeg':
            status['total_commands'] = len(self.command_history)
            status['command_history'] = self.command_history
            status['cumulative_sequence'] = [h['command'] for h in self.command_history]
        
        return status
    
    def save_history(self, filepath=None):
        """Save command execution history"""
        if self.mode != 'real_eeg':
            return
        
        if filepath is None:
            filepath = self.control_file.replace('.txt', '_history.txt')
        
        try:
            with open(filepath, 'w') as f:
                f.write("# Cumulative Command History\n")
                f.write(f"# Total commands received: {len(self.command_history)}\n")
                f.write("# Position | Command | Action | Detected At\n")
                f.write("#" + "-"*60 + "\n")
                
                for entry in self.command_history:
                    f.write(f"{entry['cumulative_position']:8d} | {entry['command']:7s} | "
                           f"{entry['action']:6s} | {entry['detected_at']}\n")
                
                f.write("\n# Final sequence: " + ', '.join([h['command'] for h in self.command_history]))
            
            print(f"Command history saved to: {filepath}")
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def stop(self):
        """Stop the control interface"""
        self.running = False
        
        if hasattr(self, 'eeg_thread'):
            self.eeg_thread.join(timeout=1.0)
        
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        # Save history for real EEG mode
        if self.mode == 'real_eeg' and self.command_history:
            self.save_history()