"""
Command Writer - Handles writing commands to control file
"""

import os
from threading import Lock


class CommandWriter:
    """Manages command history and file writing"""
    
    def __init__(self, clear_on_init=False):
        """Initialize command writer
        
        Args:
            clear_on_init: If True, clear command history on initialization
        """
        self.command_history = []
        self.command_lock = Lock()
        self.control_file_path = self._get_control_file_path()
        self.command_sent_for_current_cue = False
        
        if clear_on_init:
            # Clear the file for fresh start (important for phase1)
            self.clear_history()
            print("[CommandWriter] Cleared command history for fresh start")
        else:
            # Load existing commands
            self._load_existing_commands()
    
    def _find_project_root(self):
        """Find project root by looking for marker files"""
        current = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        markers = ['README.md', '.git', 'requirements.txt', 'setup.py']
        
        while current != os.path.dirname(current):
            # Check if this is the BCI_Wheelchair_Project directory
            if os.path.basename(current) == 'BCI_Wheelchair_Project':
                return current
            
            # Check for any marker files
            for marker in markers:
                if os.path.exists(os.path.join(current, marker)):
                    return current
            
            # Go up one directory
            current = os.path.dirname(current)
        
        # Fallback
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def _get_control_file_path(self):
        """Get the control file path relative to project root"""
        project_root = self._find_project_root()
        return os.path.join(project_root, 'eeg_cumulative_control.txt')
    
    def _load_existing_commands(self):
        """Load existing commands from file"""
        try:
            with open(self.control_file_path, 'r') as f:
                content = f.read().strip()
                # Skip comment lines
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Parse comma-separated commands
                        commands = [c.strip() for c in line.split(',') if c.strip() in ['1', '2']]
                        self.command_history = commands
                        print(f"Loaded {len(self.command_history)} existing commands")
                        return
            self.command_history = []
        except:
            self.command_history = []
    
    def write_command(self, mi_class):
        """Write command to file
        
        Args:
            mi_class: 'left' or 'right'
            
        Returns:
            bool: True if command was written
        """
        if self.command_sent_for_current_cue:
            print("WARNING: Command already sent for this cue - ignoring")
            return False
        
        command = '1' if mi_class == 'left' else '2'
        
        with self.command_lock:
            self.command_history.append(command)
            
            try:
                with open(self.control_file_path, 'w') as f:
                    f.write(', '.join(self.command_history))
                print(f"Command sent: {command} (Total: {len(self.command_history)})")
                self.command_sent_for_current_cue = True
                return True
            except Exception as e:
                print(f"ERROR: Failed to write command: {e}")
                # Remove the command we just added since write failed
                self.command_history.pop()
                return False
    
    def clear_history(self):
        """Clear command history"""
        with self.command_lock:
            self.command_history = []
            try:
                with open(self.control_file_path, 'w') as f:
                    f.write('')
                print("History cleared")
            except Exception as e:
                print(f"ERROR: Failed to clear history: {e}")
    
    def reset_for_new_cue(self):
        """Reset state for new cue"""
        self.command_sent_for_current_cue = False
    
    def get_stats(self):
        """Get command statistics"""
        with self.command_lock:
            left_count = self.command_history.count('1')
            right_count = self.command_history.count('2')
            total = len(self.command_history)
            
        return {
            'total': total,
            'left': left_count,
            'right': right_count,
            'left_percent': (left_count / total * 100) if total > 0 else 0,
            'right_percent': (right_count / total * 100) if total > 0 else 0
        }