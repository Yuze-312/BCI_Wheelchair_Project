"""
MI-Integrated Subway Surfers EEG Simulator
Enhanced version with real-time Motor Imagery control replacing keyboard input

Key changes:
- Integrates MI decoder for movement control
- Word cues trigger MI detection windows
- Maintains all ErrP functionality
- Can switch between keyboard and MI modes

Usage:
    python run_simulation_mi.py --mi-mode          # Use MI control
    python run_simulation_mi.py --keyboard-mode    # Use keyboard (default)
    python run_simulation_mi.py --simulation       # Simulate MI (no real EEG needed)
"""

import sys
import os
from pylsl import StreamInlet, resolve_streams, StreamInfo, StreamOutlet, local_clock
# Import the original simulator components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_simulation_2 import *  # Import everything from original (including hand images)

from mi_integration import MIController
# Additional imports
import argparse
import threading
import csv
import random

# Import control interface from parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import numeric markers
from simple_markers import *
# Use cumulative version for real-time with history
from control_interface_cumulative import ControlInterface

# Screen dimensions (from run_simulation_2.py main())
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900

# Additional constants needed
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Error rates - disabled for testing
# Removed error rate - simulator executes commands exactly as received

# Override the string-based LSL stream from run_simulation_2 with numeric markers
info = StreamInfo("Outlet_Info", "Markers", 1, 0, "int32", "subway-errp-001")
outlet = StreamOutlet(info)

# Override push_immediate to use numeric markers
def push_immediate(marker):
    """Push numeric marker immediately with LSL timestamp"""
    print(marker)
    outlet.push_sample([marker])
    precision_tracker.mark_event(marker)
    print(f"[LSL] Sent marker: {marker} at {get_sync_time():.3f}")

# Aliases for compatibility
World = EndlessWorld

# Simple event logger for MI integration (compatible with OptimizedEventLogger)
class UnifiedEventLogger:
    """CSV event logger for MI integration - uses numeric markers matching LSL"""
    
    # Numeric encoding mappings
    DIRECTION_MAP = {'left': 0, 'right': 1}
    ERROR_TYPE_MAP = {'': 0, 'primary': 1, 'secondary': 2, 'combined': 3}
    
    def __init__(self, filepath, difficulty=None, mi_mode=False):
        self.filepath = filepath
        self.difficulty = difficulty
        self.mi_mode = mi_mode
        self.events = []
        self.session_start_time = get_sync_time()
        self.current_trial = None
        self.trial_start_time = None
        self.cue_onset_time = None
        self.current_cue = None  # Track current cue for accuracy calculation
        self.current_gt = None  # Ground truth (0=left, 1=right)
        
        # Create log directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write header - event_type is now numeric
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # New format: timestamp, lsl_timestamp, relative_time, trial, event, classifier_out, confidence, gt
            writer.writerow(['timestamp', 'lsl_timestamp', 'rel_time', 'trial', 'event', 'classifier_out', 'confidence', 'gt'])
    
    def _add_event(self, timestamp, event_type, event_name='', trial_id='', cue_class='', 
                   predicted_class='', accuracy='', error_type='', confidence='',
                   reaction_time='', coins_collected='', artifact_flag='', details=''):
        """Add event to log with new format"""
        # Calculate relative time from cue onset
        rel_time = ''
        if self.cue_onset_time and event_type in [MARKER_RESPONSE_CORRECT, MARKER_RESPONSE_ERROR]:
            rel_time = f"{(timestamp - (self.cue_onset_time - self.session_start_time)):.3f}"
        elif event_type in [MARKER_CUE_LEFT, MARKER_CUE_RIGHT]:
            rel_time = '0.000'  # Cue onset is time zero
        
        # Classifier output and confidence (only for response events)
        classifier_out = ''
        conf = ''
        if event_type in [MARKER_RESPONSE_CORRECT, MARKER_RESPONSE_ERROR]:
            classifier_out = predicted_class  # 0=left, 1=right
            # Scale confidence 0-100% to 0-9
            if confidence:
                conf = str(min(9, int(float(confidence) * 10)))
        
        # Ground truth (current cue)
        gt = ''
        if event_type in [MARKER_CUE_LEFT, MARKER_CUE_RIGHT]:
            gt = '0' if event_type == MARKER_CUE_LEFT else '1'
            self.current_gt = gt  # Store for response events
        elif event_type in [MARKER_RESPONSE_CORRECT, MARKER_RESPONSE_ERROR]:
            gt = self.current_gt if self.current_gt else ''
        
        # Get LSL timestamp for alignment
        lsl_time = local_clock()
        
        event = {
            'timestamp': f"{timestamp:.3f}",
            'lsl_timestamp': f"{lsl_time:.6f}",
            'rel_time': rel_time,
            'trial': trial_id,
            'event': str(event_type),
            'classifier_out': classifier_out,
            'confidence': conf,
            'gt': gt
        }
        self.events.append(event)
        self._write_event(event)
    
    def start_trial(self, trial_num, difficulty):
        """Start a new trial"""
        self.current_trial = trial_num
        self.trial_start_time = get_sync_time()
        timestamp = self.trial_start_time - self.session_start_time
        # Convert difficulty to numeric (0=easy, 1=medium, 2=hard)
        diff_map = {'easy': 0, 'medium': 1, 'hard': 2}
        diff_numeric = diff_map.get(difficulty, 0)
        self._add_event(timestamp, MARKER_TRIAL_START, 'TRIAL_START', trial_id=str(trial_num), 
                       details=str(diff_numeric))
    
    def log_trial_start(self, trial_num, difficulty, is_practice):
        """Log trial start (compatibility method)"""
        self.start_trial(trial_num, difficulty)
        if is_practice:
            self.log_event(MARKER_TRIAL_START, 'PRACTICE_TRIAL', str(trial_num))
    
    def log_cue(self, cue_word, agent_position, coins_collected):
        """Log cue presentation"""
        self.cue_onset_time = get_sync_time()
        self.current_cue = cue_word.upper()  # Store current cue
        timestamp = self.cue_onset_time - self.session_start_time
        self.current_gt = '0' if cue_word.upper() == 'LEFT' else '1'  # Set ground truth
        marker = MARKER_CUE_LEFT if cue_word.upper() == 'LEFT' else MARKER_CUE_RIGHT
        # Convert cue_class to numeric (0=left, 1=right)
        cue_class_numeric = self.DIRECTION_MAP.get(cue_word.lower(), '')
        self._add_event(timestamp, marker, f'CUE_{cue_word.upper()}',
                       trial_id=str(self.current_trial),
                       cue_class=str(cue_class_numeric),
                       coins_collected=str(coins_collected),
                       details=str(agent_position))  # Just the position number
    
    def log_action(self, user_input, executed_action, agent_position, coins_collected, movement_required, confidence=0.0):
        """Log action execution with confidence"""
        timestamp = get_sync_time() - self.session_start_time
        
        # Check if there was error injection (primary error)
        is_error_injection = user_input != executed_action and user_input != "NONE"
        
        # Calculate reaction time if cue was shown
        reaction_time = ''
        if self.cue_onset_time:
            reaction_time = str(int((get_sync_time() - self.cue_onset_time) * 1000))
        
        # Determine predicted class from user's intended action (0=left, 1=right)
        predicted_class = ''
        if 'LEFT' in user_input:
            predicted_class = '0'
        elif 'RIGHT' in user_input:
            predicted_class = '1'
        
        # Check if user's intended action matches the cue
        action_correct = False
        if self.current_cue:
            if self.current_cue == 'LEFT' and 'LEFT' in user_input:
                action_correct = True
            elif self.current_cue == 'RIGHT' and 'RIGHT' in user_input:
                action_correct = True
        
        # Accuracy is whether the user's intended action matched the cue (0=false, 1=true)
        accuracy = '1' if action_correct else '0'
        
        # Determine error type (0=none, 1=primary, 2=secondary, 3=combined)
        error_type_str = ''
        if is_error_injection and not action_correct:
            error_type_str = 'combined'  # Both primary (injection) and secondary (wrong action)
        elif is_error_injection:
            error_type_str = 'primary'   # Only error injection
        elif not action_correct and user_input != "NONE":
            error_type_str = 'secondary' # User made wrong choice
        
        error_type = str(self.ERROR_TYPE_MAP.get(error_type_str, 0))
        
        # Use the already-sent marker value for consistency
        marker = MARKER_RESPONSE_ERROR if not action_correct else MARKER_RESPONSE_CORRECT
        event_name = 'RESPONSE_ERROR' if not action_correct else 'RESPONSE_CORRECT'
        self._add_event(timestamp, marker, event_name,
                       trial_id=str(self.current_trial),
                       predicted_class=predicted_class,
                       accuracy=accuracy,
                       error_type=error_type,
                       confidence=str(confidence),
                       reaction_time=reaction_time,
                       coins_collected=str(coins_collected),
                       details='')  # Details not needed with numeric encoding
        
        if is_error_injection:
            # Log ErrP event 300ms later
            threading.Timer(0.3, lambda: self._add_event(
                get_sync_time() - self.session_start_time, MARKER_RESPONSE_ERROR, 'PRIMARY_ERRP',
                trial_id=str(self.current_trial),
                error_type='1',  # primary error
                coins_collected=str(coins_collected)
            )).start()
        
        # Clear current cue after action
        self.current_cue = None
    
    def log_coin_event(self, event_type, coin_world_lane, agent_position, coins_collected):
        """Log coin-related events - DISABLED for cleaner logs"""
        # Coin events are not needed for EEG analysis
        pass
    
    def log_cue_timeout(self, agent_position, coins_collected):
        """Log cue timeout - DISABLED for cleaner logs"""
        # Clear current cue on timeout
        self.current_cue = None
        # Timeout events are not needed for simplified marker system
        pass
    
    def log_trial_complete(self, coins_collected):
        """Log trial completion"""
        timestamp = get_sync_time() - self.session_start_time
        self._add_event(timestamp, MARKER_TRIAL_START, 'TRIAL_END',
                       trial_id=str(self.current_trial),
                       coins_collected=str(coins_collected))
        # Clear current cue at trial end
        self.current_cue = None
    
    def log_event(self, marker, event_name, details=""):
        """Log a general event (compatibility method)"""
        # Only log essential markers for simplified system
        essential_markers = [MARKER_CUE_LEFT, MARKER_CUE_RIGHT, 
                           MARKER_RESPONSE_CORRECT, MARKER_RESPONSE_ERROR,
                           MARKER_TRIAL_START]
        
        if marker in essential_markers:
            timestamp = get_sync_time() - self.session_start_time
            self._add_event(timestamp, marker, event_name,
                           trial_id=str(self.current_trial) if self.current_trial else '',
                           details='')  # Simplified - no details needed
    
    def save(self):
        """Save is automatic with each event, but kept for compatibility"""
        pass
    
    def _write_event(self, event):
        """Write event to CSV"""
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['timestamp', 'lsl_timestamp', 'rel_time', 'trial', 'event', 'classifier_out', 'confidence', 'gt'])
            writer.writerow(event)


# Override WordCue class to use numeric markers
class WordCue:
    """Visual word cue system with numeric markers for EEG"""
    def __init__(self):
        self.active = False
        self.cue_word = ""
        self.start_time = 0
        self.think_duration = 0
        self.flash_timer = 0
        # For EEG alignment: track both real and game time
        self.real_start_time = 0
        self.game_time_elapsed = 0
        
    def show_cue(self, word, duration):
        """Display word cue - duration is in REAL TIME for consistent EEG epochs"""
        self.active = True
        self.cue_word = word
        self.start_time = get_sync_time()
        self.real_start_time = self.start_time  # Store for reaction time calculation
        self.think_duration = duration  # This is now REAL TIME duration
        self.flash_timer = 0
        self.game_time_elapsed = 0
        
        # Send numeric marker based on cue direction
        if word == "LEFT":
            push_immediate(MARKER_CUE_LEFT)
        else:
            push_immediate(MARKER_CUE_RIGHT)
        
    def update(self, speed_factor=1.0):
        """Update cue state - cue duration based on REAL time, not game time"""
        if self.active:
            self.flash_timer += 1
            
            # Track game time for visual effects
            frame_time = 1.0 / FPS
            self.game_time_elapsed += frame_time * speed_factor
            
            # Check if REAL TIME duration has elapsed
            real_time_elapsed = get_sync_time() - self.start_time
            
            if real_time_elapsed >= self.think_duration:
                self.active = False
                return True
        return False
    
    def get_remaining_time(self):
        """Get remaining think time in REAL TIME"""
        if self.active:
            real_time_elapsed = get_sync_time() - self.start_time
            return max(0, self.think_duration - real_time_elapsed)
        return 0
    
    def get_reaction_time(self, response_timestamp):
        """Calculate reaction time from cue onset in milliseconds"""
        if self.real_start_time > 0:
            return (response_timestamp - self.real_start_time) * 1000
        return None


def run_mi_integrated_game(control_mode='keyboard'):
    """
    Run the game with different control modes
    
    Args:
        control_mode: 'keyboard', 'mock_eeg', or 'real_eeg'
    """
    # Initialize pygame
    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    
    # Load hand images
    HandImages.load()
    
    # Select experiment settings
    print("\n" + "="*60)
    print("SUBWAY SURFERS EEG SIMULATOR")
    print("="*60)
    print(f"Control Mode: {control_mode.upper()}")
    
    if control_mode == 'keyboard':
        print("Using: Keyboard arrow keys")
    elif control_mode == 'mock_eeg':
        print("Using: Mock EEG stream (run mock_eeg_simple.py)")
    elif control_mode == 'real_eeg':
        print("Using: Real EEG file control (eeg_cumulative_control.txt)")
    
    print("="*60 + "\n")
    
    # Get experiment settings from user
    settings = select_experiment_settings()
    
    # Extract settings
    difficulty = settings['difficulty']
    total_trials = settings['trials']
    # For quick mode (5 trials), skip practice trials
    if total_trials == 5:
        practice_trials = 0
    else:
        practice_trials = settings.get('practice', PRACTICE_TRIALS)
    
    # Initialize control interface
    print("\nInitializing control interface...")
    control_interface = ControlInterface(control_mode)
    
    # Initialize MI controller only for mock_eeg mode
    mi_controller = None
    if control_mode == 'mock_eeg':
        print("Initializing Motor Imagery controller...")
        mi_controller = MIController()
        
        # Connect to mock EEG stream
        print("Connecting to mock EEG stream...")
        success = mi_controller.decoder.connect_lsl("MockEEG")
        if not success:
            print("Failed to connect to mock EEG stream!")
            print("Please run mock_eeg_simple.py in another terminal.")
            return
        print("Connected to mock EEG stream")
        
        # Start MI monitoring
        mi_controller.start_monitoring()
        print("MI monitoring started\n")
    
    # Run the main game with control integration
    main_with_mi(difficulty, total_trials, practice_trials, mi_controller, control_interface, control_mode)
    
    # Cleanup
    if control_interface:
        control_interface.stop()
    
    if mi_controller:
        mi_controller.stop_monitoring()
        if hasattr(mi_controller.decoder, 'stop'):
            mi_controller.decoder.stop()
        
        # Show MI performance stats
        print("\n" + "="*60)
        print("MI PERFORMANCE SUMMARY")
        print("="*60)
        stats = mi_controller.get_performance_stats()
        if stats:
            print(f"Total trials: {stats['total_trials']}")
            print(f"MI accuracy: {stats['accuracy']:.1%}")
            print(f"Detection rate: {stats['detection_rate']:.1%}")
            print(f"Mean reaction time: {stats['mean_reaction_time']:.0f}ms")
            print(f"Mean confidence: {stats['mean_confidence']:.1%}")
        print("="*60)


def main_with_mi(difficulty, total_trials, practice_trials, mi_controller=None, control_interface=None, control_mode='keyboard'):
    """
    Modified main function with different control modes
    Based on the original main() but with flexible control added
    """
    # Get the original main function's code with MI modifications
    # Most of this is identical to the original main() in run_simulation_2.py
    
    # Initialize screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Subway Surfers ErrP Simulator - MI Mode" if mi_controller else "Subway Surfers ErrP Simulator")
    clock = pygame.time.Clock()
    
    # Initialize global timing
    global sync_start_time
    sync_start_time = local_clock()
    verify_sync()
    
    # Create unified event logger
    session_id = int(time.time())
    
    # Always save to simulation/logs directory
    log_path = os.path.join(LOG_DIR, f"subway_errp_{session_id}.csv")
    
    errp_logger = UnifiedEventLogger(log_path, difficulty, mi_mode=mi_controller is not None)
    
    print(f"\nLogging to: {log_path}")
    
    
    # First run practice trials if any
    all_trials = practice_trials + total_trials
    
    # Track if we've entered main trials
    in_main_trials = (practice_trials == 0)  # Start as main trials if no practice
    
    # Run all trials
    for trial_num in range(1, all_trials + 1):
        # Check if transitioning from practice to main  
        if trial_num == practice_trials + 1:
            in_main_trials = True
            # Get target coins based on difficulty
            target_coins = {
                'easy': 3,
                'medium': 5,
                'hard': 7,
                'expert': 10
            }.get(difficulty, 5)
            # Show practice complete message only if there were practice trials
            if practice_trials > 0:
                # For the first main trial, show 1/total_trials
                show_trial_transition(screen, 1, total_trials, difficulty, target_coins, is_practice=False)
        
        # Update trial context for logging
        is_practice = trial_num <= practice_trials
        errp_logger.log_trial_start(trial_num, difficulty, is_practice)
        
        # Trial announcement
        if is_practice:
            print(f"\nPractice Trial {trial_num}/{practice_trials}")
        else:
            main_trial_num = trial_num - practice_trials
            print(f"\nTrial {main_trial_num}/{total_trials}")
        
        # Show trial transition screen for each trial
        if is_practice:
            show_trial_transition(screen, trial_num, practice_trials, difficulty, get_difficulty_params(difficulty)['target_coins'], is_practice=True)
        elif trial_num != practice_trials + 1:  # Don't show again for first main trial (already shown above)
            main_trial_num = trial_num - practice_trials
            show_trial_transition(screen, main_trial_num, total_trials, difficulty, get_difficulty_params(difficulty)['target_coins'], is_practice=False)
        
        # Get trial parameters
        params = get_difficulty_params(difficulty).copy()
        # No error injection - simulator executes commands exactly as received
        if DEBUG_MODE:
            print(f"[DEBUG] Phase1 mode - executing commands exactly as received")
        
        # Initialize trial components
        world = World(SCREEN_WIDTH, SCREEN_HEIGHT)
        agent = Agent(world)
        coin_manager = CoinManager(world, params['target_coins'], logger=errp_logger)
        coin_manager.min_spawn_interval = params['spawn_interval'][0]
        coin_manager.max_spawn_interval = params['spawn_interval'][1]
        word_cue = WordCue()
        
        # Trial state
        trial_running = True
        action_count = 0
        last_action_info = None
        last_cue_time = None
        cue_cooldown = 60  # Reduced initial cooldown
        current_speed_factor = 1.0
        last_cued_coin = None  # Track which coin we last showed a cue for
        
        # Define think time for this trial - fixed 5s for consistent EEG epochs
        think_time = 5.0  # Fixed 5s window for all cues
        
        # Send initial trial separator before first cue
        push_immediate(MARKER_TRIAL_START)
        
        # Main trial loop
        while trial_running:
            precision_tracker.mark_frame()
            frame_start_time = get_sync_time()
            
            # Update speed factor during cues
            if word_cue.active:
                # Use dynamic speed factor if available
                if hasattr(word_cue, 'dynamic_speed_factor'):
                    current_speed_factor = word_cue.dynamic_speed_factor
                else:
                    current_speed_factor = SLOW_MOTION_FACTOR
            else:
                current_speed_factor = 1.0
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    errp_logger.save()
                    pygame.quit()
                    exit()
                
                # Process keyboard input through control interface
                if control_interface and control_interface.mode == 'keyboard':
                    if event.type == pygame.KEYDOWN and event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                        if word_cue.active:
                            key_name = "LEFT" if event.key == pygame.K_LEFT else "RIGHT"
                            cue_match = key_name == word_cue.cue_word
                            user_intended = f"MOVE_{key_name}"
                            
                            # Execute exactly as intended - no error injection
                            executed = user_intended
                            error_injected = False
                            
                            # Send response marker based on whether action matches cue
                            if not cue_match:
                                push_immediate(MARKER_RESPONSE_ERROR)
                            else:
                                push_immediate(MARKER_RESPONSE_CORRECT)
                            
                            # Execute movement
                            if executed == "MOVE_LEFT":
                                agent.move_left()
                            elif executed == "MOVE_RIGHT":
                                agent.move_right()
                            
                            # Track action info
                            last_action_info = {
                                'cue': word_cue.cue_word,
                                'user_input': user_intended,
                                'executed': executed,
                                'error_injected': error_injected,
                                'user_intent_followed': not error_injected
                            }
                            
                            # Determine movement required
                            movement_required = "NONE"
                            if coin_manager.active_coin:
                                coin = coin_manager.active_coin
                                relative_lane = coin.get_relative_lane(agent.get_world_position())
                                if relative_lane < 0:
                                    movement_required = "LEFT"
                                elif relative_lane > 0:
                                    movement_required = "RIGHT"
                            
                            # Log action
                            # No MI confidence for keyboard
                            errp_logger.log_action(user_intended, executed, agent.get_world_position(), 
                                                 agent.coins_collected, movement_required, confidence=1.0)
                            
                            # Clear cue
                            word_cue.active = False
                            
                            # Send trial separator after action
                            push_immediate(MARKER_TRIAL_START)
                            
                            action_count += 1
                            cue_cooldown = 60
            
            # Non-keyboard control during cue window
            if control_interface and control_interface.mode != 'keyboard' and word_cue.active:
                # Check for control action
                control_action = None
                
                if control_interface.mode == 'real_eeg':
                    # Direct file-based control
                    control_action = control_interface.get_action(cue_active=True)
                elif control_interface.mode == 'mock_eeg':
                    # Mock EEG also uses file-based control
                    control_action = control_interface.get_action(cue_active=True)
                
                if control_action:
                    # Process control action similar to keyboard
                    key_name = "LEFT" if control_action == pygame.K_LEFT else "RIGHT"
                    user_intended = f"MOVE_{key_name}"
                    
                    # Execute command exactly as received - no error injection
                    print(f"[ACTION] Processing command: {key_name}")
                    executed = user_intended
                    error_injected = False
                    if DEBUG_MODE:
                        print(f"[DEBUG] Executing command exactly as received: {executed}")
                    
                    # Send response marker based on whether action matches cue
                    cue_match = key_name == word_cue.cue_word
                    if not cue_match:
                        push_immediate(MARKER_RESPONSE_ERROR)
                    else:
                        push_immediate(MARKER_RESPONSE_CORRECT)
                    
                    # Execute movement
                    if executed == "MOVE_LEFT":
                        success = agent.move_left()
                        print(f"[MOVE] LEFT command - success: {success}, position: {agent.get_world_position()}")
                        if DEBUG_MODE:
                            print(f"[DEBUG] World shifting: {agent.world.shifting}")
                    elif executed == "MOVE_RIGHT":
                        success = agent.move_right()
                        print(f"[MOVE] RIGHT command - success: {success}, position: {agent.get_world_position()}")
                        if DEBUG_MODE:
                            print(f"[DEBUG] World shifting: {agent.world.shifting}")
                    
                    # Get confidence from MI controller
                    mi_confidence = 0.0
                    if mi_controller and hasattr(mi_controller, 'last_prediction') and mi_controller.last_prediction:
                        mi_confidence = mi_controller.last_prediction.get('confidence', 0.0)
                    
                    # Track action info
                    last_action_info = {
                        'cue': word_cue.cue_word,
                        'user_input': user_intended,
                        'executed': executed,
                        'error_injected': error_injected,
                        'user_intent_followed': not error_injected,
                        'mi_detected': True,
                        'confidence': mi_confidence
                    }
                    
                    # Determine movement required
                    movement_required = "NONE"
                    if coin_manager.active_coin:
                        coin = coin_manager.active_coin
                        relative_lane = coin.get_relative_lane(agent.get_world_position())
                        if relative_lane < 0:
                            movement_required = "LEFT"
                        elif relative_lane > 0:
                            movement_required = "RIGHT"
                    
                    # Log action
                    errp_logger.log_action(user_intended, executed, agent.get_world_position(), 
                                         agent.coins_collected, movement_required, confidence=mi_confidence)
                    
                    # Clear cue
                    word_cue.active = False
                    
                    # Send trial separator after action
                    push_immediate(MARKER_TRIAL_START)
                    
                    # End MI cue
                    if mi_controller and control_interface.mode == 'mock_eeg':
                        mi_controller.end_cue()
                        mi_controller.start_feedback()  # Show MI feedback
                    
                    action_count += 1
                    cue_cooldown = 60
            
            # Update cue state
            cue_ended = word_cue.update(current_speed_factor)
            
            # Check for cue timeout
            if cue_ended and action_count == 0:
                # Timeout - send trial separator
                push_immediate(MARKER_TRIAL_START)
                errp_logger.log_cue_timeout(agent.get_world_position(), agent.coins_collected)
                
                if mi_controller and control_interface.mode == 'mock_eeg':
                    mi_controller.end_cue()
                    mi_controller.start_feedback()  # Call the method instead of setting attribute
                
                cue_cooldown = 120
                action_count = 0
            
            # Update world with speed factor
            world.update(FORWARD_SPEED * current_speed_factor)
            coin_manager.update(agent, current_speed_factor)
            
            # Clear last_cued_coin if it's no longer active
            if last_cued_coin and (last_cued_coin.collected or not coin_manager.active_coin or coin_manager.active_coin != last_cued_coin):
                last_cued_coin = None
            
            # Update cooldowns
            if cue_cooldown > 0:
                cue_cooldown -= current_speed_factor
            
            # Spawn new cue if needed
            if not word_cue.active and cue_cooldown <= 0:
                upcoming_coins = coin_manager.get_upcoming_coins(agent)
                
                # Debug: print coin status
                if not hasattr(main_with_mi, '_last_coin_debug') or time.time() - main_with_mi._last_coin_debug > 2.0:
                    main_with_mi._last_coin_debug = time.time()
                    active_coin = coin_manager.active_coin if hasattr(coin_manager, 'active_coin') else None
                    if active_coin:
                        print(f"[DEBUG] Active coin: Lane={active_coin.world_lane}, Y={active_coin.y_position:.0f}, Agent={agent.get_world_position()}")
                    else:
                        print(f"[DEBUG] No active coin, cooldown={coin_manager.spawn_cooldown:.0f}")
                
                # Check for active coin directly instead of relying on get_upcoming_coins
                active_coin = coin_manager.active_coin if hasattr(coin_manager, 'active_coin') else None
                
                if active_coin and not active_coin.collected:
                    target_coin = active_coin
                    
                    # Skip if this is the same coin we just showed a cue for
                    if last_cued_coin and target_coin == last_cued_coin:
                        # Don't show another cue for the same coin
                        cue_cooldown = 10  # Short cooldown to check for new coins
                    else:
                        # Check if coin is visible to agent
                        if target_coin.is_visible(agent.get_world_position()):
                            # Calculate minimum safe distance for 4s decision time
                            # At slow motion (1 pixel/frame), need 240 pixels for 4 seconds
                            # Add safety margin: 300 pixels minimum
                            distance_to_agent = agent.y - target_coin.y_position
                            
                            # Add randomization to prevent pattern learning
                            # Reduced minimum to ensure cues can appear
                            min_safe_distance = 250 + random.randint(0, 100)  # 250-350 pixels
                            
                            # Always show cue when coin is visible, adjust speed if needed
                            agent_pos = agent.get_world_position()
                            
                            if target_coin.world_lane < agent_pos:
                                cue_word = "LEFT"
                            elif target_coin.world_lane > agent_pos:
                                cue_word = "RIGHT"
                            else:
                                cue_word = random.choice(["LEFT", "RIGHT"])
                            
                            
                            # Adjust slow motion factor based on distance
                            if distance_to_agent < 250:
                                # Very close - need extra slow motion
                                DYNAMIC_SLOW_FACTOR = 0.1  # 10% speed
                                if DEBUG_MODE:
                                    print(f"[CUE-CLOSE] Extra slow motion! Distance={distance_to_agent:.0f}")
                            else:
                                # Normal slow motion
                                DYNAMIC_SLOW_FACTOR = SLOW_MOTION_FACTOR  # 20% speed
                            
                            word_cue.show_cue(cue_word, think_time)
                            errp_logger.log_cue(cue_word, agent.get_world_position(), agent.coins_collected)
                            last_cue_time = get_sync_time()
                            
                            # Remember this coin
                            last_cued_coin = target_coin
                            action_count = 0  # Reset action count for new cue
                            
                            # Store dynamic speed for this cue
                            word_cue.dynamic_speed_factor = DYNAMIC_SLOW_FACTOR
                            
                            if DEBUG_MODE:
                                print(f"[CUE] Coin at y={target_coin.y_position:.0f}, distance={distance_to_agent:.0f} pixels, speed={DYNAMIC_SLOW_FACTOR}")
                                
                            # Reset control interface for new cue
                            if control_interface and control_interface.mode == 'real_eeg':
                                control_interface.reset_for_new_cue()
                                print(f"[DEBUG] Reset control interface for new cue")
                            
                            # Start MI cue if using MI control
                            if mi_controller and control_interface.mode == 'mock_eeg':
                                mi_controller.start_cue(cue_word)
                            
                            cue_cooldown = 30
                        else:
                            # Coin not visible yet, check again soon
                            if DEBUG_MODE and not hasattr(main_with_mi, '_last_vis_debug') or time.time() - main_with_mi._last_vis_debug > 1.0:
                                main_with_mi._last_vis_debug = time.time()
                                print(f"[SKIP] Coin not visible: Lane={target_coin.world_lane}, Agent={agent.get_world_position()}")
                            cue_cooldown = 5
            
            # Draw everything
            draw_endless_world(screen, world, agent, coin_manager.coins, word_cue)
            # For HUD display, use correct trial numbers
            if is_practice:
                # During practice trials, show X/practice_trials
                draw_hud(screen, trial_num, practice_trials, agent, params['target_coins'], difficulty, last_action_info, is_practice=True)
            else:
                # During main trials, show only main trial numbers (not including practice)
                main_trial_num = trial_num - practice_trials
                draw_hud(screen, main_trial_num, total_trials, agent, params['target_coins'], difficulty, last_action_info, is_practice=False)
            
            # Draw MI-specific UI elements
            if mi_controller and control_interface.mode == 'mock_eeg':
                mi_controller.render_cue(screen)
                mi_controller.render_feedback(screen)
            
            # Check trial completion
            if agent.coins_collected >= params['target_coins']:
                completion_start = time.perf_counter()
                errp_logger.log_trial_complete(agent.coins_collected)
                # Skip trial complete screen - go directly to next trial transition
                trial_running = False
                completion_end = time.perf_counter()
                if DEBUG_MODE:
                    print(f"Trial completion took {(completion_end - completion_start)*1000:.1f}ms")
            
            # Frame timing
            verify_sync()
            frame_end_time = get_sync_time()
            pygame.display.flip()
            
            clock.tick(FPS)
            actual_fps = clock.get_fps()
            if actual_fps > 0 and actual_fps < FPS * 0.9:
                print(f"WARNING: FPS dropped to {actual_fps:.1f}")
        
        # Skip inter-trial interval - go directly to next trial's transition screen
    
    # Experiment complete
    errp_logger.save()
    pygame.quit()
    
    # Show timing statistics
    timing_stats = precision_tracker.get_stats()
    
    print(f"\nExperiment complete!")
    print(f"Event log saved to: {log_path}")
    print(f"Total trials: {total_trials}")
    print(f"Difficulty: {difficulty}")
    
    if timing_stats:
        print(f"\nTiming Precision Statistics:")
        print(f"  Mean frame interval: {timing_stats['mean_ms']:.2f}ms")
        print(f"  Std deviation: {timing_stats['std_ms']:.2f}ms")
        print(f"  Max frame interval: {timing_stats['max_ms']:.2f}ms")
        print(f"  Target: 16.67ms (60 FPS)")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Subway Surfers ErrP Simulator with Multiple Control Modes')
    parser.add_argument('--mode', choices=['keyboard', 'mock_eeg', 'real_eeg'], 
                        default='keyboard',
                        help='Control mode: keyboard (default), mock_eeg, or real_eeg')
    
    args = parser.parse_args()
    
    # Run the game with specified control mode
    run_mi_integrated_game(control_mode=args.mode)