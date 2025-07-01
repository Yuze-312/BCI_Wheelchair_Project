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

# Import the original simulator components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_simulation_2 import *  # Import everything from original

# Import MI integration
from mi_integration import MIController

# Additional imports
import argparse
import threading
import csv
import random

# Import control interface
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Use cumulative version for real-time with history
from control_interface_cumulative import ControlInterface

# Screen dimensions (from run_simulation_2.py main())
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 900

# Additional constants needed
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Error rates
ERROR_RATE = ERROR_P  # Use ERROR_P from run_simulation_2
PRACTICE_ERROR_RATE = ERROR_P * 0.3  # 9% for practice vs 30% for main

# Aliases for compatibility
World = EndlessWorld

# Note: OptimizedEventLogger is defined inside main(), so we'll handle it differently

# Simple event logger for MI integration (compatible with OptimizedEventLogger)
class UnifiedEventLogger:
    """CSV event logger for MI integration - compatible with OptimizedEventLogger"""
    
    def __init__(self, filepath, difficulty=None, mi_mode=False):
        self.filepath = filepath
        self.difficulty = difficulty
        self.mi_mode = mi_mode
        self.events = []
        self.session_start_time = get_sync_time()
        self.current_trial = None
        self.trial_start_time = None
        self.cue_onset_time = None
        
        # Create log directory if needed
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write header
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'event_type', 'trial_id', 'cue_class', 
                'predicted_class', 'accuracy', 'error_type', 'confidence',
                'reaction_time', 'coins_collected', 'artifact_flag', 'details'
            ])
    
    def _add_event(self, timestamp, event_type, trial_id='', cue_class='', 
                   predicted_class='', accuracy='', error_type='', confidence='',
                   reaction_time='', coins_collected='', artifact_flag='', details=''):
        """Add event to log (internal method)"""
        event = {
            'timestamp': timestamp,
            'event_type': event_type,
            'trial_id': trial_id,
            'cue_class': cue_class,
            'predicted_class': predicted_class,
            'accuracy': accuracy,
            'error_type': error_type,
            'confidence': confidence,
            'reaction_time': reaction_time,
            'coins_collected': coins_collected,
            'artifact_flag': artifact_flag,
            'details': details
        }
        self.events.append(event)
        self._write_event(event)
    
    def start_trial(self, trial_num, difficulty):
        """Start a new trial"""
        self.current_trial = trial_num
        self.trial_start_time = get_sync_time()
        timestamp = self.trial_start_time - self.session_start_time
        self._add_event(timestamp, 'trial_start', trial_id=str(trial_num), 
                       details=f'difficulty={difficulty}')
    
    def log_trial_start(self, trial_num, difficulty, is_practice):
        """Log trial start (compatibility method)"""
        self.start_trial(trial_num, difficulty)
        if is_practice:
            self.log_event('practice_trial', f'trial={trial_num}')
    
    def log_cue(self, cue_word, agent_position, coins_collected):
        """Log cue presentation"""
        self.cue_onset_time = get_sync_time()
        timestamp = self.cue_onset_time - self.session_start_time
        self._add_event(timestamp, f'cue_{cue_word.lower()}', 
                       trial_id=str(self.current_trial),
                       cue_class=cue_word.lower(),
                       coins_collected=str(coins_collected),
                       details=f'agent_pos={agent_position}')
    
    def log_action(self, user_input, executed_action, agent_position, coins_collected, movement_required):
        """Log action execution"""
        timestamp = get_sync_time() - self.session_start_time
        is_error = user_input != executed_action
        accuracy = 'False' if is_error else 'True'
        
        # Calculate reaction time if cue was shown
        reaction_time = ''
        if self.cue_onset_time:
            reaction_time = str(int((get_sync_time() - self.cue_onset_time) * 1000))
        
        # Determine predicted class from executed action
        predicted_class = ''
        if 'LEFT' in executed_action:
            predicted_class = 'left'
        elif 'RIGHT' in executed_action:
            predicted_class = 'right'
        
        self._add_event(timestamp, 'feedback_error' if is_error else 'feedback_correct',
                       trial_id=str(self.current_trial),
                       predicted_class=predicted_class,
                       accuracy=accuracy,
                       error_type='primary' if is_error else '',
                       reaction_time=reaction_time,
                       coins_collected=str(coins_collected),
                       details=f'intended={user_input}, executed={executed_action}, movement_required={movement_required}')
        
        if is_error:
            # Log ErrP event 300ms later
            threading.Timer(0.3, lambda: self._add_event(
                get_sync_time() - self.session_start_time, 'primary_errp',
                trial_id=str(self.current_trial),
                error_type='primary',
                coins_collected=str(coins_collected)
            )).start()
    
    def log_coin_event(self, event_type, coin_world_lane, agent_position, coins_collected):
        """Log coin-related events"""
        timestamp = get_sync_time() - self.session_start_time
        event_type_map = {
            'COIN_SPAWNED': 'coin_spawned',
            'COIN_COLLECTED': 'coin_collected',
            'COIN_MISSED': 'coin_missed'
        }
        mapped_type = event_type_map.get(event_type, event_type.lower())
        
        self._add_event(timestamp, mapped_type,
                       trial_id=str(self.current_trial),
                       coins_collected=str(coins_collected),
                       details=f'coin_lane={coin_world_lane}, agent_pos={agent_position}')
    
    def log_cue_timeout(self, agent_position, coins_collected):
        """Log cue timeout"""
        timestamp = get_sync_time() - self.session_start_time
        self._add_event(timestamp, 'cue_timeout',
                       trial_id=str(self.current_trial),
                       coins_collected=str(coins_collected),
                       details=f'agent_pos={agent_position}')
    
    def log_trial_complete(self, coins_collected):
        """Log trial completion"""
        timestamp = get_sync_time() - self.session_start_time
        self._add_event(timestamp, 'trial_end',
                       trial_id=str(self.current_trial),
                       coins_collected=str(coins_collected))
    
    def log_event(self, event_type, details=""):
        """Log a general event (compatibility method)"""
        timestamp = get_sync_time() - self.session_start_time
        self._add_event(timestamp, event_type.lower(),
                       trial_id=str(self.current_trial) if self.current_trial else '',
                       details=details)
    
    def save(self):
        """Save is automatic with each event, but kept for compatibility"""
        pass
    
    def _write_event(self, event):
        """Write event to CSV"""
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'timestamp', 'event_type', 'trial_id', 'cue_class', 
                'predicted_class', 'accuracy', 'error_type', 'confidence',
                'reaction_time', 'coins_collected', 'artifact_flag', 'details'
            ])
            writer.writerow(event)


def run_mi_integrated_game(control_mode='keyboard'):
    """
    Run the game with different control modes
    
    Args:
        control_mode: 'keyboard', 'mock_eeg', or 'real_eeg'
    """
    # Initialize pygame
    pygame.init()
    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
    
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
        print("Using: Real EEG file control (eeg_control.txt)")
    
    print("="*60 + "\n")
    
    # Get experiment settings from user
    settings = select_experiment_settings()
    
    # Extract settings
    difficulty = settings['difficulty']
    total_trials = settings['trials']
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
        print("✓ Connected to mock EEG stream")
        
        # Start MI monitoring
        mi_controller.start_monitoring()
        print("✓ MI monitoring started\n")
    
    # Run the main game with control integration
    main_with_mi(difficulty, total_trials, practice_trials, mi_controller, control_interface)
    
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


def main_with_mi(difficulty, total_trials, practice_trials, mi_controller=None, control_interface=None):
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
    log_path = os.path.join(LOG_DIR, f"subway_errp_{session_id}.csv")
    errp_logger = UnifiedEventLogger(log_path, difficulty, mi_mode=mi_controller is not None)
    
    print(f"\n📊 Logging to: {log_path}")
    
    # Session start
    push_immediate("SESSION_START")
    errp_logger.log_event("SESSION_START", f"difficulty={difficulty}, trials={total_trials}, mi_mode={mi_controller is not None}")
    
    # First run practice trials if any
    all_trials = practice_trials + total_trials
    
    # Track if we've entered main trials
    in_main_trials = False
    
    # Run all trials
    for trial_num in range(1, all_trials + 1):
        # Check if transitioning from practice to main
        if trial_num == practice_trials + 1 and practice_trials > 0:
            in_main_trials = True
            show_trial_transition(screen, "Practice Complete!", "Starting main experiment...")
            push_immediate("MAIN_TRIALS_START")
            errp_logger.log_event("MAIN_TRIALS_START", f"practice_complete={practice_trials}")
        
        # Update trial context for logging
        is_practice = trial_num <= practice_trials
        errp_logger.log_trial_start(trial_num, difficulty, is_practice)
        
        # Run trial with either practice or main error rate
        trial_error_rate = PRACTICE_ERROR_RATE if is_practice else ERROR_RATE
        if is_practice:
            print(f"\n🎮 Practice Trial {trial_num}/{practice_trials} (Error rate: {trial_error_rate:.0%})")
        else:
            main_trial_num = trial_num - practice_trials
            print(f"\n🎮 Trial {main_trial_num}/{total_trials} (Error rate: {trial_error_rate:.0%})")
        
        # Get trial parameters
        params = get_difficulty_params(difficulty).copy()
        params['error_rate'] = trial_error_rate
        
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
        cue_cooldown = 0
        current_speed_factor = 1.0
        
        # Define think time for this trial - fixed 5s for consistent EEG epochs
        think_time = 5.0  # Fixed 5s window for all cues
        
        # Trial start marker
        push_immediate(f"TRIAL_START_{trial_num}")
        
        # Main trial loop
        while trial_running:
            precision_tracker.mark_frame()
            frame_start_time = get_sync_time()
            
            # Update speed factor during cues
            if word_cue.active:
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
                        if word_cue.active and action_count == 0:  # Only one action per cue
                            key_name = "LEFT" if event.key == pygame.K_LEFT else "RIGHT"
                            cue_match = key_name == word_cue.cue_word
                            user_intended = f"MOVE_{key_name}"
                            
                            # Process action with error injection
                            if np.random.random() < params['error_rate']:
                                executed = f"MOVE_{'RIGHT' if key_name == 'LEFT' else 'LEFT'}"
                                error_injected = True
                            else:
                                executed = user_intended
                                error_injected = False
                            
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
                            errp_logger.log_action(user_intended, executed, agent.get_world_position(), 
                                                 agent.coins_collected, movement_required)
                            
                            # Don't disable cue - let it run for full 5 seconds
                            # This allows continued MI detection for the full window
                            # word_cue.active = False  # Commented out - cue ends naturally after 5s
                            push_immediate(f"ACTION_{executed}")
                            push_immediate("IMAGERY_ACTION_TAKEN")
                            
                            action_count += 1
                            cue_cooldown = 60
            
            # Non-keyboard control during cue window
            if control_interface and control_interface.mode != 'keyboard' and word_cue.active and action_count == 0:
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
                    
                    # Process action with error injection
                    if np.random.random() < params['error_rate']:
                        executed = f"MOVE_{'RIGHT' if key_name == 'LEFT' else 'LEFT'}"
                        error_injected = True
                    else:
                        executed = user_intended
                        error_injected = False
                    
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
                        'user_intent_followed': not error_injected,
                        'mi_detected': True,
                        'confidence': mi_controller.last_prediction.get('confidence', 0) if mi_controller and mi_controller.last_prediction else 0
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
                                         agent.coins_collected, movement_required)
                    
                    # Don't disable cue - let it run for full 5 seconds
                    # This allows continued MI detection for the full window
                    # word_cue.active = False  # Commented out - cue ends naturally after 5s
                    push_immediate(f"ACTION_{executed}")
                    push_immediate("IMAGERY_ACTION_TAKEN")
                    
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
                push_immediate(f"CUE_TIMEOUT_{word_cue.cue_word}")
                errp_logger.log_cue_timeout(agent.get_world_position(), agent.coins_collected)
                
                if mi_controller and control_interface.mode == 'mock_eeg':
                    mi_controller.end_cue()
                    mi_controller.start_feedback()  # Call the method instead of setting attribute
                
                cue_cooldown = 120
                action_count = 0
            
            # Update world with speed factor
            world.update(FORWARD_SPEED * current_speed_factor)
            coin_manager.update(agent, current_speed_factor, cue_active=word_cue.active)
            
            # Update cooldowns
            if cue_cooldown > 0:
                cue_cooldown -= current_speed_factor
            
            # Spawn new cue if needed
            if not word_cue.active and cue_cooldown <= 0:
                upcoming_coins = coin_manager.get_upcoming_coins(agent)
                
                if upcoming_coins:
                    target_coin = upcoming_coins[0]
                    agent_pos = agent.get_world_position()
                    
                    if target_coin.world_lane < agent_pos:
                        cue_word = "LEFT"
                    elif target_coin.world_lane > agent_pos:
                        cue_word = "RIGHT"
                    else:
                        cue_word = random.choice(["LEFT", "RIGHT"])
                    
                    push_immediate(f"BASELINE_START")
                    push_immediate(f"BASELINE_END")
                    
                    word_cue.show_cue(cue_word, think_time)
                    errp_logger.log_cue(cue_word, agent.get_world_position(), agent.coins_collected)
                    last_cue_time = get_sync_time()
                    
                    # Reset control interface for new cue
                    if control_interface and control_interface.mode == 'real_eeg':
                        control_interface.reset_for_new_cue()
                    
                    # Start MI cue if using MI control
                    if mi_controller and control_interface.mode == 'mock_eeg':
                        mi_controller.start_cue(cue_word)
                    
                    cue_cooldown = 30
            
            # Draw everything
            draw_endless_world(screen, world, agent, coin_manager.coins, word_cue)
            draw_hud(screen, trial_num, total_trials, agent, params['target_coins'], difficulty, last_action_info)
            
            # Draw MI-specific UI elements
            if mi_controller and control_interface.mode == 'mock_eeg':
                mi_controller.render_cue(screen)
                mi_controller.render_feedback(screen)
            
            # Check trial completion
            if agent.coins_collected >= params['target_coins']:
                completion_start = time.perf_counter()
                push_immediate(f"TRIAL_COMPLETE_{trial_num}")
                errp_logger.log_trial_complete(agent.coins_collected)
                show_trial_complete(screen, agent, params['target_coins'])
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
        
        # Inter-trial interval
        if trial_num < all_trials:
            iti_start = time.time()
            
            if not hasattr(main_with_mi, 'iti_screen'):
                main_with_mi.iti_screen = pygame.Surface(screen.get_size())
                main_with_mi.iti_screen.fill(COL['bg'])
            
            font = get_font(48)
            iti_text = font.render("Preparing next trial...", True, COL['txt'])
            iti_rect = iti_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
            
            while time.time() - iti_start < ITI:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        pygame.quit()
                        exit()
                
                screen.blit(main_with_mi.iti_screen, (0, 0))
                screen.blit(iti_text, iti_rect)
                
                remaining = ITI - (time.time() - iti_start)
                countdown_text = font.render(f"{remaining:.1f}s", True, COL['countdown'])
                countdown_rect = countdown_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 60))
                screen.blit(countdown_text, countdown_rect)
                
                pygame.display.flip()
                clock.tick(60)
    
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
        print(f"\n📊 Timing Precision Statistics:")
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