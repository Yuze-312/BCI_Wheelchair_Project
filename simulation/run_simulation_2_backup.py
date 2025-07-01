"""
Enhanced Subway Surfers Style EEG Simulator for ErrP Signal Elicitation
3-lane endless runner where agent stays centered and world moves around them

Key features:
- Always exactly 3 lanes visible (left, center, right)
- Agent always stays in center lane visually
- When agent moves, the world shifts to keep agent centered
- Coins spawn in lanes relative to agent's world position
- Smooth world scrolling animations
- Word cues for Motor Imagery commands

CRITICAL TIMING ALIGNMENT FOR EEG:
- All LSL markers use REAL TIME timestamps (not affected by slow motion)
- Cue duration is in REAL TIME (3s cue = 3s real time, even during slow motion)
- Reaction times are calculated in REAL TIME from cue onset
- ErrP latency (300ms) is in REAL TIME for consistent neural signatures
- Game visual effects use GAME TIME (affected by slow motion)

This ensures:
1. EEG epochs are properly aligned to stimulus onset
2. ERP components appear at expected latencies
3. Reaction times are comparable across trials
4. Slow motion enhances visibility without breaking timing analysis

Author: Enhanced version
Purpose: EEG research and ErrP signal elicitation with subway surfers mechanics
"""

import pygame
import random
import json
import csv
import time
import os
import math
from collections import deque
from pylsl import StreamInfo, StreamOutlet, local_clock
import numpy as np
from datetime import datetime

# ─── LSL STREAM ──────────────────────────────────────────────
info = StreamInfo("SubwaySurfers_ErrP_Markers", "Markers", 1, 0, "string", "subway-errp-001")
outlet = StreamOutlet(info)

# Essential LSL markers for EEG analysis:
# - BASELINE_START/END: For baseline correction
# - CUE_START_*/CUE_END_*: For stimulus-locked epochs
# - MI_OUTPUT_* or KEY_*: When classifier/user produces output
# - ERROR_INJECTED: When system flips the action
# - PRIMARY_ERRP: Expected ErrP component timing (300ms post-error)

# Synchronization tracking
sync_start_time = None
frame_count = 0
last_frame_time = None

# Precision timing tracking
class PrecisionTracker:
    """Track timing precision for analysis"""
    def __init__(self):
        self.event_times = deque(maxlen=1000)
        self.frame_times = deque(maxlen=1000)
        
    def mark_event(self, event_type):
        """Mark an event with high-precision timestamp"""
        timestamp = local_clock()
        perf_time = time.perf_counter()
        self.event_times.append((timestamp, perf_time, event_type))
        return timestamp
    
    def mark_frame(self):
        """Mark frame timing"""
        self.frame_times.append(time.perf_counter())
        
    def get_stats(self):
        """Get timing statistics"""
        if len(self.frame_times) < 2:
            return None
        intervals = np.diff(list(self.frame_times)) * 1000  # ms
        return {
            'mean_ms': np.mean(intervals),
            'std_ms': np.std(intervals),
            'max_ms': np.max(intervals)
        }

precision_tracker = PrecisionTracker()

# Font cache to avoid repeated font creation (performance optimization)
FONT_CACHE = {}

def get_font(size):
    """Get a cached font or create a new one"""
    if size not in FONT_CACHE:
        FONT_CACHE[size] = pygame.font.Font(None, size)
    return FONT_CACHE[size]

# Synchronized timing functions
def get_sync_time():
    """Use LSL time consistently throughout"""
    return local_clock()

def push_immediate(tag):
    """Send marker with immediate timestamp"""
    timestamp = precision_tracker.mark_event(tag)
    outlet.push_sample([tag], timestamp)
    return timestamp

def verify_sync(expected_interval=1.0/60):
    """Verify timing synchronization"""
    global last_frame_time, frame_count
    current_time = get_sync_time()
    
    if last_frame_time is not None:
        actual_interval = current_time - last_frame_time
        # Only warn for significant delays (>100ms) that affect EEG timing
        if actual_interval > 0.1:  # 100ms threshold
            print(f"SYNC WARNING: Frame took {actual_interval*1000:.1f}ms (expected {expected_interval*1000:.1f}ms)")
    
    last_frame_time = current_time
    frame_count += 1

# ─── PATHS ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(BASE_DIR, "logs", f"subway_errp_{int(time.time())}.csv")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ─── CONSTANTS ───────────────────────────────────────────────
ERROR_P = 0.3  # 30% error rate to simulate 70% MI decoder accuracy
RESP_WIN = 1.0
ITI = 2.0
FPS = 60
MAX_PROCESSING_DELAY = 0.010
FORWARD_SPEED = 5.0  # Pixels per frame for visual effects
WORLD_SHIFT_SPEED = 0.15  # Reduced for smoother, more visible transitions
MOVEMENT_EASING = 'sine'  # Options: 'cubic', 'sine', 'back', 'quadratic'
MOVEMENT_DURATION = 1.0  # Duration of movement in seconds (1 full second for clear visibility)
PAUSE_DURING_CUE = False  # Deprecated - now using slow motion
SLOW_MOTION_DURING_CUE = True  # Use slow motion effect during word cue
SLOW_MOTION_FACTOR = 0.2  # Game runs at 20% speed during cue (bullet-time effect)
DEBUG_MODE = True  # Set to True for detailed console output (turn OFF for EEG recording)
VERIFY_MODE = False  # Set to True to disable error injection for testing

# Hand image globals (will be loaded in main)
LEFT_HAND_IMG = None
RIGHT_HAND_IMG = None
HAND_IMG_SIZE = (100, 100)  # This stays constant

# EEG Experiment Parameters
BASELINE_DURATION = 0.8  # Pre-stimulus baseline in seconds (800ms)
POST_RESPONSE_DELAY = 0.5  # Delay after response before next coin (500ms)
PRACTICE_TRIALS = 3  # Number of practice trials before main experiment
SHOW_PRACTICE_FEEDBACK = True  # Show detailed feedback during practice

# MI Classifier Integration Mode
USE_KEYBOARD_AS_MI = True  # True: keyboard simulates MI, False: real EEG integration
MI_PROCESSING_DELAY = 0.5  # Simulated delay for MI classification (500ms)

# Think time range for word cues (seconds)
MIN_THINK_TIME = 1.0
MAX_THINK_TIME = 5.0

# Enhanced Colors with better contrast
COL = {
    'bg': (35, 45, 60),  # Lighter background for better contrast
    'track': (100, 100, 120),  # Brighter track
    'lane': (120, 125, 140),  # Much brighter lanes
    'lane_alt': (110, 115, 130),  # Brighter alternate lanes
    'lane_line': (180, 180, 200),  # Much brighter lane lines
    'agent': (0, 255, 136),
    'coin': (255, 215, 0),
    'collected_coin': (150, 255, 150),
    'txt': (255, 255, 255),
    'err': (255, 60, 60),
    'cue_bg': (70, 70, 100),  # Brighter cue background
    'cue_left': (100, 150, 255),
    'cue_right': (255, 150, 100),
    'countdown': (255, 255, 100),
    'highlight': (150, 170, 150),  # Brighter highlight
    'hud_bg': (0, 0, 0, 220),  # Semi-transparent HUD background
    'edge_glow': (100, 120, 180),  # Lane edge glow
}

# Controls
SUBWAY_ACTIONS = {
    pygame.K_LEFT: "MOVE_LEFT",
    pygame.K_RIGHT: "MOVE_RIGHT",
}

ACTION_CODE = {"NONE": 0, "MOVE_LEFT": 1, "MOVE_RIGHT": 2}

# ─── GAME CLASSES ─────────────────────────────────────────────

class EndlessWorld:
    """Manages the endless world with 3 visible lanes and smooth transitions"""
    def __init__(self, screen_width, screen_height, lane_width=None):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Calculate optimal lane width to fill the screen
        # We want 3 lanes to fill the entire width
        self.lane_width = screen_width // 3
        
        # World position (which lane number the agent is actually in)
        self.world_position = 0  # Start at lane 0
        
        # Visual offset for smooth scrolling (-1 to 1)
        # 0 = agent in center, -1 = agent moved one lane right, 1 = agent moved one lane left
        self.visual_offset = 0.0
        self.target_offset = 0.0
        self.shifting = False
        
        # We need to track 5 lanes for smooth transitions (2 extra for sliding in/out)
        self.total_lanes = 5
        
        # No offset needed - lanes fill the entire screen
        self.base_x_offset = -self.lane_width  # Start one lane to the left
        
    def start_move_left(self):
        """Start moving world right (agent goes left)"""
        if not self.shifting:
            self.target_offset = 1.0  # World will shift right
            self.shifting = True
            push_immediate(f"WORLD_SHIFT_START_LEFT")
            return True
        return False
    
    def start_move_right(self):
        """Start moving world left (agent goes right)"""
        if not self.shifting:
            self.target_offset = -1.0  # World will shift left
            self.shifting = True
            push_immediate(f"WORLD_SHIFT_START_RIGHT")
            return True
        return False
    
    def update(self, speed_factor=1.0):
        """Update world shifting animation with speed factor for slow motion"""
        if self.shifting:
            # Calculate progress (0 to 1)
            if not hasattr(self, 'shift_progress'):
                self.shift_progress = 0.0
                self.shift_duration = MOVEMENT_DURATION
                self.shift_timer = 0.0
            
            # Update timer with speed factor
            self.shift_timer += (1.0 / FPS) * speed_factor
            self.shift_progress = min(self.shift_timer / self.shift_duration, 1.0)
            
            # Apply selected easing function
            if MOVEMENT_EASING == 'sine':
                eased_progress = self.ease_in_out_sine(self.shift_progress)
            elif MOVEMENT_EASING == 'cubic':
                eased_progress = self.ease_in_out_cubic(self.shift_progress)
            elif MOVEMENT_EASING == 'back':
                eased_progress = self.ease_in_out_back(self.shift_progress)
            else:  # quadratic default
                eased_progress = self.ease_in_out(self.shift_progress)
            
            # Update visual offset
            self.visual_offset = self.target_offset * eased_progress
            
            # Check if animation complete
            if self.shift_progress >= 1.0:
                # Complete the shift
                if self.target_offset > 0:
                    self.world_position -= 1  # Moved left
                else:
                    self.world_position += 1  # Moved right
                    
                self.visual_offset = 0.0
                self.target_offset = 0.0
                self.shifting = False
                self.shift_progress = 0.0
                self.shift_timer = 0.0
                push_immediate(f"WORLD_SHIFT_COMPLETE_POS_{self.world_position}")
    
    def get_lane_info(self):
        """Get information about all visible lanes during transition"""
        lanes = []
        
        # Calculate positions for 5 lanes (2 extra for smooth scrolling)
        for i in range(self.total_lanes):
            # Lane index relative to view center
            view_offset = i - 2  # -2, -1, 0, 1, 2
            
            # Calculate screen position with visual offset
            x_pos = self.base_x_offset + i * self.lane_width + (self.visual_offset * self.lane_width)
            
            # Determine actual world lane number
            # This is the key fix: properly calculate which world lane this visual lane represents
            world_lane = self.world_position + view_offset
            
            # Check if lane is visible
            visible = x_pos + self.lane_width > 0 and x_pos < self.screen_width
            
            lanes.append({
                'world_lane': world_lane,
                'x': x_pos,
                'center_x': x_pos + self.lane_width // 2,
                'visible': visible,
                'alpha': self.calculate_lane_alpha(i),
                'view_offset': view_offset  # Add this for debugging
            })
        
        return lanes
    
    def calculate_lane_alpha(self, lane_index):
        """Calculate alpha for fading lanes during transition"""
        if not self.shifting:
            # Only middle 3 lanes fully visible when not shifting
            if lane_index == 0 or lane_index == 4:
                return 0
            return 255
        
        # During shift, calculate smooth fade with easing
        progress = abs(self.visual_offset)
        
        if self.visual_offset > 0:  # Shifting right
            if lane_index == 0:  # Leftmost lane fading in
                return int(255 * self.ease_in_out(progress))
            elif lane_index == 4:  # Rightmost lane fading out
                return int(255 * self.ease_in_out(1 - progress))
        else:  # Shifting left
            if lane_index == 0:  # Leftmost lane fading out
                return int(255 * self.ease_in_out(1 + self.visual_offset))
            elif lane_index == 4:  # Rightmost lane fading in
                return int(255 * self.ease_in_out(-self.visual_offset))
        
        return 255  # Middle lanes always visible
    
    def ease_in_out(self, t):
        """Smooth easing function for transitions"""
        if t < 0.5:
            return 2 * t * t
        return 1 - pow(-2 * t + 2, 2) / 2
    
    def ease_in_out_cubic(self, t):
        """Cubic easing for very smooth acceleration and deceleration"""
        if t < 0.5:
            return 4 * t * t * t
        p = 2 * t - 2
        return 1 + p * p * p / 2
    
    def ease_in_out_sine(self, t):
        """Sine-based easing for the smoothest feel"""
        return -(math.cos(math.pi * t) - 1) / 2
    
    def ease_in_out_back(self, t):
        """Back easing with slight overshoot for bouncy feel"""
        c1 = 1.70158
        c2 = c1 * 1.525
        if t < 0.5:
            return (pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2
        return (pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2
    
    def get_coin_screen_x(self, coin_world_lane):
        """Get screen X position for a coin in a specific world lane"""
        lanes = self.get_lane_info()
        for lane in lanes:
            if lane['world_lane'] == coin_world_lane and lane['visible']:
                return lane['center_x']
        return None  # Coin not visible
    
    def get_agent_lane_center(self):
        """Get the center X position of the agent's lane (always middle lane)"""
        lanes = self.get_lane_info()
        # Agent is always in the middle lane (index 2)
        return lanes[2]['center_x']

class Agent:
    """Agent that always stays in center lane visually"""
    def __init__(self, world):
        self.world = world
        self.y = 600  # Fixed Y position
        self.coins_collected = 0
        self.is_moving = False
        self.last_action = None  # Track last action for visual feedback
        self.action_was_correct = True  # Track if last action was correct
        self.error_time = 0  # Time since error occurred
        
        # Smooth movement interpolation
        self.visual_lean = 0.0  # Visual lean during movement (-1 to 1)
        self.target_lean = 0.0
        self.bounce_offset = 0.0  # Vertical bounce during movement
        
    def move_left(self):
        """Move left (world shifts right)"""
        if not self.world.shifting:
            self.world.start_move_left()
            self.is_moving = True
            self.last_action = "LEFT"
            self.target_lean = -0.3  # Lean left
            return True
        return False
    
    def move_right(self):
        """Move right (world shifts left)"""
        if not self.world.shifting:
            self.world.start_move_right()
            self.is_moving = True
            self.last_action = "RIGHT"
            self.target_lean = 0.3  # Lean right
            return True
        return False
    
    def update(self):
        """Update agent state"""
        if self.is_moving and not self.world.shifting:
            self.is_moving = False
            self.target_lean = 0.0  # Return to center
        
        # Smooth lean interpolation
        lean_diff = self.target_lean - self.visual_lean
        self.visual_lean += lean_diff * 0.15  # Smooth interpolation
        
        # Add bounce effect during movement
        if self.is_moving and self.world.shifting:
            # Bounce based on shift progress
            progress = abs(self.world.visual_offset)
            self.bounce_offset = math.sin(progress * math.pi) * 8  # Max 8 pixel bounce
        else:
            self.bounce_offset *= 0.9  # Smooth decay
        
        # Decay error visual over time
        if self.error_time > 0:
            self.error_time -= 0.02
    
    def set_action_feedback(self, was_correct):
        """Set whether the last action was correct or not"""
        self.action_was_correct = was_correct
        if not was_correct:
            self.error_time = 1.0  # Show error for 1 second
    
    def get_x_position(self):
        """Get agent's X position (always center)"""
        return self.world.get_agent_lane_center()
    
    def get_world_position(self):
        """Get agent's actual position in the endless world"""
        return self.world.world_position

class Coin:
    """Coin in the endless world"""
    def __init__(self, world_lane, y_position=-100):
        self.world_lane = world_lane  # Absolute position in world
        self.y_position = y_position
        self.collected = False
        self.radius = 25
        self.spawn_time = get_sync_time()
        
    def update(self, speed):
        """Move coin downward"""
        if not self.collected:
            self.y_position += speed
    
    def get_relative_lane(self, agent_world_pos):
        """Get lane position relative to agent (-1, 0, 1)"""
        return self.world_lane - agent_world_pos
    
    def is_visible(self, agent_world_pos):
        """Check if coin is in visible range"""
        relative = self.get_relative_lane(agent_world_pos)
        # More conservative check - only -1, 0, 1 are truly visible
        return -1 <= relative <= 1
    
    def check_collection(self, agent):
        """Check if agent collected this coin"""
        if self.collected:
            return False
            
        # Coin must be in center lane (relative position 0) and near agent
        relative_lane = self.get_relative_lane(agent.get_world_position())
        if relative_lane == 0 and abs(self.y_position - agent.y) < 50:
            self.collected = True
            return True
        return False
    
    def is_off_screen(self, screen_height):
        """Check if coin has moved off screen"""
        return self.y_position > screen_height + 50

class CoinManager:
    """Manages coins in the endless world"""
    def __init__(self, world, target_coins, logger=None):
        self.world = world
        self.target_coins = target_coins
        self.coins = []
        self.spawn_cooldown = 0
        self.min_spawn_interval = 60  # Reduced for better flow
        self.max_spawn_interval = 90
        self.active_coin = None  # Track the current active coin
        self.logger = logger  # Event logger for coin events
        
    def spawn_coin(self, agent_world_pos, coins_collected=0):
        """Spawn a single coin away from agent's current position"""
        # Only spawn if no active coin exists
        if self.active_coin and not self.active_coin.collected:
            return None
            
        # Only spawn in immediately adjacent lanes (1 lane away)
        # This ensures coins remain visible even after agent moves
        possible_offsets = [-1, 1]  # Only left or right adjacent lane
        
        # Choose a random offset
        offset = random.choice(possible_offsets)
        
        world_lane = agent_world_pos + offset
        
        # Create coin at a good starting position
        coin = Coin(world_lane, y_position=100)  # Start at y=100 for better visibility
        self.coins = [coin]  # Only one coin at a time
        self.active_coin = coin
        
        if DEBUG_MODE:
            print(f"COIN SPAWNED: Agent at world pos {agent_world_pos}, coin at world lane {world_lane} (offset {offset})")
        push_immediate(f"COIN_SPAWNED_AT_WORLD_{world_lane}_OFFSET_{offset}")
        
        # Log coin spawn event if logger available
        if self.logger:
            self.logger.log_coin_event("COIN_SPAWNED", world_lane, agent_world_pos, coins_collected)
        
        return coin
    
    def update(self, agent, speed_factor=1.0, cue_active=False):
        """Update all coins with speed factor for slow motion"""
        if self.active_coin:
            # Update coin with adjusted speed
            self.active_coin.update(FORWARD_SPEED * speed_factor)
            
            # Check collection
            if self.active_coin.check_collection(agent):
                agent.coins_collected += 1
                push_immediate(f"COIN_COLLECTED_{agent.coins_collected}")
                
                # Log coin collection event
                if self.logger:
                    self.logger.log_coin_event("COIN_COLLECTED", 
                                             self.active_coin.world_lane, 
                                             agent.get_world_position(), 
                                             agent.coins_collected)
                
                self.active_coin = None
                self.coins = []
                self.spawn_cooldown = 30  # Short cooldown after collection
            
            # Remove if off-screen
            elif self.active_coin.is_off_screen(self.world.screen_height):
                push_immediate("COIN_MISSED")
                
                # Log coin missed event
                if self.logger:
                    self.logger.log_coin_event("COIN_MISSED", 
                                             self.active_coin.world_lane, 
                                             agent.get_world_position(), 
                                             agent.coins_collected)
                
                self.active_coin = None
                self.coins = []
                self.spawn_cooldown = 60  # Longer cooldown after miss
        
        # Spawn new coin if needed (with speed factor)
        # IMPORTANT: Don't spawn new coins during active cues to prevent double cues
        if speed_factor > 0 and not cue_active:  # Only spawn when no cue is active
            self.spawn_cooldown -= speed_factor
            if self.spawn_cooldown <= 0 and not self.active_coin:
                self.spawn_coin(agent.get_world_position(), agent.coins_collected)
                self.spawn_cooldown = random.randint(self.min_spawn_interval, self.max_spawn_interval)
    
    def get_upcoming_coins(self, agent):
        """Get coins approaching the agent"""
        if not self.active_coin or self.active_coin.collected:
            return []
            
        agent_world_pos = agent.get_world_position()
        
        # Only return the coin if it's visible and approaching
        if self.active_coin.is_visible(agent_world_pos):
            if self.active_coin.y_position < agent.y and self.active_coin.y_position > 200:
                return [self.active_coin]
        
        return []

class WordCue:
    """Visual word cue system with proper timing alignment for EEG"""
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
        
        # Send marker with real-time duration for EEG alignment
        push_immediate(f"CUE_START_{word}_{duration:.1f}s_REALTIME")
        
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
                # Send end marker at precise real time
                push_immediate(f"CUE_END_{self.cue_word}_REALTIME")
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

# ─── DRAWING FUNCTIONS ───────────────────────────────────────

def draw_endless_world(screen, world, agent, coins, word_cue):
    """Draw the 3-lane endless world with smooth transitions"""
    global LEFT_HAND_IMG, RIGHT_HAND_IMG, HAND_IMG_SIZE
    screen.fill(COL['bg'])
    
    # Use cached fonts for text rendering
    font = get_font(36)
    small_font = get_font(24)
    
    # Check if in slow motion
    in_slow_motion = SLOW_MOTION_DURING_CUE and word_cue.active
    
    # Get all lane information
    lanes = world.get_lane_info()
    
    # Draw gradient background
    for y in range(0, screen.get_height(), 40):
        fade = 1.0 - (y / screen.get_height()) * 0.3
        color = tuple(int(c * fade) for c in COL['bg'])
        pygame.draw.rect(screen, color, (0, y, screen.get_width(), 40))
    
    # Draw lanes with proper alignment and enhanced visibility
    for i, lane in enumerate(lanes):
        if not lane['visible'] or lane['alpha'] == 0:
            continue
            
        # Create lane surface with alpha
        lane_surface = pygame.Surface((world.lane_width, screen.get_height()), pygame.SRCALPHA)
        
        # Alternate lane colors with better contrast
        lane_color = COL['lane'] if lane['world_lane'] % 2 == 0 else COL['lane_alt']
        lane_surface.fill((*lane_color, lane['alpha']))
        
        # Add subtle gradient to lanes
        for y in range(0, screen.get_height(), 20):
            gradient_alpha = int(lane['alpha'] * (0.8 + 0.2 * (y / screen.get_height())))
            gradient_color = tuple(int(c * 1.1) for c in lane_color)
            gradient_color = tuple(min(255, c) for c in gradient_color)
            pygame.draw.rect(lane_surface, (*gradient_color, gradient_alpha), 
                           (0, y, world.lane_width, 20))
        
        # Draw the lane
        screen.blit(lane_surface, (lane['x'], 0))
        
        # Draw enhanced lane dividers
        if i < len(lanes) - 1 and lane['alpha'] > 0:
            line_x = lane['x'] + world.lane_width
            line_alpha = min(255, lane['alpha'])
            
            # Draw glow effect for lane lines
            glow_surface = pygame.Surface((6, screen.get_height()), pygame.SRCALPHA)
            for glow_x in range(6):
                glow_alpha = int(line_alpha * 0.3 * (1 - glow_x / 6))
                pygame.draw.line(glow_surface, (*COL['edge_glow'], glow_alpha),
                               (glow_x, 0), (glow_x, screen.get_height()), 1)
            screen.blit(glow_surface, (line_x - 3, 0))
            
            # Solid line with better visibility
            pygame.draw.line(screen, (*COL['lane_line'], line_alpha), 
                           (line_x, 0), (line_x, screen.get_height()), 3)
            
            # Enhanced dashed center line
            for y in range(0, screen.get_height(), 60):
                if y % 120 < 60:  # Draw every other segment
                    pygame.draw.line(screen, (255, 255, 255, line_alpha), 
                                   (line_x - 1, y), 
                                   (line_x - 1, min(y + 40, screen.get_height())), 2)
    
    # Enhanced center lane highlight
    center_x = screen.get_width() // 2
    highlight_width = world.lane_width - 20
    highlight_left = center_x - highlight_width // 2
    
    # Create more visible highlight effect
    highlight_surface = pygame.Surface((highlight_width, screen.get_height()), pygame.SRCALPHA)
    
    # Bottom glow effect
    for y in range(screen.get_height() - 200, screen.get_height()):
        alpha = int(40 * ((y - (screen.get_height() - 200)) / 200))
        pygame.draw.rect(highlight_surface, (*COL['highlight'], alpha), 
                       (0, y, highlight_width, 2))
    
    # Side glows
    for x in range(10):
        alpha = int(30 * (1 - x / 10))
        pygame.draw.line(highlight_surface, (*COL['highlight'], alpha),
                       (x, 0), (x, screen.get_height()), 1)
        pygame.draw.line(highlight_surface, (*COL['highlight'], alpha),
                       (highlight_width - x - 1, 0), 
                       (highlight_width - x - 1, screen.get_height()), 1)
    
    screen.blit(highlight_surface, (highlight_left, 0))
    
    # Enhanced moving road lines for speed effect
    line_offset = (pygame.time.get_ticks() // 10) % 60
    for i in range(-1, screen.get_height() // 60 + 2):
        y = i * 60 + line_offset
        for lane in lanes:
            if lane['visible'] and lane['alpha'] > 100:
                line_alpha = min(150, int(lane['alpha'] * 0.6))
                # Draw with glow
                pygame.draw.rect(screen, (200, 200, 200, line_alpha), 
                               (lane['center_x'] - 4, y, 8, 25))
                pygame.draw.rect(screen, (255, 255, 255, line_alpha // 2), 
                               (lane['center_x'] - 2, y + 2, 4, 21))
    
    # Draw coins with enhanced effects
    for coin in coins:
        if not coin.collected:
            relative_pos = coin.get_relative_lane(agent.get_world_position())
            
            if -1 <= relative_pos <= 1:
                coin_x = world.get_coin_screen_x(coin.world_lane)
                
                if coin_x is not None:
                    # Enhanced coin shadow
                    shadow_offset = 5
                    shadow_surface = pygame.Surface((coin.radius * 3, coin.radius * 3), pygame.SRCALPHA)
                    pygame.draw.circle(shadow_surface, (0, 0, 0, 100), 
                                     (coin.radius * 1.5 + shadow_offset, coin.radius * 1.5 + shadow_offset), 
                                     coin.radius + 3)
                    screen.blit(shadow_surface, (coin_x - coin.radius * 1.5, coin.y_position - coin.radius * 1.5))
                    
                    # Enhanced coin glow with pulsing
                    pulse = abs(math.sin(pygame.time.get_ticks() * 0.003))
                    glow_radius = coin.radius * (1.5 + 0.3 * pulse)
                    glow_surface = pygame.Surface((glow_radius * 2.5, glow_radius * 2.5), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surface, (255, 215, 0, 80), 
                                     (glow_radius * 1.25, glow_radius * 1.25), glow_radius)
                    screen.blit(glow_surface, (coin_x - glow_radius * 1.25, coin.y_position - glow_radius * 1.25))
                    
                    # Main coin with multiple layers
                    pygame.draw.circle(screen, COL['coin'], (int(coin_x), int(coin.y_position)), coin.radius)
                    pygame.draw.circle(screen, (255, 200, 0), (int(coin_x), int(coin.y_position)), coin.radius - 3)
                    pygame.draw.circle(screen, (255, 230, 100), (int(coin_x), int(coin.y_position)), coin.radius - 6)
                    
                    # Shine effect
                    pygame.draw.circle(screen, (255, 250, 200), 
                                     (int(coin_x - coin.radius * 0.3), int(coin.y_position - coin.radius * 0.3)), 
                                     coin.radius // 3)
                    
                    # Coin symbol with better contrast
                    coin_text = font.render("$", True, (100, 70, 0))
                    text_rect = coin_text.get_rect(center=(int(coin_x), int(coin.y_position)))
                    screen.blit(coin_text, text_rect)
                    
                    # Enhanced direction indicators
                    distance = agent.y - coin.y_position
                    if 100 < distance < 300 and relative_pos != 0 and not world.shifting:
                        arrow_y = coin.y_position + 50
                        pulse = abs(math.sin(pygame.time.get_ticks() * 0.006))
                        arrow_alpha = int(150 + 100 * pulse)
                        arrow_size = 8 + int(4 * pulse)
                        
                        if relative_pos < 0:
                            # Left arrow with glow
                            for j in range(3):
                                glow_alpha = arrow_alpha // (j + 1)
                                arrow_points = [
                                    (coin_x - 25 - j*2, arrow_y),
                                    (coin_x - 15 + j, arrow_y - arrow_size - j),
                                    (coin_x - 15 + j, arrow_y + arrow_size + j)
                                ]
                                pygame.draw.polygon(screen, (*COL['cue_left'], glow_alpha), arrow_points)
                        else:
                            # Right arrow with glow
                            for j in range(3):
                                glow_alpha = arrow_alpha // (j + 1)
                                arrow_points = [
                                    (coin_x + 25 + j*2, arrow_y),
                                    (coin_x + 15 - j, arrow_y - arrow_size - j),
                                    (coin_x + 15 - j, arrow_y + arrow_size + j)
                                ]
                                pygame.draw.polygon(screen, (*COL['cue_right'], glow_alpha), arrow_points)
    
    # Draw agent with enhanced visibility and smooth interpolation
    agent_x = agent.get_x_position() + agent.visual_lean * 30  # Lean offset
    agent_y = agent.y - agent.bounce_offset  # Bounce offset
    
    # Determine agent color based on action correctness
    if agent.error_time > 0:
        # Red color for errors with pulsing effect
        error_intensity = agent.error_time
        agent_color = (
            int(255 * error_intensity + COL['agent'][0] * (1 - error_intensity)),
            int(60 * error_intensity + COL['agent'][1] * (1 - error_intensity)),
            int(60 * error_intensity + COL['agent'][2] * (1 - error_intensity))
        )
        glow_color = (255, 60, 60)  # Red glow for errors
    else:
        agent_color = COL['agent']  # Normal green
        glow_color = COL['agent']
    
    # Enhanced agent trail effect during movement
    if world.shifting:
        trail_alpha = int(150 * abs(world.visual_offset))
        trail_offset = world.visual_offset * world.lane_width * 0.3
        
        # Draw enhanced motion blur
        for i in range(5):
            blur_x = agent_x - trail_offset * (i + 1) * 0.2
            blur_alpha = max(0, min(255, trail_alpha - i * 25))
            if blur_alpha > 0:
                blur_surface = pygame.Surface((80, 80), pygame.SRCALPHA)
                pygame.draw.circle(blur_surface, (*agent_color, blur_alpha), (40, 40), 30 - i * 4)
                screen.blit(blur_surface, (blur_x - 40, agent_y - 40))
    
    # Enhanced agent shadow
    shadow_surface = pygame.Surface((90, 40), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow_surface, (0, 0, 0, 150), (0, 0, 90, 40))
    screen.blit(shadow_surface, (agent_x - 45, agent_y + 25))
    
    # Enhanced agent glow
    glow_pulse = abs(math.sin(pygame.time.get_ticks() * 0.002))
    glow_size = 40 + int(10 * glow_pulse)
    glow_surface = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
    for i in range(3):
        glow_alpha = 60 - i * 15
        pygame.draw.circle(glow_surface, (*glow_color, glow_alpha), 
                         (glow_size, glow_size), glow_size - i * 10)
    screen.blit(glow_surface, (agent_x - glow_size, agent_y - glow_size))
    
    # Agent body with multiple layers
    pygame.draw.circle(screen, agent_color, (int(agent_x), int(agent_y)), 30)
    pygame.draw.circle(screen, tuple(int(c * 0.8) for c in agent_color), 
                      (int(agent_x), int(agent_y)), 26)
    pygame.draw.circle(screen, agent_color, (int(agent_x), int(agent_y)), 22)
    
    # Enhanced highlight
    highlight_color = tuple(min(255, int(c * 1.5)) for c in agent_color)
    pygame.draw.circle(screen, highlight_color, (int(agent_x - 8), int(agent_y - 8)), 12)
    
    # Direction indicator with glow
    pygame.draw.circle(screen, (255, 255, 255), (int(agent_x), int(agent_y - 3)), 8)
    pygame.draw.circle(screen, tuple(int(c * 0.6) for c in agent_color), 
                      (int(agent_x), int(agent_y - 3)), 5)
    
    # Enhanced movement arrows during shift
    if world.shifting:
        progress = abs(world.visual_offset)
        
        # Determine arrow color based on action correctness
        if hasattr(agent, 'last_action'):
            if agent.action_was_correct:
                arrow_color = COL['cue_left'] if world.target_offset > 0 else COL['cue_right']
            else:
                arrow_color = COL['err']  # Red arrows for incorrect action
        else:
            arrow_color = COL['cue_left'] if world.target_offset > 0 else COL['cue_right']
        
        if world.target_offset > 0:  # Moving left
            for i in range(4):
                arrow_x = agent_x - 50 - i * 25
                arrow_alpha = max(0, min(255, int(255 * progress * (1 - i * 0.2))))
                if arrow_alpha > 0:
                    # Arrow with glow
                    for j in range(2):
                        glow_alpha = arrow_alpha // (j + 1)
                        arrow_points = [
                            (arrow_x - 12 - j, agent_y),
                            (arrow_x + j, agent_y - 10 - j),
                            (arrow_x + j, agent_y + 10 + j)
                        ]
                        pygame.draw.polygon(screen, (*arrow_color, glow_alpha), arrow_points)
                    
                    # Add "X" mark if error
                    if not agent.action_was_correct and i == 0:
                        cross_size = 20
                        cross_alpha = min(255, arrow_alpha + 50)
                        pygame.draw.line(screen, (*COL['err'], cross_alpha), 
                                       (arrow_x - cross_size, agent_y - cross_size), 
                                       (arrow_x + cross_size, agent_y + cross_size), 4)
                        pygame.draw.line(screen, (*COL['err'], cross_alpha), 
                                       (arrow_x - cross_size, agent_y + cross_size), 
                                       (arrow_x + cross_size, agent_y - cross_size), 4)
        else:  # Moving right
            for i in range(4):
                arrow_x = agent_x + 50 + i * 25
                arrow_alpha = max(0, min(255, int(255 * progress * (1 - i * 0.2))))
                if arrow_alpha > 0:
                    # Arrow with glow
                    for j in range(2):
                        glow_alpha = arrow_alpha // (j + 1)
                        arrow_points = [
                            (arrow_x + 12 + j, agent_y),
                            (arrow_x - j, agent_y - 10 - j),
                            (arrow_x - j, agent_y + 10 + j)
                        ]
                        pygame.draw.polygon(screen, (*arrow_color, glow_alpha), arrow_points)
                    
                    # Add "X" mark if error
                    if not agent.action_was_correct and i == 0:
                        cross_size = 20
                        cross_alpha = min(255, arrow_alpha + 50)
                        pygame.draw.line(screen, (*COL['err'], cross_alpha), 
                                       (arrow_x - cross_size, agent_y - cross_size), 
                                       (arrow_x + cross_size, agent_y + cross_size), 4)
                        pygame.draw.line(screen, (*COL['err'], cross_alpha), 
                                       (arrow_x - cross_size, agent_y + cross_size), 
                                       (arrow_x + cross_size, agent_y - cross_size), 4)
    
    # Draw slow motion effect if active
    if in_slow_motion:
        # Add blue tint overlay for bullet-time effect
        slow_mo_overlay = pygame.Surface((screen.get_width(), screen.get_height()))
        slow_mo_overlay.set_alpha(30)
        slow_mo_overlay.fill((100, 150, 255))  # Light blue tint
        screen.blit(slow_mo_overlay, (0, 0))
        
        # Draw speed lines for motion effect
        line_count = 15
        for i in range(line_count):
            # Random speed lines from edges
            y = random.randint(0, screen.get_height())
            line_length = random.randint(50, 150)
            alpha = random.randint(20, 60)
            
            # Create line surface with alpha
            line_surface = pygame.Surface((line_length, 2), pygame.SRCALPHA)
            line_surface.fill((255, 255, 255, alpha))
            
            # Draw from left and right edges
            if i % 2 == 0:
                screen.blit(line_surface, (0, y))
            else:
                screen.blit(line_surface, (screen.get_width() - line_length, y))
        
        # Add "SLOW MOTION" text
        slow_mo_font = get_font(24)
        slow_mo_text = slow_mo_font.render("SLOW MOTION", True, (200, 220, 255))
        text_rect = slow_mo_text.get_rect(center=(screen.get_width() // 2, 40))
        
        # Add glow effect to text
        for i in range(3):
            glow_surf = slow_mo_font.render("SLOW MOTION", True, (100, 150, 255))
            glow_surf.set_alpha(50 - i * 15)
            glow_rect = glow_surf.get_rect(center=(screen.get_width() // 2 + i, 40 + i))
            screen.blit(glow_surf, glow_rect)
        
        screen.blit(slow_mo_text, text_rect)
    
    # Draw word cue
    if word_cue.active:
        draw_word_cue(screen, word_cue)

def draw_word_cue(screen, word_cue):
    """Draw the word cue overlay - enhanced design"""
    screen_width, screen_height = screen.get_size()
    
    # Semi-transparent overlay
    overlay = pygame.Surface((screen_width, screen_height))
    overlay.set_alpha(80)  # Lighter overlay to keep game visible
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Cue box dimensions
    cue_width, cue_height = 360, 160
    cue_x = screen_width // 2 - cue_width // 2
    cue_y = 200  # Position it lower to avoid overlap with HUD
    
    # Background with gradient effect
    cue_surface = pygame.Surface((cue_width, cue_height), pygame.SRCALPHA)
    
    # Draw gradient background
    for y in range(cue_height):
        fade = 1.0 - (y / cue_height) * 0.3
        color = tuple(int(c * fade) for c in COL['cue_bg'])
        pygame.draw.rect(cue_surface, (*color, 240), (0, y, cue_width, 1))
    
    screen.blit(cue_surface, (cue_x, cue_y))
    
    # Border with glow
    border_color = COL['cue_left'] if word_cue.cue_word == "LEFT" else COL['cue_right']
    
    # Outer glow
    for i in range(3):
        glow_alpha = 100 - i * 30
        pygame.draw.rect(screen, (*border_color, glow_alpha), 
                        (cue_x - 2 - i*2, cue_y - 2 - i*2, 
                         cue_width + 4 + i*4, cue_height + 4 + i*4), 
                        2, border_radius=15)
    
    # Main border
    border_thickness = 3 + int(2 * math.sin(word_cue.flash_timer * 0.1))
    pygame.draw.rect(screen, border_color, 
                    (cue_x - 2, cue_y - 2, cue_width + 4, cue_height + 4), 
                    border_thickness, border_radius=15)
    
    # Enhanced direction arrow
    arrow_y = cue_y + 45
    arrow_size = 20
    arrow_pulse = abs(math.sin(word_cue.flash_timer * 0.08))
    
    if word_cue.cue_word == "LEFT":
        # Left arrow with glow effect
        for i in range(3):
            glow_size = arrow_size + i * 2
            glow_alpha = 200 - i * 50
            arrow_points = [
                (cue_x + 50 - i*2, arrow_y),
                (cue_x + 50 + glow_size, arrow_y - glow_size),
                (cue_x + 50 + glow_size, arrow_y + glow_size)
            ]
            pygame.draw.polygon(screen, (*border_color, int(glow_alpha * arrow_pulse)), arrow_points)
    else:
        # Right arrow with glow effect
        for i in range(3):
            glow_size = arrow_size + i * 2
            glow_alpha = 200 - i * 50
            arrow_points = [
                (cue_x + cue_width - 50 + i*2, arrow_y),
                (cue_x + cue_width - 50 - glow_size, arrow_y - glow_size),
                (cue_x + cue_width - 50 - glow_size, arrow_y + glow_size)
            ]
            pygame.draw.polygon(screen, (*border_color, int(glow_alpha * arrow_pulse)), arrow_points)
    
    # Display hand image alongside word cue
    if word_cue.cue_word == "LEFT":
        if LEFT_HAND_IMG:
            # Position left hand image to the left of center
            hand_x = cue_x + 30
            hand_y = cue_y + 10
            
            # Add subtle glow effect behind hand
            glow_surface = pygame.Surface((HAND_IMG_SIZE[0] + 20, HAND_IMG_SIZE[1] + 20), pygame.SRCALPHA)
            glow_alpha = int(80 * arrow_pulse)
            for i in range(3):
                alpha = glow_alpha - i * 20
                if alpha > 0:
                    pygame.draw.ellipse(glow_surface, (*border_color, alpha), 
                                      (10 - i*3, 10 - i*3, 
                                       HAND_IMG_SIZE[0] + i*6, HAND_IMG_SIZE[1] + i*6))
            screen.blit(glow_surface, (hand_x - 10, hand_y - 10))
            
            # Draw the hand image
            screen.blit(LEFT_HAND_IMG, (hand_x, hand_y))
        else:
            if DEBUG_MODE:
                print("WARNING: LEFT_HAND_IMG is None when trying to display")
        
    elif word_cue.cue_word == "RIGHT":
        if RIGHT_HAND_IMG:
            # Position right hand image to the right of center
            hand_x = cue_x + cue_width - HAND_IMG_SIZE[0] - 30
            hand_y = cue_y + 10
            
            # Add subtle glow effect behind hand
            glow_surface = pygame.Surface((HAND_IMG_SIZE[0] + 20, HAND_IMG_SIZE[1] + 20), pygame.SRCALPHA)
            glow_alpha = int(80 * arrow_pulse)
            for i in range(3):
                alpha = glow_alpha - i * 20
                if alpha > 0:
                    pygame.draw.ellipse(glow_surface, (*border_color, alpha), 
                                      (10 - i*3, 10 - i*3, 
                                       HAND_IMG_SIZE[0] + i*6, HAND_IMG_SIZE[1] + i*6))
            screen.blit(glow_surface, (hand_x - 10, hand_y - 10))
            
            # Draw the hand image
            screen.blit(RIGHT_HAND_IMG, (hand_x, hand_y))
        else:
            if DEBUG_MODE:
                print("WARNING: RIGHT_HAND_IMG is None when trying to display")
    
    # Word text with shadow
    font = get_font(72)
    # Shadow
    shadow_text = font.render(word_cue.cue_word, True, (0, 0, 0))
    shadow_rect = shadow_text.get_rect(center=(screen_width // 2 + 3, cue_y + 45 + 3))
    screen.blit(shadow_text, shadow_rect)
    # Main text
    cue_text = font.render(word_cue.cue_word, True, border_color)
    text_rect = cue_text.get_rect(center=(screen_width // 2, cue_y + 45))
    screen.blit(cue_text, text_rect)
    
    # "IMAGINE" instruction with better styling
    imagine_font = get_font(28)
    imagine_text = imagine_font.render("IMAGINE MOVEMENT", True, (220, 220, 220))
    imagine_rect = imagine_text.get_rect(center=(screen_width // 2, cue_y + 85))
    screen.blit(imagine_text, imagine_rect)
    
    # Enhanced timer bar
    remaining = word_cue.get_remaining_time()
    total_time = word_cue.think_duration
    time_ratio = remaining / total_time if total_time > 0 else 0
    
    bar_width = 280
    bar_height = 20
    bar_x = screen_width // 2 - bar_width // 2
    bar_y = cue_y + 115
    
    # Background bar with glow
    pygame.draw.rect(screen, (20, 20, 20), 
                    (bar_x - 3, bar_y - 3, bar_width + 6, bar_height + 6), 
                    border_radius=12)
    pygame.draw.rect(screen, (60, 60, 60), 
                    (bar_x - 2, bar_y - 2, bar_width + 4, bar_height + 4), 
                    2, border_radius=11)
    
    # Progress bar with gradient
    fill_width = int(bar_width * time_ratio)
    if fill_width > 0:
        bar_color = COL['countdown'] if time_ratio > 0.3 else COL['err']
        
        # Draw gradient fill
        progress_surface = pygame.Surface((fill_width, bar_height), pygame.SRCALPHA)
        for x in range(fill_width):
            fade = 1.0 - (x / bar_width) * 0.2
            gradient_color = tuple(int(c * fade) for c in bar_color)
            pygame.draw.line(progress_surface, (*gradient_color, 255), 
                           (x, 0), (x, bar_height), 1)
        
        # Create rounded corners mask
        pygame.draw.rect(screen, bar_color, 
                        (bar_x, bar_y, fill_width, bar_height), 
                        border_radius=9)
    
    # Timer text (simplified - no redundant display)
    timer_font = get_font(26)
    timer_text = timer_font.render(f"{remaining:.1f}s", True, COL['txt'])
    timer_rect = timer_text.get_rect(midright=(bar_x + bar_width + 40, bar_y + bar_height // 2))
    screen.blit(timer_text, timer_rect)
    
    # PAUSED indicator with animation
    if PAUSE_DURING_CUE:
        pause_font = get_font(32)
        pause_alpha = int(180 + 75 * math.sin(pygame.time.get_ticks() * 0.003))
        pause_text = pause_font.render("GAME PAUSED", True, (255, 255, 255, pause_alpha))
        pause_rect = pause_text.get_rect(center=(screen_width // 2, screen_height - 60))
        screen.blit(pause_text, pause_rect)

def draw_hud(screen, trial_num, total_trials, agent, target_coins, difficulty, last_action_info=None):
    """Draw the heads-up display with enhanced visibility"""
    font = get_font(36)
    small_font = get_font(28)
    
    # Top bar with gradient
    bar_height = 70
    bar_surface = pygame.Surface((screen.get_width(), bar_height), pygame.SRCALPHA)
    
    # Draw gradient background
    for y in range(bar_height):
        alpha = int(240 * (1 - y / bar_height))
        pygame.draw.rect(bar_surface, (0, 0, 0, alpha), (0, y, screen.get_width(), 1))
    
    screen.blit(bar_surface, (0, 0))
    
    # Add bottom border
    pygame.draw.line(screen, COL['lane_line'], (0, bar_height - 1), 
                    (screen.get_width(), bar_height - 1), 2)
    
    # Trial info with shadow
    trial_text = f"Trial {trial_num}/{total_trials}"
    # Shadow
    trial_shadow = font.render(trial_text, True, (0, 0, 0))
    screen.blit(trial_shadow, (23, 18))
    # Main text
    trial_surface = font.render(trial_text, True, COL['txt'])
    screen.blit(trial_surface, (20, 15))
    
    # Enhanced difficulty badge
    diff_text = difficulty.upper()
    diff_color = {
        'easy': COL['agent'],
        'medium': COL['countdown'],
        'hard': COL['err']
    }.get(difficulty, COL['txt'])
    
    diff_surface = small_font.render(diff_text, True, diff_color)
    diff_rect = diff_surface.get_rect(left=trial_surface.get_rect().right + 40, centery=30)
    
    # Badge background with glow
    badge_padding = 12
    badge_rect = diff_rect.inflate(badge_padding * 2, badge_padding)
    
    # Glow effect
    for i in range(2):
        glow_rect = badge_rect.inflate(i * 4, i * 4)
        pygame.draw.rect(screen, (*diff_color, 100 - i * 40), glow_rect, 2, border_radius=8)
    
    # Main badge
    pygame.draw.rect(screen, (*diff_color, 30), badge_rect, border_radius=8)
    pygame.draw.rect(screen, diff_color, badge_rect, 2, border_radius=8)
    screen.blit(diff_surface, diff_rect)
    
    # Coins collected (center) with enhanced styling
    coins_text = f"Coins: {agent.coins_collected}/{target_coins}"
    coins_color = COL['collected_coin'] if agent.coins_collected >= target_coins else COL['coin']
    
    # Add coin icon
    coin_icon_x = screen.get_width() // 2 - 120
    pygame.draw.circle(screen, COL['coin'], (coin_icon_x, 30), 15)
    pygame.draw.circle(screen, (255, 200, 0), (coin_icon_x, 30), 12)
    pygame.draw.circle(screen, (255, 240, 100), (coin_icon_x - 5, 25), 5)
    
    # Coins text with shadow
    coins_shadow = font.render(coins_text, True, (0, 0, 0))
    coins_rect_shadow = coins_shadow.get_rect(center=(screen.get_width() // 2 + 3, 33))
    screen.blit(coins_shadow, coins_rect_shadow)
    
    coins_surface = font.render(coins_text, True, coins_color)
    coins_rect = coins_surface.get_rect(center=(screen.get_width() // 2, 30))
    screen.blit(coins_surface, coins_rect)
    
    # Enhanced progress bar
    progress_width = 250
    progress_height = 10
    progress_x = screen.get_width() // 2 - progress_width // 2
    progress_y = 50
    
    # Background with border
    pygame.draw.rect(screen, (20, 20, 20), 
                    (progress_x - 2, progress_y - 2, progress_width + 4, progress_height + 4), 
                    border_radius=6)
    pygame.draw.rect(screen, (60, 60, 60), 
                    (progress_x - 1, progress_y - 1, progress_width + 2, progress_height + 2), 
                    1, border_radius=5)
    
    if target_coins > 0:
        fill_width = int(progress_width * (agent.coins_collected / target_coins))
        if fill_width > 0:
            # Gradient fill
            for x in range(fill_width):
                fade = 1.0 - (x / progress_width) * 0.3
                gradient_color = tuple(int(c * fade) for c in coins_color)
                pygame.draw.line(screen, gradient_color, 
                               (progress_x + x, progress_y), 
                               (progress_x + x, progress_y + progress_height), 1)
            
            # Add shine effect
            shine_width = min(30, fill_width)
            shine_x = progress_x + fill_width - shine_width
            for x in range(shine_width):
                shine_alpha = int(100 * (x / shine_width))
                pygame.draw.line(screen, (255, 255, 255, shine_alpha), 
                               (shine_x + x, progress_y), 
                               (shine_x + x, progress_y + progress_height), 1)
    
    # World position (right) with icon
    world_icon = "🌍"
    world_text = f"Lane: {agent.get_world_position()}"
    world_surface = small_font.render(world_text, True, COL['txt'])
    world_rect = world_surface.get_rect(right=screen.get_width() - 20, centery=30)
    screen.blit(world_surface, world_rect)
    
    # Action feedback display (if provided)
    if last_action_info and DEBUG_MODE:
        info_y = bar_height + 10
        info_bg = pygame.Surface((450, 130), pygame.SRCALPHA)
        info_bg.fill((0, 0, 0, 200))
        screen.blit(info_bg, (10, info_y))
        
        # Show what happened
        debug_font = get_font(24)
        lines = [
            f"Cue: {last_action_info.get('cue', 'N/A')}",
            f"You pressed: {last_action_info.get('user_input', 'N/A').replace('MOVE_', '')}",
            f"Agent moved: {last_action_info.get('executed', 'N/A').replace('MOVE_', '')}",
            f"Error Injected: {last_action_info.get('error_injected', False)} | Intent Followed: {last_action_info.get('user_intent_followed', 'N/A')}",
        ]
        
        # Add timing precision info
        timing_stats = precision_tracker.get_stats()
        if timing_stats:
            lines.append(f"Frame timing: {timing_stats['mean_ms']:.1f}±{timing_stats['std_ms']:.1f}ms (max: {timing_stats['max_ms']:.1f}ms)")
        
        for i, line in enumerate(lines):
            text_surface = debug_font.render(line, True, COL['txt'])
            screen.blit(text_surface, (20, info_y + 10 + i * 22))

# ─── EXPERIMENT FUNCTIONS ────────────────────────────────────

def select_experiment_settings():
    """Initial settings selection screen with enhanced UI"""
    screen = pygame.display.set_mode((900, 700))
    pygame.display.set_caption("Subway Surfers EEG Simulator - Settings")
    clock = pygame.time.Clock()
    
    font = get_font(42)
    small_font = get_font(28)
    
    # Settings
    settings = {
        'difficulty': None,
        'trials': None,
        'practice': PRACTICE_TRIALS  # Add practice trials by default
    }
    
    while True:
        screen.fill(COL['bg'])
        
        # Add subtle background pattern
        for y in range(0, screen.get_height(), 40):
            alpha = 20 + int(10 * math.sin(y * 0.02))
            pygame.draw.rect(screen, (50, 60, 80, alpha), (0, y, screen.get_width(), 20))
        
        # Single title without duplication
        title = font.render("EEG Motor Imagery Experiment", True, COL['txt'])
        title_rect = title.get_rect(center=(450, 60))
        screen.blit(title, title_rect)
        
        subtitle = small_font.render("3-Lane Endless Runner with Word Cues", True, COL['cue_left'])
        subtitle_rect = subtitle.get_rect(center=(450, 100))
        screen.blit(subtitle, subtitle_rect)
        
        # Instructions box with enhanced styling
        inst_y = 150
        inst_box = pygame.Rect(50, inst_y, 800, 120)
        
        # Box background with gradient
        inst_surface = pygame.Surface((800, 120), pygame.SRCALPHA)
        for y in range(120):
            fade = 1.0 - (y / 120) * 0.3
            color = tuple(int(c * fade) for c in (50, 50, 70))
            pygame.draw.rect(inst_surface, (*color, 200), (0, y, 800, 1))
        screen.blit(inst_surface, (50, inst_y))
        
        # Box border with glow
        for i in range(2):
            glow_alpha = 150 - i * 50
            pygame.draw.rect(screen, (*COL['lane_line'], glow_alpha), 
                           inst_box.inflate(i * 4, i * 4), 2, border_radius=10)
        
        instructions = [
            "• Agent moves based on YOUR KEY PRESS (not the cue word)",
            "• Press the arrow key matching the cue word shown",
            "• 30% of the time, the system will flip your action (MI decoder error)",
            "• This creates ErrP signals when you see incorrect execution"
        ]
        
        for i, inst in enumerate(instructions):
            inst_surface = small_font.render(inst, True, COL['txt'])
            screen.blit(inst_surface, (70, inst_y + 20 + i * 25))
        
        # Difficulty selection with enhanced visuals
        y_offset = 300
        diff_title = font.render("Select Difficulty:", True, COL['txt'])
        screen.blit(diff_title, (50, y_offset))
        
        difficulties = [
            ('1', 'easy', 'Easy', '3 coins, 3-5s think time', COL['agent']),
            ('2', 'medium', 'Medium', '5 coins, 2-4s think time', COL['countdown']),
            ('3', 'hard', 'Hard', '7 coins, 1-3s think time', COL['err'])
        ]
        
        for i, (key, value, name, desc, color) in enumerate(difficulties):
            selected = settings['difficulty'] == value
            
            # Option box with enhanced styling
            option_rect = pygame.Rect(70, y_offset + 50 + i * 45, 700, 40)
            
            if selected:
                # Selected state with glow
                for j in range(2):
                    glow_rect = option_rect.inflate(j * 4, j * 4)
                    pygame.draw.rect(screen, (*color, 100 - j * 40), glow_rect, 2, border_radius=5)
                pygame.draw.rect(screen, (*color, 40), option_rect, border_radius=5)
                pygame.draw.rect(screen, color, option_rect, 3, border_radius=5)
            else:
                # Hover effect
                pygame.draw.rect(screen, (*color, 20), option_rect, border_radius=5)
                pygame.draw.rect(screen, (*color, 80), option_rect, 1, border_radius=5)
            
            text_color = color if selected else (180, 180, 180)
            text = small_font.render(f"[{key}] {name}: {desc}", True, text_color)
            screen.blit(text, (80, y_offset + 60 + i * 45))
        
        # Number of trials with enhanced visuals
        y_offset = 480
        trials_title = font.render("Number of Trials:", True, COL['txt'])
        screen.blit(trials_title, (50, y_offset))
        
        trial_options = [
            ('Q', 5, 'Quick (5 trials)'),
            ('M', 10, 'Medium (10 trials)'),
            ('L', 20, 'Long (20 trials)')
        ]
        
        for i, (key, value, desc) in enumerate(trial_options):
            selected = settings['trials'] == value
            
            option_rect = pygame.Rect(70, y_offset + 50 + i * 45, 400, 40)
            
            if selected:
                # Selected state with glow
                for j in range(2):
                    glow_rect = option_rect.inflate(j * 4, j * 4)
                    pygame.draw.rect(screen, (*COL['collected_coin'], 100 - j * 40), glow_rect, 2, border_radius=5)
                pygame.draw.rect(screen, (*COL['collected_coin'], 40), option_rect, border_radius=5)
                pygame.draw.rect(screen, COL['collected_coin'], option_rect, 3, border_radius=5)
            else:
                pygame.draw.rect(screen, (*COL['collected_coin'], 20), option_rect, border_radius=5)
                pygame.draw.rect(screen, (*COL['collected_coin'], 80), option_rect, 1, border_radius=5)
            
            color = COL['collected_coin'] if selected else (180, 180, 180)
            text = small_font.render(f"[{key}] {desc}", True, color)
            screen.blit(text, (80, y_offset + 60 + i * 45))
        
        # Start button with animation
        if settings['difficulty'] and settings['trials']:
            start_rect = pygame.Rect(300, 630, 300, 50)
            
            # Pulsing effect
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.003))
            
            # Glow effect
            for i in range(3):
                glow_rect = start_rect.inflate(i * 8, i * 4)
                glow_alpha = int((100 - i * 25) * pulse)
                pygame.draw.rect(screen, (*COL['agent'], glow_alpha), glow_rect, 
                               border_radius=25)
            
            # Main button
            pygame.draw.rect(screen, COL['agent'], start_rect, border_radius=25)
            
            # Inner highlight
            highlight_rect = start_rect.inflate(-20, -20)
            pygame.draw.rect(screen, tuple(min(255, int(c * 1.2)) for c in COL['agent']), 
                           highlight_rect, 2, border_radius=20)
            
            start_text = font.render("Press SPACE", True, (0, 0, 0))
            start_rect_text = start_text.get_rect(center=(450, 655))
            screen.blit(start_text, start_rect_text)
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    settings['difficulty'] = 'easy'
                elif event.key == pygame.K_2:
                    settings['difficulty'] = 'medium'
                elif event.key == pygame.K_3:
                    settings['difficulty'] = 'hard'
                elif event.key == pygame.K_q:
                    settings['trials'] = 5
                elif event.key == pygame.K_m:
                    settings['trials'] = 10
                elif event.key == pygame.K_l:
                    settings['trials'] = 20
                elif event.key == pygame.K_SPACE and settings['difficulty'] and settings['trials']:
                    return settings
        
        clock.tick(60)

def get_difficulty_params(difficulty):
    """Get parameters based on difficulty"""
    params = {
        'easy': {
            'target_coins': 3,
            'min_think_time': 3.0,
            'max_think_time': 5.0,
            'spawn_interval': (120, 180)
        },
        'medium': {
            'target_coins': 5,
            'min_think_time': 2.0,
            'max_think_time': 4.0,
            'spawn_interval': (90, 150)
        },
        'hard': {
            'target_coins': 7,
            'min_think_time': 1.0,
            'max_think_time': 3.0,
            'spawn_interval': (60, 120)
        }
    }
    return params[difficulty]

def show_trial_transition(screen, trial_num, total_trials, difficulty, target_coins, is_practice=False):
    """Show transition screen between trials with enhanced visuals"""
    font = get_font(56)
    small_font = get_font(36)
    
    screen.fill(COL['bg'])
    
    # Draw animated background lanes with parallax effect
    lane_width = 200
    for i in range(5):
        x = i * lane_width - 100 + int(20 * math.sin(pygame.time.get_ticks() * 0.001 + i))
        color = COL['lane'] if i % 2 == 0 else COL['lane_alt']
        
        # Add gradient to lanes
        lane_surface = pygame.Surface((lane_width, screen.get_height()), pygame.SRCALPHA)
        for y in range(0, screen.get_height(), 20):
            fade = 1.0 - (y / screen.get_height()) * 0.3
            lane_color = tuple(int(c * fade) for c in color)
            pygame.draw.rect(lane_surface, (*lane_color, 200), (0, y, lane_width, 20))
        screen.blit(lane_surface, (x, 0))
    
    # Dark overlay with gradient
    overlay = pygame.Surface(screen.get_size())
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Animated particles in background
    for i in range(20):
        particle_y = (pygame.time.get_ticks() * 0.1 + i * 50) % screen.get_height()
        particle_x = 100 + i * 60
        particle_alpha = int(100 * abs(math.sin(i + pygame.time.get_ticks() * 0.001)))
        pygame.draw.circle(screen, (*COL['lane_line'], particle_alpha), 
                         (int(particle_x), int(particle_y)), 2)
    
    # Trial info box with enhanced styling
    box_width, box_height = 600, 400
    box_x = screen.get_width() // 2 - box_width // 2
    box_y = screen.get_height() // 2 - box_height // 2
    
    # Box shadow
    shadow_surface = pygame.Surface((box_width + 20, box_height + 20), pygame.SRCALPHA)
    pygame.draw.rect(shadow_surface, (0, 0, 0, 100), (0, 0, box_width + 20, box_height + 20), 
                    border_radius=25)
    screen.blit(shadow_surface, (box_x - 10, box_y - 10))
    
    # Box background with gradient
    box_surface = pygame.Surface((box_width, box_height), pygame.SRCALPHA)
    for y in range(box_height):
        fade = 1.0 - (y / box_height) * 0.2
        color = tuple(int(c * fade) for c in COL['cue_bg'])
        pygame.draw.rect(box_surface, (*color, 240), (0, y, box_width, 1))
    screen.blit(box_surface, (box_x, box_y))
    
    # Box border with glow
    for i in range(3):
        glow_alpha = 150 - i * 40
        pygame.draw.rect(screen, (*COL['lane_line'], glow_alpha), 
                        (box_x - i*2, box_y - i*2, box_width + i*4, box_height + i*4), 
                        2, border_radius=20)
    
    # Content
    y = box_y + 50
    
    # Trial number with shadow effect
    if is_practice:
        text = f"PRACTICE {trial_num}/{total_trials}"
        text_color = COL['collected_coin']  # Green for practice
    else:
        text = f"Trial {trial_num}/{total_trials}"
        text_color = COL['txt']
    
    trial_text = font.render(text, True, text_color)
    trial_shadow = font.render(text, True, (0, 0, 0))
    trial_rect = trial_text.get_rect(center=(screen.get_width() // 2, y))
    shadow_rect = trial_shadow.get_rect(center=(screen.get_width() // 2 + 3, y + 3))
    screen.blit(trial_shadow, shadow_rect)
    screen.blit(trial_text, trial_rect)
    
    # Goal with coin icon
    y += 80
    # Draw coin icon
    coin_x = screen.get_width() // 2 - 100
    pygame.draw.circle(screen, COL['coin'], (coin_x, y), 20)
    pygame.draw.circle(screen, (255, 200, 0), (coin_x, y), 17)
    pygame.draw.circle(screen, (255, 240, 100), (coin_x - 7, y - 7), 7)
    
    goal_text = small_font.render(f"Collect {target_coins} coins", True, COL['coin'])
    goal_rect = goal_text.get_rect(center=(screen.get_width() // 2 + 20, y))
    screen.blit(goal_text, goal_rect)
    
    # Instructions with cleaner design
    y += 60
    
    # "Watch for word cues" header
    header_text = small_font.render("Watch for word cues:", True, COL['txt'])
    header_rect = header_text.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(header_text, header_rect)
    
    # Word cue examples with better spacing
    y += 40
    arrow_offset = int(10 * math.sin(pygame.time.get_ticks() * 0.003))
    
    # LEFT cue with arrow
    left_group_x = screen.get_width() // 2 - 100
    # Draw arrow first (to the left of text)
    arrow_x = left_group_x - 60 + arrow_offset
    pygame.draw.polygon(screen, COL['cue_left'], [
        (arrow_x, y),
        (arrow_x + 15, y - 8),
        (arrow_x + 15, y + 8)
    ])
    # Then text
    left_text = small_font.render("LEFT", True, COL['cue_left'])
    left_rect = left_text.get_rect(center=(left_group_x, y))
    screen.blit(left_text, left_rect)
    
    # "or" text in the middle
    or_text = small_font.render("or", True, COL['txt'])
    or_rect = or_text.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(or_text, or_rect)
    
    # RIGHT cue with arrow
    right_group_x = screen.get_width() // 2 + 100
    # Draw text first
    right_text = small_font.render("RIGHT", True, COL['cue_right'])
    right_rect = right_text.get_rect(center=(right_group_x, y))
    screen.blit(right_text, right_rect)
    # Then arrow (to the right of text)
    arrow_x = right_group_x + 60 - arrow_offset
    pygame.draw.polygon(screen, COL['cue_right'], [
        (arrow_x, y),
        (arrow_x - 15, y - 8),
        (arrow_x - 15, y + 8)
    ])
    
    # "Imagine the movement!" instruction
    y += 40
    imagine_text = small_font.render("Imagine the movement!", True, COL['txt'])
    imagine_rect = imagine_text.get_rect(center=(screen.get_width() // 2, y))
    screen.blit(imagine_text, imagine_rect)
    
    # Start prompt with enhanced animation
    start_text = font.render("Press SPACE to start", True, COL['agent'])
    start_rect = start_text.get_rect(center=(screen.get_width() // 2, box_y + box_height - 60))
    
    # Multi-layer pulsing effect
    pulse = abs(math.sin(pygame.time.get_ticks() * 0.003))
    
    # Outer glow
    for i in range(4):
        glow_alpha = int((80 - i * 15) * pulse)
        glow_surface = pygame.Surface((start_rect.width + 60 + i*20, start_rect.height + 30 + i*10))
        glow_surface.set_alpha(glow_alpha)
        glow_surface.fill(COL['agent'])
        screen.blit(glow_surface, (start_rect.x - 30 - i*10, start_rect.y - 15 - i*5))
    
    # Button background
    button_rect = start_rect.inflate(60, 30)
    pygame.draw.rect(screen, COL['agent'], button_rect, border_radius=20)
    
    # Inner highlight
    highlight_rect = button_rect.inflate(-20, -10)
    pygame.draw.rect(screen, tuple(min(255, int(c * 1.3)) for c in COL['agent']), 
                    highlight_rect, 2, border_radius=15)
    
    # Text shadow
    shadow_text = font.render("Press SPACE to start", True, (0, 0, 0))
    shadow_rect = shadow_text.get_rect(center=(screen.get_width() // 2 + 2, box_y + box_height - 58))
    screen.blit(shadow_text, shadow_rect)
    
    screen.blit(start_text, start_rect)
    
    pygame.display.flip()
    
    # Wait for space (with proper timing)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return
        
        # Maintain frame rate during wait
        clock.tick(60)

def show_trial_complete(screen, agent, target_coins):
    """Show trial completion screen (FAST VERSION for EEG timing)"""
    font = get_font(72)
    small_font = get_font(42)
    
    # Quick success display without blocking animation
    overlay = pygame.Surface(screen.get_size())
    overlay.set_alpha(180)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    
    # Simple success message
    complete_text = font.render("Trial Complete!", True, COL['collected_coin'])
    complete_rect = complete_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 - 50))
    screen.blit(complete_text, complete_rect)
    
    # Score display
    coins_text = small_font.render(f"Coins: {agent.coins_collected}/{target_coins}", True, COL['coin'])
    coins_rect = coins_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 20))
    screen.blit(coins_text, coins_rect)
    
    # Continue prompt
    continue_text = small_font.render("Starting next trial...", True, COL['txt'])
    continue_rect = continue_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 80))
    screen.blit(continue_text, continue_rect)
    
    pygame.display.flip()
    
    # Brief non-blocking pause (500ms instead of 2600ms)
    pause_start = time.time()
    clock = pygame.time.Clock()
    while time.time() - pause_start < 0.5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                exit()
        clock.tick(60)
    
    return  # Old animation code removed for EEG timing precision

# ─── MAIN EXPERIMENT ─────────────────────────────────────────

def main():
    # Initialize Pygame once at the start
    pygame.init()
    
    # Load hand images for motor imagery cues
    global LEFT_HAND_IMG, RIGHT_HAND_IMG, HAND_IMG_SIZE
    
    try:
        left_path = os.path.join(os.path.dirname(__file__), 'left_hand.png')
        right_path = os.path.join(os.path.dirname(__file__), 'right_hand.png')
        
        if os.path.exists(left_path) and os.path.exists(right_path):
            LEFT_HAND_IMG = pygame.image.load(left_path)
            RIGHT_HAND_IMG = pygame.image.load(right_path)
            # Scale images to appropriate size
            LEFT_HAND_IMG = pygame.transform.scale(LEFT_HAND_IMG, HAND_IMG_SIZE)
            RIGHT_HAND_IMG = pygame.transform.scale(RIGHT_HAND_IMG, HAND_IMG_SIZE)
            print("Hand images loaded successfully")
            print(f"  LEFT_HAND_IMG size: {LEFT_HAND_IMG.get_size()}")
            print(f"  RIGHT_HAND_IMG size: {RIGHT_HAND_IMG.get_size()}")
        else:
            print(f"Hand images not found at {os.path.dirname(__file__)}")
            if not os.path.exists(left_path):
                print(f"  Missing: {left_path}")
            if not os.path.exists(right_path):
                print(f"  Missing: {right_path}")
    except Exception as e:
        print(f"Warning: Could not load hand images: {e}")
    
    # Get experiment settings
    settings = select_experiment_settings()
    difficulty = settings['difficulty']
    total_trials = settings['trials']
    
    # Get difficulty parameters
    params = get_difficulty_params(difficulty)
    
    # Create main game window (larger than settings window)
    screen = pygame.display.set_mode((1400, 900))
    pygame.display.set_caption("EEG Motor Imagery Experiment")
    clock = pygame.time.Clock()
    
    # Setup enhanced logging for ErrP analysis
    log_header = [
        # Timing & Trial Info
        "timestamp", "trial", "frame",
        # Action & Error Core
        "user_input", "executed_action", "is_error", "error_type",
        # Motor Imagery Context
        "cue_word", "cue_onset_time", "action_onset_time", "reaction_time",
        # Error Response
        "error_onset",
        # Task State
        "movement_required", "task_difficulty",
        # Performance
        "coins_collected", "trial_accuracy", "consecutive_errors"
    ]
    
    # Optimized Single-File CSV Event Logger
    class OptimizedEventLogger:
        """Single CSV file event logger optimized for EEG analysis"""
        
        def __init__(self, filepath):
            self.filepath = filepath
            self.session_start_time = get_sync_time()
            self.session_start_iso = datetime.now().isoformat()
            
            # Initialize event buffer with header
            self.headers = [
                'timestamp', 'event_type', 'trial_id', 'cue_class', 'predicted_class',
                'accuracy', 'error_type', 'confidence', 'reaction_time', 
                'coins_collected', 'artifact_flag', 'details'
            ]
            self.events = []
            
            # Current state tracking
            self.current_trial = {
                'trial_id': 0,
                'cue_onset_time': None,
                'cue_type': None,
                'imagery_start': None,
                'consecutive_errors': 0
            }
            
            # Add session start event
            self._add_event(
                timestamp=0.0,
                event_type='session_start',
                details=f'session_id={self.session_start_iso}'
            )
        
        def _add_event(self, timestamp, event_type, trial_id='', cue_class='', 
                       predicted_class='', accuracy='', error_type='', confidence='',
                       reaction_time='', coins_collected='', artifact_flag='', details=''):
            """Add an event to the buffer"""
            self.events.append({
                'timestamp': f'{timestamp:.3f}',
                'event_type': event_type,
                'trial_id': str(trial_id),
                'cue_class': cue_class,
                'predicted_class': predicted_class,
                'accuracy': accuracy,
                'error_type': error_type,
                'confidence': confidence,
                'reaction_time': reaction_time,
                'coins_collected': str(coins_collected) if coins_collected else '',
                'artifact_flag': artifact_flag,
                'details': details
            })
        
        def start_trial(self, trial_num, difficulty):
            """Initialize new trial tracking"""
            timestamp = get_sync_time() - self.session_start_time
            
            self.current_trial = {
                'trial_id': trial_num,
                'cue_onset_time': None,
                'cue_type': None,
                'imagery_start': None,
                'consecutive_errors': 0
            }
            
            self._add_event(
                timestamp=timestamp,
                event_type='trial_start',
                trial_id=trial_num,
                details=f'difficulty={difficulty}'
            )
        
        def log_cue(self, cue_word, agent_position, coins_collected):
            """Log cue presentation event"""
            timestamp = get_sync_time() - self.session_start_time
            self.current_trial['cue_onset_time'] = timestamp
            self.current_trial['cue_type'] = cue_word
            self.current_trial['imagery_start'] = timestamp
            
            self._add_event(
                timestamp=timestamp,
                event_type=f'cue_{cue_word.lower()}',
                trial_id=self.current_trial['trial_id'],
                cue_class=cue_word.lower(),
                details=f'agent_pos={agent_position}'
            )
        
        def log_action(self, user_input, executed_action, agent_position, 
                       coins_collected, movement_required):
            """Log an action event with all ErrP-relevant information"""
            timestamp = get_sync_time() - self.session_start_time
            
            # End imagery period if active
            if self.current_trial['imagery_start']:
                confidence = 0.7 + (0.2 if user_input == executed_action else -0.1)
                self._add_event(
                    timestamp=timestamp,
                    event_type='imagery_end',
                    trial_id=self.current_trial['trial_id'],
                    confidence=f'{confidence:.2f}'
                )
            
            # Calculate reaction time
            reaction_time = 0
            if self.current_trial['cue_onset_time'] and user_input != "NONE":
                reaction_time = (timestamp - self.current_trial['cue_onset_time']) * 1000
            
            # Determine accuracy and error type
            user_intent_followed = (user_input == executed_action)
            is_primary_error = not user_intent_followed and user_input != "NONE"
            
            is_secondary_error = False
            if self.current_trial['cue_type']:
                expected_action = f"MOVE_{self.current_trial['cue_type']}"
                is_secondary_error = (user_input != expected_action and user_input != "NONE")
            
            # Classify error type
            if not is_primary_error and not is_secondary_error:
                error_type = ''
                accuracy = True
            elif is_primary_error and not is_secondary_error:
                error_type = 'primary'
                accuracy = False
            elif not is_primary_error and is_secondary_error:
                error_type = 'secondary'
                accuracy = False
            else:
                error_type = 'combined'
                accuracy = False
            
            # Log feedback event based on user intent
            feedback_type = 'feedback_error' if is_primary_error else 'feedback_correct'
            self._add_event(
                timestamp=timestamp,
                event_type=feedback_type,
                trial_id=self.current_trial['trial_id'],
                cue_class=self.current_trial['cue_type'].lower() if self.current_trial['cue_type'] else '',
                predicted_class=executed_action.replace('MOVE_', '').lower() if executed_action != 'NONE' else '',
                accuracy=str(accuracy),
                error_type=error_type,
                reaction_time=f'{reaction_time:.0f}' if reaction_time > 0 else '',
                coins_collected=coins_collected,
                details=f'user_input={user_input}'
            )
            
            # Log ErrP event if primary error occurred
            if is_primary_error:
                # ErrP latency is in REAL TIME for consistent neural response timing
                # This is critical: ErrP components occur at fixed latencies (250-350ms)
                # regardless of game speed
                error_latency = 0.3  # 300ms typical ErrP latency in REAL TIME
                confidence = 0.85 if not is_secondary_error else 0.65
                
                self._add_event(
                    timestamp=timestamp + error_latency,
                    event_type='primary_errp',
                    trial_id=self.current_trial['trial_id'],
                    confidence=f'{confidence:.2f}',
                    details=f'latency={error_latency*1000:.0f}ms_realtime'
                )
                
                self.current_trial['consecutive_errors'] += 1
            else:
                self.current_trial['consecutive_errors'] = 0
            
            # Clear cue after action
            if user_input != "NONE":
                self.current_trial['cue_onset_time'] = None
                self.current_trial['cue_type'] = None
                self.current_trial['imagery_start'] = None
            
            return is_primary_error
        
        def log_coin_event(self, event_type, coin_world_lane, agent_position, coins_collected):
            """Log coin-related events"""
            timestamp = get_sync_time() - self.session_start_time
            relative_pos = coin_world_lane - agent_position
            
            self._add_event(
                timestamp=timestamp,
                event_type=event_type.lower(),
                trial_id=self.current_trial['trial_id'],
                coins_collected=coins_collected,
                details=f'relative_pos={relative_pos}'
            )
        
        def log_cue_timeout(self, agent_position, coins_collected):
            """Log when a cue times out without action"""
            timestamp = get_sync_time() - self.session_start_time
            
            # End imagery period
            if self.current_trial['imagery_start']:
                self._add_event(
                    timestamp=timestamp,
                    event_type='imagery_end',
                    trial_id=self.current_trial['trial_id'],
                    confidence='0.0',
                    details='timeout'
                )
            
            self._add_event(
                timestamp=timestamp,
                event_type='cue_timeout',
                trial_id=self.current_trial['trial_id'],
                cue_class=self.current_trial['cue_type'].lower() if self.current_trial['cue_type'] else '',
                details='no_action'
            )
            
            self.current_trial['cue_onset_time'] = None
            self.current_trial['cue_type'] = None
            self.current_trial['imagery_start'] = None
        
        def log_trial_complete(self, coins_collected):
            """Log trial completion"""
            timestamp = get_sync_time() - self.session_start_time
            
            self._add_event(
                timestamp=timestamp,
                event_type='trial_end',
                trial_id=self.current_trial['trial_id'],
                coins_collected=coins_collected,
                details=f'consecutive_errors={self.current_trial["consecutive_errors"]}'
            )
        
        def save(self):
            """Save all logged data to CSV file"""
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                writer.writerows(self.events)
            
            print(f"Optimized event log saved to: {self.filepath}")
            print(f"Total events logged: {len(self.events)}")
            
            # Print summary statistics
            trial_count = sum(1 for e in self.events if e['event_type'] == 'trial_start')
            error_count = sum(1 for e in self.events if 'errp' in e['event_type'])
            print(f"Trials: {trial_count}, ErrP events: {error_count}")
    # Keep the old logger for backward compatibility
    class ErrPLogger:
        def __init__(self, filepath):
            self.filepath = filepath
            self.event_header = [
                # Event Info
                "timestamp", "trial", "event_type", "event_details",
                # Action & Error Data (when applicable)
                "user_input", "executed_action", "is_error", "error_type",
                # Motor Imagery Context
                "cue_word", "cue_onset_time", "reaction_time",
                # Task State
                "agent_world_position", "movement_required",
                # Performance
                "coins_collected", "trial_accuracy", "consecutive_errors"
            ]
            self.rows = [self.event_header]
            self.current_trial = {
                'trial': 0,
                'cue_onset_time': None,
                'cue_word': None,
                'total_actions': 0,
                'correct_actions': 0,
                'consecutive_errors': 0,
                'difficulty': 'medium',
                'trial_start_time': None
            }
            
        def start_trial(self, trial_num, difficulty):
            """Initialize new trial tracking and log trial start event"""
            timestamp = get_sync_time()
            self.current_trial = {
                'trial': trial_num,
                'cue_onset_time': None,
                'cue_word': None,
                'total_actions': 0,
                'correct_actions': 0,
                'consecutive_errors': 0,
                'difficulty': difficulty,
                'trial_start_time': timestamp
            }
            
            # Log trial start event
            self.log_event("TRIAL_START", f"difficulty={difficulty}", agent_position=0)
            
        def log_event(self, event_type, event_details="", agent_position=None, coins_collected=0):
            """Log any event with minimal redundancy"""
            timestamp = get_sync_time()
            
            # Calculate trial accuracy
            trial_accuracy = 0.0
            if self.current_trial['total_actions'] > 0:
                trial_accuracy = self.current_trial['correct_actions'] / self.current_trial['total_actions']
            
            row = [
                # Event Info
                f"{timestamp:.6f}",
                self.current_trial['trial'],
                event_type,
                event_details,
                # Action & Error Data (empty for non-action events)
                "", "", "", "",
                # Motor Imagery Context
                self.current_trial['cue_word'] or "",
                f"{self.current_trial['cue_onset_time']:.6f}" if self.current_trial['cue_onset_time'] else "",
                "",
                # Task State
                agent_position if agent_position is not None else "",
                "",
                # Performance
                coins_collected,
                f"{trial_accuracy:.3f}",
                self.current_trial['consecutive_errors']
            ]
            
            self.rows.append(row)
            
        def log_cue(self, cue_word, agent_position, coins_collected):
            """Log cue presentation event"""
            self.current_trial['cue_onset_time'] = get_sync_time()
            self.current_trial['cue_word'] = cue_word
            self.log_event("CUE_PRESENTED", f"word={cue_word}", agent_position, coins_collected)
            
        def log_action(self, user_input, executed_action, agent_position, coins_collected, movement_required):
            """Log an action event with all ErrP-relevant information"""
            timestamp = get_sync_time()
            
            # Determine error types
            is_error = user_input != executed_action and user_input != "NONE"
            
            # Classify error type
            if not is_error:
                error_type = "NO_ERROR"
            else:
                # This is a system error (MI decoder error)
                error_type = "MI_DECODER_ERROR"
            
            # Also check if user pressed wrong key (different from system error)
            user_error = False
            if self.current_trial['cue_word']:
                expected_action = f"MOVE_{self.current_trial['cue_word']}"
                if user_input != expected_action and user_input != "NONE":
                    user_error = True
                    error_type = "USER_ERROR" if not is_error else "USER_ERROR+MI_ERROR"
            
            # Calculate reaction time if there was a cue
            reaction_time = None
            if self.current_trial['cue_onset_time'] and user_input != "NONE":
                reaction_time = timestamp - self.current_trial['cue_onset_time']
            
            # Track performance
            if user_input != "NONE":
                self.current_trial['total_actions'] += 1
                if not is_error:
                    self.current_trial['correct_actions'] += 1
                    self.current_trial['consecutive_errors'] = 0
                else:
                    self.current_trial['consecutive_errors'] += 1
            
            # Calculate trial accuracy
            trial_accuracy = 0.0
            if self.current_trial['total_actions'] > 0:
                trial_accuracy = self.current_trial['correct_actions'] / self.current_trial['total_actions']
            
            # Create row with action data
            row = [
                # Event Info
                f"{timestamp:.6f}",
                self.current_trial['trial'],
                "ACTION_EXECUTED" if not is_error else "ACTION_ERROR",
                f"{user_input}->{executed_action}",
                # Action & Error Data
                user_input,
                executed_action,
                is_error,
                error_type,
                # Motor Imagery Context
                self.current_trial['cue_word'] or "",
                f"{self.current_trial['cue_onset_time']:.6f}" if self.current_trial['cue_onset_time'] else "",
                f"{reaction_time:.3f}" if reaction_time else "",
                # Task State
                agent_position,
                movement_required,
                # Performance
                coins_collected,
                f"{trial_accuracy:.3f}",
                self.current_trial['consecutive_errors']
            ]
            
            self.rows.append(row)
            
            # Clear cue after action
            if user_input != "NONE":
                self.current_trial['cue_onset_time'] = None
                self.current_trial['cue_word'] = None
                
            return is_error
            
        def log_coin_event(self, event_type, coin_world_lane, agent_position, coins_collected):
            """Log coin-related events"""
            details = f"coin_lane={coin_world_lane}, relative_pos={coin_world_lane - agent_position}"
            self.log_event(event_type, details, agent_position, coins_collected)
            
        def log_cue_timeout(self, agent_position, coins_collected):
            """Log when a cue times out without action"""
            self.log_event("CUE_TIMEOUT", f"cue={self.current_trial['cue_word']}", agent_position, coins_collected)
            self.current_trial['cue_onset_time'] = None
            self.current_trial['cue_word'] = None
            
        def log_trial_complete(self, coins_collected):
            """Log trial completion"""
            trial_duration = get_sync_time() - self.current_trial['trial_start_time']
            self.log_event("TRIAL_COMPLETE", f"duration={trial_duration:.2f}s", coins_collected=coins_collected)
            
        def save(self):
            """Save all logged data"""
            with open(self.filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.rows)
            print(f"Event-based ErrP log saved to: {self.filepath}")
            print(f"Total events logged: {len(self.rows) - 1}")
    
    # Initialize optimized CSV event logger
    errp_logger = OptimizedEventLogger(LOG_PATH)
    
    # Add practice trials to the beginning
    practice_trials = settings.get('practice', PRACTICE_TRIALS)
    all_trials = []
    
    # Add practice trials
    for i in range(practice_trials):
        all_trials.append(('practice', i + 1))
    
    # Add main trials
    for i in range(total_trials):
        all_trials.append(('main', i + 1))
    
    # Run all trials
    for trial_type, trial_num in all_trials:
        trial_setup_start = time.perf_counter()
        
        # Show transition screen
        if trial_type == 'practice':
            show_trial_transition(screen, trial_num, practice_trials, difficulty, params['target_coins'], is_practice=True)
        else:
            show_trial_transition(screen, trial_num, total_trials, difficulty, params['target_coins'])
        
        transition_end = time.perf_counter()
        
        # Initialize trial
        world = EndlessWorld(screen.get_width(), screen.get_height())
        agent = Agent(world)
        coin_manager = CoinManager(world, params['target_coins'], logger=errp_logger)
        coin_manager.min_spawn_interval = params['spawn_interval'][0]
        coin_manager.max_spawn_interval = params['spawn_interval'][1]
        word_cue = WordCue()
        
        # Spawn first coin immediately
        coin_manager.spawn_coin(agent.get_world_position(), agent.coins_collected)
        
        init_end = time.perf_counter()
        
        if DEBUG_MODE:
            print(f"\n=== TRIAL {trial_num} SETUP TIMING ===")
            print(f"Transition screen: {(transition_end - trial_setup_start)*1000:.1f}ms")
            print(f"Object initialization: {(init_end - transition_end)*1000:.1f}ms")
            print(f"Total setup: {(init_end - trial_setup_start)*1000:.1f}ms")
        
        # Trial state
        trial_running = True
        step = 0
        last_cue_time = 0
        cue_cooldown = 0
        action_count = 0
        last_action_info = None  # For debug display
        
        # Initialize ErrP logger for this trial (skip for practice)
        if trial_type == 'main':
            errp_logger.start_trial(trial_num, difficulty)
        
        # Send trial start marker
        marker_prefix = "PRACTICE_" if trial_type == 'practice' else ""
        trial_start_ts = push_immediate(f"{marker_prefix}TRIAL_START_{trial_num}_{difficulty}")
        
        while trial_running:
            frame_start_time = get_sync_time()
            precision_tracker.mark_frame()
            
            # Handle events
            intended_action = "NONE"
            key_event_ts = None
            
            # Get events (keyboard simulation of MI classifier)
            events = pygame.event.get()
            
            # In real EEG integration, this would be replaced with:
            # if word_cue.active and not action_taken:
            #     mi_output = get_mi_classifier_output()  # Returns "MOVE_LEFT", "MOVE_RIGHT", or "NONE"
            #     if mi_output != "NONE":
            #         intended_action = mi_output
            #         key_event_ts = get_sync_time()
            #         outlet.push_sample([f"MI_OUTPUT_{intended_action}"], key_event_ts)
            #         action_taken = True  # Prevent multiple classifications per cue
            
            for event in events:
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    errp_logger.save()
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN and event.key in SUBWAY_ACTIONS and word_cue.active:
                    # Keyboard simulates MI classifier output
                    intended_action = SUBWAY_ACTIONS[event.key]
                    key_event_ts = get_sync_time()  # Timestamp when "classifier" produces output
                    outlet.push_sample([f"KEY_{intended_action}"], key_event_ts)
            
            # Process action if any (only one action allowed per cue)
            if intended_action != "NONE" and word_cue.active and action_count == 0:
                processing_start = get_sync_time()
                step += 1
                action_count += 1
                
                # Store the cue word before clearing it
                original_cue_word = word_cue.cue_word
                
                # IMPORTANT: The action is based on what key the user pressed, NOT the cue word
                # This allows us to detect when users press the wrong key
                if DEBUG_MODE:
                    print(f"\n=== ACTION PROCESSING ===")
                    print(f"Cue shown: {original_cue_word}")
                    print(f"User pressed: {intended_action}")
                
                # Simulate MI decoder error (reduced rate for practice)
                actual_action = intended_action
                error_injected = False
                
                # Use reduced error rate during practice
                error_rate = ERROR_P * 0.3 if trial_type == 'practice' else ERROR_P  # 9% for practice, 30% for main
                
                if not VERIFY_MODE and random.random() < error_rate:
                    # Inject error - flip the direction
                    actual_action = "MOVE_RIGHT" if intended_action == "MOVE_LEFT" else "MOVE_LEFT"
                    error_injected = True
                    error_ts = push_immediate("ERROR_INJECTED")
                    if DEBUG_MODE:
                        print(f"Error injected! Action flipped to: {actual_action}")
                else:
                    if DEBUG_MODE:
                        print(f"No error injected. Executing user's action: {actual_action}")
                
                # Determine movement required
                if coin_manager.active_coin:
                    coin = coin_manager.active_coin
                    relative_lane = coin.get_relative_lane(agent.get_world_position())
                    if relative_lane < 0:
                        movement_required = "LEFT"
                    elif relative_lane > 0:
                        movement_required = "RIGHT"
                    else:
                        movement_required = "NONE"
                else:
                    movement_required = "NONE"
                
                # Log the action with ErrP analysis data
                is_error_occurred = errp_logger.log_action(
                    intended_action, 
                    actual_action, 
                    agent.get_world_position(),
                    agent.coins_collected,
                    movement_required
                )
                
                # Check multiple aspects of correctness
                user_pressed_correct_key = (original_cue_word == intended_action.replace("MOVE_", ""))
                final_action_correct = (original_cue_word == "LEFT" and actual_action == "MOVE_LEFT") or \
                                     (original_cue_word == "RIGHT" and actual_action == "MOVE_RIGHT")
                
                # The key insight: an error occurs when the system doesn't execute what the user intended
                user_intent_followed = (intended_action == actual_action)
                
                if DEBUG_MODE:
                    print(f"User pressed correct key: {user_pressed_correct_key}")
                    print(f"Final action correct: {final_action_correct}")
                    print(f"User intent followed: {user_intent_followed}")
                    print(f"======================")
                
                # Set agent feedback based on whether user intent was followed
                agent.set_action_feedback(user_intent_followed)
                
                # Store action info for debug display
                last_action_info = {
                    'cue': original_cue_word,
                    'user_input': intended_action,
                    'executed': actual_action,
                    'error_injected': error_injected,
                    'user_correct': user_pressed_correct_key,
                    'final_correct': final_action_correct,
                    'user_intent_followed': user_intent_followed
                }
                
                # Execute action
                action_executed = False
                if actual_action == "MOVE_LEFT":
                    action_executed = agent.move_left()
                elif actual_action == "MOVE_RIGHT":
                    action_executed = agent.move_right()
                
                # Don't disable cue - let it run for full 5 seconds
                # This allows continued MI detection for the full window
                # word_cue.active = False  # Commented out - cue ends naturally after 5s
                
                # Send action marker
                push_immediate(f"ACTION_{actual_action}_POS_{agent.get_world_position()}")
                
                # Send feedback marker
                # Error occurs when user intent is not followed (i.e., error injection changed the action)
                if not user_intent_followed:
                    push_immediate("FEEDBACK_ERROR")
                else:
                    push_immediate("FEEDBACK_CORRECT")
                
                # Set cooldown after action
                cue_cooldown = 90  # 1.5 second cooldown at 60fps
            # Note: No need to log every frame - only log events!
            
            # Update game state (IMPORTANT: update step counter)
            step += 1
            
            # Check if game should be in slow motion
            if SLOW_MOTION_DURING_CUE and word_cue.active:
                speed_factor = SLOW_MOTION_FACTOR
            else:
                speed_factor = 1.0
            
            # Update world and agent with speed factor
            world.update(speed_factor)
            agent.update()
            
            # Update coins with speed factor and cue state
            coin_manager.update(agent, speed_factor=speed_factor, cue_active=word_cue.active)
            
            # Update word cue with speed factor
            cue_ended = word_cue.update(speed_factor)
            
            if cue_ended:
                # Log if cue ended without action
                if action_count == 0:
                    push_immediate("CUE_TIMEOUT_NO_ACTION")
                    errp_logger.log_cue_timeout(agent.get_world_position(), agent.coins_collected)
                    # Add cooldown after timeout to prevent immediate new cue
                    cue_cooldown = 120  # 2 second cooldown at 60fps
                action_count = 0
            
            # Update cooldowns with speed factor
            if cue_cooldown > 0:
                cue_cooldown -= speed_factor
            
            # Check for new cue opportunity
            if not word_cue.active and cue_cooldown <= 0:
                # Look for the active coin
                if coin_manager.active_coin and not coin_manager.active_coin.collected:
                    coin = coin_manager.active_coin
                    relative_lane = coin.get_relative_lane(agent.get_world_position())
                    
                    # Verify coin is actually on screen and valid
                    # Coins spawn at Y=-100, so wait until they're properly on screen
                    if coin.y_position < 50:  # Too high up, not visible yet
                        # Coin just spawned, wait for it to come into view
                        if DEBUG_MODE and coin.y_position < 0:
                            print(f"⚠️ Coin just spawned, waiting... Y={coin.y_position}")
                    elif coin.y_position > 500:  # Too low, about to be collected/missed
                        # Coin is too close, no time for cue
                        pass
                    # Only cue if coin is properly visible and in good position
                    elif coin.is_visible(agent.get_world_position()) and coin.y_position > 150 and coin.y_position < 450:
                        # Only cue for coins that require movement
                        if relative_lane != 0:
                            # Determine direction needed
                            cue_word = "LEFT" if relative_lane < 0 else "RIGHT"
                            
                            # Show cue with fixed 5s think time for consistent EEG epochs
                            think_time = 5.0  # Fixed 5s window for all cues
                            
                            # CRITICAL FOR EEG: Send baseline marker BEFORE cue
                            baseline_start = push_immediate("BASELINE_START")
                            
                            # Schedule cue presentation after baseline period
                            # This ensures clean EEG baseline for epoch extraction
                            baseline_frames = int(BASELINE_DURATION * FPS)
                            
                            # We'll need to wait for baseline before showing cue
                            # For now, show immediately but mark the baseline period
                            push_immediate(f"BASELINE_END")
                            
                            word_cue.show_cue(cue_word, think_time)
                            # Log the cue event
                            errp_logger.log_cue(cue_word, agent.get_world_position(), agent.coins_collected)
                            last_cue_time = get_sync_time()
                            
                            action_count = 0
                            cue_cooldown = 30  # Short cooldown after showing cue
            
            # Draw everything
            draw_endless_world(screen, world, agent, coin_manager.coins, word_cue)
            draw_hud(screen, trial_num, total_trials, agent, params['target_coins'], difficulty, last_action_info)
            
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
            
            # Precise frame timing for synchronization
            verify_sync()  # Check timing consistency
            frame_end_time = get_sync_time()
            pygame.display.flip()
            
            # Track frame timing for sync verification
            clock.tick(FPS)
            actual_fps = clock.get_fps()
            if actual_fps > 0 and actual_fps < FPS * 0.9:  # More than 10% drop
                print(f"WARNING: FPS dropped to {actual_fps:.1f}")
        
        # Inter-trial interval (non-blocking)
        iti_start = time.time()
        # Pre-create ITI screen once to avoid repeated Surface creation
        if not hasattr(main, 'iti_screen'):
            main.iti_screen = pygame.Surface(screen.get_size())
            main.iti_screen.fill(COL['bg'])
        
        # Show ITI message
        font = get_font(48)
        iti_text = font.render("Preparing next trial...", True, COL['txt'])
        iti_rect = iti_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2))
        
        while time.time() - iti_start < ITI:
            # Clear event queue properly to prevent buildup
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    exit()
            
            # Show ITI screen
            screen.blit(main.iti_screen, (0, 0))
            screen.blit(iti_text, iti_rect)
            
            # Show countdown
            remaining = ITI - (time.time() - iti_start)
            countdown_text = font.render(f"{remaining:.1f}s", True, COL['countdown'])
            countdown_rect = countdown_text.get_rect(center=(screen.get_width() // 2, screen.get_height() // 2 + 60))
            screen.blit(countdown_text, countdown_rect)
            
            pygame.display.flip()
            clock.tick(60)  # Maintain timing
    
    # Experiment complete
    errp_logger.save()
    pygame.quit()
    
    # Show timing statistics
    timing_stats = precision_tracker.get_stats()
    
    print(f"\nExperiment complete!")
    print(f"Event log saved to: {LOG_PATH}")
    print(f"Total trials: {total_trials}")
    print(f"Difficulty: {difficulty}")
    
    if timing_stats:
        print(f"\n📊 Timing Precision Statistics:")
        print(f"  Mean frame interval: {timing_stats['mean_ms']:.2f}ms")
        print(f"  Std deviation: {timing_stats['std_ms']:.2f}ms")
        print(f"  Max frame interval: {timing_stats['max_ms']:.2f}ms")
        print(f"  Target: 16.67ms (60 FPS)")
        
        # Assess precision quality
        if timing_stats['mean_ms'] < 20 and timing_stats['std_ms'] < 5:
            print(f"  ✅ Excellent timing precision for ERP research")
        elif timing_stats['mean_ms'] < 25:
            print(f"  ⚠️  Acceptable timing, but could be better")
        else:
            print(f"  ❌ Poor timing precision - may affect ERP quality")
    print("\nKey features:")
    print("✓ 3-lane view with agent always centered")
    print("✓ Endless world that shifts around the agent")
    print("✓ Coins spawn away from agent to encourage movement")
    print("✓ Word cues with configurable think time (1-5s)")
    print("✓ 30% error injection for ErrP elicitation")
    print("\nOptimized CSV Event Format:")
    print("✓ Single file with all events in chronological order")
    print("✓ Event types: session_start, trial_start/end, cue_*, feedback_*, primary_errp")
    print("✓ Includes motor imagery periods (imagery_end events)")
    print("✓ Error classification: primary (system), secondary (user), combined")
    print("✓ Rich details field for additional context")
    print("\nAdvantages of Single CSV Format:")
    print("- Simple to load with pandas: df = pd.read_csv(log_path)")
    print("- All events in one timeline - easy to analyze sequences")
    print("- Compatible with standard data analysis tools")
    print("- Sparse format - empty cells for non-applicable fields")
    print("- Easy to filter by event_type, trial_id, etc.")

def read_csv_log(filepath):
    """Utility function to read and display CSV log contents using pandas"""
    try:
        import pandas as pd
        
        print(f"\nReading CSV log: {filepath}")
        print("=" * 60)
        
        # Load the data
        df = pd.read_csv(filepath)
        
        print(f"Total events: {len(df)}")
        print(f"Event types: {df['event_type'].value_counts().to_dict()}")
        
        # Show trial summary
        trials = df[df['event_type'] == 'trial_start']
        print(f"\nTrials: {len(trials)}")
        
        # Show error summary
        errors = df[df['event_type'] == 'primary_errp']
        print(f"Primary errors: {len(errors)}")
        
        # Show first few events
        print("\nFirst 10 events:")
        print(df[['timestamp', 'event_type', 'trial_id', 'cue_class', 'predicted_class', 'error_type']].head(10))
        
        # Calculate average reaction time
        feedback_events = df[df['reaction_time'].notna()]
        if not feedback_events.empty:
            avg_rt = feedback_events['reaction_time'].astype(float).mean()
            print(f"\nAverage reaction time: {avg_rt:.0f}ms")
        
        print("=" * 60)
        print("\nTo analyze in detail:")
        print("df = pd.read_csv(filepath)")
        print("df[df['event_type'] == 'primary_errp']  # View all error events")
        print("df[df['trial_id'] == '1']  # View all events from trial 1")
        
    except ImportError:
        print("pandas not installed. Use: pip install pandas")
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    main()