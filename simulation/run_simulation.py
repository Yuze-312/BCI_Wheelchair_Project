import pygame, random, json, csv, time, os, threading
from collections import deque
from pylsl import StreamInfo, StreamOutlet, local_clock

# ─── LSL STREAM ──────────────────────────────────────────────
info   = StreamInfo("ErrP_Markers", "Markers", 1, 0, "string", "errp-maze-001")
outlet = StreamOutlet(info)

# Synchronized timing functions
def get_sync_time():
    """Use LSL time consistently throughout"""
    return local_clock()

def push_immediate(tag):
    """Send marker with immediate timestamp"""
    timestamp = get_sync_time()
    outlet.push_sample([tag], timestamp)
    return timestamp

# ─── PATHS ───────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MAZE_CFG  = os.path.join(BASE_DIR, "configs", "maze_layout.json") 
LOG_PATH  = os.path.join(BASE_DIR, "logs", f"errp_{int(time.time())}.csv")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

# ─── CONSTANTS ───────────────────────────────────────────────
ERROR_P, RESP_WIN, ITI, FPS = 0.5, 1.0, 1.5, 60
MAX_PROCESSING_DELAY = 0.010  # 10ms warning threshold
COL = dict(bg=(30,30,30), wall=(200,200,200), path=(50,50,50),
           star=(255,215,0), dot=(0,180,0), err=(180,0,0), txt=(255,255,255))

# Control schemes for different modes
BCI_ACTIONS = {
    pygame.K_LEFT:  "TURN_LEFT",
    pygame.K_RIGHT: "TURN_RIGHT", 
    pygame.K_UP:    "MOVE_FORWARD",
}

CLASSIC_ACTIONS = {
    pygame.K_UP:    "UP",
    pygame.K_DOWN:  "DOWN",
    pygame.K_LEFT:  "LEFT",
    pygame.K_RIGHT: "RIGHT",
}

BCI_ACTION_CODE = {"NONE": 0, "TURN_LEFT": 1, "TURN_RIGHT": 2, "MOVE_FORWARD": 3}
CLASSIC_ACTION_CODE = {"NONE": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4}

# ─── HELPERS ─────────────────────────────────────────────────
def load_maze_with_size(cfg, mid, size_setting):
    """Load maze with size-aware configuration"""
    with open(cfg) as f:
        maze_data = json.load(f)
    
    if mid in maze_data:
        m = maze_data[mid]
        return m["layout"], tuple(m["start"])
    else:
        # Fallback: if small maze doesn't exist, create a scaled version
        if size_setting == "SMALL" and mid.startswith("small_"):
            base_id = mid.replace("small_", "")
            if base_id in maze_data:
                print(f"Warning: {mid} not found, using scaled {base_id}")
                return scale_maze_down(maze_data[base_id]["layout"], maze_data[base_id]["start"])
        
        raise ValueError(f"Maze {mid} not found in configuration")

def scale_maze_down(layout, start):
    """Scale down a large maze to create a smaller version"""
    # Simple scaling: take every other row/column from the center area
    rows, cols = len(layout), len(layout[0])
    
    # Extract 5x5 area from center of 9x9 maze
    start_row, start_col = 2, 2  # Start from (2,2) to get center 5x5
    small_layout = []
    
    for r in range(5):
        row = []
        for c in range(5):
            orig_r = start_row + r
            orig_c = start_col + c
            if orig_r < rows and orig_c < cols:
                row.append(layout[orig_r][orig_c])
            else:
                row.append(1)  # Wall if out of bounds
        small_layout.append(row)
    
    # Adjust start position for smaller maze
    new_start = (1, 1)  # Safe starting position
    
    # Ensure start position is valid (not a wall)
    if small_layout[new_start[1]][new_start[0]] == 1:
        # Find first open space
        for r in range(5):
            for c in range(5):
                if small_layout[r][c] == 0:
                    new_start = (c, r)
                    break
    
    return small_layout, new_start

def stars(layout):
    return {(x,y) for y,row in enumerate(layout)
                   for x,c in enumerate(row) if c == 0}

def get_direction_vector(direction):
    """Convert direction angle to movement vector"""
    if direction == 0:    return (0, -1)  # North
    elif direction == 90: return (1, 0)   # East  
    elif direction == 180: return (0, 1)  # South
    elif direction == 270: return (-1, 0) # West
    return (0, 0)

def get_direction_name(direction):
    """Convert direction angle to name"""
    names = {0: "NORTH", 90: "EAST", 180: "SOUTH", 270: "WEST"}
    return names.get(direction, "UNKNOWN")

def get_maze_config(size_setting):
    """Get maze configuration based on size setting"""
    if size_setting == "SMALL":
        return {
            "maze_ids": ["small_maze1", "small_maze2", "small_maze3", "small_maze4"],
            "cell_size": 80,  # Larger cells for small mazes
            "expected_duration": "30-60 seconds per trial"
        }
    else:  # LARGE
        return {
            "maze_ids": ["maze1", "maze2", "maze3", "maze4"], 
            "cell_size": 60,  # Current cell size
            "expected_duration": "60-120 seconds per trial"
        }

def select_experiment_settings():
    """Enhanced selection screen for both mode and maze size"""
    pygame.init()
    font = pygame.font.SysFont(None, 36)
    small_font = pygame.font.SysFont(None, 24)
    screen = pygame.display.set_mode((900, 700))
    clock = pygame.time.Clock()
    
    selected_mode = None
    selected_size = None
    
    while True:
        screen.fill((30, 30, 30))
        
        # Title
        title = font.render("ErrP Simulator - Experiment Settings", True, (255, 255, 255))
        screen.blit(title, (450 - title.get_width()//2, 50))
        
        # Mode Selection
        mode_title = font.render("1. Select Control Mode:", True, (200, 200, 255))
        screen.blit(mode_title, (50, 150))
        
        mode1_color = (100, 255, 100) if selected_mode == "BCI" else (150, 150, 150)
        mode2_color = (100, 255, 100) if selected_mode == "CLASSIC" else (150, 150, 150)
        
        mode1 = small_font.render("Press '1': BCI Mode (3-class: Turn Left/Right + Move Forward)", True, mode1_color)
        mode2 = small_font.render("Press '2': Classic Mode (4-direction: Up/Down/Left/Right)", True, mode2_color)
        
        screen.blit(mode1, (70, 190))
        screen.blit(mode2, (70, 220))
        
        # Size Selection
        size_title = font.render("2. Select Maze Size:", True, (200, 200, 255))
        screen.blit(size_title, (50, 300))
        
        small_color = (100, 255, 100) if selected_size == "SMALL" else (150, 150, 150)
        large_color = (100, 255, 100) if selected_size == "LARGE" else (150, 150, 150)
        
        size1 = small_font.render("Press 'S': Small Mazes (5x5 - Quick trials, more errors)", True, small_color)
        size2 = small_font.render("Press 'L': Large Mazes (9x9 - Longer trials, more navigation)", True, large_color)
        
        screen.blit(size1, (70, 340))
        screen.blit(size2, (70, 370))
        
        # Configuration Preview
        if selected_mode and selected_size:
            preview_title = font.render("3. Configuration Preview:", True, (200, 255, 200))
            screen.blit(preview_title, (50, 450))
            
            config_text = []
            config_text.append(f"Mode: {selected_mode}")
            config_text.append(f"Size: {selected_size}")
            
            if selected_size == "SMALL":
                config_text.append("Maze dimensions: 5x5")
                config_text.append("Expected trial duration: 30-60 seconds")
                config_text.append("Difficulty: Easier navigation, more frequent errors")
            else:
                config_text.append("Maze dimensions: 9x9") 
                config_text.append("Expected trial duration: 60-120 seconds")
                config_text.append("Difficulty: Complex navigation, strategic planning")
            
            for i, text in enumerate(config_text):
                color = (255, 255, 255) if i < 2 else (200, 200, 200)
                line = small_font.render(text, True, color)
                screen.blit(line, (70, 490 + i * 25))
            
            # Start button
            start_text = font.render("Press SPACE to start experiment", True, (100, 255, 100))
            screen.blit(start_text, (450 - start_text.get_width()//2, 620))
        
        # Instructions
        instr = small_font.render("ESC to quit", True, (150, 150, 150))
        screen.blit(instr, (450 - instr.get_width()//2, 660))
        
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    selected_mode = "BCI"
                elif event.key == pygame.K_2:
                    selected_mode = "CLASSIC"
                elif event.key == pygame.K_s:
                    selected_size = "SMALL"
                elif event.key == pygame.K_l:
                    selected_size = "LARGE"
                elif event.key == pygame.K_SPACE and selected_mode and selected_size:
                    return selected_mode, selected_size
        
        clock.tick(60)

def enhanced_transition(scr, font, k, tot, mode, size):
    scr.fill((0,0,0))
    t1 = font.render(f"Trial {k}/{tot} - {mode} Mode ({size} Maze)", True, COL["txt"])
    t2 = font.render("Press SPACE to start", True, COL["txt"])
    
    if mode == "BCI":
        t3 = font.render("Controls: LEFT/RIGHT arrows to turn, UP arrow to move forward", True, COL["txt"])
    else:
        t3 = font.render("Controls: Arrow keys for direct movement (UP/DOWN/LEFT/RIGHT)", True, COL["txt"])
    
    if size == "SMALL":
        t4 = font.render("Small maze: Quick navigation, frequent decision points", True, (150, 150, 255))
    else:
        t4 = font.render("Large maze: Complex navigation, strategic planning required", True, (150, 150, 255))
    
    w,h = scr.get_size()
    scr.blit(t1,(w//2-t1.get_width()//2, h//2-80))
    scr.blit(t2,(w//2-t2.get_width()//2, h//2-40))
    scr.blit(t3,(w//2-t3.get_width()//2, h//2))
    scr.blit(t4,(w//2-t4.get_width()//2, h//2+40))
    pygame.display.flip()
    
    while True:
        for e in pygame.event.get():
            if e.type==pygame.QUIT or (e.type==pygame.KEYDOWN and e.key==pygame.K_ESCAPE):
                pygame.quit(); exit()
            if e.type==pygame.KEYDOWN and e.key==pygame.K_SPACE:
                return

# ─── MAIN EXPERIMENT ────────────────────────────────────────

# Select both mode and size at startup
selected_mode, selected_size = select_experiment_settings()

# Get size-specific configuration
maze_config = get_maze_config(selected_size)
MAZE_IDS = maze_config["maze_ids"]
CELL = maze_config["cell_size"]

# ─── INIT PYGAME (AFTER GETTING CELL SIZE) ─────────────────────────────────────────
pygame.init()
font  = pygame.font.SysFont(None, 24)
clock = pygame.time.Clock()
pygame.display.set_mode((CELL*9, CELL*9))

arrow = pygame.Surface((CELL//2, CELL//3), pygame.SRCALPHA)
pygame.draw.polygon(arrow, COL["txt"],
                    [(CELL//2, CELL//6), (0,0), (0,CELL//3)])

# Set up mode-specific variables
if selected_mode == "BCI":
    ACTIONS = BCI_ACTIONS
    ACTION_CODE = BCI_ACTION_CODE
    use_agent_direction = True
else:
    ACTIONS = CLASSIC_ACTIONS  
    ACTION_CODE = CLASSIC_ACTION_CODE
    use_agent_direction = False

# Enhanced logging with size information
if use_agent_direction:
    log_header = [
        "trial","maze_id","maze_size","step","lsl_timestamp",
        "x","y","facing_direction", 
        "intended","actual","intended_code","actual_code",
        "moved","blocked","turned",
        "error_flag","err_ts","resp_ts","rt",
        "stars_left","processing_delay_ms","timing_warning"
    ]
else:
    log_header = [
        "trial","maze_id","maze_size","step","lsl_timestamp",
        "x","y",
        "intended","actual","intended_code","actual_code", 
        "moved","blocked",
        "error_flag","err_ts","resp_ts","rt",
        "stars_left","processing_delay_ms","timing_warning"
    ]

rows = [log_header]

for trial, mid in enumerate(MAZE_IDS, 1):
    enhanced_transition(pygame.display.get_surface(), font, trial, len(MAZE_IDS), selected_mode, selected_size)

    # Use size-aware maze loading
    layout, (ax, ay) = load_maze_with_size(MAZE_CFG, mid, selected_size)
    R,C  = len(layout), len(layout[0])
    scr  = pygame.display.set_mode((C*CELL, R*CELL))
    reward = stars(layout); reward.discard((ax,ay))
    
    # Initialize last movement direction for classic mode
    if use_agent_direction:
        agent_dir = 0  # 0=North, 90=East, 180=South, 270=West
        last_movement_dir = None
    else:
        agent_dir = None
        last_movement_dir = "RIGHT"  # Default starting direction

    # draw helpers -----------------------------------------------------------
    def draw_static():
        for y,row in enumerate(layout):
            for x,c in enumerate(row):
                pygame.draw.rect(scr,
                                 COL["wall" if c else "path"],
                                 (x*CELL, y*CELL, CELL, CELL))
    def draw_rewards():
        for x,y in reward:
            pygame.draw.circle(scr, COL["star"],
                               (x*CELL+CELL//2, y*CELL+CELL//2), CELL//8)
    
    def draw_avatar(err, direction_info):
        cx, cy = ax*CELL+CELL//2, ay*CELL+CELL//2
        pygame.draw.circle(scr, COL["err" if err else "dot"], (cx, cy), CELL//4)
        
        if use_agent_direction and direction_info:
            # BCI Mode: Draw direction indicator based on agent's facing direction
            direction_angles = {"NORTH": 90, "EAST": 0, "SOUTH": 270, "WEST": 180}
            rot = direction_angles.get(direction_info, 0)
            rotated_arrow = pygame.transform.rotate(arrow, rot)
            scr.blit(rotated_arrow, rotated_arrow.get_rect(center=(cx, cy)))
        elif not use_agent_direction and direction_info:
            # Classic Mode: Draw arrow showing direction of last movement
            movement_angles = {"UP": 90, "DOWN": 270, "LEFT": 180, "RIGHT": 0}
            rot = movement_angles.get(direction_info, 0)
            rotated_arrow = pygame.transform.rotate(arrow, rot)
            scr.blit(rotated_arrow, rotated_arrow.get_rect(center=(cx, cy)))
        else:
            # Default arrow pointing right
            scr.blit(arrow, arrow.get_rect(center=(cx, cy)))

    # first frame
    scr.fill(COL["bg"]); draw_static(); draw_rewards()
    if use_agent_direction:
        draw_avatar(False, get_direction_name(agent_dir))
    else:
        draw_avatar(False, last_movement_dir)
    pygame.display.flip()
    maze_start_ts = push_immediate("MAZE_START")

    step, err_active = 0, False
    pending_response = None  # For tracking response timing

    run = True
    while run:
        # ─── IMMEDIATE EVENT PROCESSING (TIMING CRITICAL) ──────────────────
        frame_start = get_sync_time()
        intended = actual = "NONE"
        got_action = False
        key_event_ts = None
        response_event_ts = None

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); exit()
            
            elif e.type == pygame.KEYDOWN:
                if e.key in ACTIONS and not got_action:
                    # CRITICAL: Mark immediately before any processing
                    intended = ACTIONS[e.key]
                    key_event_ts = push_immediate(f"KEY_{intended}")
                    got_action = True
                    
                elif e.key == pygame.K_SPACE and err_active:
                    # CRITICAL: Mark response immediately
                    response_event_ts = push_immediate("RESP")
                    if pending_response:
                        rt = response_event_ts - pending_response['err_ts']
                        pending_response['resp_ts'] = response_event_ts
                        pending_response['rt'] = rt
                    err_active = False

        if not got_action:
            clock.tick(FPS)
            continue

        # ─── GAME LOGIC PROCESSING ─────────────────────────────────────────
        processing_start = get_sync_time()
        step += 1

        # Initialize movement variables
        moved = False
        blocked = False
        turned = False

        # STEP 1: Determine final action (intended vs error) - FIXED TIMING
        actual = intended  # Default: no error
        error_flag = 0
        err_ts = None

        # Check for error injection BEFORE executing any action
        if random.random() < ERROR_P and not err_active:
            error_decision_time = get_sync_time()  # Track error decision timing
            
            if use_agent_direction:
                # BCI Mode error injection
                alt_actions = []
                
                if intended == "TURN_LEFT":
                    alt_actions = ["TURN_RIGHT", "MOVE_FORWARD"]
                elif intended == "TURN_RIGHT": 
                    alt_actions = ["TURN_LEFT", "MOVE_FORWARD"]
                elif intended == "MOVE_FORWARD":
                    alt_actions = ["TURN_LEFT", "TURN_RIGHT"]
                
                # Filter alternatives that are valid (for MOVE_FORWARD check if movement is possible)
                valid_alts = []
                for alt_action in alt_actions:
                    if alt_action in ["TURN_LEFT", "TURN_RIGHT"]:
                        valid_alts.append(alt_action)
                    elif alt_action == "MOVE_FORWARD":
                        # Check if forward movement is possible
                        dx, dy = get_direction_vector(agent_dir)
                        nx, ny = ax + dx, ay + dy
                        if 0 <= nx < C and 0 <= ny < R and not layout[ny][nx]:
                            valid_alts.append(alt_action)
                
                if valid_alts:
                    actual = random.choice(valid_alts)
                    error_flag = 1
                    err_active = True
                    # CRITICAL: Mark error immediately after decision
                    err_ts = push_immediate("ERR")
                    pending_response = {'err_ts': err_ts, 'resp_ts': None, 'rt': None}
                    
                    # Optional: Track error decision delay for analysis
                    error_delay = err_ts - error_decision_time
                    if error_delay > 0.001:  # 1ms threshold
                        print(f"WARNING: Error marking delayed by {error_delay*1000:.1f}ms")
            
            else:
                # Classic Mode error injection
                direction_map = {
                    "UP": (0, -1),
                    "DOWN": (0, 1), 
                    "LEFT": (-1, 0),
                    "RIGHT": (1, 0)
                }
                
                alt_directions = [d for d in direction_map.keys() if d != intended]
                valid_alts = []
                
                for alt_dir in alt_directions:
                    dx, dy = direction_map[alt_dir]
                    nx, ny = ax + dx, ay + dy
                    if 0 <= nx < C and 0 <= ny < R and not layout[ny][nx]:
                        valid_alts.append(alt_dir)
                
                if valid_alts:
                    actual = random.choice(valid_alts)
                    error_flag = 1
                    err_active = True
                    # CRITICAL: Mark error immediately after decision
                    err_ts = push_immediate("ERR")
                    pending_response = {'err_ts': err_ts, 'resp_ts': None, 'rt': None}

        # STEP 2: Execute the final action (only once!) - FIXED DOUBLE ACTION BUG
        action_execution_start = get_sync_time()
        
        if use_agent_direction:
            # BCI Mode: Execute the determined action
            if actual == "TURN_LEFT":
                agent_dir = (agent_dir - 90) % 360
                turned = True
                
            elif actual == "TURN_RIGHT":
                agent_dir = (agent_dir + 90) % 360
                turned = True
                
            elif actual == "MOVE_FORWARD":
                dx, dy = get_direction_vector(agent_dir)
                nx, ny = ax + dx, ay + dy
                
                if 0 <= nx < C and 0 <= ny < R and not layout[ny][nx]:
                    ax, ay = nx, ny
                    moved = True
                    blocked = False
                else:
                    moved = False
                    blocked = True
        else:
            # Classic Mode: Execute the determined action  
            direction_map = {
                "UP": (0, -1),
                "DOWN": (0, 1), 
                "LEFT": (-1, 0),
                "RIGHT": (1, 0)
            }
            
            if actual in direction_map:
                dx, dy = direction_map[actual]
                nx, ny = ax + dx, ay + dy
                
                if 0 <= nx < C and 0 <= ny < R and not layout[ny][nx]:
                    ax, ay = nx, ny
                    moved = True
                    blocked = False
                    last_movement_dir = actual  # Update arrow direction
                else:
                    moved = False
                    blocked = True

        # Update rewards if moved
        if moved:
            reward.discard((ax, ay))

        # Mark movement/action with precise timing
        move_marker_ts = None
        if moved or (use_agent_direction and turned):
            move_marker_ts = push_immediate(f"ACTION_{actual}")

        processing_end = get_sync_time()
        processing_delay = processing_end - processing_start
        timing_warning = processing_delay > MAX_PROCESSING_DELAY

        if timing_warning:
            print(f"WARNING: Processing delay {processing_delay*1000:.1f}ms > {MAX_PROCESSING_DELAY*1000}ms threshold")

        # ─── RENDERING (NON-TIMING CRITICAL) ───────────────────────────────
        scr.fill(COL["bg"]); draw_static(); draw_rewards()
        
        if use_agent_direction:
            draw_avatar(error_flag, get_direction_name(agent_dir))
            # Show timing info and agent state in HUD
            hud1 = font.render(f"Stars left: {len(reward)} | Action: {actual} | Facing: {get_direction_name(agent_dir)}", True, COL["txt"])
        else:
            draw_avatar(error_flag, last_movement_dir)
            # Show timing info for classic mode
            hud1 = font.render(f"Stars left: {len(reward)} | Action: {actual} | Last move: {last_movement_dir}", True, COL["txt"])
        
        timing_info = f"Delay: {processing_delay*1000:.1f}ms" + (" ⚠" if timing_warning else "")
        hud2 = font.render(timing_info, True, COL["err"] if timing_warning else COL["txt"])
        scr.blit(hud1, (10, R*CELL-50))
        scr.blit(hud2, (10, R*CELL-30))
        pygame.display.flip()

        # ─── LOGGING (IMPROVED TIMESTAMP CONSISTENCY) ──────────────────────
        # Use the key event timestamp as primary reference for consistency
        primary_timestamp = key_event_ts if key_event_ts else frame_start
        
        resp_ts_str = f"{pending_response['resp_ts']:.6f}" if pending_response and pending_response['resp_ts'] else ""
        rt_str = f"{pending_response['rt']:.6f}" if pending_response and pending_response['rt'] else ""
        err_ts_str = f"{err_ts:.6f}" if err_ts else ""
        
        if use_agent_direction:
            # BCI mode logging with direction info and size
            row = [
                trial, mid, selected_size, step, f"{primary_timestamp:.6f}",
                ax, ay, agent_dir,
                intended, actual,
                ACTION_CODE[intended], ACTION_CODE[actual],
                int(moved), int(blocked), int(turned),
                error_flag, 
                err_ts_str,
                resp_ts_str, rt_str,
                len(reward),
                f"{processing_delay*1000:.3f}",
                int(timing_warning)
            ]
        else:
            # Classic mode logging without direction info but with size
            row = [
                trial, mid, selected_size, step, f"{primary_timestamp:.6f}",
                ax, ay,
                intended, actual,
                ACTION_CODE[intended], ACTION_CODE[actual], 
                int(moved), int(blocked),
                error_flag,
                err_ts_str,
                resp_ts_str, rt_str,
                len(reward),
                f"{processing_delay*1000:.3f}",
                int(timing_warning)
            ]
        rows.append(row)

        # Clear completed response tracking
        if pending_response and pending_response['resp_ts']:
            pending_response = None

        if not reward:
            push_immediate("MAZE_END")
            run = False

        clock.tick(FPS)

    time.sleep(ITI)

# ─── SAVE CSV ────────────────────────────────────────────────
with open(LOG_PATH, "w", newline="") as f:
    csv.writer(f).writerows(rows)

pygame.quit()
print("Experiment complete – log:", LOG_PATH)
print(f"Mode used: {selected_mode}")
print(f"Maze size: {selected_size}")
print(f"Expected trial duration: {maze_config['expected_duration']}")
if selected_mode == "BCI":
    print("Control scheme: LEFT/RIGHT arrows to turn, UP arrow to move forward")
else:
    print("Control scheme: Arrow keys for direct movement (UP/DOWN/LEFT/RIGHT)")
print(f"Check timing_warning column for frames with >10ms processing delays")
print("FIXES APPLIED:")
print("✓ Error injection bug fixed - no more double actions")
print("✓ ERR marker timing improved - immediate marking after error decision")
print("✓ Timestamp consistency improved for better EEG synchronization")
print("✓ Flexible maze size system implemented - Small (5x5) and Large (9x9) mazes")