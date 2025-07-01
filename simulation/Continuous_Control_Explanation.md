# Continuous Control in BCI Systems

## What is Continuous Control?

Continuous control means the BCI output is a **smooth, analog signal** rather than discrete commands. Think of it like:
- **Discrete Control**: Digital buttons (LEFT or RIGHT)
- **Continuous Control**: Analog joystick (any direction, any speed)

## Examples to Illustrate

### 1. Cursor Control (Classic BCI Paradigm)
```
Discrete: Click to move cursor one grid space left/right
Continuous: Cursor moves smoothly following your mental command intensity

User thinks "move right" softly → Cursor drifts slowly right
User thinks "move right" strongly → Cursor moves quickly right
```

### 2. Wheelchair Control
```
Discrete: Forward/Back/Left/Right commands
Continuous: 
- Speed: 0-100% based on intention strength  
- Direction: 0-360° based on imagined movement vector
```

### 3. In Our Game Context
```
Current (Discrete):
- Press LEFT → Agent jumps to left lane
- Press RIGHT → Agent jumps to right lane

Continuous Alternative:
- Mental effort controls horizontal position (-1.0 to +1.0)
- Agent smoothly slides between lanes
- Can stop anywhere, even between lanes
```

## How Continuous Control Works in MI-BCI

### Signal Processing
```python
# Discrete (Current)
if mi_classifier.predict(eeg_features) == "LEFT":
    execute_left_command()

# Continuous
mi_strength = mi_regressor.predict(eeg_features)  # Returns -1.0 to 1.0
agent.position += mi_strength * speed_factor
```

### Mental Imagery Mapping
- **Left Motor Imagery**: Negative values (-1.0 to 0)
- **Right Motor Imagery**: Positive values (0 to +1.0)
- **Rest/Neutral**: Values near 0
- **Strength of Imagery**: Magnitude of value

## Why Continuous Control Matters for ErrP Research

### 1. Different Error Types
```
Discrete Errors:
- Binary: Either correct or wrong
- Clear error moment when wrong action executes

Continuous Errors:
- Gradual: Small deviations accumulate
- Fuzzy boundaries: When exactly is it "wrong"?
- Magnitude matters: 10° off vs 90° off
```

### 2. Error Detection Complexity
In continuous control, errors can be:
- **Undershoot**: Not moving far enough
- **Overshoot**: Moving too far
- **Lag**: Correct direction but delayed
- **Drift**: Gradual deviation from intended path
- **Oscillation**: Overcorrecting back and forth

### 3. Neural Signatures Differ

**Discrete Control ErrP**:
- Sharp, time-locked to the error event
- "Oh no, that's wrong!" response
- Binary classification: error or no error

**Continuous Control ErrP**:
- May show graded responses based on error magnitude
- Continuous monitoring signals
- "This is getting worse..." response
- Multiple error-checking timepoints

## Visual Example

```
Discrete Control Timeline:
[CUE: LEFT] → [USER THINKS LEFT] → [SYSTEM GOES RIGHT] → [ERROR!]
                                                           ↑
                                                    Clear ErrP here

Continuous Control Timeline:
[TARGET: 75°] → [USER TRIES TO REACH 75°] → [CURSOR DRIFTS TO 60°... 65°... 80°... 85°...]
                                              ↑        ↑        ↑        ↑
                                         Small error  OK   Small error  Larger error
                                         
ErrP magnitude correlates with deviation size
```

## Implementation Example for Our Simulator

```python
class ContinuousLaneControl:
    def __init__(self):
        self.agent_position = 0.0  # -1.0 (left) to 1.0 (right)
        self.target_position = 0.0
        self.control_signal = 0.0
        
    def update_from_mi(self, mi_output):
        # MI output: -1 (strong left) to +1 (strong right)
        # Add realistic noise and delays
        noisy_signal = mi_output + np.random.normal(0, 0.1)
        
        # Smooth integration (like a low-pass filter)
        self.control_signal = 0.8 * self.control_signal + 0.2 * noisy_signal
        
        # Update position
        self.agent_position += self.control_signal * 0.05
        self.agent_position = np.clip(self.agent_position, -1.0, 1.0)
        
        # Calculate error
        error = abs(self.target_position - self.agent_position)
        return error
        
    def draw_agent(self, screen):
        # Agent can be anywhere on the continuum
        x = screen_center + (self.agent_position * lane_width * 1.5)
        # Smooth position, not snapped to lanes
```

## Research Advantages

1. **More Realistic**: Most real-world BCI applications need continuous control
2. **Richer Error Signals**: Magnitude and direction of errors provide more data
3. **Natural Interaction**: Mimics how we actually control things in real life
4. **Better for Learning**: Can study error correction and adaptation over time

## The Challenge

Continuous control is harder to implement and analyze because:
- No clear "correct/incorrect" moments
- Need to define error thresholds
- More complex signal processing
- Requires regression instead of classification
- More susceptible to noise and drift

But it's more representative of real BCI challenges and likely generates different (possibly richer) ErrP patterns for study.