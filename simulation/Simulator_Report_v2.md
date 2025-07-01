# ErrP Elicitation Simulator v2.0 - Technical Report

## Executive Summary

The Enhanced Subway Surfers Style EEG Simulator is a specialized tool designed for eliciting Error-related Potentials (ErrP) in Brain-Computer Interface (BCI) research. This second-generation simulator implements a 3-lane endless runner paradigm with sophisticated error injection mechanisms to study neural responses when BCI systems fail to execute user intentions.

## Key Innovations

### 1. Paradigm Design
- **3-Lane Endless Runner**: Agent remains visually centered while the world shifts
- **Motor Imagery Integration**: Word cues ("LEFT"/"RIGHT") simulate MI-BCI commands
- **70% Accuracy Simulation**: 30% error rate mimics realistic MI decoder performance
- **Bullet-Time Mechanics**: Slow motion (20% speed) during cues for enhanced reaction time

### 2. Error Detection Architecture

#### Primary Errors (ErrP-inducing)
- **Definition**: System executes opposite of user's intended action
- **Neural Marker**: Generates error-related potentials
- **Visual Feedback**: Agent turns red, error sound plays
- **Example**: User presses LEFT → System moves RIGHT

#### Secondary Errors (Non-ErrP)
- **Definition**: User makes incorrect choice relative to cue
- **Neural Marker**: No ErrP expected (user's conscious decision)
- **Visual Feedback**: Standard feedback only
- **Example**: Cue shows LEFT, user intentionally presses RIGHT

### 3. Technical Implementation

#### Core Systems
```python
# Synchronized timing with LSL
StreamInfo("SubwaySurfers_ErrP_Markers", "Markers", 1, 0, "string", "subway-errp-001")

# Speed factor for bullet-time
SLOW_MOTION_DURING_CUE = True
SLOW_MOTION_FACTOR = 0.2  # 20% speed

# Error injection
ERROR_P = 0.3  # 30% error rate
```

#### Movement System
- **Smooth Transitions**: Time-based easing functions (sine, cubic, quadratic)
- **Visual Feedback**: Agent lean animation and vertical bounce
- **Frame-Independent**: Consistent behavior across different frame rates

#### Data Logging
- **Format**: Optimized CSV with comprehensive event tracking
- **Fields**: timestamp, event_type, trial_id, cue_class, predicted_class, accuracy, error_type, confidence, reaction_time, coins_collected, artifact_flag, details
- **Synchronization**: LSL timestamps for precise EEG alignment

## Experimental Protocol

### Trial Structure
1. **Baseline Period**: Normal gameplay (2-3 seconds)
2. **Cue Presentation**: Word appears with slow motion effect
3. **Think Time**: 1-5 seconds (difficulty dependent)
4. **Action Window**: User responds via keyboard
5. **Execution**: 70% correct, 30% flipped (error injection)
6. **Feedback**: Visual and auditory response
7. **Inter-Trial Interval**: Variable cooldown period

### Difficulty Levels
- **Easy**: 3 coins per trial, 3-5s think time
- **Medium**: 5 coins per trial, 2-4s think time  
- **Hard**: 7 coins per trial, 1-3s think time

## Key Features for EEG Research

### 1. Temporal Precision
- Sub-millisecond LSL timestamp accuracy
- Frame timing verification (60 FPS target)
- Pre/post visual markers for offline analysis

### 2. Artifact Reduction
- Smooth animations minimize eye movements
- Centered agent reduces saccades
- Predictable visual flow pattern

### 3. Engagement Optimization
- Dynamic coin collection mechanic
- Progressive difficulty scaling
- Immediate performance feedback

## Performance Metrics

### System Requirements
- **Python**: 3.8+
- **Dependencies**: pygame, pylsl, numpy, pandas
- **Hardware**: 60Hz display recommended
- **OS**: Cross-platform (Windows/Mac/Linux)

### Data Output
- **Event Rate**: ~10-20 events per minute
- **File Size**: ~50KB per 10-minute session
- **Processing Latency**: <10ms typical

## Advantages Over Version 1

1. **Improved Error Logic**: Clear distinction between system errors and user choices
2. **Enhanced Visuals**: Better contrast, lane visibility, and feedback mechanisms
3. **Bullet-Time Innovation**: Maintains game flow while extending reaction windows
4. **Simplified Logging**: Single CSV format for easier analysis
5. **Better Synchronization**: Robust LSL integration with timing verification

## Future Enhancements

### Planned Features
1. **Adaptive Difficulty**: Real-time adjustment based on performance
2. **Multi-Modal Feedback**: Haptic/audio enhancement options
3. **Online Classification**: Real-time ErrP detection integration
4. **Extended Paradigms**: Additional game modes for varied ErrP elicitation

### Research Applications
- ErrP characterization in MI-BCI contexts
- Error awareness and learning studies
- BCI reliability improvement
- Human-AI interaction research

## Conclusion

The Enhanced Subway Surfers Style EEG Simulator v2.0 represents a significant advancement in ErrP elicitation tools. By combining engaging gameplay with precise experimental control, it provides researchers with a powerful platform for studying error processing in BCI systems. The bullet-time innovation and improved error detection logic make it particularly suitable for investigating the neural correlates of human-machine interaction failures.

## Technical Specifications

### Configuration Parameters
```python
ERROR_P = 0.3                    # Error injection rate
RESP_WIN = 1.0                   # Response window (seconds)
ITI = 2.0                        # Inter-trial interval
FPS = 60                         # Target frame rate
SLOW_MOTION_FACTOR = 0.2         # Bullet-time speed
MOVEMENT_DURATION = 0.35         # Animation duration
```

### Event Types Logged
- `trial_start` / `trial_end`
- `cue_presented` / `cue_timeout`
- `user_action` / `simulated_action`
- `primary_errp` / `feedback_error`
- `coin_collected` / `coin_missed`
- `imagery_start` / `imagery_end`

### Analysis Pipeline
```python
# Example analysis code
df = pd.read_csv('logs/subway_errp_*.csv')
primary_errors = df[df['event_type'] == 'primary_errp']
error_rate = len(primary_errors) / len(df[df['event_type'].str.contains('feedback')])
mean_rt = df['reaction_time'].astype(float).mean()
```

---

*Developed for EEG/BCI Research*  
*Version 2.0 - December 2024*