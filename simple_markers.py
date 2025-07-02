"""
Simple numeric markers for EEG experiments using single digits 1-9
This is cleaner and follows common EEG practices
"""

# Core experimental markers (1-4)
MARKER_TRIAL_START = 1
MARKER_CUE_LEFT = 2
MARKER_CUE_RIGHT = 3
MARKER_CUE_END = 4

# Response/Feedback markers (5-7)
MARKER_RESPONSE_CORRECT = 5
MARKER_RESPONSE_ERROR = 6    # Error feedback (for ErrP epochs)
MARKER_NO_RESPONSE = 7       # Timeout

# Special markers (8-9)
MARKER_BASELINE = 8          # Baseline period
MARKER_SESSION = 9           # Session start/end

# Human-readable names for logging
MARKER_NAMES = {
    1: "TRIAL_START",
    2: "CUE_LEFT",
    3: "CUE_RIGHT",
    4: "CUE_END",
    5: "RESPONSE_CORRECT",
    6: "RESPONSE_ERROR",
    7: "NO_RESPONSE",
    8: "BASELINE",
    9: "SESSION"
}

# Common marker combinations for analysis
STIMULUS_MARKERS = [2, 3]        # Left and right cues
RESPONSE_MARKERS = [5, 6, 7]     # All response types
ERROR_MARKERS = [6]              # For ErrP analysis
CUE_MARKERS = [2, 3]            # For MI epoch extraction