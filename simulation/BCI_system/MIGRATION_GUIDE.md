# BCI System Migration Guide

This guide helps you migrate from the monolithic `eeg_model_integration.py` to the new modular BCI system.

## Overview

The new modular architecture separates concerns into distinct components:
- **StreamManager**: Handles LSL stream connections and data buffering
- **MIClassifier**: Encapsulates CSP+LDA classification logic
- **CommandWriter**: Manages command file writing
- **VotingController**: Handles the 4-second voting window
- **EEGProcessor**: Main orchestrator that coordinates all components

## Quick Migration

### Option 1: Use Compatibility Wrapper (Recommended)

Simply replace your import:

```python
# Old
from eeg_model_integration import TrainedModelEEGProcessor

# New (maintains exact same interface)
from eeg_model_integration_v2 import TrainedModelEEGProcessor
```

No other code changes needed! The wrapper maintains full backward compatibility.

### Option 2: Use New Modular System

```python
# Direct usage of new system
from BCI_system import EEGProcessor

processor = EEGProcessor(debug=False, phase='real')
processor.initialize()  # Replaces setup_streams()
processor.run()        # Replaces process_eeg_continuous()
```

## Key Differences

### 1. Initialization
- Old: `processor.setup_streams()`
- New: `processor.initialize()`

### 2. Main Loop
- Old: `processor.process_eeg_continuous()`
- New: `processor.run()`

### 3. Component Access
In the new system, components are accessible as attributes:
```python
processor.stream_manager    # Stream handling
processor.classifier       # MI classification
processor.command_writer   # Command output
processor.voting_controller # Voting window logic
```

## Benefits of New Architecture

1. **Maintainability**: Each component has a single responsibility
2. **Testability**: Components can be tested in isolation
3. **Extensibility**: Easy to add new features or replace components
4. **Debugging**: Issues are easier to trace to specific components

## Timing Guarantees

The new system maintains all critical timing requirements:
- 4-second voting window collection
- Decision at exactly t=4s
- 1 second remaining for action execution
- Command index tracking to prevent pre-cue command execution

## File Locations

- Old monolithic file: `simulation/eeg_model_integration.py`
- New modular system: `simulation/BCI_system/`
- Compatibility wrapper: `simulation/eeg_model_integration_v2.py`

## Testing

Run the test suite to verify functionality:
```bash
python test_bci_refactoring.py
```

## Common Issues

1. **Import Errors**: Make sure `BCI_system` is in your Python path
2. **Model Path**: The system automatically finds models in the MI directory
3. **Dependencies**: All original dependencies (numpy, scipy, pylsl) are still required