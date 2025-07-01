# Best Practices for CLAUDE.md in Research Projects

## What TO Include

### 1. **Research Context & Constraints**
```markdown
## Research Context
- Studying ErrP signals for BCI error detection
- Participants: 20-30 healthy adults
- EEG System: 32-channel, 500Hz sampling
- Timeline: Data collection by March 2025
```

### 2. **Key Design Decisions & Rationale**
```markdown
## Design Decisions
- **Why 30% error rate**: Based on Smith et al. 2023 showing optimal ErrP elicitation
- **Why word cues instead of arrows**: Reduces eye movement artifacts
- **Why slow-motion not pause**: Maintains flow state (pilot study feedback)
```

### 3. **Domain Knowledge**
```markdown
## EEG/ErrP Background
- ErrP latency: 200-400ms post-error
- Expected amplitude: 5-10µV
- Critical electrode sites: FCz, Cz
- Confounds: EMG from frustration, eye movements
```

### 4. **Current Problems & Priorities**
```markdown
## Known Issues (Prioritized)
1. **Temporal jitter**: ~17ms uncertainty affects ERP quality [CRITICAL]
2. **Artificial errors**: Not real MI failures, affects validity [ACKNOWLEDGED]
3. **No artifact detection**: Trials contaminated by blinks [TODO]
```

### 5. **Experimental Insights**
```markdown
## Insights from Testing
- Users prefer sine easing over cubic (less jarring)
- 0.35s movement duration optimal (tried 0.2-0.5s)
- Debug output overwhelming during actual experiments
```

## What NOT to Include

### 1. **Code Documentation**
❌ "The agent.move_left() function moves the agent left"
✅ "Agent stays centered, world shifts to create movement illusion (reduces saccades)"

### 2. **Verbose Implementation Details**
❌ Full code snippets or line-by-line explanations
✅ High-level architecture and why it matters

### 3. **Outdated Information**
❌ Keeping old decisions that have been superseded
✅ Brief note: "Previously used HDF5, switched to CSV for simplicity"

### 4. **Generic Programming Notes**
❌ "Remember to handle errors properly"
✅ "LSL timeout errors common on Mac - use try/except"

## Ideal Structure

```markdown
# ErrP Simulator Project Memory

## Quick Start
- Environment: `conda activate bci_final`
- Run: `python run_simulation_2.py`
- Analyze: `python analyze_logs_example.py`

## Research Goals
- Primary: Elicit reliable ErrP signals for BCI error detection
- Secondary: Compare error types (system vs user errors)
- Constraint: Must minimize eye movement artifacts

## Architecture Decisions
- **Centered agent**: Reduces saccades, world moves instead
- **CSV logging**: Simple, compatible with MNE-Python
- **30% errors**: Optimal for ErrP based on literature

## Current State (Dec 2024)
- ✅ Basic gameplay working
- ✅ Slow-motion during cues
- ✅ LSL markers synchronized
- ⚠️ Artificial error injection (not real MI)
- ❌ No temporal precision beyond frame rate

## Critical for EEG Analysis
- Markers sent at stimulus onset, not screen refresh
- 500ms baseline needed before cues
- Watch for movement artifacts during responses

## Next Sessions Should Focus On
1. Implement sub-frame timing precision
2. Add practice/calibration phase
3. Reduce debug output for actual experiments
```

## Tips for Maintenance

1. **Update After Major Changes**: Not every small fix
2. **Focus on "Why" Not "What"**: Code shows what, memory explains why
3. **Include Failed Attempts**: "Tried X, didn't work because Y"
4. **Keep It Scannable**: Use headers and bullet points
5. **Add Session Boundaries**: "--- Session Dec 15: Added slow motion ---"

## For Your Project Specifically

Most valuable additions would be:
- EEG recording parameters and requirements
- Pilot study findings
- Rationale for game mechanics choices
- Integration points with your analysis pipeline
- Supervisor feedback and priorities
```

This approach treats CLAUDE.md as a **research notebook** that captures institutional knowledge, not just code documentation.