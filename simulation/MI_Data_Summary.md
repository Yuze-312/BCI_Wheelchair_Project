# Comprehensive Analysis of MI (Motor Imagery) Data

## Overview
The MI folder contains motor imagery EEG data from a Brain-Computer Interface (BCI) experiment. The data appears to be from a 3-class motor imagery paradigm recorded using OpenVibe software.

## Data Format and Structure

### 1. File Organization
```
MI/
├── EEG data/
│   ├── T-005/
│   │   ├── Info.txt (Task: VR no noise)
│   │   ├── Session 1/ (5 CSV files)
│   │   ├── Session 2/ (5 CSV files)
│   │   └── Test/ (3 CSV files + 2 report files)
│   └── T-008/
│       ├── Session 1/ (6 CSV files)
│       ├── Session 2/ (5 CSV files)
│       └── Test/ (3 CSV files + 2 report files)
└── OpenVibe_marked_CSV.py (data loading script)
```

### 2. Recording Parameters
- **Sampling Rate**: 512 Hz (confirmed from data)
- **Number of Channels**: 16 EEG channels
- **File Format**: OpenVibe CSV format
- **Recording Type**: Continuous EEG with event markers

### 3. CSV File Structure
Each CSV file contains 21 columns:
1. `Time:512Hz` - Timestamp in seconds
2. `Epoch` - Epoch counter
3. `Channel 1` through `Channel 16` - EEG data (in microvolts)
4. `Event Id` - Event markers (empty for most rows)
5. `Event Date` - Event timestamp (usually empty)
6. `Event Duration` - Duration of events (usually empty)

### 4. Event Coding Scheme
The data uses a 4-event coding system:
- **Event ID 0**: Rest/Preparation period (inter-trial interval)
- **Event ID 1**: Motor imagery class 1 (likely left hand)
- **Event ID 2**: Motor imagery class 2 (likely right hand)
- **Event ID 3**: Motor imagery class 3 (likely feet or tongue)

### 5. Experimental Paradigm

#### Trial Structure
The experiment follows an alternating block design:
1. Rest period (Event 0) - ~11-13 seconds
2. Motor imagery trial (Event 1, 2, or 3) - ~11-13 seconds
3. Return to rest (Event 0)

#### Timing Details
- **Average inter-event interval**: 12-16 seconds
- **Min interval**: ~10.4 seconds
- **Max interval**: ~26.3 seconds
- Events appear to be pseudo-randomly ordered

#### Session Organization
Each participant completed:
- **Session 1**: 5-6 runs (training data)
- **Session 2**: 5 runs (training data)
- **Test Session**: 3 runs + performance evaluation

### 6. Report Files
Report files track online BCI performance:
- `Achieved Value`: Classifier prediction (1, 2, or 3)
- `Aiming Value`: True label/target class
- `Classification percentage`: Running accuracy (0-1)

Example entries show the progression of classification accuracy as the system learns.

### 7. Data Processing Pipeline

The `OpenVibe_marked_CSV.py` script provides a data loading class with:
- **Default epoch length**: 1 second
- **Default overlap**: 0.5 (50%)
- **Functionality**:
  - Loads multiple CSV files
  - Extracts epochs between event markers
  - Groups data by Event ID (class labels)
  - Supports parallel processing for efficiency

### 8. Electrode Montage
While the specific electrode positions for the 16-channel setup aren't documented in the MI folder, related files suggest:
- Standard 10-20 system placement
- Focus on motor cortex areas (likely C3, C4, Cz regions)
- 16 channels selected from a larger montage

### 9. Key Observations

1. **Data Quality**: High sampling rate (512 Hz) suitable for motor imagery analysis
2. **Trial Count**: Each run contains 8-10 motor imagery trials (excluding rest)
3. **Class Balance**: Relatively balanced across the 3 motor imagery classes
4. **Continuous Recording**: Data is recorded continuously with event markers
5. **Real-time Feedback**: Report files suggest online classification was performed

### 10. Usage Recommendations

For analysis:
1. Use the `OpenVibe_marked_CSV.py` script to load data
2. Extract epochs around motor imagery events (1, 2, 3)
3. Use rest periods (Event 0) for baseline correction
4. Standard epoch window: -0.5 to +4 seconds around events
5. Consider the 50% overlap for epoch extraction

### Summary
This is a well-structured 3-class motor imagery dataset with:
- 2 participants (T-005, T-008)
- Multiple training and test sessions
- Clear event marking system
- High-quality continuous EEG recordings
- Built-in performance tracking through report files

The data is suitable for developing and testing motor imagery classification algorithms, with sufficient trials for machine learning approaches.