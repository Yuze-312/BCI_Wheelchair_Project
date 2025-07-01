# MI Classifiers Organization

This directory separates MI classifiers into two distinct approaches:

## 📁 Structure

```
classifiers/
├── universal/           # Cross-subject classifiers
│   ├── train_universal_classifier.py
│   └── README.md
│
└── subject_specific/    # Individual classifiers
    ├── train_subject_classifier.py
    ├── collect_calibration_data.py
    └── README.md
```

## 🌍 Universal Classifier
- **Purpose**: Works across multiple people without calibration
- **Accuracy**: 50-70% (often shows bias)
- **Use case**: Initial testing, no calibration time
- **Problem**: Inter-subject variability limits performance

## 👤 Subject-Specific Classifier
- **Purpose**: Optimized for individual brain patterns
- **Accuracy**: 70-90% (much more reliable)
- **Use case**: Personal BCI systems, research
- **Requirement**: 10-20 minute calibration session

## 🔄 Automatic Participant Discovery

Both classifiers now automatically discover participants from the data folder:

### Universal Classifier
```bash
# Automatically finds and combines ALL participants
python universal/train_universal_classifier.py

# Creates: MI/models/mi_improved_classifier.pkl
```

### Subject-Specific Classifiers
```bash
# Train models for ALL discovered participants
python subject_specific/train_subject_classifier.py

# Or train just one participant
python subject_specific/train_subject_classifier.py --participant T-005

# Creates: MI/models/subject_T-005_current.pkl (for each participant)
```

### Compare Approaches
```bash
# See performance difference between universal vs subject-specific
python compare_approaches.py
```

## 🎯 Which to Choose?

**Universal if:**
- Testing system functionality
- No time for calibration
- Comparing across subjects
- Developing transfer learning

**Subject-Specific if:**
- Need reliable control
- Have 10-20 minutes for calibration
- Building personal BCI system
- Conducting serious experiments

## 🔬 Research Direction
Your ErrP-enhanced adaptive BCI could bridge these approaches:
1. Start with universal model
2. Detect errors using ErrP
3. Adapt to individual over time
4. No explicit calibration needed!