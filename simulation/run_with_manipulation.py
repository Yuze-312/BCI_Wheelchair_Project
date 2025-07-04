#!/usr/bin/env python3
"""
Run the BCI system with manipulation enabled for ErrP elicitation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from BCI_system import EEGProcessor

# Configuration
MANIPULATION_RATE = 0.75  # 75% success rate = 25% errors
PARTICIPANT_ID = 'T-005'  # Change to your participant ID

def main():
    print("="*60)
    print("Starting BCI System with ErrP Elicitation")
    print(f"Manipulation Rate: {MANIPULATION_RATE:.0%} success")
    print(f"Expected Error Rate: {(1-MANIPULATION_RATE)*100:.0f}%")
    print(f"Participant: {PARTICIPANT_ID}")
    print("="*60)
    
    # Create processor with manipulation
    processor = EEGProcessor(
        debug=False,
        phase='real',
        participant_id=PARTICIPANT_ID,
        manipulation_rate=MANIPULATION_RATE
    )
    
    # Initialize and run
    processor.initialize()
    processor.run()

if __name__ == "__main__":
    main()