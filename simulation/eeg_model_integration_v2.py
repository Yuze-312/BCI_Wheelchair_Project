"""
EEG Model Integration v2 - Uses modular BCI system components
Maintains backward compatibility with original TrainedModelEEGProcessor interface
"""

from BCI_system import EEGProcessor


class TrainedModelEEGProcessor(EEGProcessor):
    """
    Backward compatibility wrapper for the original TrainedModelEEGProcessor.
    This ensures existing code continues to work with the new modular architecture.
    """
    
    def __init__(self, debug=False, phase='real', participant_id='unknown', error_rate=0.0, manipulation_rate=0.0):
        """Initialize with same parameters as original
        
        Args:
            debug: Enable debug output
            phase: 'real' for actual EEG, 'phase1' for fake classifier
            participant_id: Participant ID for phase1 data collection
            error_rate: Error injection rate for phase1
            manipulation_rate: For real mode - target success rate via manipulation
        """
        # Initialize parent with same parameters
        super().__init__(debug=debug, phase=phase, participant_id=participant_id, error_rate=error_rate, manipulation_rate=manipulation_rate)
        
        # Additional attributes for compatibility
        self.classifier_loaded = False
        
    def setup_streams(self):
        """Setup and connect to LSL streams (original method name)"""
        # Maps to new initialize() method
        self.initialize()
        self.classifier_loaded = True
        
    def process_eeg_continuous(self):
        """Process EEG data continuously (original method name)"""
        # Maps to new run() method
        self.run()


# For backward compatibility - allow direct import
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='EEG Model Integration for MI Detection')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--phase', choices=['phase1', 'real'], default='real',
                       help='Phase 1: Fake classifier, Phase 2+: Real EEG')
    parser.add_argument('--participant', type=str, default='unknown',
                       help='Participant ID for phase1 data collection (e.g., T-009)')
    parser.add_argument('--error-rate', type=float, default=0.0,
                       help='Error injection rate for phase1 (default: 0.0 for data collection)')
    parser.add_argument('--manipulation-rate', type=float, default=0.0,
                       help='For real mode: target success rate via manipulation (0.75 = 75% success, 0.0 = no manipulation)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BCI CONTROL SYSTEM - Using Modular Architecture")
    print("=" * 80)
    
    processor = TrainedModelEEGProcessor(
        debug=args.debug, 
        phase=args.phase,
        participant_id=args.participant,
        error_rate=args.error_rate,
        manipulation_rate=args.manipulation_rate
    )
    processor.setup_streams()
    processor.process_eeg_continuous()