"""
Main script to process ErrP data from BCI wheelchair experiments

Usage:
    python process_errp_data.py --participant T-001 --session "Session 3"
    python process_errp_data.py --participant T-001 --sessions "Session 1" "Session 2" "Session 3"
    python process_errp_data.py --participant T-001 --all-sessions
    python process_errp_data.py --all-participants
"""

import argparse
import os
from pathlib import Path
import logging
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Add the parent directory to the path so we can import the errps module
sys.path.insert(0, str(Path(__file__).parent.parent))

from errps.pipeline import ErrPPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_base_path() -> Path:
    """Get the base path to EEG data"""
    # Get the path relative to this script
    script_dir = Path(__file__).parent.parent  # Go up to BCI_Wheelchair_Project
    data_path = script_dir / "EEG_data"
    
    if not data_path.exists():
        # Try absolute path as fallback
        data_path = Path("C:/Users/yuzeb/BCI_Final/BCI_Wheelchair_Project/EEG_data")
    
    if not data_path.exists():
        raise ValueError(f"EEG data directory not found. Expected at: {data_path}")
    
    return data_path


def get_available_participants(base_path: Path) -> List[str]:
    """Get list of available participants"""
    participants = [d.name for d in base_path.iterdir() if d.is_dir() and d.name.startswith('T-')]
    return sorted(participants)


def get_available_sessions(base_path: Path, participant: str) -> List[str]:
    """Get list of available sessions for a participant"""
    participant_path = base_path / participant
    if not participant_path.exists():
        return []
    
    sessions = [d.name for d in participant_path.iterdir() if d.is_dir() and d.name.startswith('Session')]
    return sorted(sessions)


def process_and_merge_sessions(pipeline: ErrPPipeline, participant: str, sessions: List[str], 
                              base_path: Path, output_dir: Path) -> dict:
    """Process multiple sessions and merge the data"""
    
    all_error_epochs = []
    all_correct_epochs = []
    session_results = []
    
    for session in sessions:
        session_path = base_path / participant / session
        
        if not session_path.exists():
            logger.warning(f"Session path not found: {session_path}")
            continue
        
        logger.info(f"\nProcessing {participant} - {session}")
        logger.info(f"Path: {session_path}")
        
        try:
            # Load data
            session_data = pipeline.loader.load_session(str(session_path))
            
            # Preprocess
            preprocessed = pipeline.preprocessor.preprocess(
                session_data['eeg_data'],
                apply_spatial_filter='car',
                remove_powerline=True
            )
            
            # Extract epochs
            comparison_epochs = pipeline.extractor.extract_comparison_epochs(
                preprocessed,
                session_data['events']
            )
            
            # Collect epochs
            error_epochs = comparison_epochs['error']
            correct_epochs = comparison_epochs['correct']
            
            if len(error_epochs['epochs']) > 0:
                all_error_epochs.extend(error_epochs['epochs'])
            if len(correct_epochs['epochs']) > 0:
                all_correct_epochs.extend(correct_epochs['epochs'])
            
            session_results.append({
                'session': session,
                'n_error': len(error_epochs['epochs']),
                'n_correct': len(correct_epochs['epochs']),
                'metadata': {
                    'session_path': str(session_path),
                    'sampling_rate': pipeline.sampling_rate
                }
            })
            
            logger.info(f"  Extracted {len(error_epochs['epochs'])} error epochs")
            logger.info(f"  Extracted {len(correct_epochs['epochs'])} correct epochs")
            
        except Exception as e:
            logger.error(f"Error processing {session}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create merged results
    # Determine actual epoch length from the data
    if all_error_epochs:
        actual_epoch_length = all_error_epochs[0].shape[0]
    elif all_correct_epochs:
        actual_epoch_length = all_correct_epochs[0].shape[0]
    else:
        # Fallback to calculated length
        actual_epoch_length = int(np.round((pipeline.extractor.epoch_params['epoch_end'] - 
                                          pipeline.extractor.epoch_params['epoch_start']) * pipeline.sampling_rate))
    
    merged_results = {
        'participant': participant,
        'sessions': session_results,
        'error_epochs': np.array(all_error_epochs) if all_error_epochs else np.array([]),
        'correct_epochs': np.array(all_correct_epochs) if all_correct_epochs else np.array([]),
        'times': np.linspace(pipeline.extractor.epoch_params['epoch_start'], 
                            pipeline.extractor.epoch_params['epoch_end'], 
                            actual_epoch_length),
        'channel_names': [f'Channel_{i}' for i in range(1, 17)],
        'sampling_rate': pipeline.sampling_rate,
        'processing_params': {
            'preprocess': pipeline.preprocessor.params,
            'epochs': pipeline.extractor.epoch_params
        }
    }
    
    logger.info(f"\nMerged results:")
    logger.info(f"  Total error epochs: {len(merged_results['error_epochs'])}")
    logger.info(f"  Total correct epochs: {len(merged_results['correct_epochs'])}")
    
    # Save merged data
    output_file = output_dir / f"{participant}_merged_errp_data.npz"
    np.savez(output_file,
             error_epochs=merged_results['error_epochs'],
             correct_epochs=merged_results['correct_epochs'],
             times=merged_results['times'],
             sessions=session_results,
             channel_names=merged_results['channel_names'],
             sampling_rate=merged_results['sampling_rate'])
    
    logger.info(f"Merged data saved to: {output_file}")
    
    return merged_results


def main():
    parser = argparse.ArgumentParser(description='Process ErrP data from BCI wheelchair experiments')
    
    # Participant selection
    parser.add_argument('--participant', '-p', type=str, 
                       help='Participant ID (e.g., T-001)')
    parser.add_argument('--all-participants', action='store_true',
                       help='Process all participants')
    
    # Session selection
    parser.add_argument('--session', '-s', type=str,
                       help='Single session to process (e.g., "Session 3")')
    parser.add_argument('--sessions', nargs='+',
                       help='Multiple sessions to process')
    parser.add_argument('--all-sessions', action='store_true',
                       help='Process all sessions for the participant')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default='errp_results',
                       help='Output directory for results (default: errp_results)')
    parser.add_argument('--sampling-rate', type=int, default=512,
                       help='EEG sampling rate in Hz (default: 512)')
    
    # Processing options
    parser.add_argument('--no-intermediate', action='store_true',
                       help="Don't save intermediate results")
    parser.add_argument('--classifier', choices=['lda', 'svm', 'rf'], default='lda',
                       help='Classifier type (default: lda)')
    
    args = parser.parse_args()
    
    # Get base path
    base_path = get_base_path()
    logger.info(f"EEG data directory: {base_path}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = ErrPPipeline(sampling_rate=args.sampling_rate)
    pipeline.params['analysis']['classifier_type'] = args.classifier
    
    # Determine which participants to process
    if args.all_participants:
        participants = get_available_participants(base_path)
        logger.info(f"Found {len(participants)} participants: {participants}")
    elif args.participant:
        participants = [args.participant]
    else:
        # Show available participants and exit
        available = get_available_participants(base_path)
        print("\nAvailable participants:")
        for p in available:
            sessions = get_available_sessions(base_path, p)
            print(f"  {p}: {len(sessions)} sessions")
        print("\nPlease specify --participant or --all-participants")
        return
    
    # Process each participant
    for participant in participants:
        # Check if participant exists
        if not (base_path / participant).exists():
            logger.error(f"Participant {participant} not found")
            continue
        
        # Determine which sessions to process
        if args.session:
            sessions = [args.session]
        elif args.sessions:
            sessions = args.sessions
        elif args.all_sessions or args.all_participants:
            sessions = get_available_sessions(base_path, participant)
            logger.info(f"{participant} has sessions: {sessions}")
        else:
            # Show available sessions and exit
            available = get_available_sessions(base_path, participant)
            print(f"\nAvailable sessions for {participant}:")
            for s in available:
                session_path = base_path / participant / s
                files = list(session_path.glob("*.csv"))
                print(f"  {s}: {len(files)} files")
            print("\nPlease specify --session, --sessions, or --all-sessions")
            continue
        
        # Process and merge all sessions for this participant
        try:
            merged_results = process_and_merge_sessions(
                pipeline, participant, sessions, base_path, output_dir
            )
            
            # Print summary
            print(f"\n{participant}:")
            for session_info in merged_results['sessions']:
                print(f"  {session_info['session']}: {session_info['n_error']} error epochs, {session_info['n_correct']} correct epochs")
            print(f"  Total: {len(merged_results['error_epochs'])} error epochs, {len(merged_results['correct_epochs'])} correct epochs")
            
        except Exception as e:
            logger.error(f"Error processing {participant}: {e}")
            import traceback
            traceback.print_exc()
        
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()