#!/usr/bin/env python
"""
Process new participant data from CSV to pkl format
This script takes raw CSV data and creates the processed pkl file needed for training
"""

import os
import sys
import argparse
from datetime import datetime

# Add MI module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import MIProcessingPipeline

def main():
    parser = argparse.ArgumentParser(description='Process MI data for a new participant')
    parser.add_argument('--participant', type=str, required=True,
                       help='Participant ID (e.g., T-009)')
    parser.add_argument('--sessions', nargs='+', default=None,
                       help='Sessions to process (e.g., Session 1 Session 2). Default: all')
    parser.add_argument('--base-path', type=str, default='MI/EEG data',
                       help='Base path to EEG data folder (default: MI/EEG data)')
    parser.add_argument('--output-dir', type=str, default='MI/processed_data',
                       help='Output directory for processed data (default: MI/processed_data)')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"MI Data Processing Pipeline")
    print(f"{'='*70}")
    print(f"Participant: {args.participant}")
    print(f"Base path: {args.base_path}")
    print(f"Sessions: {args.sessions if args.sessions else 'All available'}")
    print(f"{'='*70}\n")
    
    # Initialize pipeline
    try:
        pipeline = MIProcessingPipeline(base_path=args.base_path, sampling_rate=512)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return
    
    # Process participant data
    try:
        # Process the data
        processed_data = pipeline.process_participant(
            participant_id=args.participant,
            sessions=args.sessions
        )
        
        # Save processed data
        os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, f"{args.participant}_processed.pkl")
        pipeline.save_processed_data(processed_data, output_file)
        
        print(f"\n{'='*70}")
        print(f"✓ Processing complete!")
        print(f"✓ Processed data saved to: {output_file}")
        print(f"{'='*70}")
        
        # Print summary statistics
        if 'epochs' in processed_data and len(processed_data['epochs']) > 0:
            print(f"\nSummary:")
            print(f"  Total epochs: {len(processed_data['epochs'])}")
            print(f"  Epoch shape: {processed_data['epochs'].shape}")
            print(f"  Labels: {processed_data['labels']}")
            
            # Class distribution
            import numpy as np
            unique, counts = np.unique(processed_data['labels'], return_counts=True)
            print(f"\nClass distribution:")
            for label, count in zip(unique, counts):
                class_name = 'Left' if label == 1 else 'Right' if label == 2 else 'Rest'
                print(f"  {class_name}: {count} trials ({count/len(processed_data['labels'])*100:.1f}%)")
        
        print(f"\nNext steps:")
        print(f"1. Train subject-specific model:")
        print(f"   python MI/classifiers/subject_specific/train_subject_classifier.py {args.participant}")
        print(f"\n2. Or use the processed data with other training scripts")
        
    except Exception as e:
        print(f"\nError processing data: {e}")
        print(f"Make sure the data exists at: {args.base_path}/{args.participant}/")
        print(f"Expected structure:")
        print(f"  {args.base_path}/")
        print(f"    {args.participant}/")
        print(f"      Session 1/")
        print(f"        *.csv files")
        print(f"      Session 2/")
        print(f"        *.csv files")
        print(f"      Info.txt")


if __name__ == "__main__":
    main()