"""
Example script for analyzing CSV event logs from the ErrP simulation
"""

import pandas as pd
import sys

def analyze_errp_log(filepath):
    """Analyze an ErrP simulation log file"""
    
    # Load the data
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} events from {filepath}")
    
    # Basic statistics
    print("\n=== EVENT SUMMARY ===")
    print(df['event_type'].value_counts())
    
    # Trial analysis
    print("\n=== TRIAL ANALYSIS ===")
    trials = df[df['event_type'] == 'trial_start']
    print(f"Total trials: {len(trials)}")
    
    # Error analysis
    print("\n=== ERROR ANALYSIS ===")
    primary_errors = df[df['event_type'] == 'primary_errp']
    feedback_errors = df[df['event_type'] == 'feedback_error']
    print(f"Primary errors (ErrP events): {len(primary_errors)}")
    print(f"Total error feedbacks: {len(feedback_errors)}")
    
    if len(feedback_errors) > 0:
        error_rate = len(feedback_errors) / len(df[df['event_type'].str.contains('feedback')])
        print(f"Error rate: {error_rate:.1%}")
    
    # Reaction time analysis
    print("\n=== REACTION TIME ANALYSIS ===")
    feedback_events = df[df['reaction_time'].notna() & (df['reaction_time'] != '')]
    if not feedback_events.empty:
        rt_values = feedback_events['reaction_time'].astype(float)
        print(f"Mean RT: {rt_values.mean():.0f}ms")
        print(f"Std RT: {rt_values.std():.0f}ms")
        print(f"Min RT: {rt_values.min():.0f}ms")
        print(f"Max RT: {rt_values.max():.0f}ms")
    
    # Motor imagery analysis
    print("\n=== MOTOR IMAGERY ANALYSIS ===")
    mi_events = df[df['event_type'] == 'imagery_end']
    if not mi_events.empty:
        confidence_values = mi_events['confidence'].astype(float)
        print(f"Mean MI confidence: {confidence_values.mean():.2f}")
        print(f"MI periods: {len(mi_events)}")
    
    # Error type breakdown
    print("\n=== ERROR TYPE BREAKDOWN ===")
    error_types = df[df['error_type'].notna() & (df['error_type'] != '')]
    if not error_types.empty:
        print(error_types['error_type'].value_counts())
    
    # Example: View events from a specific trial
    print("\n=== EXAMPLE: TRIAL 1 EVENTS ===")
    trial1 = df[df['trial_id'] == '1']
    if not trial1.empty:
        print(trial1[['timestamp', 'event_type', 'cue_class', 'predicted_class', 'error_type', 'details']].to_string())
    
    return df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default to most recent log
        import os
        import glob
        logs = glob.glob("logs/subway_errp_*.csv")
        if logs:
            filepath = max(logs, key=os.path.getctime)
            print(f"Using most recent log: {filepath}")
        else:
            print("No log files found in logs/ directory")
            sys.exit(1)
    
    df = analyze_errp_log(filepath)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("You can continue exploring with:")
    print(f"  df = pd.read_csv('{filepath}')")
    print("  df[df['event_type'] == 'primary_errp']  # View error events")
    print("  df.groupby('trial_id')['event_type'].value_counts()  # Events per trial")