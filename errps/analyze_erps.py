# analyze_erps.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import stats
import argparse
import logging
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''

# Set matplotlib backend to avoid Qt issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Now your regular imports
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt  # Import after setting backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ERPAnalyzer:
    def __init__(self, sampling_rate: int = 512):
        self.sampling_rate = sampling_rate
        self.times = None
        
    def load_participant_data(self, participant_id: str, sessions: list = None, data_dir: str = './processed_data'):
        """Load specified sessions for a participant"""
        participant_path = Path(data_dir) / participant_id
        
        if not participant_path.exists():
            raise ValueError(f"No processed data found for {participant_id}")
        
        all_error_epochs = []
        all_correct_epochs = []
        sessions_loaded = []
        
        # Get all available session files
        session_files = sorted(participant_path.glob('*_epochs.pkl'))
        
        for session_file in session_files:
            session_name = session_file.stem.replace('_epochs', '')
            
            # Skip if specific sessions requested and this isn't one of them
            if sessions and session_name not in sessions:
                continue
                
            with open(session_file, 'rb') as f:
                data = pickle.load(f)
            
            if data['epochs']['error']['data'].size > 0:
                all_error_epochs.append(data['epochs']['error']['data'])
            if data['epochs']['correct']['data'].size > 0:
                all_correct_epochs.append(data['epochs']['correct']['data'])
                
            # Store times (same for all sessions)
            if self.times is None and 'times' in data['epochs']['error']:
                self.times = data['epochs']['error']['times']
                
            sessions_loaded.append(session_name)
        
        # Concatenate all sessions
        self.error_epochs = np.vstack(all_error_epochs) if all_error_epochs else np.array([])
        self.correct_epochs = np.vstack(all_correct_epochs) if all_correct_epochs else np.array([])
        
        logger.info(f"Loaded {participant_id} - Sessions: {', '.join(sessions_loaded)}")
        logger.info(f"  Error epochs: {self.error_epochs.shape[0] if self.error_epochs.size > 0 else 0}")
        logger.info(f"  Correct epochs: {self.correct_epochs.shape[0] if self.correct_epochs.size > 0 else 0}")
        
        return self
    
    def compute_erps(self):
        """Compute average ERPs and difference waves"""
        if self.error_epochs.size == 0 or self.correct_epochs.size == 0:
            raise ValueError("No epochs available for ERP computation")
            
        self.error_erp = self.error_epochs.mean(axis=0)
        self.correct_erp = self.correct_epochs.mean(axis=0)
        self.difference_wave = self.error_erp - self.correct_erp
        
        # Compute standard errors
        self.error_sem = self.error_epochs.std(axis=0) / np.sqrt(len(self.error_epochs))
        self.correct_sem = self.correct_epochs.std(axis=0) / np.sqrt(len(self.correct_epochs))
        
    def find_peak_latencies(self, channel: int, time_windows: dict):
        """Find peak latencies in specified time windows"""
        peaks = {}
        
        for component, (start_time, end_time) in time_windows.items():
            # Convert time to samples
            start_idx = np.argmin(np.abs(self.times - start_time))
            end_idx = np.argmin(np.abs(self.times - end_time))
            
            # Find peak in difference wave
            window_data = self.difference_wave[start_idx:end_idx, channel]
            
            if component in ['Ne', 'ERN']:  # Negative components
                peak_idx = np.argmin(window_data)
            else:  # Positive components
                peak_idx = np.argmax(window_data)
            
            peak_latency = self.times[start_idx + peak_idx]
            peak_amplitude = window_data[peak_idx]
            
            peaks[component] = {
                'latency': peak_latency,
                'amplitude': peak_amplitude
            }
        
        return peaks
    
    def plot_channel_erps(self, channels: list, channel_names: list = None, save_path: str = None):
        """Plot ERPs for specified channels"""
        if channel_names is None:
            channel_names = [f'Ch{ch+1}' for ch in channels]
        
        fig, axes = plt.subplots(len(channels), 1, figsize=(10, 3*len(channels)), 
                                sharex=True, sharey=True)
        
        if len(channels) == 1:
            axes = [axes]
        
        for idx, (ch, name) in enumerate(zip(channels, channel_names)):
            ax = axes[idx]
            
            # Plot ERPs with confidence intervals
            ax.plot(self.times, self.error_erp[:, ch], 'r-', linewidth=2, label='Error')
            ax.fill_between(self.times, 
                          self.error_erp[:, ch] - self.error_sem[:, ch],
                          self.error_erp[:, ch] + self.error_sem[:, ch],
                          alpha=0.3, color='red')
            
            ax.plot(self.times, self.correct_erp[:, ch], 'b-', linewidth=2, label='Correct')
            ax.fill_between(self.times,
                          self.correct_erp[:, ch] - self.correct_sem[:, ch],
                          self.correct_erp[:, ch] + self.correct_sem[:, ch],
                          alpha=0.3, color='blue')
            
            # Mark typical ErrP components
            ax.axvspan(0.05, 0.10, alpha=0.1, color='yellow', label='Ne/ERN')
            ax.axvspan(0.20, 0.40, alpha=0.1, color='green', label='Pe')
            
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.set_ylabel(f'{name} (uV)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.2, 0.8)
            
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ERP plot to {save_path}")
            
        return fig
    
    def plot_difference_waves(self, channels: list, channel_names: list = None, save_path: str = None):
        """Plot difference waves for specified channels"""
        if channel_names is None:
            channel_names = [f'Ch{ch+1}' for ch in channels]
            
        plt.figure(figsize=(10, 6))
        
        for ch, name in zip(channels, channel_names):
            plt.plot(self.times, self.difference_wave[:, ch], linewidth=2, label=name)
        
        plt.axvline(0, color='k', linestyle='--', alpha=0.5)
        plt.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        # Mark components
        plt.axvspan(0.05, 0.10, alpha=0.1, color='yellow')
        plt.axvspan(0.20, 0.40, alpha=0.1, color='green')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Error - Correct (uV)')
        plt.title('Difference Waves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(-0.2, 0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved difference wave plot to {save_path}")
    
    def statistical_analysis(self, time_windows: dict):
        """Perform statistical tests on ERP components"""
        results = {}
        
        for component, (start_time, end_time) in time_windows.items():
            start_idx = np.argmin(np.abs(self.times - start_time))
            end_idx = np.argmin(np.abs(self.times - end_time))
            
            # Average amplitude in time window for each trial
            error_amps = self.error_epochs[:, start_idx:end_idx, :].mean(axis=1)
            correct_amps = self.correct_epochs[:, start_idx:end_idx, :].mean(axis=1)
            
            # Perform t-tests for each channel
            channel_results = []
            for ch in range(error_amps.shape[1]):
                t_stat, p_value = stats.ttest_ind(error_amps[:, ch], correct_amps[:, ch])
                effect_size = (error_amps[:, ch].mean() - correct_amps[:, ch].mean()) / \
                            np.sqrt((error_amps[:, ch].var() + correct_amps[:, ch].var()) / 2)
                
                channel_results.append({
                    'channel': ch,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'mean_diff': error_amps[:, ch].mean() - correct_amps[:, ch].mean()
                })
            
            results[component] = sorted(channel_results, key=lambda x: abs(x['effect_size']), reverse=True)
        
        return results


def main():
    """Command line interface for ERP analysis"""
    parser = argparse.ArgumentParser(description='Analyze ERPs from processed EEG data')
    parser.add_argument('--data-dir', type=str, default='./processed_data',
                        help='Directory containing processed epochs')
    parser.add_argument('--participants', nargs='*', type=str,
                        help='Participant IDs to analyze (e.g., T-001 T-002). If not specified, analyze all.')
    parser.add_argument('--sessions', nargs='*', type=str,
                        help='Session names to include (e.g., "Session 2" "Session 3"). If not specified, use all.')
    parser.add_argument('--channels', nargs='*', type=int, default=[3, 5, 7, 9],
                        help='Channel indices to plot (0-based)')
    parser.add_argument('--channel-names', nargs='*', type=str,
                        help='Channel names for plotting')
    parser.add_argument('--output-dir', type=str, default='./erp_results',
                        help='Directory to save analysis results')
    parser.add_argument('--show-plots', action='store_true',
                        help='Display plots interactively')
    
    args = parser.parse_args()
    
    # Find participants to analyze
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_path}")
        return
    
    if args.participants:
        participant_ids = args.participants
    else:
        # Find all participants with processed data
        participant_ids = [d.name for d in data_path.iterdir() if d.is_dir()]
        participant_ids = sorted([p for p in participant_ids if p.startswith('T-')])
    
    if not participant_ids:
        print("No participants found in processed data directory!")
        return
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Define time windows for ErrP components
    time_windows = {
        'Ne/ERN': (0.15, 0.2),   # 150-200ms
        'Pe': (0.25, 0.35),       # 250-350ms
    }
    
    # Process each participant
    for participant_id in participant_ids:
        print(f"\n{'='*60}")
        print(f"Analyzing participant: {participant_id}")
        print(f"{'='*60}")
        
        try:
            # Initialize analyzer
            analyzer = ERPAnalyzer()
            
            # Load participant data
            analyzer.load_participant_data(participant_id, sessions=args.sessions, data_dir=args.data_dir)
            
            # Compute ERPs
            analyzer.compute_erps()
            
            # Create participant output directory
            participant_output = output_path / participant_id
            participant_output.mkdir(exist_ok=True)
            
            # Find peak latencies
            print("\nPeak Analysis:")
            for ch_idx, ch_name in enumerate(args.channel_names or [f'Ch{ch+1}' for ch in args.channels]):
                if ch_idx < len(args.channels):
                    channel = args.channels[ch_idx]
                    peaks = analyzer.find_peak_latencies(channel=channel, time_windows=time_windows)
                    print(f"\n{ch_name} (Channel {channel+1}):")
                    for component, values in peaks.items():
                        print(f"  {component}: {values['latency']*1000:.1f}ms, {values['amplitude']:.2f}uV")
            
            # Plot ERPs
            erp_fig = analyzer.plot_channel_erps(
                channels=args.channels,
                channel_names=args.channel_names,
                save_path=participant_output / 'erp_waveforms.png'
            )
            
            # Plot difference waves
            analyzer.plot_difference_waves(
                channels=args.channels,
                channel_names=args.channel_names,
                save_path=participant_output / 'difference_waves.png'
            )
            
            # Statistical analysis
            stats_results = analyzer.statistical_analysis(time_windows)
            
            # Save statistical results
            with open(participant_output / 'statistical_results.txt', 'w') as f:
                f.write(f"Statistical Analysis for {participant_id}\n")
                f.write("="*50 + "\n\n")
                
                for component, results in stats_results.items():
                    f.write(f"{component} Time Window:\n")
                    f.write("-"*30 + "\n")
                    
                    # Show top 5 channels with strongest effects
                    for i, ch_result in enumerate(results[:5]):
                        f.write(f"Channel {ch_result['channel']+1}:\n")
                        f.write(f"  Mean difference: {ch_result['mean_diff']:.2f} uV\n")
                        f.write(f"  t-statistic: {ch_result['t_stat']:.2f}\n")
                        f.write(f"  p-value: {ch_result['p_value']:.4f}\n")
                        f.write(f"  Effect size (Cohen's d): {ch_result['effect_size']:.2f}\n\n")
            
            # Print summary
            print("\nStatistical Summary:")
            for component, results in stats_results.items():
                sig_channels = [r for r in results if r['p_value'] < 0.05]
                print(f"\n{component}: {len(sig_channels)} channels with significant differences (p<0.05)")
                if sig_channels:
                    top_channel = sig_channels[0]
                    print(f"  Strongest effect: Channel {top_channel['channel']+1} "
                          f"(d={top_channel['effect_size']:.2f}, p={top_channel['p_value']:.4f})")
            
            print(f"\nResults saved to: {participant_output}")
            
        except Exception as e:
            print(f"Error analyzing {participant_id}: {str(e)}")
            continue
    
    if args.show_plots:
        plt.show()
    else:
        plt.close('all')
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Processed {len(participant_ids)} participants")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()