"""
Analyze ErrP data from BCI wheelchair experiments
Focus on ERP analysis and visualization without classification

Usage:
    python analyze_errp_data.py --participant T-001 --sessions "Session 3" "Session 4" "Session 5"
    python analyze_errp_data.py --participant T-001 --all-sessions
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import sys

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from errps.data_loader import ErrPDataLoader
from errps.preprocessor import ErrPPreprocessor
from errps.epoch_extractor import ErrPEpochExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPAnalyzer:
    """Analyze ErrP signals from multiple sessions"""
    
    def __init__(self, sampling_rate: int = 512):
        self.sampling_rate = sampling_rate
        self.loader = ErrPDataLoader(sampling_rate)
        self.preprocessor = ErrPPreprocessor(sampling_rate)
        self.extractor = ErrPEpochExtractor(sampling_rate)
        
    def process_sessions(self, participant: str, sessions: list, base_path: Path) -> dict:
        """Process and merge multiple sessions for analysis"""
        
        all_error_epochs = []
        all_correct_epochs = []
        session_info = []
        
        for session in sessions:
            session_path = base_path / participant / session
            
            if not session_path.exists():
                logger.warning(f"Session path not found: {session_path}")
                continue
                
            logger.info(f"\nProcessing {session}...")
            
            try:
                # Load session data
                session_data = self.loader.load_session(str(session_path))
                
                # Preprocess
                preprocessed = self.preprocessor.preprocess(
                    session_data['eeg_data'],
                    apply_spatial_filter='car',
                    remove_powerline=True
                )
                
                # Extract epochs
                epochs = self.extractor.extract_comparison_epochs(
                    preprocessed,
                    session_data['events']
                )
                
                # Collect epochs
                if len(epochs['error']['epochs']) > 0:
                    all_error_epochs.extend(epochs['error']['epochs'])
                if len(epochs['correct']['epochs']) > 0:
                    all_correct_epochs.extend(epochs['correct']['epochs'])
                
                session_info.append({
                    'session': session,
                    'n_error': len(epochs['error']['epochs']),
                    'n_correct': len(epochs['correct']['epochs'])
                })
                
                logger.info(f"  Found {len(epochs['error']['epochs'])} error epochs")
                logger.info(f"  Found {len(epochs['correct']['epochs'])} correct epochs")
                
            except Exception as e:
                logger.error(f"Error processing {session}: {e}")
                continue
        
        # Convert to arrays
        results = {
            'participant': participant,
            'sessions': session_info,
            'error_epochs': np.array(all_error_epochs) if all_error_epochs else np.array([]),
            'correct_epochs': np.array(all_correct_epochs) if all_correct_epochs else np.array([]),
            'times': self.extractor.epoch_params['epoch_start'] + \
                    np.arange(int((self.extractor.epoch_params['epoch_end'] - 
                             self.extractor.epoch_params['epoch_start']) * self.sampling_rate)) / self.sampling_rate
        }
        
        logger.info(f"\nTotal epochs collected:")
        logger.info(f"  Error: {len(results['error_epochs'])}")
        logger.info(f"  Correct: {len(results['correct_epochs'])}")
        
        return results
    
    def analyze_erps(self, data: dict) -> dict:
        """Perform ERP analysis on merged data"""
        
        if len(data['error_epochs']) == 0 or len(data['correct_epochs']) == 0:
            logger.warning("Insufficient epochs for analysis")
            return {}
        
        # Compute ERPs
        error_erp = np.mean(data['error_epochs'], axis=0)
        correct_erp = np.mean(data['correct_epochs'], axis=0)
        
        # Compute standard errors
        error_se = np.std(data['error_epochs'], axis=0) / np.sqrt(len(data['error_epochs']))
        correct_se = np.std(data['correct_epochs'], axis=0) / np.sqrt(len(data['correct_epochs']))
        
        # Difference wave
        difference_wave = error_erp - correct_erp
        
        # Statistical comparison
        n_times, n_channels = error_erp.shape
        p_values = np.zeros((n_times, n_channels))
        
        for t in range(n_times):
            for ch in range(n_channels):
                _, p = stats.ttest_ind(
                    data['error_epochs'][:, t, ch],
                    data['correct_epochs'][:, t, ch]
                )
                p_values[t, ch] = p
        
        # Find significant time windows (corrected for multiple comparisons)
        alpha = 0.05
        significant_mask = p_values < alpha
        
        return {
            'error_erp': error_erp,
            'correct_erp': correct_erp,
            'error_se': error_se,
            'correct_se': correct_se,
            'difference_wave': difference_wave,
            'p_values': p_values,
            'significant_mask': significant_mask,
            'n_error': len(data['error_epochs']),
            'n_correct': len(data['correct_epochs'])
        }
    
    def create_visualization(self, data: dict, analysis: dict, output_path: Path):
        """Create comprehensive ErrP visualization"""
        
        # Select channels to display
        channels_to_plot = [0, 3, 7, 11]  # Fz, C3, Cz, Pz typically
        n_channels = min(len(channels_to_plot), analysis['error_erp'].shape[1])
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        times_ms = data['times'] * 1000  # Convert to milliseconds
        
        # Title
        sessions_str = ", ".join([s['session'] for s in data['sessions']])
        fig.suptitle(f"ErrP Analysis - {data['participant']} - {sessions_str}", fontsize=16)
        
        # Plot ERPs for each channel
        for idx, ch in enumerate(channels_to_plot[:n_channels]):
            # ERP comparison
            ax1 = plt.subplot(n_channels, 3, idx*3 + 1)
            
            # Plot ERPs with confidence intervals
            ax1.plot(times_ms, analysis['error_erp'][:, ch], 'r-', linewidth=2, label='Error')
            ax1.fill_between(times_ms, 
                           analysis['error_erp'][:, ch] - 2*analysis['error_se'][:, ch],
                           analysis['error_erp'][:, ch] + 2*analysis['error_se'][:, ch],
                           color='red', alpha=0.2)
            
            ax1.plot(times_ms, analysis['correct_erp'][:, ch], 'b-', linewidth=2, label='Correct')
            ax1.fill_between(times_ms,
                           analysis['correct_erp'][:, ch] - 2*analysis['correct_se'][:, ch],
                           analysis['correct_erp'][:, ch] + 2*analysis['correct_se'][:, ch],
                           color='blue', alpha=0.2)
            
            # Mark significant regions
            sig_mask = analysis['significant_mask'][:, ch]
            if sig_mask.any():
                sig_regions = self._find_continuous_regions(sig_mask)
                for start, end in sig_regions:
                    ax1.axvspan(times_ms[start], times_ms[end], alpha=0.2, color='gray')
            
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Amplitude (μV)')
            ax1.set_title(f'Channel {ch+1} - ERPs')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-200, 800)
            
            # Difference wave
            ax2 = plt.subplot(n_channels, 3, idx*3 + 2)
            ax2.plot(times_ms, analysis['difference_wave'][:, ch], 'g-', linewidth=2)
            ax2.fill_between(times_ms, 0, analysis['difference_wave'][:, ch],
                           where=sig_mask, alpha=0.3, color='green',
                           label='p<0.05')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Difference (μV)')
            ax2.set_title(f'Channel {ch+1} - Difference Wave')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-200, 800)
            
            # P-values
            ax3 = plt.subplot(n_channels, 3, idx*3 + 3)
            ax3.semilogy(times_ms, analysis['p_values'][:, ch], 'k-')
            ax3.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
            ax3.axhline(y=0.01, color='orange', linestyle='--', label='p=0.01')
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('p-value')
            ax3.set_title(f'Channel {ch+1} - Statistical Significance')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(-200, 800)
            ax3.set_ylim(0.001, 1)
        
        # Add summary text
        ax_text = plt.subplot(n_channels, 3, n_channels*3)
        ax_text.axis('off')
        
        summary = f"Summary Statistics\n" + "="*30 + "\n\n"
        summary += f"Total Epochs:\n"
        summary += f"  Error: {analysis['n_error']}\n"
        summary += f"  Correct: {analysis['n_correct']}\n\n"
        
        summary += f"Sessions Included:\n"
        for s in data['sessions']:
            summary += f"  {s['session']}: {s['n_error']} errors, {s['n_correct']} correct\n"
        
        # Find peak differences
        for ch in channels_to_plot[:n_channels]:
            diff = analysis['difference_wave'][:, ch]
            # ERN window (0-150ms)
            ern_mask = (times_ms >= 0) & (times_ms <= 150)
            if ern_mask.any():
                ern_peak_idx = np.argmin(diff[ern_mask])
                ern_peak_time = times_ms[ern_mask][ern_peak_idx]
                ern_peak_amp = diff[ern_mask][ern_peak_idx]
                
                # Pe window (200-500ms)
                pe_mask = (times_ms >= 200) & (times_ms <= 500)
                pe_peak_idx = np.argmax(diff[pe_mask])
                pe_peak_time = times_ms[pe_mask][pe_peak_idx]
                pe_peak_amp = diff[pe_mask][pe_peak_idx]
                
                summary += f"\nChannel {ch+1} Peaks:\n"
                summary += f"  ERN: {ern_peak_amp:.2f}μV at {ern_peak_time:.0f}ms\n"
                summary += f"  Pe: {pe_peak_amp:.2f}μV at {pe_peak_time:.0f}ms\n"
        
        ax_text.text(0.1, 0.9, summary, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
        plt.close()
        
        # Create a second figure for topographical analysis if we have all channels
        if analysis['error_erp'].shape[1] >= 16:
            self._create_topographical_plot(data, analysis, 
                                           output_path.with_stem(output_path.stem + '_topo'))
    
    def _find_continuous_regions(self, mask):
        """Find continuous True regions in a boolean mask"""
        regions = []
        in_region = False
        start = 0
        
        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i-1))
                in_region = False
        
        if in_region:
            regions.append((start, len(mask)-1))
        
        return regions
    
    def _create_topographical_plot(self, data, analysis, output_path):
        """Create topographical plots at key time points"""
        
        fig = plt.figure(figsize=(20, 12))
        times_ms = data['times'] * 1000
        
        # Key time points for ErrP
        time_points = {
            'Baseline': -50,
            'ERN (100ms)': 100,
            'Pe (300ms)': 300,
            'Late (500ms)': 500
        }
        
        n_plots = len(time_points)
        
        for idx, (label, target_time) in enumerate(time_points.items()):
            # Find closest time index
            time_idx = np.argmin(np.abs(times_ms - target_time))
            actual_time = times_ms[time_idx]
            
            # Error topography
            ax1 = plt.subplot(3, n_plots, idx + 1)
            error_values = analysis['error_erp'][time_idx, :]
            self._plot_channel_values(ax1, error_values, f'Error - {label}\n({actual_time:.0f}ms)')
            
            # Correct topography
            ax2 = plt.subplot(3, n_plots, idx + n_plots + 1)
            correct_values = analysis['correct_erp'][time_idx, :]
            self._plot_channel_values(ax2, correct_values, f'Correct - {label}\n({actual_time:.0f}ms)')
            
            # Difference topography
            ax3 = plt.subplot(3, n_plots, idx + 2*n_plots + 1)
            diff_values = analysis['difference_wave'][time_idx, :]
            self._plot_channel_values(ax3, diff_values, f'Difference - {label}\n({actual_time:.0f}ms)')
        
        plt.suptitle(f"Topographical Analysis - {data['participant']}", fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_channel_values(self, ax, values, title):
        """Simple channel value visualization"""
        # This is a simplified visualization - you could implement proper topographical plots
        # with electrode positions if you have them
        
        n_channels = len(values)
        grid_size = int(np.sqrt(n_channels))
        
        # Reshape values into a grid (simplified representation)
        grid_values = values.reshape(grid_size, grid_size) if n_channels == grid_size**2 else values[:grid_size**2].reshape(grid_size, grid_size)
        
        im = ax.imshow(grid_values, cmap='RdBu_r', aspect='equal')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('μV', rotation=270, labelpad=15)


def main():
    parser = argparse.ArgumentParser(description='Analyze ErrP data from BCI wheelchair experiments')
    
    # Participant selection
    parser.add_argument('--participant', '-p', type=str, required=True,
                       help='Participant ID (e.g., T-001)')
    
    # Session selection
    parser.add_argument('--sessions', nargs='+',
                       help='Sessions to analyze (e.g., "Session 3" "Session 4")')
    parser.add_argument('--all-sessions', action='store_true',
                       help='Analyze all sessions')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str, default='errp_analysis',
                       help='Output directory for results (default: errp_analysis)')
    parser.add_argument('--sampling-rate', type=int, default=512,
                       help='EEG sampling rate in Hz (default: 512)')
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path("C:/Users/yuzeb/BCI_Final/BCI_Wheelchair_Project/EEG_data")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine sessions
    if args.all_sessions:
        participant_path = base_path / args.participant
        if participant_path.exists():
            sessions = [d.name for d in participant_path.iterdir() if d.is_dir() and d.name.startswith('Session')]
            sessions = sorted(sessions)
        else:
            logger.error(f"Participant directory not found: {participant_path}")
            return
    elif args.sessions:
        sessions = args.sessions
    else:
        logger.error("Please specify --sessions or --all-sessions")
        return
    
    logger.info(f"Analyzing {args.participant} - Sessions: {sessions}")
    
    # Initialize analyzer
    analyzer = ErrPAnalyzer(sampling_rate=args.sampling_rate)
    
    # Process sessions
    data = analyzer.process_sessions(args.participant, sessions, base_path)
    
    if len(data['error_epochs']) == 0 or len(data['correct_epochs']) == 0:
        logger.error("No epochs found for analysis")
        return
    
    # Analyze ERPs
    analysis = analyzer.analyze_erps(data)
    
    # Create visualization
    output_file = output_dir / f"{args.participant}_errp_analysis.png"
    analyzer.create_visualization(data, analysis, output_file)
    
    # Save processed data
    np.savez(output_dir / f"{args.participant}_errp_data.npz",
             error_epochs=data['error_epochs'],
             correct_epochs=data['correct_epochs'],
             times=data['times'],
             sessions=data['sessions'])
    
    logger.info(f"\nAnalysis complete!")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()