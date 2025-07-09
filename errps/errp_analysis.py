"""
ErrP Analysis Module
Analyzes processed ErrP data and creates visualizations

Usage:
    python errp_analysis.py --participant T-001 --input-file errp_results/T-001_merged_errp_data.npz
    python errp_analysis.py --participant T-001  # Uses default path
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import logging

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None
    logging.warning("statsmodels not installed - FDR correction will be skipped")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrPAnalysis:
    """Analyze and visualize processed ErrP data"""
    
    def __init__(self):
        # Updated channel names to match your montage
        self.channel_names = ['Fz', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4',
                             'C3', 'C1', 'Cz', 'C2', 'C4',
                             'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4']
        
    def load_processed_data(self, file_path: str) -> dict:
        """Load processed ErrP data from npz file"""
        
        data = np.load(file_path, allow_pickle=True)
        
        # Handle sessions loading - it might be a list or array
        sessions = []
        if 'sessions' in data:
            sessions_data = data['sessions']
            if hasattr(sessions_data, 'tolist'):
                sessions = sessions_data.tolist()
            elif hasattr(sessions_data, 'item'):
                sessions = sessions_data.item()
            else:
                sessions = sessions_data
        
        # Handle channel names
        channel_names = self.channel_names
        if 'channel_names' in data:
            cn_data = data['channel_names']
            if hasattr(cn_data, 'tolist'):
                channel_names = cn_data.tolist()
            else:
                channel_names = cn_data
        
        # Handle sampling rate
        sampling_rate = 512
        if 'sampling_rate' in data:
            sr_data = data['sampling_rate']
            if hasattr(sr_data, 'item'):
                sampling_rate = sr_data.item()
            else:
                sampling_rate = int(sr_data)
        
        loaded_data = {
            'error_epochs': data['error_epochs'],
            'correct_epochs': data['correct_epochs'],
            'times': data['times'],
            'sessions': sessions,
            'channel_names': channel_names,
            'sampling_rate': sampling_rate
        }
        
        logger.info(f"Loaded data:")
        logger.info(f"  Error epochs: {loaded_data['error_epochs'].shape}")
        logger.info(f"  Correct epochs: {loaded_data['correct_epochs'].shape}")
        logger.info(f"  Time points: {len(loaded_data['times'])}")
        if sessions:
            logger.info(f"  Sessions: {len(sessions)}")
        
        return loaded_data
    
    def compute_erps(self, data: dict) -> dict:
        """Compute ERPs and statistical analysis"""
        
        # Compute ERPs
        error_erp = np.mean(data['error_epochs'], axis=0)
        correct_erp = np.mean(data['correct_epochs'], axis=0)
        
        # Compute standard errors
        error_se = np.std(data['error_epochs'], axis=0) / np.sqrt(len(data['error_epochs']))
        correct_se = np.std(data['correct_epochs'], axis=0) / np.sqrt(len(data['correct_epochs']))
        
        # Difference wave
        difference_wave = error_erp - correct_erp
        
        # Statistical testing
        n_times, n_channels = error_erp.shape
        p_values = np.zeros((n_times, n_channels))
        t_values = np.zeros((n_times, n_channels))
        
        for t in range(n_times):
            for ch in range(n_channels):
                t_stat, p_val = stats.ttest_ind(
                    data['error_epochs'][:, t, ch],
                    data['correct_epochs'][:, t, ch]
                )
                t_values[t, ch] = t_stat
                p_values[t, ch] = p_val
        
        # Multiple comparison correction (FDR)
        if multipletests is not None:
            significant_mask = np.zeros_like(p_values, dtype=bool)
            for ch in range(n_channels):
                _, p_corrected, _, _ = multipletests(p_values[:, ch], alpha=0.05, method='fdr_bh')
                significant_mask[:, ch] = p_corrected < 0.05
        else:
            # Simple threshold without correction
            significant_mask = p_values < 0.05
        
        return {
            'error_erp': error_erp,
            'correct_erp': correct_erp,
            'error_se': error_se,
            'correct_se': correct_se,
            'difference_wave': difference_wave,
            'p_values': p_values,
            't_values': t_values,
            'significant_mask': significant_mask,
            'n_error': len(data['error_epochs']),
            'n_correct': len(data['correct_epochs'])
        }
    
    def plot_individual_epochs(self, data: dict, n_epochs: int = 20, output_path: Path = None):
        """Plot individual epochs"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Ensure times matches epoch length
        n_time_points = data['error_epochs'].shape[1] if len(data['error_epochs']) > 0 else data['correct_epochs'].shape[1]
        if len(data['times']) != n_time_points:
            logger.warning(f"Time vector mismatch: {len(data['times'])} vs {n_time_points} samples")
            # Recreate time vector with correct length
            epoch_duration = 1.0  # 1 second total (-0.2 to 0.8)
            times = np.linspace(-0.2, 0.8, n_time_points)
        else:
            times = data['times']
        
        times_ms = times * 1000
        
        # Select central channel (Cz - typically channel 3)
        ch = 3
        
        # Plot error epochs
        ax = axes[0, 0]
        n_error_plot = min(n_epochs, len(data['error_epochs']))
        for i in range(n_error_plot):
            ax.plot(times_ms, data['error_epochs'][i, :, ch], 'r-', alpha=0.3)
        ax.plot(times_ms, np.mean(data['error_epochs'][:, :, ch], axis=0), 'r-', 
                linewidth=3, label='Average')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'Error Epochs (n={len(data["error_epochs"])}) - Channel Cz')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-200, 800)
        
        # Plot correct epochs
        ax = axes[0, 1]
        n_correct_plot = min(n_epochs, len(data['correct_epochs']))
        for i in range(n_correct_plot):
            ax.plot(times_ms, data['correct_epochs'][i, :, ch], 'b-', alpha=0.3)
        ax.plot(times_ms, np.mean(data['correct_epochs'][:, :, ch], axis=0), 'b-', 
                linewidth=3, label='Average')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'Correct Epochs (n={len(data["correct_epochs"])}) - Channel Cz')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-200, 800)
        
        # Plot all channels - error
        ax = axes[1, 0]
        for ch_idx in range(data['error_epochs'].shape[2]):
            ax.plot(times_ms, np.mean(data['error_epochs'][:, :, ch_idx], axis=0), 
                   alpha=0.7, label=f'Ch{ch_idx+1}' if ch_idx < 4 else '')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('Error ERPs - All Channels')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-200, 800)
        
        # Plot all channels - correct
        ax = axes[1, 1]
        for ch_idx in range(data['correct_epochs'].shape[2]):
            ax.plot(times_ms, np.mean(data['correct_epochs'][:, :, ch_idx], axis=0), 
                   alpha=0.7, label=f'Ch{ch_idx+1}' if ch_idx < 4 else '')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('Correct ERPs - All Channels')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-200, 800)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Epochs plot saved to {output_path}")
        plt.close()
    
    def plot_erp_analysis(self, data: dict, analysis: dict, output_path: Path = None):
        """Create comprehensive ERP analysis plot"""
        
        # Select key channels: Fz, FCz, Cz, CPz
        channel_indices = [0, 3, 8, 13]  # Fz=0, Fcz=3, Cz=8, Cpz=13
        channel_labels = ['Fz', 'FCz', 'Cz', 'CPz']
        n_channels = len(channel_indices)
        
        fig = plt.figure(figsize=(20, 16))
        
        # Ensure times matches analysis data
        n_time_points = analysis['error_erp'].shape[0]
        if len(data['times']) != n_time_points:
            times = np.linspace(-0.2, 0.8, n_time_points)
        else:
            times = data['times']
        times_ms = times * 1000
        
        for idx, (ch, ch_label) in enumerate(zip(channel_indices, channel_labels)):
            # ERP comparison
            ax1 = plt.subplot(n_channels, 4, idx*4 + 1)
            
            # Plot with confidence intervals
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
                    ax1.axvspan(times_ms[start], times_ms[end], alpha=0.2, color='yellow')
            
            ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax1.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Amplitude (μV)')
            ax1.set_title(f'{ch_label} - ERPs')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-200, 800)
            
            # Difference wave
            ax2 = plt.subplot(n_channels, 4, idx*4 + 2)
            ax2.plot(times_ms, analysis['difference_wave'][:, ch], 'g-', linewidth=2)
            ax2.fill_between(times_ms, 0, analysis['difference_wave'][:, ch],
                           where=sig_mask, alpha=0.3, color='green')
            
            # Mark ErrP components
            ax2.axvspan(0, 150, alpha=0.1, color='red', label='ERN')
            ax2.axvspan(200, 500, alpha=0.1, color='blue', label='Pe')
            
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax2.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Time (ms)')
            ax2.set_ylabel('Difference (μV)')
            ax2.set_title(f'{ch_label} - Difference Wave')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-200, 800)
            
            # T-values
            ax3 = plt.subplot(n_channels, 4, idx*4 + 3)
            ax3.plot(times_ms, analysis['t_values'][:, ch], 'k-', linewidth=2)
            ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='t=2')
            ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
            ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Time (ms)')
            ax3.set_ylabel('t-value')
            ax3.set_title(f'{ch_label} - T-statistics')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_xlim(-200, 800)
            
            # P-values
            ax4 = plt.subplot(n_channels, 4, idx*4 + 4)
            ax4.semilogy(times_ms, analysis['p_values'][:, ch], 'k-')
            ax4.axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
            ax4.axhline(y=0.01, color='orange', linestyle='--', label='p=0.01')
            ax4.axhline(y=0.001, color='red', linestyle='--', label='p=0.001')
            ax4.set_xlabel('Time (ms)')
            ax4.set_ylabel('p-value')
            ax4.set_title(f'{ch_label} - Significance')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(-200, 800)
            ax4.set_ylim(0.0001, 1)
        
        # Add summary
        plt.figtext(0.5, 0.98, f"ErrP Analysis - Error: {analysis['n_error']} epochs, Correct: {analysis['n_correct']} epochs", 
                   ha='center', fontsize=14, weight='bold')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"ERP analysis saved to {output_path}")
        plt.close()
    
    def plot_component_summary(self, data: dict, analysis: dict, output_path: Path = None):
        """Plot summary of ErrP components across channels"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Ensure times matches analysis data
        n_time_points = analysis['difference_wave'].shape[0]
        if len(data['times']) != n_time_points:
            times = np.linspace(-0.2, 0.8, n_time_points)
        else:
            times = data['times']
        times_ms = times * 1000
        
        # Define component windows
        ern_window = (50, 150)  # ERN: 50-150ms
        pe_window = (200, 400)  # Pe: 200-400ms
        
        # 1. Grand average difference wave
        ax = axes[0, 0]
        grand_avg_diff = np.mean(analysis['difference_wave'], axis=1)
        grand_avg_se = np.std(analysis['difference_wave'], axis=1) / np.sqrt(analysis['difference_wave'].shape[1])
        
        ax.plot(times_ms, grand_avg_diff, 'g-', linewidth=3)
        ax.fill_between(times_ms, grand_avg_diff - 2*grand_avg_se, 
                       grand_avg_diff + 2*grand_avg_se, color='green', alpha=0.2)
        ax.axvspan(ern_window[0], ern_window[1], alpha=0.1, color='red', label='ERN')
        ax.axvspan(pe_window[0], pe_window[1], alpha=0.1, color='blue', label='Pe')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title('Grand Average Difference Wave')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-200, 800)
        
        # 2. Component amplitudes by channel
        ax = axes[0, 1]
        
        # Extract component peaks
        ern_mask = (times_ms >= ern_window[0]) & (times_ms <= ern_window[1])
        pe_mask = (times_ms >= pe_window[0]) & (times_ms <= pe_window[1])
        
        ern_peaks = []
        pe_peaks = []
        
        for ch in range(analysis['difference_wave'].shape[1]):
            diff = analysis['difference_wave'][:, ch]
            ern_peaks.append(np.min(diff[ern_mask]))
            pe_peaks.append(np.max(diff[pe_mask]))
        
        x = np.arange(len(ern_peaks))
        width = 0.35
        
        ax.bar(x - width/2, ern_peaks, width, label='ERN', color='red', alpha=0.7)
        ax.bar(x + width/2, pe_peaks, width, label='Pe', color='blue', alpha=0.7)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Peak Amplitude (μV)')
        ax.set_title('ErrP Component Peaks by Channel')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ch{i+1}' for i in range(len(ern_peaks))], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # 3. Significant time points
        ax = axes[1, 0]
        sig_proportion = np.mean(analysis['significant_mask'], axis=1)
        ax.plot(times_ms, sig_proportion * 100, 'k-', linewidth=2)
        ax.fill_between(times_ms, 0, sig_proportion * 100, alpha=0.3, color='gray')
        ax.axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% channels')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('% Channels Significant')
        ax.set_title('Proportion of Significant Channels')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-200, 800)
        ax.set_ylim(0, 100)
        
        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "Summary Statistics\n" + "="*30 + "\n\n"
        summary_text += f"Total Epochs:\n"
        summary_text += f"  Error: {analysis['n_error']}\n"
        summary_text += f"  Correct: {analysis['n_correct']}\n\n"
        
        if 'sessions' in data and data['sessions']:
            summary_text += "Sessions Included:\n"
            for session in data['sessions']:
                summary_text += f"  {session['session']}: "
                summary_text += f"{session['n_error']} err, {session['n_correct']} corr\n"
        
        # Component analysis
        summary_text += f"\nGrand Average Components:\n"
        ern_peak_time = times_ms[ern_mask][np.argmin(grand_avg_diff[ern_mask])]
        ern_peak_amp = np.min(grand_avg_diff[ern_mask])
        pe_peak_time = times_ms[pe_mask][np.argmax(grand_avg_diff[pe_mask])]
        pe_peak_amp = np.max(grand_avg_diff[pe_mask])
        
        summary_text += f"  ERN: {ern_peak_amp:.2f}μV at {ern_peak_time:.0f}ms\n"
        summary_text += f"  Pe: {pe_peak_amp:.2f}μV at {pe_peak_time:.0f}ms\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Component summary saved to {output_path}")
        plt.close()
    
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
    
    def plot_all_channels_comparison(self, data: dict, analysis: dict, output_path: Path = None):
        """Plot all channels with error vs correct comparison"""
        
        fig = plt.figure(figsize=(24, 16))
        times_ms = data['times'] * 1000 if len(data['times']) == analysis['error_erp'].shape[0] else np.linspace(-200, 800, analysis['error_erp'].shape[0])
        
        n_channels = min(16, analysis['error_erp'].shape[1])
        
        # Create 4x4 grid for 16 channels
        for ch in range(n_channels):
            ax = plt.subplot(4, 4, ch + 1)
            
            # Plot error ERP
            ax.plot(times_ms, analysis['error_erp'][:, ch], 'r-', linewidth=2, label='Error')
            ax.fill_between(times_ms, 
                           analysis['error_erp'][:, ch] - analysis['error_se'][:, ch],
                           analysis['error_erp'][:, ch] + analysis['error_se'][:, ch],
                           color='red', alpha=0.2)
            
            # Plot correct ERP
            ax.plot(times_ms, analysis['correct_erp'][:, ch], 'b-', linewidth=2, label='Correct')
            ax.fill_between(times_ms,
                           analysis['correct_erp'][:, ch] - analysis['correct_se'][:, ch],
                           analysis['correct_erp'][:, ch] + analysis['correct_se'][:, ch],
                           color='blue', alpha=0.2)
            
            # Mark significant regions
            sig_mask = analysis['significant_mask'][:, ch]
            if sig_mask.any():
                sig_regions = self._find_continuous_regions(sig_mask)
                for start, end in sig_regions:
                    ax.axvspan(times_ms[start], times_ms[end], alpha=0.15, color='yellow')
            
            # Add vertical lines for stimulus and key time points
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Channel label
            ch_name = self.channel_names[ch] if ch < len(self.channel_names) else f'Ch{ch+1}'
            ax.set_title(ch_name, fontsize=11, fontweight='bold')
            
            # Only show x-axis labels on bottom row
            if ch >= 12:
                ax.set_xlabel('Time (ms)', fontsize=9)
            else:
                ax.set_xticklabels([])
            
            # Only show y-axis labels on left column
            if ch % 4 == 0:
                ax.set_ylabel('Amplitude (μV)', fontsize=9)
            else:
                ax.set_yticklabels([])
            
            # Add legend only to first subplot
            if ch == 0:
                ax.legend(loc='upper right', fontsize=9)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-200, 800)
            ax.tick_params(axis='both', labelsize=8)
            
            # Add difference value in corner
            diff = analysis['difference_wave'][:, ch]
            max_diff = np.max(np.abs(diff))
            ax.text(0.02, 0.98, f'Max Δ: {max_diff:.2f}μV', 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.suptitle(f'All Channels ERP Comparison\nError (n={analysis["n_error"]}) vs Correct (n={analysis["n_correct"]})', 
                    fontsize=16, y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"All channels comparison saved to {output_path}")
        plt.close()
    
    def generate_report(self, data: dict, participant: str, output_dir: Path):
        """Generate complete analysis report"""
        
        # Compute ERPs and statistics
        analysis = self.compute_erps(data)
        
        # Create plots
        self.plot_individual_epochs(
            data, 
            n_epochs=20,
            output_path=output_dir / f"{participant}_epochs.png"
        )
        
        self.plot_erp_analysis(
            data,
            analysis,
            output_path=output_dir / f"{participant}_erp_analysis.png"
        )
        
        self.plot_all_channels_comparison(
            data,
            analysis,
            output_path=output_dir / f"{participant}_all_channels_comparison.png"
        )
        
        self.plot_component_summary(
            data,
            analysis,
            output_path=output_dir / f"{participant}_component_summary.png"
        )
        
        # Save analysis results
        np.savez(output_dir / f"{participant}_analysis_results.npz",
                 **analysis,
                 participant=participant)
        
        logger.info(f"\nAnalysis complete for {participant}")
        logger.info(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze processed ErrP data')
    parser.add_argument('--participant', '-p', type=str, required=True,
                       help='Participant ID (e.g., T-001)')
    parser.add_argument('--input-file', '-i', type=str,
                       help='Path to processed data file (npz)')
    parser.add_argument('--output-dir', '-o', type=str, default='errp_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Setup paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input file
    if args.input_file:
        input_file = Path(args.input_file)
    else:
        # Default path
        input_file = Path('errp_results') / f"{args.participant}_merged_errp_data.npz"
    
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.info("Please run process_errp_data.py first to generate the processed data")
        return
    
    # Initialize analyzer
    analyzer = ErrPAnalysis()
    
    # Load data
    logger.info(f"Loading processed data from: {input_file}")
    data = analyzer.load_processed_data(input_file)
    
    # Generate analysis report
    analyzer.generate_report(data, args.participant, output_dir)


if __name__ == "__main__":
    main()