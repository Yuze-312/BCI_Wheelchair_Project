# analyze_session_variability.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn as sns
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_all_sessions(participant_id='T-002', data_dir='processed_data'):
    """Load all sessions for a participant separately"""
    participant_path = Path(data_dir) / participant_id
    
    sessions_data = {}
    
    # Get all session files
    session_files = sorted(participant_path.glob('*.pkl'))
    
    for session_file in session_files:
        # Extract session name
        session_name = session_file.stem.replace('_epochs', '')
        
        with open(session_file, 'rb') as f:
            data = pickle.load(f)
        
        sessions_data[session_name] = {
            'error': data['epochs']['error'],
            'correct': data['epochs']['correct'],
            'file': session_file.name
        }
        
        print(f"Loaded {session_name}:")
        print(f"  Error epochs: {data['epochs']['error']['data'].shape[0] if data['epochs']['error']['data'].size > 0 else 0}")
        print(f"  Correct epochs: {data['epochs']['correct']['data'].shape[0] if data['epochs']['correct']['data'].size > 0 else 0}")
    
    return sessions_data

def plot_session_erps(sessions_data, channel=0, save_dir='session_analysis'):
    """Plot ERPs for each session separately"""
    Path(save_dir).mkdir(exist_ok=True)
    
    n_sessions = len(sessions_data)
    fig, axes = plt.subplots(n_sessions, 1, figsize=(12, 4*n_sessions), sharex=True)
    
    if n_sessions == 1:
        axes = [axes]
    
    session_stats = {}
    
    for idx, (session_name, session_data) in enumerate(sorted(sessions_data.items())):
        ax = axes[idx]
        
        error_data = session_data['error']['data']
        correct_data = session_data['correct']['data']
        
        if 'times' in session_data['error']:
            times = session_data['error']['times']
        else:
            times = np.linspace(-0.2, 0.8, error_data.shape[1])
        
        # Calculate means and SEMs
        if error_data.size > 0:
            error_mean = error_data[:, :, channel].mean(axis=0)
            error_sem = error_data[:, :, channel].std(axis=0) / np.sqrt(len(error_data))
            
            ax.plot(times, error_mean, 'r-', linewidth=2, label=f'Error (n={len(error_data)})')
            ax.fill_between(times, error_mean-error_sem, error_mean+error_sem, 
                          alpha=0.3, color='red')
        
        if correct_data.size > 0:
            correct_mean = correct_data[:, :, channel].mean(axis=0)
            correct_sem = correct_data[:, :, channel].std(axis=0) / np.sqrt(len(correct_data))
            
            ax.plot(times, correct_mean, 'b-', linewidth=2, label=f'Correct (n={len(correct_data)})')
            ax.fill_between(times, correct_mean-correct_sem, correct_mean+correct_sem, 
                          alpha=0.3, color='blue')
        
        # Mark components
        ax.axvspan(0.15, 0.20, alpha=0.1, color='yellow')
        ax.axvspan(0.25, 0.35, alpha=0.1, color='green')
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        
        ax.set_ylabel('Amplitude (μV)')
        ax.set_title(f'{session_name} - Channel {channel+1}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-10, 25)
        
        # Calculate session statistics
        if error_data.size > 0 and correct_data.size > 0:
            # Pe window statistics (250-350ms)
            pe_start = int(0.45 * 512)  # Assuming 512 Hz
            pe_end = int(0.55 * 512)
            
            error_pe = error_data[:, pe_start:pe_end, channel].max(axis=1)
            correct_pe = correct_data[:, pe_start:pe_end, channel].max(axis=1)
            
            session_stats[session_name] = {
                'error_pe_mean': error_pe.mean(),
                'error_pe_std': error_pe.std(),
                'correct_pe_mean': correct_pe.mean(),
                'correct_pe_std': correct_pe.std(),
                'effect_size': (error_pe.mean() - correct_pe.mean()) / np.sqrt((error_pe.var() + correct_pe.var()) / 2)
            }
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/session_erps_ch{channel+1}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return session_stats

def plot_session_variability(sessions_data, channel=0, save_dir='session_analysis'):
    """Plot individual trial variability within each session"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (session_name, session_data) in enumerate(sorted(sessions_data.items())):
        if idx >= 4:  # Only plot first 4 sessions
            break
            
        ax = axes[idx]
        
        error_data = session_data['error']['data']
        
        if error_data.size > 0:
            if 'times' in session_data['error']:
                times = session_data['error']['times']
            else:
                times = np.linspace(-0.2, 0.8, error_data.shape[1])
            
            # Plot individual error trials
            for trial in range(len(error_data)):
                ax.plot(times, error_data[trial, :, channel], 'r-', alpha=0.3, linewidth=0.5)
            
            # Plot mean on top
            error_mean = error_data[:, :, channel].mean(axis=0)
            ax.plot(times, error_mean, 'k-', linewidth=3, label='Mean')
            
            ax.set_title(f'{session_name} - Error Trials (n={len(error_data)})')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (μV)')
            ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            ax.axhline(0, color='k', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-50, 80)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/trial_variability_ch{channel+1}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_pe_distributions(sessions_data, channel=0, save_dir='session_analysis'):
    """Plot Pe amplitude distributions for each session"""
    Path(save_dir).mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    all_session_stats = []
    
    for idx, (session_name, session_data) in enumerate(sorted(sessions_data.items())):
        if idx >= 4:
            break
            
        ax = axes[idx]
        
        error_data = session_data['error']['data']
        correct_data = session_data['correct']['data']
        
        if error_data.size > 0 and correct_data.size > 0:
            # Extract Pe amplitudes (250-350ms window)
            pe_start = int(0.45 * 512)
            pe_end = int(0.55 * 512)
            
            error_pe = error_data[:, pe_start:pe_end, channel].max(axis=1)
            correct_pe = correct_data[:, pe_start:pe_end, channel].max(axis=1)
            
            # Plot distributions
            bins = np.linspace(-20, 80, 30)
            ax.hist(error_pe, bins=bins, alpha=0.6, color='red', label='Error', density=True)
            ax.hist(correct_pe, bins=bins, alpha=0.6, color='blue', label='Correct', density=True)
            
            # Add means
            ax.axvline(error_pe.mean(), color='darkred', linestyle='--', linewidth=2)
            ax.axvline(correct_pe.mean(), color='darkblue', linestyle='--', linewidth=2)
            
            ax.set_title(f'{session_name}')
            ax.set_xlabel('Pe Amplitude (μV)')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Calculate statistics
            t_stat, p_value = stats.ttest_ind(error_pe, correct_pe)
            effect_size = (error_pe.mean() - correct_pe.mean()) / np.sqrt((error_pe.var() + correct_pe.var()) / 2)
            
            stats_text = f'p={p_value:.3f}\nd={effect_size:.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            all_session_stats.append({
                'session': session_name,
                'error_mean': error_pe.mean(),
                'error_std': error_pe.std(),
                'correct_mean': correct_pe.mean(),
                'correct_std': correct_pe.std(),
                'p_value': p_value,
                'effect_size': effect_size
            })
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pe_distributions_ch{channel+1}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return all_session_stats

def plot_session_comparison(sessions_data, channel=0, save_dir='session_analysis'):
    """Create a summary comparison across sessions"""
    Path(save_dir).mkdir(exist_ok=True)
    
    session_names = []
    error_means = []
    error_stds = []
    correct_means = []
    correct_stds = []
    n_errors = []
    n_corrects = []
    
    for session_name, session_data in sorted(sessions_data.items()):
        error_data = session_data['error']['data']
        correct_data = session_data['correct']['data']
        
        if error_data.size > 0 and correct_data.size > 0:
            # Pe window
            pe_start = int(0.45 * 512)
            pe_end = int(0.55 * 512)
            
            error_pe = error_data[:, pe_start:pe_end, channel].max(axis=1)
            correct_pe = correct_data[:, pe_start:pe_end, channel].max(axis=1)
            
            session_names.append(session_name.replace('Session_', 'S'))
            error_means.append(error_pe.mean())
            error_stds.append(error_pe.std())
            correct_means.append(correct_pe.mean())
            correct_stds.append(correct_pe.std())
            n_errors.append(len(error_data))
            n_corrects.append(len(correct_data))
    
    # Create comparison plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    x = np.arange(len(session_names))
    width = 0.35
    
    # 1. Mean Pe amplitudes
    ax1.bar(x - width/2, error_means, width, yerr=error_stds, 
            label='Error', color='red', alpha=0.7, capsize=5)
    ax1.bar(x + width/2, correct_means, width, yerr=correct_stds,
            label='Correct', color='blue', alpha=0.7, capsize=5)
    ax1.set_ylabel('Pe Amplitude (μV)')
    ax1.set_title('Mean Pe Amplitude by Session')
    ax1.set_xticks(x)
    ax1.set_xticklabels(session_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Effect sizes
    effect_sizes = [(error_means[i] - correct_means[i]) / 
                   np.sqrt((error_stds[i]**2 + correct_stds[i]**2) / 2)
                   for i in range(len(session_names))]
    
    ax2.bar(x, effect_sizes, color='green', alpha=0.7)
    ax2.axhline(0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(0.5, color='k', linestyle='--', alpha=0.3, label='Medium effect')
    ax2.axhline(0.8, color='k', linestyle='--', alpha=0.3, label='Large effect')
    ax2.set_ylabel("Cohen's d")
    ax2.set_title('Effect Size by Session')
    ax2.set_xticks(x)
    ax2.set_xticklabels(session_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Trial counts
    ax3.bar(x - width/2, n_errors, width, label='Error trials', color='red', alpha=0.7)
    ax3.bar(x + width/2, n_corrects, width, label='Correct trials', color='blue', alpha=0.7)
    ax3.set_ylabel('Number of trials')
    ax3.set_title('Trial Counts by Session')
    ax3.set_xticks(x)
    ax3.set_xticklabels(session_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/session_comparison_ch{channel+1}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Run the complete session variability analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ErrP variability across sessions')
    parser.add_argument('--participant', type=str, default='T-002',
                       help='Participant ID to analyze')
    parser.add_argument('--channel', type=int, default=0,
                       help='Channel index to analyze (0-based)')
    parser.add_argument('--data-dir', type=str, default='processed_data',
                       help='Directory containing processed data')
    parser.add_argument('--save-dir', type=str, default='session_analysis',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    print(f"Analyzing session variability for {args.participant}")
    print("="*60)
    
    # Load all sessions
    sessions_data = load_all_sessions(args.participant, args.data_dir)
    
    if not sessions_data:
        print("No sessions found!")
        return
    
    print(f"\nFound {len(sessions_data)} sessions")
    print("="*60)
    
    # Run analyses
    print("\n1. Plotting ERPs by session...")
    session_stats = plot_session_erps(sessions_data, args.channel, args.save_dir)
    
    print("\n2. Plotting trial variability...")
    plot_session_variability(sessions_data, args.channel, args.save_dir)
    
    print("\n3. Plotting Pe distributions...")
    pe_stats = plot_pe_distributions(sessions_data, args.channel, args.save_dir)
    
    print("\n4. Creating session comparison...")
    plot_session_comparison(sessions_data, args.channel, args.save_dir)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for stats in pe_stats:
        print(f"\n{stats['session']}:")
        print(f"  Error Pe: {stats['error_mean']:.1f} ± {stats['error_std']:.1f} μV")
        print(f"  Correct Pe: {stats['correct_mean']:.1f} ± {stats['correct_std']:.1f} μV")
        print(f"  Effect size (d): {stats['effect_size']:.2f}")
        print(f"  P-value: {stats['p_value']:.4f}")
    
    print(f"\nAll plots saved to: {args.save_dir}/")

if __name__ == "__main__":
    main()