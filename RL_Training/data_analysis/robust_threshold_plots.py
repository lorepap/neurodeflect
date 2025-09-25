#!/usr/bin/env python3
"""
Robust Enhanced Threshold Analysis with Reward Distributions and FCT Percentiles

Uses seaborn for better plotting compatibility and focuses on reward distributions 
and detailed FCT analysis as requested.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_robust_threshold_plots():
    """
    Create robust enhanced visualizations with reward distributions and FCT percentiles.
    """
    print("=== ROBUST ENHANCED THRESHOLD ANALYSIS ===")
    
    # Load the data
    output_path = Path("threshold_analysis")
    
    # Load rewards data
    rewards_file = output_path / "all_rewards_by_threshold.csv"
    if not rewards_file.exists():
        print("❌ Rewards data not found. Please run enhanced_threshold_plots.py first.")
        return
    
    rewards_df = pd.read_csv(rewards_file)
    print(f"📊 Loaded {len(rewards_df):,} reward records")
    
    # Load original dataset for FCT analysis
    dataset_path = "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    dataset = pd.read_csv(dataset_path)
    print(f"📊 Loaded {len(dataset):,} original records")
    
    # Set up plotting
    plt.style.use('default')
    sns.set_style("whitegrid")
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']  # Blue, Green, Orange, Red
    
    # Create comprehensive figure
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Enhanced Deflection Threshold Analysis\nReward Distributions and FCT Analysis', 
                fontsize=16, fontweight='bold')
    
    thresholds = sorted(rewards_df['threshold'].unique())
    print(f"🎯 Analyzing thresholds: {thresholds}")
    
    # 1. Reward Distribution Boxplot (using seaborn)
    ax1 = axes[0, 0]
    sns.boxplot(data=rewards_df, x='threshold', y='reward', ax=ax1, palette=colors)
    ax1.set_title('Reward Distribution by Threshold', fontweight='bold')
    ax1.set_xlabel('Deflection Threshold')
    ax1.set_ylabel('Reward')
    
    # Add mean markers
    for i, t in enumerate(thresholds):
        mean_reward = rewards_df[rewards_df['threshold'] == t]['reward'].mean()
        ax1.scatter(i, mean_reward, color='red', s=80, marker='D', zorder=5)
        ax1.text(i, mean_reward + 0.02, f'{mean_reward:.3f}', 
                ha='center', va='bottom', fontweight='bold', color='red')
    
    # 2. Reward Distribution Violin Plot
    ax2 = axes[0, 1]
    sns.violinplot(data=rewards_df, x='threshold', y='reward', ax=ax2, palette=colors)
    ax2.set_title('Reward Distribution Density', fontweight='bold')
    ax2.set_xlabel('Deflection Threshold')
    ax2.set_ylabel('Reward')
    
    # 3. FCT Mean vs 95th Percentile
    ax3 = axes[0, 2]
    fct_stats = []
    for t in thresholds:
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        mean_fct = threshold_data['FCT'].mean() * 1000
        p95_fct = threshold_data['FCT'].quantile(0.95) * 1000
        p50_fct = threshold_data['FCT'].median() * 1000
        fct_stats.append({'threshold': t, 'mean': mean_fct, 'p95': p95_fct, 'median': p50_fct})
    
    fct_df = pd.DataFrame(fct_stats)
    
    x = np.arange(len(thresholds))
    width = 0.25
    
    ax3.bar(x - width, fct_df['mean'], width, label='Mean FCT', alpha=0.8, color=colors[0])
    ax3.bar(x, fct_df['median'], width, label='Median FCT', alpha=0.8, color=colors[1])
    ax3.bar(x + width, fct_df['p95'], width, label='95th Percentile FCT', alpha=0.8, color=colors[2])
    
    ax3.set_title('FCT Statistics by Threshold', fontweight='bold')
    ax3.set_xlabel('Deflection Threshold')
    ax3.set_ylabel('FCT (ms)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax3.legend()
    
    # Add value labels
    for i, (mean_val, med_val, p95_val) in enumerate(zip(fct_df['mean'], fct_df['median'], fct_df['p95'])):
        ax3.text(i - width, mean_val + 0.1, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i, med_val + 0.1, f'{med_val:.1f}', ha='center', va='bottom', fontsize=8)
        ax3.text(i + width, p95_val + 0.1, f'{p95_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Reward vs FCT Scatter
    ax4 = axes[1, 0]
    for i, t in enumerate(thresholds):
        threshold_data = rewards_df[rewards_df['threshold'] == t]
        ax4.scatter(threshold_data['fct'] * 1000, threshold_data['reward'], 
                   alpha=0.6, s=15, color=colors[i], label=f'Threshold {t:.2f}')
    
    ax4.set_title('Reward vs FCT Relationship', fontweight='bold')
    ax4.set_xlabel('FCT (ms)')
    ax4.set_ylabel('Reward')
    ax4.legend()
    
    # 5. Deflection Rate Analysis
    ax5 = axes[1, 1]
    deflection_rates = []
    for t in thresholds:
        rate = rewards_df[rewards_df['threshold'] == t]['action'].mean() * 100
        deflection_rates.append(rate)
    
    bars = ax5.bar(range(len(thresholds)), deflection_rates, color=colors, alpha=0.8)
    ax5.set_title('Deflection Rate by Threshold', fontweight='bold')
    ax5.set_xlabel('Deflection Threshold')
    ax5.set_ylabel('Deflection Rate (%)')
    ax5.set_xticks(range(len(thresholds)))
    ax5.set_xticklabels([f'{t:.2f}' for t in thresholds])
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, deflection_rates)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 6. Queue Utilization by Threshold
    ax6 = axes[1, 2]
    sns.boxplot(data=rewards_df, x='threshold', y='queue_utilization', ax=ax6, palette=colors)
    ax6.set_title('Queue Utilization Distribution', fontweight='bold')
    ax6.set_xlabel('Deflection Threshold')
    ax6.set_ylabel('Queue Utilization')
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    # 7. Out-of-Order Rate
    ax7 = axes[2, 0]
    ooo_rates = []
    for t in thresholds:
        rate = rewards_df[rewards_df['threshold'] == t]['ooo'].mean() * 100
        ooo_rates.append(rate)
    
    bars = ax7.bar(range(len(thresholds)), ooo_rates, color=colors, alpha=0.8)
    ax7.set_title('Out-of-Order Rate by Threshold', fontweight='bold')
    ax7.set_xlabel('Deflection Threshold')
    ax7.set_ylabel('OOO Rate (%)')
    ax7.set_xticks(range(len(thresholds)))
    ax7.set_xticklabels([f'{t:.2f}' for t in thresholds])
    
    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, ooo_rates)):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 8. FCT Percentile Analysis
    ax8 = axes[2, 1]
    percentiles = [50, 75, 90, 95, 99]
    
    for i, t in enumerate(thresholds):
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        fct_percentiles = [threshold_data['FCT'].quantile(p/100) * 1000 for p in percentiles]
        ax8.plot(percentiles, fct_percentiles, marker='o', linewidth=2, markersize=6,
               label=f'Threshold {t:.2f}', color=colors[i])
    
    ax8.set_title('FCT Percentile Analysis', fontweight='bold')
    ax8.set_xlabel('Percentile')
    ax8.set_ylabel('FCT (ms)')
    ax8.legend()
    ax8.set_xticks(percentiles)
    
    # 9. Performance Summary
    ax9 = axes[2, 2]
    
    # Create summary metrics
    summary_data = []
    for t in thresholds:
        reward_data = rewards_df[rewards_df['threshold'] == t]
        orig_data = dataset[dataset['deflection_threshold'] == t]
        
        summary_data.append({
            'Threshold': f'{t:.2f}',
            'Reward': reward_data['reward'].mean(),
            'FCT (ms)': orig_data['FCT'].mean() * 1000,
            'P95 FCT (ms)': orig_data['FCT'].quantile(0.95) * 1000,
            'Deflection %': reward_data['action'].mean() * 100,
            'OOO %': reward_data['ooo'].mean() * 100
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create text summary
    ax9.axis('off')
    summary_text = "PERFORMANCE SUMMARY\\n" + "="*30 + "\\n"
    
    for _, row in summary_df.iterrows():
        summary_text += f"Threshold {row['Threshold']}:\\n"
        summary_text += f"  Reward: {row['Reward']:7.4f}\\n"
        summary_text += f"  FCT: {row['FCT (ms)']:6.1f}ms\\n"
        summary_text += f"  P95 FCT: {row['P95 FCT (ms)']:6.1f}ms\\n"
        summary_text += f"  Deflection: {row['Deflection %']:4.1f}%\\n"
        summary_text += f"  OOO: {row['OOO %']:6.1f}%\\n\\n"
    
    # Find best threshold
    best_idx = summary_df['Reward'].idxmax()
    best_threshold = summary_df.iloc[best_idx]
    
    summary_text += f"🏆 BEST: Threshold {best_threshold['Threshold']}\\n"
    summary_text += f"   Reward: {best_threshold['Reward']:.4f}\\n"
    summary_text += f"   FCT: {best_threshold['FCT (ms)']:.1f}ms\\n"
    summary_text += f"   P95 FCT: {best_threshold['P95 FCT (ms)']:.1f}ms"
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_path / "enhanced_threshold_analysis_robust.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved enhanced analysis plot: {plot_file}")
    plt.close()
    
    # Create focused FCT analysis
    create_focused_fct_analysis(dataset, thresholds, colors, output_path)
    
    # Print detailed statistics
    print_detailed_statistics(summary_df, thresholds)

def create_focused_fct_analysis(dataset, thresholds, colors, output_path):
    """Create focused FCT analysis with multiple percentiles."""
    
    print("📊 Creating focused FCT analysis...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. FCT Distribution Comparison
    for i, t in enumerate(thresholds):
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        # Use a sample for better visualization
        sample_data = threshold_data['FCT'].sample(min(5000, len(threshold_data))) * 1000
        
        ax1.hist(sample_data, bins=50, alpha=0.6, label=f'Threshold {t:.2f}', 
                color=colors[i], density=True)
    
    ax1.set_title('FCT Distribution Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('FCT (ms)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Detailed Percentile Analysis
    percentiles = range(50, 100, 5)  # 50th to 95th percentile in 5% steps
    
    for i, t in enumerate(thresholds):
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        fct_percentiles = [threshold_data['FCT'].quantile(p/100) * 1000 for p in percentiles]
        
        ax2.plot(percentiles, fct_percentiles, marker='o', linewidth=2, markersize=4,
               label=f'Threshold {t:.2f}', color=colors[i])
    
    ax2.set_title('Detailed FCT Percentile Analysis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('FCT (ms)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(50, 100, 10))
    
    # Highlight 95th percentile
    ax2.axvline(x=95, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.text(95.5, ax2.get_ylim()[1] * 0.9, '95th Percentile', rotation=90, 
             color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path / "focused_fct_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Focused FCT analysis created")

def print_detailed_statistics(summary_df, thresholds):
    """Print detailed statistics."""
    
    print(f"\\n📊 DETAILED STATISTICS:")
    print(f"{'Threshold':<10} {'Reward':<8} {'FCT(ms)':<8} {'P95 FCT':<8} {'Deflect%':<9} {'OOO%':<7}")
    print(f"{'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*7}")
    
    for _, row in summary_df.iterrows():
        print(f"{row['Threshold']:<10} "
              f"{row['Reward']:<8.4f} "
              f"{row['FCT (ms)']:<8.1f} "
              f"{row['P95 FCT (ms)']:<8.1f} "
              f"{row['Deflection %']:<9.1f} "
              f"{row['OOO %']:<7.1f}")
    
    # Best/worst analysis
    best_idx = summary_df['Reward'].idxmax()
    worst_idx = summary_df['Reward'].idxmin()
    
    best = summary_df.iloc[best_idx]
    worst = summary_df.iloc[worst_idx]
    
    print(f"\\n🏆 BEST THRESHOLD: {best['Threshold']}")
    print(f"   Reward: {best['Reward']:.4f}")
    print(f"   Mean FCT: {best['FCT (ms)']:.1f}ms")
    print(f"   95th FCT: {best['P95 FCT (ms)']:.1f}ms")
    print(f"   Deflection: {best['Deflection %']:.1f}%")
    
    print(f"\\n❌ WORST THRESHOLD: {worst['Threshold']}")
    print(f"   Reward: {worst['Reward']:.4f}")
    print(f"   Mean FCT: {worst['FCT (ms)']:.1f}ms")
    print(f"   95th FCT: {worst['P95 FCT (ms)']:.1f}ms")
    print(f"   Deflection: {worst['Deflection %']:.1f}%")
    
    # Improvements
    reward_improvement = ((best['Reward'] - worst['Reward']) / abs(worst['Reward']) * 100)
    fct_improvement = ((worst['FCT (ms)'] - best['FCT (ms)']) / worst['FCT (ms)'] * 100)
    p95_improvement = ((worst['P95 FCT (ms)'] - best['P95 FCT (ms)']) / worst['P95 FCT (ms)'] * 100)
    
    print(f"\\n📈 IMPROVEMENTS WITH OPTIMAL THRESHOLD:")
    print(f"   Reward: {reward_improvement:+.1f}%")
    print(f"   Mean FCT: {fct_improvement:+.1f}%")
    print(f"   95th FCT: {p95_improvement:+.1f}%")

if __name__ == "__main__":
    create_robust_threshold_plots()
