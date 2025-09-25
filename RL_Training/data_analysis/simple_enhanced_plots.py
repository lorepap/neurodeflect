#!/usr/bin/env python3
"""
Simple Enhanced Threshold Analysis with Reward Distributions and FCT Percentiles

Uses pure matplotlib for maximum compatibility. Creates the requested comparison
of reward distributions and FCT percentile analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_simple_enhanced_plots():
    """
    Create enhanced visualizations using only matplotlib for compatibility.
    """
    print("=== SIMPLE ENHANCED THRESHOLD ANALYSIS ===")
    
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
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']  # Blue, Green, Orange, Red
    
    thresholds = sorted(rewards_df['threshold'].unique())
    print(f"🎯 Analyzing thresholds: {thresholds}")
    
    # Create the main comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Deflection Threshold Analysis\\nReward Distributions and FCT Analysis', 
                fontsize=16, fontweight='bold')
    
    # 1. Reward Distribution Histograms
    ax1 = axes[0, 0]
    for i, t in enumerate(thresholds):
        threshold_rewards = rewards_df[rewards_df['threshold'] == t]['reward']
        ax1.hist(threshold_rewards, bins=30, alpha=0.7, label=f'Threshold {t:.2f}', 
                color=colors[i], density=True)
    
    ax1.set_title('Reward Distribution Comparison', fontweight='bold')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Statistics Bar Chart
    ax2 = axes[0, 1]
    reward_stats = []
    for t in thresholds:
        threshold_rewards = rewards_df[rewards_df['threshold'] == t]['reward']
        reward_stats.append({
            'threshold': t,
            'mean': threshold_rewards.mean(),
            'median': threshold_rewards.median(),
            'std': threshold_rewards.std()
        })
    
    reward_df = pd.DataFrame(reward_stats)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax2.bar(x - width/2, reward_df['mean'], width, label='Mean Reward', 
           alpha=0.8, color=colors[0])
    ax2.bar(x + width/2, reward_df['median'], width, label='Median Reward', 
           alpha=0.8, color=colors[1])
    
    ax2.set_title('Reward Statistics by Threshold', fontweight='bold')
    ax2.set_xlabel('Deflection Threshold')
    ax2.set_ylabel('Reward')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax2.legend()
    
    # Add value labels
    for i, (mean_val, med_val) in enumerate(zip(reward_df['mean'], reward_df['median'])):
        ax2.text(i - width/2, mean_val + 0.005, f'{mean_val:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(i + width/2, med_val + 0.005, f'{med_val:.3f}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. FCT Mean vs 95th Percentile (as requested)
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
    
    bars1 = ax3.bar(x - width, fct_df['mean'], width, label='Mean FCT', 
                   alpha=0.8, color=colors[0])
    bars2 = ax3.bar(x, fct_df['median'], width, label='Median FCT', 
                   alpha=0.8, color=colors[1])
    bars3 = ax3.bar(x + width, fct_df['p95'], width, label='95th Percentile FCT', 
                   alpha=0.8, color=colors[2])
    
    ax3.set_title('FCT: Mean and 95th Percentile Comparison', fontweight='bold')
    ax3.set_xlabel('Deflection Threshold')
    ax3.set_ylabel('FCT (ms)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax3.legend()
    
    # Add value labels
    for i in range(len(thresholds)):
        ax3.text(i - width, fct_df.iloc[i]['mean'] + 0.1, f"{fct_df.iloc[i]['mean']:.1f}", 
                ha='center', va='bottom', fontsize=8)
        ax3.text(i, fct_df.iloc[i]['median'] + 0.1, f"{fct_df.iloc[i]['median']:.1f}", 
                ha='center', va='bottom', fontsize=8)
        ax3.text(i + width, fct_df.iloc[i]['p95'] + 0.1, f"{fct_df.iloc[i]['p95']:.1f}", 
                ha='center', va='bottom', fontsize=8)
    
    # 4. FCT Distribution Comparison
    ax4 = axes[1, 0]
    for i, t in enumerate(thresholds):
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        # Sample data for better visualization
        sample_data = threshold_data['FCT'].sample(min(3000, len(threshold_data))) * 1000
        ax4.hist(sample_data, bins=40, alpha=0.6, label=f'Threshold {t:.2f}', 
                color=colors[i], density=True)
    
    ax4.set_title('FCT Distribution Comparison', fontweight='bold')
    ax4.set_xlabel('FCT (ms)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. FCT Percentile Analysis (detailed)
    ax5 = axes[1, 1]
    percentiles = [50, 60, 70, 80, 90, 95, 99]
    
    for i, t in enumerate(thresholds):
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        fct_percentiles = [threshold_data['FCT'].quantile(p/100) * 1000 for p in percentiles]
        
        ax5.plot(percentiles, fct_percentiles, marker='o', linewidth=2, markersize=6,
               label=f'Threshold {t:.2f}', color=colors[i])
    
    ax5.set_title('FCT Percentile Analysis', fontweight='bold')
    ax5.set_xlabel('Percentile')
    ax5.set_ylabel('FCT (ms)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(percentiles)
    
    # Highlight 95th percentile
    ax5.axvline(x=95, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax5.text(95.5, ax5.get_ylim()[1] * 0.9, '95th', rotation=90, 
             color='red', fontweight='bold')
    
    # 6. Performance Summary Table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary data
    summary_data = []
    for t in thresholds:
        reward_data = rewards_df[rewards_df['threshold'] == t]
        orig_data = dataset[dataset['deflection_threshold'] == t]
        
        summary_data.append([
            f'{t:.2f}',
            f'{reward_data["reward"].mean():.4f}',
            f'{orig_data["FCT"].mean() * 1000:.1f}',
            f'{orig_data["FCT"].quantile(0.95) * 1000:.1f}',
            f'{reward_data["action"].mean() * 100:.1f}%'
        ])
    
    # Create table
    table_data = [['Threshold', 'Reward', 'Mean FCT\\n(ms)', '95th FCT\\n(ms)', 'Deflection\\nRate']]
    table_data.extend(summary_data)
    
    table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     bbox=[0, 0.2, 1, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best threshold (lowest reward value means best performance)
    best_idx = np.argmax([float(row[1]) for row in summary_data])
    for i in range(len(table_data[0])):
        table[(best_idx + 1, i)].set_facecolor('#FFE082')
    
    ax6.set_title('Performance Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = output_path / "enhanced_threshold_analysis_simple.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved enhanced analysis plot: {plot_file}")
    plt.close()
    
    # Print detailed analysis
    print_analysis_summary(thresholds, reward_df, fct_df)

def print_analysis_summary(thresholds, reward_df, fct_df):
    """Print detailed analysis summary."""
    
    print(f"\\n📊 ENHANCED ANALYSIS SUMMARY:")
    print(f"{'='*60}")
    
    print(f"\\n🎯 REWARD DISTRIBUTION ANALYSIS:")
    print(f"{'Threshold':<10} {'Mean':<8} {'Median':<8} {'Std Dev':<8}")
    print(f"{'-'*40}")
    
    for _, row in reward_df.iterrows():
        print(f"{row['threshold']:<10.2f} "
              f"{row['mean']:<8.4f} "
              f"{row['median']:<8.4f} "
              f"{row['std']:<8.4f}")
    
    print(f"\\n⏱️ FCT ANALYSIS (Mean vs 95th Percentile):")
    print(f"{'Threshold':<10} {'Mean FCT':<10} {'95th FCT':<10} {'Difference':<12}")
    print(f"{'-'*50}")
    
    for _, row in fct_df.iterrows():
        diff = row['p95'] - row['mean']
        print(f"{row['threshold']:<10.2f} "
              f"{row['mean']:<10.1f} "
              f"{row['p95']:<10.1f} "
              f"{diff:<12.1f}")
    
    # Best threshold analysis
    best_reward_idx = reward_df['mean'].idxmax()
    best_fct_idx = fct_df['mean'].idxmin()  # Lower FCT is better
    
    print(f"\\n🏆 OPTIMAL THRESHOLD ANALYSIS:")
    print(f"   Best Reward: Threshold {reward_df.iloc[best_reward_idx]['threshold']:.2f} "
          f"(Reward: {reward_df.iloc[best_reward_idx]['mean']:.4f})")
    print(f"   Best FCT: Threshold {fct_df.iloc[best_fct_idx]['threshold']:.2f} "
          f"(FCT: {fct_df.iloc[best_fct_idx]['mean']:.1f}ms)")
    
    if best_reward_idx == best_fct_idx:
        print(f"   ✅ CONSISTENT WINNER: Threshold {reward_df.iloc[best_reward_idx]['threshold']:.2f}")
    else:
        print(f"   ⚠️ Different optimal thresholds for reward vs FCT")
    
    # Performance improvements
    worst_reward = reward_df['mean'].min()
    best_reward = reward_df['mean'].max()
    worst_fct = fct_df['mean'].max()
    best_fct = fct_df['mean'].min()
    
    reward_improvement = ((best_reward - worst_reward) / abs(worst_reward)) * 100
    fct_improvement = ((worst_fct - best_fct) / worst_fct) * 100
    
    print(f"\\n📈 PERFORMANCE IMPROVEMENTS:")
    print(f"   Reward improvement: {reward_improvement:+.1f}%")
    print(f"   FCT improvement: {fct_improvement:+.1f}%")
    
    print(f"\\n🎯 RECOMMENDATION:")
    best_overall = reward_df.iloc[best_reward_idx]['threshold']
    print(f"   Deploy threshold {best_overall:.2f} for optimal network performance")
    print(f"   Expected reward: {reward_df.iloc[best_reward_idx]['mean']:.4f}")
    print(f"   Expected mean FCT: {fct_df.iloc[best_reward_idx]['mean']:.1f}ms")
    print(f"   Expected 95th FCT: {fct_df.iloc[best_reward_idx]['p95']:.1f}ms")

if __name__ == "__main__":
    create_simple_enhanced_plots()
