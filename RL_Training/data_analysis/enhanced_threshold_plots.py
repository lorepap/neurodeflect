#!/usr/bin/env python3
"""
Enhanced Threshold Analysis with Reward Distributions and FCT Percentiles

This script creates comprehensive visualizations including:
1. Reward distribution comparisons across thresholds
2. Mean and 95th percentile FCT analysis
3. Statistical distribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add the parent directory to import the environment
sys.path.append('..')
from environments.deflection_env import DatacenterDeflectionEnv

def compute_rewards_for_plotting(dataset_path: str, output_path: Path):
    """
    Compute rewards for all records in the dataset for plotting purposes.
    """
    print("   Computing rewards for all records...")
    
    # Load dataset
    dataset = pd.read_csv(dataset_path)
    
    # Create environment to compute rewards (without normalization for analysis)
    env = DatacenterDeflectionEnv(dataset_path, normalize_states=False)
    
    # Compute rewards for each record
    rewards_data = []
    
    for idx, row in dataset.iterrows():
        if idx % 20000 == 0:
            print(f"      Processing record {idx:,}/{len(dataset):,} ({idx/len(dataset)*100:.1f}%)")
        
        try:
            # Extract state features
            queue_util = row['occupancy'] / (row['capacity'] + 1e-8)
            total_occupancy = row['total_occupancy']
            ttl_priority = (250 - row['ttl']) / 250.0
            state = np.array([queue_util, total_occupancy, ttl_priority])
            
            # Simulate reward computation
            env.current_episode_data = {
                'dataset_indices': [idx]
            }
            env.current_step = 1  # Set to 1 so step_idx becomes 0
            
            # Compute reward
            reward = env._compute_reward(state, row['action'], state)
            
            # Store reward data
            rewards_data.append({
                'threshold': row['deflection_threshold'],
                'action': row['action'],
                'reward': reward,
                'queue_utilization': queue_util,
                'total_occupancy': total_occupancy,
                'ttl_priority': ttl_priority,
                'fct': row['FCT'],
                'ooo': row['ooo'],
                'packet_delay': row['end_time'] - row['start_time']
            })
            
        except Exception as e:
            if idx < 10:  # Only print first few errors
                print(f"      Error processing record {idx}: {e}")
            continue
    
    # Convert to DataFrame and save
    rewards_df = pd.DataFrame(rewards_data)
    rewards_file = output_path / "all_rewards_by_threshold.csv"
    rewards_df.to_csv(rewards_file, index=False)
    print(f"   ✅ Saved rewards data: {rewards_file}")
    
    return rewards_df

def create_enhanced_threshold_plots(dataset_path: str, output_dir: str = "threshold_analysis"):
    """
    Create enhanced visualizations with reward distributions and FCT percentiles.
    """
    print("=== ENHANCED THRESHOLD ANALYSIS WITH DISTRIBUTIONS ===")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Check if we have the full rewards data, if not compute it
    full_rewards_file = output_path / "all_rewards_by_threshold.csv"
    
    if not full_rewards_file.exists():
        print("📊 Computing reward data on-the-fly...")
        rewards_df = compute_rewards_for_plotting(dataset_path, output_path)
    else:
        print("📊 Loading existing rewards data...")
        rewards_df = pd.read_csv(full_rewards_file)
    
    print(f"   Loaded {len(rewards_df):,} reward records")
    
    # Also load the original dataset for FCT percentiles
    print("📊 Loading original dataset for FCT analysis...")
    dataset = pd.read_csv(dataset_path)
    print(f"   Loaded {len(dataset):,} original records")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create comprehensive figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define subplot grid: 3 rows, 3 columns
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Reward Distributions (Box plots) - Top Left
    ax1 = fig.add_subplot(gs[0, 0])
    thresholds = sorted(rewards_df['threshold'].unique())
    
    # Prepare data for boxplot with proper conversion to list of arrays
    reward_data = []
    for t in thresholds:
        threshold_rewards = rewards_df[rewards_df['threshold'] == t]['reward'].values
        reward_data.append(threshold_rewards)
    
    # Create box plot using matplotlib's axes method
    bp = ax1.boxplot(reward_data, labels=[f'{t:.2f}' for t in thresholds], 
                     patch_artist=True, notch=True, whis=1.5)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_title('Reward Distribution by Threshold', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Deflection Threshold')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Add mean markers
    for i, t in enumerate(thresholds):
        mean_reward = rewards_df[rewards_df['threshold'] == t]['reward'].mean()
        ax1.scatter(i+1, mean_reward, color='red', s=50, marker='D', zorder=5)
        ax1.text(i+1, mean_reward + 0.02, f'{mean_reward:.3f}', 
                ha='center', va='bottom', fontsize=9, color='red', fontweight='bold')
    
    # 2. Reward Distributions (Violin plots) - Top Center
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create violin plot with proper data handling
    try:
        violin_parts = ax2.violinplot(reward_data, positions=range(1, len(thresholds)+1), 
                                     showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(violin_parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
    except Exception as e:
        print(f"   Warning: Violin plot failed ({e}), using alternative visualization")
        # Alternative: overlapping histograms
        for i, t in enumerate(thresholds):
            threshold_rewards = rewards_df[rewards_df['threshold'] == t]['reward']
            ax2.hist(threshold_rewards, bins=30, alpha=0.5, label=f'{t:.2f}', 
                    color=colors[i], density=True)
        ax2.legend()
    
    ax2.set_title('Reward Distribution Density by Threshold', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Deflection Threshold')
    ax2.set_ylabel('Reward')
    ax2.set_xticks(range(1, len(thresholds)+1))
    ax2.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax2.grid(True, alpha=0.3)
    
    # 3. FCT Mean and 95th Percentile - Top Right
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Calculate FCT statistics by threshold
    fct_stats = []
    for t in thresholds:
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        mean_fct = threshold_data['FCT'].mean() * 1000  # Convert to ms
        p95_fct = threshold_data['FCT'].quantile(0.95) * 1000  # 95th percentile
        fct_stats.append({'threshold': t, 'mean_fct': mean_fct, 'p95_fct': p95_fct})
    
    fct_df = pd.DataFrame(fct_stats)
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, fct_df['mean_fct'], width, label='Mean FCT', 
                   alpha=0.8, color='skyblue', edgecolor='navy')
    bars2 = ax3.bar(x + width/2, fct_df['p95_fct'], width, label='95th Percentile FCT', 
                   alpha=0.8, color='lightcoral', edgecolor='darkred')
    
    ax3.set_title('FCT: Mean vs 95th Percentile by Threshold', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Deflection Threshold')
    ax3.set_ylabel('FCT (ms)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Reward vs FCT Scatter - Middle Left
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Create scatter plot colored by threshold
    for i, t in enumerate(thresholds):
        threshold_rewards = rewards_df[rewards_df['threshold'] == t]
        ax4.scatter(threshold_rewards['fct'] * 1000, threshold_rewards['reward'], 
                   alpha=0.6, s=20, color=colors[i], label=f'Threshold {t:.2f}')
    
    ax4.set_title('Reward vs FCT by Threshold', fontsize=12, fontweight='bold')
    ax4.set_xlabel('FCT (ms)')
    ax4.set_ylabel('Reward')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Deflection Rate Analysis - Middle Center
    ax5 = fig.add_subplot(gs[1, 1])
    
    deflection_stats = []
    for t in thresholds:
        threshold_data = rewards_df[rewards_df['threshold'] == t]
        deflection_rate = threshold_data['action'].mean() * 100
        deflection_stats.append(deflection_rate)
    
    bars = ax5.bar(range(len(thresholds)), deflection_stats, 
                  color=colors, alpha=0.8, edgecolor='black')
    
    ax5.set_title('Deflection Rate by Threshold', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Deflection Threshold')
    ax5.set_ylabel('Deflection Rate (%)')
    ax5.set_xticks(range(len(thresholds)))
    ax5.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Queue Utilization Distribution - Middle Right
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Prepare queue utilization data properly
    queue_util_data = []
    for t in thresholds:
        util_values = rewards_df[rewards_df['threshold'] == t]['queue_utilization'].values * 100
        queue_util_data.append(util_values)
    
    try:
        box_plot2 = ax6.boxplot(queue_util_data, labels=[f'{t:.2f}' for t in thresholds], 
                               patch_artist=True)
        
        # Color the boxes
        for patch, color in zip(box_plot2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    except Exception as e:
        print(f"   Warning: Queue util boxplot failed ({e}), using bar chart")
        # Alternative: bar chart with means
        queue_means = [np.mean(data) for data in queue_util_data]
        ax6.bar(range(len(thresholds)), queue_means, color=colors, alpha=0.7)
        ax6.set_xticks(range(len(thresholds)))
        ax6.set_xticklabels([f'{t:.2f}' for t in thresholds])
    
    ax6.set_title('Queue Utilization Distribution by Threshold', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Deflection Threshold')
    ax6.set_ylabel('Queue Utilization (%)')
    ax6.grid(True, alpha=0.3)
    
    # 7. OOO Rate Comparison - Bottom Left
    ax7 = fig.add_subplot(gs[2, 0])
    
    ooo_stats = []
    for t in thresholds:
        threshold_data = rewards_df[rewards_df['threshold'] == t]
        ooo_rate = threshold_data['ooo'].mean() * 100
        ooo_stats.append(ooo_rate)
    
    bars3 = ax7.bar(range(len(thresholds)), ooo_stats, 
                   color=colors, alpha=0.8, edgecolor='black')
    
    ax7.set_title('Out-of-Order Rate by Threshold', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Deflection Threshold')
    ax7.set_ylabel('OOO Rate (%)')
    ax7.set_xticks(range(len(thresholds)))
    ax7.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax7.grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 8. Performance Summary Heatmap - Bottom Center & Right
    ax8 = fig.add_subplot(gs[2, 1:])
    
    # Create performance metrics matrix
    metrics_data = []
    metric_names = ['Reward Score', 'FCT (ms)', '95th FCT (ms)', 'Deflection %', 'OOO %', 'Queue Util %']
    
    for t in thresholds:
        threshold_rewards = rewards_df[rewards_df['threshold'] == t]
        threshold_orig = dataset[dataset['deflection_threshold'] == t]
        
        metrics = [
            threshold_rewards['reward'].mean(),
            threshold_rewards['fct'].mean() * 1000,
            threshold_orig['FCT'].quantile(0.95) * 1000,
            threshold_rewards['action'].mean() * 100,
            threshold_rewards['ooo'].mean() * 100,
            threshold_rewards['queue_utilization'].mean() * 100
        ]
        metrics_data.append(metrics)
    
    metrics_matrix = np.array(metrics_data).T  # Transpose for heatmap
    
    # Normalize each metric for better visualization
    normalized_matrix = np.zeros_like(metrics_matrix)
    for i in range(len(metric_names)):
        row = metrics_matrix[i]
        if i == 0:  # Reward (higher is better)
            normalized_matrix[i] = (row - row.min()) / (row.max() - row.min())
        else:  # Other metrics (lower is generally better)
            normalized_matrix[i] = 1 - (row - row.min()) / (row.max() - row.min())
    
    im = ax8.imshow(normalized_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax8.set_xticks(range(len(thresholds)))
    ax8.set_xticklabels([f'Threshold {t:.2f}' for t in thresholds])
    ax8.set_yticks(range(len(metric_names)))
    ax8.set_yticklabels(metric_names)
    
    # Add text annotations with actual values
    for i in range(len(metric_names)):
        for j in range(len(thresholds)):
            value = metrics_matrix[i, j]
            if i == 0:  # Reward
                text = f'{value:.3f}'
            elif i in [1, 2]:  # FCT metrics
                text = f'{value:.1f}'
            else:  # Percentage metrics
                text = f'{value:.1f}%'
            
            ax8.text(j, i, text, ha='center', va='center', 
                    color='white' if normalized_matrix[i, j] < 0.5 else 'black',
                    fontweight='bold', fontsize=9)
    
    ax8.set_title('Performance Metrics Heatmap\n(Green=Better, Red=Worse)', 
                 fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax8, shrink=0.8)
    cbar.set_label('Normalized Performance (0=Worst, 1=Best)', rotation=270, labelpad=20)
    
    # Add main title
    fig.suptitle('Comprehensive Deflection Threshold Analysis\nReward Distributions and Performance Metrics', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save the plot
    plot_file = output_path / "enhanced_threshold_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved enhanced analysis plot: {plot_file}")
    
    # Print detailed statistics
    print(f"\n📊 DETAILED STATISTICS:")
    print(f"{'Threshold':<10} {'Mean Reward':<12} {'Mean FCT':<10} {'95th FCT':<10} {'Deflect%':<9} {'OOO%':<7} {'QUtil%':<8}")
    print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*9} {'-'*7} {'-'*8}")
    
    for i, t in enumerate(thresholds):
        print(f"{t:<10.2f} "
              f"{metrics_data[i][0]:<12.4f} "
              f"{metrics_data[i][1]:<10.1f} "
              f"{metrics_data[i][2]:<10.1f} "
              f"{metrics_data[i][3]:<9.1f} "
              f"{metrics_data[i][4]:<7.1f} "
              f"{metrics_data[i][5]:<8.1f}")
    
    plt.close()
    
    # Create additional focused plots
    create_focused_distribution_plots(rewards_df, dataset, thresholds, output_path, colors)
    
    print(f"\n✅ Enhanced analysis complete! All plots saved in: {output_path}")

def create_focused_distribution_plots(rewards_df, dataset, thresholds, output_path, colors):
    """Create focused distribution comparison plots."""
    
    print("\n📊 Creating focused distribution plots...")
    
    # 1. Reward Distribution Comparison (Detailed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram comparison
    for i, t in enumerate(thresholds):
        threshold_rewards = rewards_df[rewards_df['threshold'] == t]['reward']
        ax1.hist(threshold_rewards, bins=50, alpha=0.6, label=f'Threshold {t:.2f}', 
                color=colors[i], density=True)
    
    ax1.set_title('Reward Distribution Comparison (Histograms)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # KDE comparison
    for i, t in enumerate(thresholds):
        threshold_rewards = rewards_df[rewards_df['threshold'] == t]['reward']
        threshold_rewards.plot.kde(ax=ax2, label=f'Threshold {t:.2f}', 
                                  color=colors[i], linewidth=2)
    
    ax2.set_title('Reward Distribution Comparison (KDE)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Reward')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "reward_distributions_detailed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. FCT Percentile Analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    percentiles = [50, 75, 90, 95, 99]
    
    for i, t in enumerate(thresholds):
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        fct_percentiles = [threshold_data['FCT'].quantile(p/100) * 1000 for p in percentiles]
        
        ax.plot(percentiles, fct_percentiles, marker='o', linewidth=2, markersize=6,
               label=f'Threshold {t:.2f}', color=colors[i])
    
    ax.set_title('FCT Percentile Analysis by Threshold', fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentile')
    ax.set_ylabel('FCT (ms)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(percentiles)
    
    # Add value annotations at 95th percentile
    for i, t in enumerate(thresholds):
        threshold_data = dataset[dataset['deflection_threshold'] == t]
        p95_fct = threshold_data['FCT'].quantile(0.95) * 1000
        ax.annotate(f'{p95_fct:.1f}ms', xy=(95, p95_fct), xytext=(10, 5), 
                   textcoords='offset points', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_path / "fct_percentile_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Focused distribution plots created")

def main():
    """Main execution function."""
    dataset_path = "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    
    if not Path(dataset_path).exists():
        print("❌ Dataset not found!")
        return
    
    create_enhanced_threshold_plots(dataset_path)

if __name__ == "__main__":
    main()
