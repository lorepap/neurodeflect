#!/usr/bin/env python3
"""
Reward Trend Analysis by Deflection Threshold

This script analyzes how rewards computed by the RL environment vary across 
different deflection thresholds in the original dataset. This helps understand:
1. Which thresholds lead to better network performance
2. How reward components change with threshold settings
3. Optimal threshold ranges for deployment

The analysis uses the same reward function as the RL training environment.
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

def analyze_reward_trends_by_threshold(dataset_path: str, output_dir: str = "threshold_analysis"):
    """
    Analyze reward trends across different deflection thresholds.
    
    Args:
        dataset_path: Path to the threshold dataset
        output_dir: Directory to save analysis results
    """
    print("=== REWARD TREND ANALYSIS BY THRESHOLD ===")
    print(f"Dataset: {dataset_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load dataset directly
    print("\n1. Loading dataset...")
    dataset = pd.read_csv(dataset_path)
    print(f"   Loaded {len(dataset):,} records")
    print(f"   Columns: {list(dataset.columns)}")
    
    # Check for required columns
    required_columns = ['deflection_threshold', 'action', 'FCT', 'ooo', 'end_time', 'start_time', 'RequesterID']
    missing_columns = [col for col in required_columns if col not in dataset.columns]
    if missing_columns:
        print(f"   WARNING: Missing columns: {missing_columns}")
        return
    
    # Analyze threshold distribution
    print("\n2. Threshold distribution analysis...")
    thresholds = sorted(dataset['deflection_threshold'].unique())
    print(f"   Found {len(thresholds)} unique thresholds: {thresholds}")
    
    threshold_counts = dataset['deflection_threshold'].value_counts().sort_index()
    print(f"   Records per threshold:")
    for threshold, count in threshold_counts.items():
        pct = count / len(dataset) * 100
        print(f"     {threshold:4.2f}: {count:8,} records ({pct:5.1f}%)")
    
    # Create environment to compute rewards (without normalization for analysis)
    print("\n3. Setting up reward computation...")
    try:
        env = DatacenterDeflectionEnv(dataset_path, normalize_states=False)
        print("   Environment created successfully")
    except Exception as e:
        print(f"   Error creating environment: {e}")
        return
    
    # Compute rewards for each record
    print("\n4. Computing rewards for all records...")
    rewards_data = []
    
    for idx, row in dataset.iterrows():
        if idx % 10000 == 0:
            print(f"   Processing record {idx:,}/{len(dataset):,} ({idx/len(dataset)*100:.1f}%)")
        
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
                print(f"   Error processing record {idx}: {e}")
            continue
    
    # Convert to DataFrame
    rewards_df = pd.DataFrame(rewards_data)
    print(f"   Successfully computed rewards for {len(rewards_df):,} records")
    
    # Analyze rewards by threshold
    print("\n5. Analyzing reward trends by threshold...")
    
    # Summary statistics by threshold
    reward_stats = rewards_df.groupby('threshold').agg({
        'reward': ['count', 'mean', 'std', 'min', 'max', 'median'],
        'action': 'mean',  # Deflection rate
        'fct': 'mean',
        'ooo': 'mean',
        'packet_delay': 'mean',
        'queue_utilization': 'mean'
    }).round(4)
    
    # Flatten column names
    reward_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in reward_stats.columns]
    
    print("   Reward statistics by threshold:")
    print(reward_stats.to_string())
    
    # Save detailed results
    results_file = output_path / "reward_trends_by_threshold.csv"
    reward_stats.to_csv(results_file)
    print(f"   Saved detailed results to: {results_file}")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Reward distribution by threshold (box plots)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Reward Analysis by Deflection Threshold', fontsize=16, fontweight='bold')
    
    # Box plot of rewards
    ax1 = axes[0, 0]
    rewards_df.boxplot(column='reward', by='threshold', ax=ax1)
    ax1.set_title('Reward Distribution by Threshold')
    ax1.set_xlabel('Deflection Threshold')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    
    # Mean reward by threshold
    ax2 = axes[0, 1]
    mean_rewards = rewards_df.groupby('threshold')['reward'].mean()
    mean_rewards.plot(kind='bar', ax=ax2, color='skyblue', edgecolor='navy')
    ax2.set_title('Mean Reward by Threshold')
    ax2.set_xlabel('Deflection Threshold')
    ax2.set_ylabel('Mean Reward')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(mean_rewards.values):
        ax2.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Deflection rate by threshold
    ax3 = axes[1, 0]
    deflection_rates = rewards_df.groupby('threshold')['action'].mean()
    deflection_rates.plot(kind='bar', ax=ax3, color='lightcoral', edgecolor='darkred')
    ax3.set_title('Deflection Rate by Threshold')
    ax3.set_xlabel('Deflection Threshold')
    ax3.set_ylabel('Deflection Rate')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(deflection_rates.values):
        ax3.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Correlation: Threshold vs FCT
    ax4 = axes[1, 1]
    fct_by_threshold = rewards_df.groupby('threshold')['fct'].mean()
    fct_by_threshold.plot(kind='line', marker='o', ax=ax4, color='green', linewidth=2, markersize=6)
    ax4.set_title('Mean FCT by Threshold')
    ax4.set_xlabel('Deflection Threshold')
    ax4.set_ylabel('Mean FCT (seconds)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file1 = output_path / "reward_trends_overview.png"
    plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
    print(f"   Saved overview plot: {plot_file1}")
    plt.close()
    
    # 2. Detailed reward components analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Detailed Performance Metrics by Deflection Threshold', fontsize=16, fontweight='bold')
    
    metrics = ['reward', 'fct', 'ooo', 'packet_delay', 'queue_utilization']
    titles = ['Reward', 'Flow Completion Time', 'Out-of-Order Rate', 'Packet Delay', 'Queue Utilization']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        metric_by_threshold = rewards_df.groupby('threshold')[metric].mean()
        metric_by_threshold.plot(kind='line', marker='o', ax=ax, linewidth=2, markersize=6)
        ax.set_title(f'Mean {title} by Threshold')
        ax.set_xlabel('Deflection Threshold')
        ax.set_ylabel(f'Mean {title}')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Remove the empty subplot
    axes[1, 2].remove()
    
    plt.tight_layout()
    plot_file2 = output_path / "performance_metrics_by_threshold.png"
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    print(f"   Saved metrics plot: {plot_file2}")
    plt.close()
    
    # 3. Action-specific analysis
    print("\n7. Action-specific analysis...")
    
    action_analysis = rewards_df.groupby(['threshold', 'action']).agg({
        'reward': ['count', 'mean', 'std'],
        'fct': 'mean',
        'ooo': 'mean',
        'packet_delay': 'mean'
    }).round(4)
    
    print("   Reward by threshold and action:")
    print(action_analysis.to_string())
    
    # Plot action-specific rewards
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Reward Analysis by Action Type', fontsize=14, fontweight='bold')
    
    # Separate forward and deflect rewards
    forward_rewards = rewards_df[rewards_df['action'] == 0].groupby('threshold')['reward'].mean()
    deflect_rewards = rewards_df[rewards_df['action'] == 1].groupby('threshold')['reward'].mean()
    
    # Plot side by side
    x = np.arange(len(thresholds))
    width = 0.35
    
    ax1.bar(x - width/2, forward_rewards, width, label='Forward (Action 0)', alpha=0.8, color='blue')
    ax1.bar(x + width/2, deflect_rewards, width, label='Deflect (Action 1)', alpha=0.8, color='red')
    ax1.set_title('Mean Reward by Action and Threshold')
    ax1.set_xlabel('Deflection Threshold')
    ax1.set_ylabel('Mean Reward')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reward difference (Forward - Deflect)
    reward_diff = forward_rewards - deflect_rewards
    reward_diff.plot(kind='bar', ax=ax2, color='purple', alpha=0.7)
    ax2.set_title('Reward Advantage: Forward vs Deflect')
    ax2.set_xlabel('Deflection Threshold')
    ax2.set_ylabel('Reward Difference (Forward - Deflect)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plot_file3 = output_path / "action_specific_rewards.png"
    plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
    print(f"   Saved action analysis plot: {plot_file3}")
    plt.close()
    
    # 8. Statistical summary
    print("\n8. Statistical Summary:")
    print(f"   Best performing threshold (highest mean reward): {mean_rewards.idxmax():.2f} (reward: {mean_rewards.max():.4f})")
    print(f"   Worst performing threshold (lowest mean reward): {mean_rewards.idxmin():.2f} (reward: {mean_rewards.min():.4f})")
    print(f"   Reward range: {mean_rewards.max() - mean_rewards.min():.4f}")
    
    # Threshold with lowest FCT
    fct_by_threshold = rewards_df.groupby('threshold')['fct'].mean()
    print(f"   Threshold with lowest FCT: {fct_by_threshold.idxmin():.2f} (FCT: {fct_by_threshold.min():.6f}s)")
    
    # Threshold with lowest OOO rate
    ooo_by_threshold = rewards_df.groupby('threshold')['ooo'].mean()
    print(f"   Threshold with lowest OOO rate: {ooo_by_threshold.idxmin():.2f} (OOO rate: {ooo_by_threshold.min():.4f})")
    
    # Save raw rewards data for further analysis
    rewards_file = output_path / "all_rewards_by_threshold.csv"
    rewards_df.to_csv(rewards_file, index=False)
    print(f"   Saved all rewards data: {rewards_file}")
    
    print(f"\n✅ Analysis complete! Results saved in: {output_path}")
    
    return rewards_df, reward_stats

def main():
    """Main execution function."""
    # Look for the threshold dataset
    dataset_files = [
        "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv",
        "../extracted_results/threshold_experiment_results.csv",
        "threshold_experiment_results.csv", 
        "../threshold_experiment_results.csv"
    ]
    
    dataset_path = None
    for file_path in dataset_files:
        if Path(file_path).exists():
            dataset_path = file_path
            break
    
    if dataset_path is None:
        print("ERROR: Could not find threshold dataset file!")
        print("Expected locations:")
        for path in dataset_files:
            print(f"  - {path}")
        return
    
    # Run analysis
    try:
        rewards_df, reward_stats = analyze_reward_trends_by_threshold(dataset_path)
        
        print("\n=== ANALYSIS SUMMARY ===")
        print("Key insights:")
        print("1. Reward trends show how different thresholds affect network performance")
        print("2. Lower thresholds may lead to more deflections but potentially better load balancing")
        print("3. Higher thresholds may preserve more direct paths but risk congestion")
        print("4. The optimal threshold balances deflection benefits with path efficiency")
        
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
