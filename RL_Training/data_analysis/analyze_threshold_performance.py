#!/usr/bin/env python3
"""
Analyze threshold performance from the combined dataset.
Compare FCT, deflection rates, and per-packet delay across different thresholds.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_threshold_performance(csv_path):
    """Analyze performance differences between deflection thresholds."""
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"Thresholds: {sorted(df['deflection_threshold'].unique())}")
    
    # Calculate per-packet delay (assuming this is related to timestamp differences in flow)
    # For simplicity, we'll use FCT/flow_packet_count as an approximation
    df['per_packet_delay'] = df['FCT'] / df['flow_packet_count']
    
    # Group by threshold for analysis
    threshold_groups = df.groupby('deflection_threshold')
    
    # Create comprehensive analysis
    results = {}
    
    print("\n" + "="*80)
    print("THRESHOLD PERFORMANCE ANALYSIS")
    print("="*80)
    
    for threshold in sorted(df['deflection_threshold'].unique()):
        subset = df[df['deflection_threshold'] == threshold]
        
        results[threshold] = {
            'sample_count': len(subset),
            'fct_mean': subset['FCT'].mean(),
            'fct_std': subset['FCT'].std(),
            'fct_median': subset['FCT'].median(),
            'fct_95th': subset['FCT'].quantile(0.95),
            'deflection_rate_mean': subset['deflection_rate'].mean(),
            'deflection_rate_std': subset['deflection_rate'].std(),
            'per_packet_delay_mean': subset['per_packet_delay'].mean(),
            'per_packet_delay_std': subset['per_packet_delay'].std(),
            'per_packet_delay_median': subset['per_packet_delay'].median(),
            'per_packet_delay_95th': subset['per_packet_delay'].quantile(0.95)
        }
        
        print(f"\nThreshold: {threshold}")
        print(f"  Samples: {results[threshold]['sample_count']}")
        print(f"  FCT (Flow Completion Time):")
        print(f"    Mean: {results[threshold]['fct_mean']:.6f}s")
        print(f"    Std:  {results[threshold]['fct_std']:.6f}s") 
        print(f"    Median: {results[threshold]['fct_median']:.6f}s")
        print(f"    95th percentile: {results[threshold]['fct_95th']:.6f}s")
        print(f"  Deflection Rate:")
        print(f"    Mean: {results[threshold]['deflection_rate_mean']:.4f}")
        print(f"    Std:  {results[threshold]['deflection_rate_std']:.4f}")
        print(f"  Per-Packet Delay:")
        print(f"    Mean: {results[threshold]['per_packet_delay_mean']:.6f}s")
        print(f"    Std:  {results[threshold]['per_packet_delay_std']:.6f}s")
        print(f"    Median: {results[threshold]['per_packet_delay_median']:.6f}s")
        print(f"    95th percentile: {results[threshold]['per_packet_delay_95th']:.6f}s")
    
    # Statistical significance tests
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    thresholds = sorted(df['deflection_threshold'].unique())
    
    # FCT comparison
    print("\nFCT Comparisons (Mann-Whitney U test):")
    for i in range(len(thresholds)):
        for j in range(i+1, len(thresholds)):
            t1, t2 = thresholds[i], thresholds[j]
            group1 = df[df['deflection_threshold'] == t1]['FCT']
            group2 = df[df['deflection_threshold'] == t2]['FCT']
            
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"  {t1} vs {t2}: p={p_value:.6f} {significance}")
    
    # Deflection rate comparison
    print("\nDeflection Rate Comparisons (Mann-Whitney U test):")
    for i in range(len(thresholds)):
        for j in range(i+1, len(thresholds)):
            t1, t2 = thresholds[i], thresholds[j]
            group1 = df[df['deflection_threshold'] == t1]['deflection_rate']
            group2 = df[df['deflection_threshold'] == t2]['deflection_rate']
            
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            print(f"  {t1} vs {t2}: p={p_value:.6f} {significance}")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # fig.suptitle('Threshold Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. FCT Box Plot - Create individual plots for each threshold
    thresholds = sorted(df['deflection_threshold'].unique())
    threshold_labels = []
    fct_means = []
    fct_stds = []
    
    # Collect statistics for plotting
    for t in thresholds:
        subset = df[df['deflection_threshold'] == t]
        threshold_labels.append(str(t))
        fct_means.append(subset['FCT'].mean())
        fct_stds.append(subset['FCT'].std())
    
    # Create bar plot with error bars instead of boxplot
    x_pos = np.arange(len(threshold_labels))
    axes[0,0].bar(x_pos, fct_means, yerr=fct_stds, capsize=5, alpha=0.7)
    axes[0,0].set_title('Flow Completion Time by Threshold')
    axes[0,0].set_xlabel('Deflection Threshold')
    axes[0,0].set_ylabel('FCT (seconds)')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(threshold_labels)
    
    # 2. Deflection Rate Bar Plot
    deflection_means = []
    deflection_stds = []
    for t in thresholds:
        subset = df[df['deflection_threshold'] == t]
        deflection_means.append(subset['deflection_rate'].mean())
        deflection_stds.append(subset['deflection_rate'].std())
    
    axes[0,1].bar(x_pos, deflection_means, yerr=deflection_stds, capsize=5, alpha=0.7, color='orange')
    axes[0,1].set_title('Deflection Rate by Threshold')
    axes[0,1].set_xlabel('Deflection Threshold')
    axes[0,1].set_ylabel('Deflection Rate')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(threshold_labels)
    
    # 3. Per-Packet Delay Bar Plot
    delay_means = []
    delay_stds = []
    for t in thresholds:
        subset = df[df['deflection_threshold'] == t]
        delay_means.append(subset['per_packet_delay'].mean())
        delay_stds.append(subset['per_packet_delay'].std())
    
    axes[0,2].bar(x_pos, delay_means, yerr=delay_stds, capsize=5, alpha=0.7, color='green')
    axes[0,2].set_title('Per-Packet Delay by Threshold')
    axes[0,2].set_xlabel('Deflection Threshold')
    axes[0,2].set_ylabel('Per-Packet Delay (seconds)')
    axes[0,2].set_xticks(x_pos)
    axes[0,2].set_xticklabels(threshold_labels)
    
    # 4. FCT vs Deflection Rate Scatter
    for threshold in thresholds:
        subset = df[df['deflection_threshold'] == threshold]
        axes[1,0].scatter(subset['deflection_rate'], subset['FCT'], 
                         label=f'Threshold {threshold}', alpha=0.6, s=20)
    axes[1,0].set_xlabel('Deflection Rate')
    axes[1,0].set_ylabel('FCT (seconds)')
    axes[1,0].set_title('FCT vs Deflection Rate')
    axes[1,0].legend()
    
    # 5. Mean metrics comparison - use matplotlib directly
    means_df = pd.DataFrame(results).T
    width = 0.35
    x_pos = np.arange(len(thresholds))
    
    axes[1,1].bar(x_pos - width/2, means_df['fct_mean'], width, label='FCT Mean', alpha=0.7)
    axes[1,1].bar(x_pos + width/2, means_df['per_packet_delay_mean'], width, label='Per-Packet Delay Mean', alpha=0.7)
    axes[1,1].set_title('Mean FCT and Per-Packet Delay by Threshold')
    axes[1,1].set_xlabel('Threshold')
    axes[1,1].set_ylabel('Time (seconds)')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels([str(t) for t in thresholds])
    axes[1,1].legend()
    
    # 6. CDF of FCT
    for threshold in thresholds:
        subset = df[df['deflection_threshold'] == threshold]
        sorted_fct = np.sort(subset['FCT'])
        y = np.arange(1, len(sorted_fct) + 1) / len(sorted_fct)
        axes[1,2].plot(sorted_fct, y, label=f'Threshold {threshold}', linewidth=2)
    axes[1,2].set_xlabel('FCT (seconds)')
    axes[1,2].set_ylabel('CDF')
    axes[1,2].set_title('Cumulative Distribution of FCT')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/practical_deflection/RL_Training/threshold_performance_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("Visualization saved as 'threshold_performance_analysis.png'")
    
    # Summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Best and worst thresholds for different metrics
    best_fct = min(results, key=lambda x: results[x]['fct_mean'])
    worst_fct = max(results, key=lambda x: results[x]['fct_mean'])
    
    lowest_deflection = min(results, key=lambda x: results[x]['deflection_rate_mean'])
    highest_deflection = max(results, key=lambda x: results[x]['deflection_rate_mean'])
    
    best_per_packet = min(results, key=lambda x: results[x]['per_packet_delay_mean'])
    worst_per_packet = max(results, key=lambda x: results[x]['per_packet_delay_mean'])
    
    print(f"Best FCT performance: Threshold {best_fct} ({results[best_fct]['fct_mean']:.6f}s)")
    print(f"Worst FCT performance: Threshold {worst_fct} ({results[worst_fct]['fct_mean']:.6f}s)")
    print(f"Lowest deflection rate: Threshold {lowest_deflection} ({results[lowest_deflection]['deflection_rate_mean']:.4f})")
    print(f"Highest deflection rate: Threshold {highest_deflection} ({results[highest_deflection]['deflection_rate_mean']:.4f})")
    print(f"Best per-packet delay: Threshold {best_per_packet} ({results[best_per_packet]['per_packet_delay_mean']:.6f}s)")
    print(f"Worst per-packet delay: Threshold {worst_per_packet} ({results[worst_per_packet]['per_packet_delay_mean']:.6f}s)")
    
    # Performance improvement calculations
    fct_improvement = (results[worst_fct]['fct_mean'] - results[best_fct]['fct_mean']) / results[worst_fct]['fct_mean'] * 100
    deflection_reduction = (results[highest_deflection]['deflection_rate_mean'] - results[lowest_deflection]['deflection_rate_mean']) / results[highest_deflection]['deflection_rate_mean'] * 100
    
    print(f"\nFCT improvement from worst to best: {fct_improvement:.2f}%")
    print(f"Deflection rate reduction from highest to lowest: {deflection_reduction:.2f}%")
    
    return df, results

if __name__ == "__main__":
    csv_path = "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    df, results = analyze_threshold_performance(csv_path)
    
    plt.show()
