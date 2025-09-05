#!/usr/bin/env python3
"""
Threshold Impact Analysis: FCT and One-Way Delay Comparison

This script analyzes the impact of different deflection thresholds on:
1. Flow Completion Time (FCT) distributions
2. Packet one-way delays
3. Deflection rates and patterns
4. Network performance metrics

The goal is to understand how deflection threshold settings affect network performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_threshold_datasets(base_path="/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims"):
    """Load all threshold datasets"""
    datasets = {}
    thresholds = [15000, 25000, 37500, 50000]
    
    print("Loading threshold datasets...")
    for threshold in thresholds:
        file_path = f"{base_path}/threshold_dataset_{threshold}.csv"
        if Path(file_path).exists():
            print(f"  Loading threshold {threshold}...")
            df = pd.read_csv(file_path)
            datasets[threshold] = df
            print(f"    {len(df)} packets, {df['FlowID'].nunique()} flows")
        else:
            print(f"  ⚠ File not found: {file_path}")
    
    return datasets

def analyze_fct_by_threshold(datasets):
    """Analyze FCT distributions across thresholds"""
    print("\n" + "="*60)
    print("FCT ANALYSIS BY THRESHOLD")
    print("="*60)
    
    fct_stats = []
    
    for threshold, df in datasets.items():
        # Get flow-level FCT data (one FCT per flow)
        flow_data = df.groupby('FlowID').agg({
            'FCT': 'first',
            'flow_packet_count': 'first',
            'deflection_count': 'first',
            'deflection_rate': 'first'
        }).reset_index()
        
        fct_values = flow_data['FCT'].values
        
        stats_dict = {
            'threshold': threshold,
            'num_flows': len(flow_data),
            'mean_fct': np.mean(fct_values),
            'median_fct': np.median(fct_values),
            'std_fct': np.std(fct_values),
            'min_fct': np.min(fct_values),
            'max_fct': np.max(fct_values),
            'q25_fct': np.percentile(fct_values, 25),
            'q75_fct': np.percentile(fct_values, 75),
            'q95_fct': np.percentile(fct_values, 95),
            'q99_fct': np.percentile(fct_values, 99),
            'mean_deflection_rate': np.mean(flow_data['deflection_rate']),
            'flows_with_deflections': len(flow_data[flow_data['deflection_count'] > 0]),
            'percent_flows_deflected': len(flow_data[flow_data['deflection_count'] > 0]) / len(flow_data) * 100
        }
        
        fct_stats.append(stats_dict)
        
        print(f"\nThreshold {threshold}:")
        print(f"  Flows: {stats_dict['num_flows']}")
        print(f"  Mean FCT: {stats_dict['mean_fct']:.6f}s")
        print(f"  Median FCT: {stats_dict['median_fct']:.6f}s")
        print(f"  95th percentile FCT: {stats_dict['q95_fct']:.6f}s")
        print(f"  Max FCT: {stats_dict['max_fct']:.6f}s")
        print(f"  Flows with deflections: {stats_dict['flows_with_deflections']} ({stats_dict['percent_flows_deflected']:.1f}%)")
        print(f"  Mean deflection rate: {stats_dict['mean_deflection_rate']:.3f}")
    
    return pd.DataFrame(fct_stats)

def analyze_packet_delays_by_threshold(datasets):
    """Analyze packet one-way delays across thresholds"""
    print("\n" + "="*60)
    print("PACKET ONE-WAY DELAY ANALYSIS BY THRESHOLD")
    print("="*60)
    
    delay_stats = []
    
    for threshold, df in datasets.items():
        # Calculate packet delays (relative to flow start)
        df_delays = df.copy()
        df_delays['packet_delay'] = df_delays['timestamp'] - df_delays['flow_start_time']
        
        # Overall packet delay statistics
        delays = df_delays['packet_delay'].values
        
        stats_dict = {
            'threshold': threshold,
            'num_packets': len(delays),
            'mean_delay': np.mean(delays),
            'median_delay': np.median(delays),
            'std_delay': np.std(delays),
            'min_delay': np.min(delays),
            'max_delay': np.max(delays),
            'q95_delay': np.percentile(delays, 95),
            'q99_delay': np.percentile(delays, 99)
        }
        
        delay_stats.append(stats_dict)
        
        print(f"\nThreshold {threshold}:")
        print(f"  Packets: {stats_dict['num_packets']:,}")
        print(f"  Mean delay: {stats_dict['mean_delay']:.6f}s")
        print(f"  Median delay: {stats_dict['median_delay']:.6f}s")
        print(f"  95th percentile delay: {stats_dict['q95_delay']:.6f}s")
        print(f"  Max delay: {stats_dict['max_delay']:.6f}s")
    
    return pd.DataFrame(delay_stats)

def perform_statistical_tests(datasets):
    """Perform statistical tests to compare thresholds"""
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON TESTS")
    print("="*60)
    
    thresholds = list(datasets.keys())
    
    # FCT comparison
    print("\nFCT Statistical Tests:")
    fct_data = {}
    for threshold, df in datasets.items():
        flow_data = df.groupby('FlowID')['FCT'].first()
        fct_data[threshold] = flow_data.values
    
    # Compare consecutive thresholds
    for i in range(len(thresholds) - 1):
        t1, t2 = thresholds[i], thresholds[i + 1]
        statistic, p_value = stats.mannwhitneyu(fct_data[t1], fct_data[t2], alternative='two-sided')
        
        mean1, mean2 = np.mean(fct_data[t1]), np.mean(fct_data[t2])
        change = ((mean2 - mean1) / mean1) * 100
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"  {t1} vs {t2}: p={p_value:.2e} {significance}")
        print(f"    Mean FCT change: {change:+.1f}% ({mean1:.6f}s → {mean2:.6f}s)")
    
    # Packet delay comparison
    print("\nPacket Delay Statistical Tests:")
    delay_data = {}
    for threshold, df in datasets.items():
        df_copy = df.copy()
        df_copy['packet_delay'] = df_copy['timestamp'] - df_copy['flow_start_time']
        # Sample for statistical tests (too many points otherwise)
        sample_size = min(10000, len(df_copy))
        delay_data[threshold] = df_copy['packet_delay'].sample(sample_size).values
    
    for i in range(len(thresholds) - 1):
        t1, t2 = thresholds[i], thresholds[i + 1]
        statistic, p_value = stats.mannwhitneyu(delay_data[t1], delay_data[t2], alternative='two-sided')
        
        mean1, mean2 = np.mean(delay_data[t1]), np.mean(delay_data[t2])
        change = ((mean2 - mean1) / mean1) * 100
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"  {t1} vs {t2}: p={p_value:.2e} {significance}")
        print(f"    Mean delay change: {change:+.1f}% ({mean1:.6f}s → {mean2:.6f}s)")

def create_visualizations(datasets, fct_stats, delay_stats, output_dir="threshold_analysis"):
    """Create comprehensive visualizations"""
    print(f"\nCreating visualizations in {output_dir}/...")
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Figure 1: FCT Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FCT Analysis Across Deflection Thresholds', fontsize=16, fontweight='bold')
    
    # FCT box plot - create manually with simplified approach
    threshold_list = list(datasets.keys())
    
    # Create FCT comparison without problematic boxplot
    axes[0,0].bar(range(len(threshold_list)), fct_stats['mean_fct'], alpha=0.7, 
                  color=['blue', 'green', 'orange', 'red'])
    axes[0,0].set_xlabel('Deflection Threshold')
    axes[0,0].set_ylabel('Mean Flow Completion Time (s)')
    axes[0,0].set_title('Mean FCT by Threshold')
    axes[0,0].set_xticks(range(len(threshold_list)))
    axes[0,0].set_xticklabels([str(t) for t in threshold_list])
    axes[0,0].grid(True, alpha=0.3)
    
    # FCT improvement over thresholds (percentage change from baseline)
    baseline_mean = fct_stats['mean_fct'].iloc[0]
    pct_improvement = ((baseline_mean - fct_stats['mean_fct']) / baseline_mean * 100).values
    axes[0,1].bar(range(len(threshold_list)), pct_improvement, alpha=0.7, 
                  color=['gray' if x <= 0 else 'green' for x in pct_improvement])
    axes[0,1].set_xlabel('Deflection Threshold')
    axes[0,1].set_ylabel('FCT Improvement (%)')
    axes[0,1].set_title('FCT Improvement vs Baseline (15000)')
    axes[0,1].set_xticks(range(len(threshold_list)))
    axes[0,1].set_xticklabels([str(t) for t in threshold_list])
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Deflection impact
    axes[1,0].bar(fct_stats['threshold'].astype(str), fct_stats['percent_flows_deflected'].values, 
                  color='lightcoral', alpha=0.7, edgecolor='darkred')
    axes[1,0].set_xlabel('Deflection Threshold')
    axes[1,0].set_ylabel('% Flows with Deflections')
    axes[1,0].set_title('Deflection Impact by Threshold')
    axes[1,0].grid(True, alpha=0.3)
    
    # Mean deflection rate
    axes[1,1].plot(fct_stats['threshold'].values, fct_stats['mean_deflection_rate'].values, 'mo-', 
                   linewidth=3, markersize=10)
    axes[1,1].set_xlabel('Deflection Threshold')
    axes[1,1].set_ylabel('Mean Deflection Rate')
    axes[1,1].set_title('Average Deflection Rate by Threshold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fct_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Packet Delay Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Packet One-Way Delay Analysis Across Thresholds', fontsize=16, fontweight='bold')
    
    # Delay distributions - use bar chart instead of boxplot
    axes[0,0].bar(range(len(threshold_list)), delay_stats['mean_delay'], alpha=0.7,
                  color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'])
    axes[0,0].set_xlabel('Deflection Threshold')
    axes[0,0].set_ylabel('Mean Packet One-Way Delay (s)')
    axes[0,0].set_title('Mean Packet Delay by Threshold')
    axes[0,0].set_xticks(range(len(threshold_list)))
    axes[0,0].set_xticklabels([str(t) for t in threshold_list])
    axes[0,0].grid(True, alpha=0.3)
    
    # Delay improvement over thresholds (percentage change from baseline)
    baseline_delay = delay_stats['mean_delay'].iloc[0]
    delay_pct_improvement = ((baseline_delay - delay_stats['mean_delay']) / baseline_delay * 100).values
    axes[0,1].bar(range(len(threshold_list)), delay_pct_improvement, alpha=0.7,
                  color=['gray' if x <= 0 else 'blue' for x in delay_pct_improvement])
    axes[0,1].set_xlabel('Deflection Threshold')
    axes[0,1].set_ylabel('Delay Improvement (%)')
    axes[0,1].set_title('Packet Delay Improvement vs Baseline (15000)')
    axes[0,1].set_xticks(range(len(threshold_list)))
    axes[0,1].set_xticklabels([str(t) for t in threshold_list])
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Delay vs FCT correlation
    axes[1,0].scatter(delay_stats['mean_delay'].values, fct_stats['mean_fct'].values, s=100, c=fct_stats['threshold'].values, 
                     cmap='viridis', alpha=0.8, edgecolors='black')
    for i, threshold in enumerate(fct_stats['threshold'].values):
        axes[1,0].annotate(f'{threshold}', 
                          (delay_stats.iloc[i]['mean_delay'], fct_stats.iloc[i]['mean_fct']),
                          xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    axes[1,0].set_xlabel('Mean Packet Delay (s)')
    axes[1,0].set_ylabel('Mean FCT (s)')
    axes[1,0].set_title('Packet Delay vs FCT Correlation')
    axes[1,0].grid(True, alpha=0.3)
    
    # Performance comparison
    thresholds = fct_stats['threshold'].values
    x_pos = np.arange(len(thresholds))
    width = 0.35
    
    axes[1,1].bar(x_pos - width/2, fct_stats['mean_fct'].values, width, label='Mean FCT', alpha=0.8)
    axes[1,1].bar(x_pos + width/2, delay_stats['mean_delay'].values, width, label='Mean Delay', alpha=0.8)
    axes[1,1].set_xlabel('Deflection Threshold')
    axes[1,1].set_ylabel('Time (s)')
    axes[1,1].set_title('Performance Metrics Comparison')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(thresholds)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/delay_threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def create_separate_statistics_trends(fct_stats, delay_stats, output_dir="threshold_analysis"):
    """Create separate trend graphs for each statistic and save as individual PNG files"""
    
    print("\n" + "="*60)
    print("CREATING SEPARATE STATISTICS TRENDS GRAPHS")
    print("="*60)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Color schemes for different statistics
    colors = {
        'mean': '#2E86AB',     # Blue
        'median': '#A23B72',   # Purple
        'q95': '#F18F01'       # Orange
    }
    
    # 1. FCT Mean Trend
    plt.figure(figsize=(10, 6))
    plt.plot(fct_stats['threshold'].values, fct_stats['mean_fct'].values, 
             'o-', color=colors['mean'], linewidth=3, markersize=10, markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors['mean'])
    plt.xlabel('Deflection Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Mean FCT (s)', fontsize=12, fontweight='bold')
    plt.title('Mean Flow Completion Time Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (threshold, fct) in enumerate(zip(fct_stats['threshold'].values, fct_stats['mean_fct'].values)):
        plt.annotate(f'{fct:.4f}s', (threshold, fct), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fct_mean_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fct_mean_trend.png")
    
    # 2. FCT Median Trend
    plt.figure(figsize=(10, 6))
    plt.plot(fct_stats['threshold'].values, fct_stats['median_fct'].values, 
             'o-', color=colors['median'], linewidth=3, markersize=10, markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors['median'])
    plt.xlabel('Deflection Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Median FCT (s)', fontsize=12, fontweight='bold')
    plt.title('Median Flow Completion Time Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (threshold, fct) in enumerate(zip(fct_stats['threshold'].values, fct_stats['median_fct'].values)):
        plt.annotate(f'{fct:.4f}s', (threshold, fct), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fct_median_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fct_median_trend.png")
    
    # 3. FCT 95th Percentile Trend
    plt.figure(figsize=(10, 6))
    plt.plot(fct_stats['threshold'].values, fct_stats['q95_fct'].values, 
             'o-', color=colors['q95'], linewidth=3, markersize=10, markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors['q95'])
    plt.xlabel('Deflection Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('95th Percentile FCT (s)', fontsize=12, fontweight='bold')
    plt.title('95th Percentile Flow Completion Time Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (threshold, fct) in enumerate(zip(fct_stats['threshold'].values, fct_stats['q95_fct'].values)):
        plt.annotate(f'{fct:.4f}s', (threshold, fct), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fct_95th_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: fct_95th_trend.png")
    
    # 4. Packet Delay Mean Trend
    plt.figure(figsize=(10, 6))
    plt.plot(delay_stats['threshold'].values, delay_stats['mean_delay'].values, 
             'o-', color=colors['mean'], linewidth=3, markersize=10, markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors['mean'])
    plt.xlabel('Deflection Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Packet Delay (s)', fontsize=12, fontweight='bold')
    plt.title('Mean Packet One-Way Delay Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (threshold, delay) in enumerate(zip(delay_stats['threshold'].values, delay_stats['mean_delay'].values)):
        plt.annotate(f'{delay:.5f}s', (threshold, delay), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/delay_mean_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: delay_mean_trend.png")
    
    # 5. Packet Delay Median Trend
    plt.figure(figsize=(10, 6))
    plt.plot(delay_stats['threshold'].values, delay_stats['median_delay'].values, 
             'o-', color=colors['median'], linewidth=3, markersize=10, markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors['median'])
    plt.xlabel('Deflection Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('Median Packet Delay (s)', fontsize=12, fontweight='bold')
    plt.title('Median Packet One-Way Delay Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (threshold, delay) in enumerate(zip(delay_stats['threshold'].values, delay_stats['median_delay'].values)):
        plt.annotate(f'{delay:.5f}s', (threshold, delay), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/delay_median_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: delay_median_trend.png")
    
    # 6. Packet Delay 95th Percentile Trend
    plt.figure(figsize=(10, 6))
    plt.plot(delay_stats['threshold'].values, delay_stats['q95_delay'].values, 
             'o-', color=colors['q95'], linewidth=3, markersize=10, markerfacecolor='white', 
             markeredgewidth=2, markeredgecolor=colors['q95'])
    plt.xlabel('Deflection Threshold', fontsize=12, fontweight='bold')
    plt.ylabel('95th Percentile Packet Delay (s)', fontsize=12, fontweight='bold')
    plt.title('95th Percentile Packet One-Way Delay Trend', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (threshold, delay) in enumerate(zip(delay_stats['threshold'].values, delay_stats['q95_delay'].values)):
        plt.annotate(f'{delay:.5f}s', (threshold, delay), 
                    textcoords="offset points", xytext=(0,10), ha='center', 
                    fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/delay_95th_trend.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: delay_95th_trend.png")
    
    print(f"\n✓ All trend graphs saved to: {output_dir}/")
    print("  Individual trend files:")
    print("    - fct_mean_trend.png")
    print("    - fct_median_trend.png") 
    print("    - fct_95th_trend.png")
    print("    - delay_mean_trend.png")
    print("    - delay_median_trend.png")
    print("    - delay_95th_trend.png")

def generate_summary_report(fct_stats, delay_stats, output_dir="threshold_analysis"):
    """Generate a comprehensive summary report"""
    report_file = f"{output_dir}/threshold_impact_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("DEFLECTION THRESHOLD IMPACT ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY:\n")
        f.write("-" * 20 + "\n")
        
        # Key findings
        fct_improvement = ((fct_stats.iloc[-1]['mean_fct'] - fct_stats.iloc[0]['mean_fct']) / 
                          fct_stats.iloc[0]['mean_fct']) * 100
        delay_improvement = ((delay_stats.iloc[-1]['mean_delay'] - delay_stats.iloc[0]['mean_delay']) / 
                           delay_stats.iloc[0]['mean_delay']) * 100
        deflection_change = fct_stats.iloc[-1]['mean_deflection_rate'] - fct_stats.iloc[0]['mean_deflection_rate']
        
        f.write(f"• FCT change from lowest to highest threshold: {fct_improvement:+.1f}%\n")
        f.write(f"• Packet delay change from lowest to highest threshold: {delay_improvement:+.1f}%\n")
        f.write(f"• Deflection rate change: {deflection_change:+.3f}\n")
        f.write(f"• Optimal threshold (lowest mean FCT): {fct_stats.loc[fct_stats['mean_fct'].idxmin(), 'threshold']}\n")
        f.write(f"• Threshold with lowest delays: {delay_stats.loc[delay_stats['mean_delay'].idxmin(), 'threshold']}\n\n")
        
        f.write("FCT ANALYSIS BY THRESHOLD:\n")
        f.write("-" * 30 + "\n")
        for _, row in fct_stats.iterrows():
            f.write(f"Threshold {row['threshold']}:\n")
            f.write(f"  Mean FCT: {row['mean_fct']:.6f}s\n")
            f.write(f"  Median FCT: {row['median_fct']:.6f}s\n")
            f.write(f"  95th percentile FCT: {row['q95_fct']:.6f}s\n")
            f.write(f"  Flows with deflections: {row['percent_flows_deflected']:.1f}%\n")
            f.write(f"  Mean deflection rate: {row['mean_deflection_rate']:.3f}\n\n")
        
        f.write("PACKET DELAY ANALYSIS BY THRESHOLD:\n")
        f.write("-" * 40 + "\n")
        for _, row in delay_stats.iterrows():
            f.write(f"Threshold {row['threshold']}:\n")
            f.write(f"  Mean delay: {row['mean_delay']:.6f}s\n")
            f.write(f"  Median delay: {row['median_delay']:.6f}s\n")
            f.write(f"  95th percentile delay: {row['q95_delay']:.6f}s\n\n")
    
    print(f"Summary report saved: {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze threshold impact on FCT and packet delays")
    parser.add_argument("--data-path", default="/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims",
                       help="Path to threshold dataset files")
    parser.add_argument("--output-dir", default="threshold_analysis", help="Output directory for results")
    
    args = parser.parse_args()
    
    print("DEFLECTION THRESHOLD IMPACT ANALYSIS")
    print("=" * 60)
    
    # Load datasets
    datasets = load_threshold_datasets(args.data_path)
    
    if not datasets:
        print("No datasets found! Please check the data path.")
        return
    
    # Analyze FCT
    fct_stats = analyze_fct_by_threshold(datasets)
    
    # Analyze packet delays
    delay_stats = analyze_packet_delays_by_threshold(datasets)
    
    # Statistical tests
    perform_statistical_tests(datasets)
    
    # Create visualizations
    create_visualizations(datasets, fct_stats, delay_stats, args.output_dir)
    
    # Create separate statistics trend graphs
    create_separate_statistics_trends(fct_stats, delay_stats, args.output_dir)
    
    # Generate summary
    generate_summary_report(fct_stats, delay_stats, args.output_dir)
    
    # Save detailed statistics
    Path(args.output_dir).mkdir(exist_ok=True)
    fct_stats.to_csv(f"{args.output_dir}/fct_statistics_by_threshold.csv", index=False)
    delay_stats.to_csv(f"{args.output_dir}/delay_statistics_by_threshold.csv", index=False)
    
    print(f"\nAnalysis complete! Results saved in: {args.output_dir}/")
    print("Files generated:")
    print("  - fct_threshold_analysis.png")
    print("  - delay_threshold_analysis.png")
    print("  - fct_mean_trend.png")
    print("  - fct_median_trend.png")
    print("  - fct_95th_trend.png")
    print("  - delay_mean_trend.png")
    print("  - delay_median_trend.png")
    print("  - delay_95th_trend.png")
    print("  - threshold_impact_summary.txt")
    print("  - fct_statistics_by_threshold.csv")
    print("  - delay_statistics_by_threshold.csv")

if __name__ == "__main__":
    main()
