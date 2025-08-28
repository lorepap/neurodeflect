#!/usr/bin/env python3
"""
Simple Threshold Impact Analysis

This script provides a basic analysis of how different deflection thresholds
impact packet latency and network performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_threshold_data(dataset_path):
    """Analyze threshold impact on network performance."""
    print("Loading threshold dataset...")
    df = pd.read_csv(dataset_path)
    
    # Calculate latency in milliseconds
    df['latency_ms'] = (df['end_time'] - df['start_time']) * 1000
    
    # Calculate utilization metrics
    df['queue_utilization'] = df['occupancy'] / (df['capacity'] + 1e-8)
    df['total_utilization'] = df['total_occupancy'] / (df['total_capacity'] + 1e-8)
    
    # Calculate hops
    df['hops'] = 250 - df['ttl']
    
    print(f"Dataset loaded: {len(df)} records")
    print(f"Threshold values: {sorted(df['deflection_threshold'].unique())}")
    print(f"Actions present: {sorted(df['action'].unique())}")
    print(f"Latency range: {df['latency_ms'].min():.3f} - {df['latency_ms'].max():.3f} ms")
    
    return df

def analyze_by_threshold(df):
    """Analyze metrics grouped by threshold."""
    print("\n" + "="*60)
    print("THRESHOLD IMPACT ANALYSIS")
    print("="*60)
    
    results = []
    
    for threshold in sorted(df['deflection_threshold'].unique()):
        subset = df[df['deflection_threshold'] == threshold]
        
        stats = {
            'Threshold': threshold,
            'Packets': len(subset),
            'Mean_Latency_ms': subset['latency_ms'].mean(),
            'Median_Latency_ms': subset['latency_ms'].median(),
            'P95_Latency_ms': subset['latency_ms'].quantile(0.95),
            'P99_Latency_ms': subset['latency_ms'].quantile(0.99),
            'Std_Latency_ms': subset['latency_ms'].std(),
            'Mean_Queue_Util': subset['queue_utilization'].mean(),
            'Mean_Total_Util': subset['total_utilization'].mean(),
            'Mean_Hops': subset['hops'].mean(),
            'OOO_Rate': subset['ooo'].mean(),
            'Forward_Rate': (subset['action'] == 0).mean(),
            'Deflect_Rate': (subset['action'] == 1).mean(),
            'Drop_Rate': (subset['action'] == 2).mean() if 2 in subset['action'].values else 0.0
        }
        results.append(stats)
    
    results_df = pd.DataFrame(results)
    
    print("\nPERFORMANCE SUMMARY BY THRESHOLD:")
    print("-" * 80)
    
    # Print key metrics in a readable format
    for _, row in results_df.iterrows():
        print(f"\nThreshold {row['Threshold']}:")
        print(f"  Packets analyzed: {row['Packets']:,}")
        print(f"  Mean latency: {row['Mean_Latency_ms']:.3f} ms")
        print(f"  P95 latency: {row['P95_Latency_ms']:.3f} ms")
        print(f"  P99 latency: {row['P99_Latency_ms']:.3f} ms")
        print(f"  Queue utilization: {row['Mean_Queue_Util']:.3f}")
        print(f"  Out-of-order rate: {row['OOO_Rate']:.4f}")
        print(f"  Forward rate: {row['Forward_Rate']:.1%}")
        print(f"  Deflect rate: {row['Deflect_Rate']:.1%}")
    
    return results_df

def create_simple_plots(df, output_dir):
    """Create simple plots for threshold analysis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Mean latency by threshold
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Mean latency comparison
    ax1 = axes[0, 0]
    threshold_stats = df.groupby('deflection_threshold')['latency_ms'].agg(['mean', 'std'])
    ax1.bar(threshold_stats.index, threshold_stats['mean'], 
           yerr=threshold_stats['std'], capsize=5, alpha=0.7, color='lightcoral')
    ax1.set_xlabel('Deflection Threshold')
    ax1.set_ylabel('Mean Latency (ms)')
    ax1.set_title('Mean Latency by Threshold')
    ax1.grid(True, alpha=0.3)
    
    # Latency percentiles
    ax2 = axes[0, 1]
    percentiles = df.groupby('deflection_threshold')['latency_ms'].quantile([0.5, 0.95, 0.99]).unstack()
    percentiles.plot(kind='bar', ax=ax2, color=['blue', 'orange', 'red'])
    ax2.set_xlabel('Deflection Threshold')
    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('Latency Percentiles by Threshold')
    ax2.legend(['Median', 'P95', 'P99'])
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
    
    # Action distribution
    ax3 = axes[1, 0]
    action_dist = df.groupby(['deflection_threshold', 'action']).size().unstack(fill_value=0)
    action_percentages = action_dist.div(action_dist.sum(axis=1), axis=0) * 100
    action_percentages.plot(kind='bar', ax=ax3, color=['lightblue', 'orange', 'red'])
    ax3.set_xlabel('Deflection Threshold')
    ax3.set_ylabel('Action Percentage (%)')
    ax3.set_title('Action Distribution by Threshold')
    ax3.legend(['Forward', 'Deflect', 'Drop'])
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=0)
    
    # Queue utilization vs threshold
    ax4 = axes[1, 1]
    util_stats = df.groupby('deflection_threshold').agg({
        'queue_utilization': 'mean',
        'total_utilization': 'mean'
    })
    util_stats.plot(kind='bar', ax=ax4, color=['steelblue', 'darkgreen'])
    ax4.set_xlabel('Deflection Threshold')
    ax4.set_ylabel('Utilization')
    ax4.set_title('Queue and Total Utilization by Threshold')
    ax4.legend(['Queue Util', 'Total Util'])
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
    
    plt.tight_layout()
    
    plot_path = output_dir / "threshold_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to: {plot_path}")
    
    return fig

def create_latency_histogram(df, output_dir):
    """Create latency distribution histogram."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    thresholds = sorted(df['deflection_threshold'].unique())
    colors = ['blue', 'orange', 'green', 'red']
    
    for i, threshold in enumerate(thresholds):
        data = df[df['deflection_threshold'] == threshold]['latency_ms']
        ax.hist(data, bins=50, alpha=0.7, label=f'Threshold {threshold}', 
               color=colors[i % len(colors)], density=True)
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Density')
    ax.set_title('Latency Distribution by Deflection Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = Path(output_dir) / "latency_distributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Latency distribution plot saved to: {plot_path}")
    
    return fig

def find_optimal_thresholds(results_df):
    """Find optimal thresholds for different objectives."""
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*60)
    
    # Best for different metrics
    best_latency = results_df.loc[results_df['Mean_Latency_ms'].idxmin()]
    best_p99 = results_df.loc[results_df['P99_Latency_ms'].idxmin()]
    lowest_ooo = results_df.loc[results_df['OOO_Rate'].idxmin()]
    
    print(f"\n🏆 BEST AVERAGE LATENCY:")
    print(f"   Threshold: {best_latency['Threshold']}")
    print(f"   Mean latency: {best_latency['Mean_Latency_ms']:.3f} ms")
    print(f"   P99 latency: {best_latency['P99_Latency_ms']:.3f} ms")
    print(f"   Deflect rate: {best_latency['Deflect_Rate']:.1%}")
    
    print(f"\n🏆 BEST P99 LATENCY:")
    print(f"   Threshold: {best_p99['Threshold']}")
    print(f"   P99 latency: {best_p99['P99_Latency_ms']:.3f} ms")
    print(f"   Mean latency: {best_p99['Mean_Latency_ms']:.3f} ms")
    print(f"   Queue utilization: {best_p99['Mean_Queue_Util']:.3f}")
    
    print(f"\n🏆 LOWEST OUT-OF-ORDER RATE:")
    print(f"   Threshold: {lowest_ooo['Threshold']}")
    print(f"   OOO rate: {lowest_ooo['OOO_Rate']:.4f}")
    print(f"   Mean latency: {lowest_ooo['Mean_Latency_ms']:.3f} ms")
    print(f"   Deflect rate: {lowest_ooo['Deflect_Rate']:.1%}")

def main():
    """Main analysis function."""
    dataset_path = "combined_threshold_dataset.csv"
    output_dir = "threshold_analysis_results"
    
    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return 1
    
    try:
        # Load and analyze data
        df = analyze_threshold_data(dataset_path)
        
        # Analyze by threshold
        results_df = analyze_by_threshold(df)
        
        # Create plots
        print(f"\nCreating analysis plots in: {output_dir}")
        create_simple_plots(df, output_dir)
        create_latency_histogram(df, output_dir)
        
        # Find optimal thresholds
        find_optimal_thresholds(results_df)
        
        # Save results
        output_path = Path(output_dir)
        results_df.to_csv(output_path / "threshold_comparison.csv", index=False)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📁 Results saved in: {output_dir}")
        print(f"📊 Plots created: threshold_analysis.png, latency_distributions.png")
        print(f"📄 Summary saved: threshold_comparison.csv")
        
        return 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
