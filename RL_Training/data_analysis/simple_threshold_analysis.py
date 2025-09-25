#!/usr/bin/env python3
"""
Simplified Reward Trend Analysis by Deflection Threshold

Focuses on key insights without complex plotting that might cause errors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def analyze_threshold_rewards_simple(dataset_path: str):
    """
    Simple analysis of reward trends by threshold with robust plotting.
    """
    print("=== SIMPLIFIED REWARD TREND ANALYSIS ===")
    
    # Load the pre-computed results
    results_file = Path("threshold_analysis/reward_trends_by_threshold.csv")
    
    if results_file.exists():
        print("Loading pre-computed results...")
        results = pd.read_csv(results_file)
        print(results.to_string(index=False))
        
        print("\n=== KEY INSIGHTS ===")
        
        # Find best and worst performing thresholds
        best_threshold = results.loc[results['reward_mean'].idxmax()]
        worst_threshold = results.loc[results['reward_mean'].idxmin()]
        
        print(f"🏆 BEST THRESHOLD: {best_threshold['threshold']}")
        print(f"   Mean Reward: {best_threshold['reward_mean']:.4f}")
        print(f"   Deflection Rate: {best_threshold['action_mean']:.1%}")
        print(f"   Mean FCT: {best_threshold['fct_mean']:.6f}s")
        print(f"   OOO Rate: {best_threshold['ooo_mean']:.1%}")
        
        print(f"\n❌ WORST THRESHOLD: {worst_threshold['threshold']}")
        print(f"   Mean Reward: {worst_threshold['reward_mean']:.4f}")
        print(f"   Deflection Rate: {worst_threshold['action_mean']:.1%}")
        print(f"   Mean FCT: {worst_threshold['fct_mean']:.6f}s")
        print(f"   OOO Rate: {worst_threshold['ooo_mean']:.1%}")
        
        print(f"\n📊 REWARD PERFORMANCE RANKING:")
        sorted_results = results.sort_values('reward_mean', ascending=False)
        for i, row in sorted_results.iterrows():
            print(f"   {int(i)+1}. Threshold {row['threshold']:4.2f}: {row['reward_mean']:7.4f} reward")
        
        print(f"\n🎯 DEFLECTION RATE ANALYSIS:")
        for _, row in results.iterrows():
            print(f"   Threshold {row['threshold']:4.2f}: {row['action_mean']:6.1%} deflection rate")
        
        print(f"\n⚡ PERFORMANCE METRICS ANALYSIS:")
        print(f"   Threshold | FCT (ms) | OOO Rate | Queue Util | Reward")
        print(f"   ----------|----------|----------|------------|--------")
        for _, row in results.iterrows():
            fct_ms = row['fct_mean'] * 1000  # Convert to milliseconds
            print(f"   {row['threshold']:8.2f} | {fct_ms:8.2f} | {row['ooo_mean']:8.1%} | {row['queue_utilization_mean']:10.1%} | {row['reward_mean']:7.4f}")
        
        # Trends analysis
        print(f"\n📈 TREND ANALYSIS:")
        
        # FCT trend
        fct_trend = "INCREASING" if results['fct_mean'].is_monotonic_increasing else "DECREASING" if results['fct_mean'].is_monotonic_decreasing else "MIXED"
        print(f"   FCT trend as threshold increases: {fct_trend}")
        
        # OOO trend  
        ooo_trend = "INCREASING" if results['ooo_mean'].is_monotonic_increasing else "DECREASING" if results['ooo_mean'].is_monotonic_decreasing else "MIXED"
        print(f"   OOO rate trend as threshold increases: {ooo_trend}")
        
        # Deflection rate trend
        deflection_trend = "INCREASING" if results['action_mean'].is_monotonic_increasing else "DECREASING" if results['action_mean'].is_monotonic_decreasing else "MIXED"
        print(f"   Deflection rate trend as threshold increases: {deflection_trend}")
        
        # Reward trend
        reward_trend = "INCREASING" if results['reward_mean'].is_monotonic_increasing else "DECREASING" if results['reward_mean'].is_monotonic_decreasing else "MIXED"
        print(f"   Reward trend as threshold increases: {reward_trend}")
        
        # Create simple, robust plots
        print(f"\n📊 Creating simplified visualizations...")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Threshold Performance Analysis', fontsize=14, fontweight='bold')
        
        # 1. Mean reward by threshold
        axes[0,0].bar(results['threshold'], results['reward_mean'], alpha=0.7, color='skyblue', edgecolor='navy')
        axes[0,0].set_title('Mean Reward by Threshold')
        axes[0,0].set_xlabel('Deflection Threshold')
        axes[0,0].set_ylabel('Mean Reward')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(results['reward_mean']):
            axes[0,0].text(results['threshold'].iloc[i], v + 0.002, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Deflection rate by threshold
        axes[0,1].bar(results['threshold'], results['action_mean'] * 100, alpha=0.7, color='lightcoral', edgecolor='darkred')
        axes[0,1].set_title('Deflection Rate by Threshold')
        axes[0,1].set_xlabel('Deflection Threshold')
        axes[0,1].set_ylabel('Deflection Rate (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(results['action_mean'] * 100):
            axes[0,1].text(results['threshold'].iloc[i], v + 0.1, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. FCT by threshold
        axes[1,0].plot(results['threshold'], results['fct_mean'] * 1000, marker='o', linewidth=2, markersize=6, color='green')
        axes[1,0].set_title('Mean FCT by Threshold')
        axes[1,0].set_xlabel('Deflection Threshold')
        axes[1,0].set_ylabel('Mean FCT (ms)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. OOO rate by threshold
        axes[1,1].plot(results['threshold'], results['ooo_mean'] * 100, marker='o', linewidth=2, markersize=6, color='orange')
        axes[1,1].set_title('Out-of-Order Rate by Threshold')
        axes[1,1].set_xlabel('Deflection Threshold')
        axes[1,1].set_ylabel('OOO Rate (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = "threshold_analysis/threshold_performance_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   Saved plot: {plot_file}")
        plt.close()
        
        # Key correlations
        print(f"\n🔗 CORRELATION ANALYSIS:")
        correlations = {
            'Threshold vs Reward': np.corrcoef(results['threshold'], results['reward_mean'])[0,1],
            'Threshold vs FCT': np.corrcoef(results['threshold'], results['fct_mean'])[0,1], 
            'Threshold vs OOO Rate': np.corrcoef(results['threshold'], results['ooo_mean'])[0,1],
            'Threshold vs Deflection Rate': np.corrcoef(results['threshold'], results['action_mean'])[0,1],
            'Deflection Rate vs Reward': np.corrcoef(results['action_mean'], results['reward_mean'])[0,1],
            'FCT vs Reward': np.corrcoef(results['fct_mean'], results['reward_mean'])[0,1]
        }
        
        for metric, corr in correlations.items():
            direction = "POSITIVE" if corr > 0.1 else "NEGATIVE" if corr < -0.1 else "WEAK"
            print(f"   {metric}: {corr:6.3f} ({direction})")
        
        print(f"\n🎯 OPTIMAL THRESHOLD RECOMMENDATION:")
        if best_threshold['threshold'] == 0.3:
            print(f"   ✅ Lower thresholds (0.3) perform best")
            print(f"   💡 Reason: More aggressive deflection helps load balancing")
        elif best_threshold['threshold'] == 1.0:
            print(f"   ✅ Higher thresholds (1.0) perform best") 
            print(f"   💡 Reason: Conservative deflection preserves direct paths")
        else:
            print(f"   ✅ Moderate threshold ({best_threshold['threshold']}) performs best")
            print(f"   💡 Reason: Balanced deflection strategy")
            
        print(f"   📋 Deploy with threshold = {best_threshold['threshold']} for optimal performance")
        
    else:
        print("❌ No pre-computed results found. Please run the full analysis first.")

if __name__ == "__main__":
    analyze_threshold_rewards_simple("dummy_path")
