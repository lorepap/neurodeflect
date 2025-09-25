#!/usr/bin/env python3
"""
Analyze deflection trends across different thresholds.
Verify hypothesis: Lower thresholds should lead to higher deflection counts.
"""

import pandas as pd
import numpy as np

def main():
    base_path = '/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims'
    # Load the dataset
    print("Loading combined_threshold_dataset.csv...")
    df = pd.read_csv(f'{base_path}/combined_threshold_dataset.csv')
    
    print(f"Dataset loaded: {len(df)} observations")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Check unique thresholds and their interpretation
    thresholds = sorted(df['deflection_threshold'].unique())
    print(f"Unique thresholds: {thresholds}")
    
    # Map normalized thresholds to actual byte values (based on 50000 byte buffer)
    # 0.3 = 15000, 0.5 = 25000, 0.75 = 37500, 1.0 = 50000
    threshold_mapping = {
        0.3: 15000,
        0.5: 25000, 
        0.75: 37500,
        1.0: 50000
    }
    
    print("Threshold mapping (normalized → bytes):")
    for norm, bytes_val in threshold_mapping.items():
        if norm in thresholds:
            print(f"  {norm} → {bytes_val} bytes")
    print()
    
    # Analyze deflection counts by threshold
    print("=" * 60)
    print("DEFLECTION ANALYSIS BY THRESHOLD")
    print("=" * 60)
    
    deflection_summary = []
    
    for threshold in sorted(thresholds):
        subset = df[df['deflection_threshold'] == threshold]
        
        # Basic metrics - using ACTION column correctly
        total_packets = len(subset)
        deflected_packets = len(subset[subset['action'] == 1])  # action=1 means DEFLECTION
        forwarded_packets = len(subset[subset['action'] == 0])  # action=0 means forward
        
        # Flow-level analysis
        unique_flows = subset['RequesterID'].nunique()
        flows_with_deflections = subset[subset['action'] == 1]['RequesterID'].nunique()
        
        # Rates and averages
        deflection_rate = deflected_packets / total_packets if total_packets > 0 else 0
        flow_deflection_rate = flows_with_deflections / unique_flows if unique_flows > 0 else 0
        # avg_packets_per_flow = subset['flow_packet_count'].mean()
        avg_fct = subset['FCT'].mean()
        
        # Out-of-order analysis (separate from deflection)
        total_ooo_packets = subset['ooo'].sum()
        avg_ooo_per_flow = subset['ooo'].mean()
        
        # Convert threshold to bytes for display
        threshold_bytes = threshold_mapping.get(threshold, f"{threshold}*50000")
        
        deflection_summary.append({
            'threshold_norm': threshold,
            'threshold_bytes': threshold_bytes,
            'total_packets': total_packets,
            'unique_flows': unique_flows,
            'deflected_packets': deflected_packets,
            'flows_with_deflections': flows_with_deflections,
            'forwarded_packets': forwarded_packets,
            'deflection_rate': deflection_rate,
            'flow_deflection_rate': flow_deflection_rate,
            'total_ooo_packets': total_ooo_packets,
            'avg_ooo_per_flow': avg_ooo_per_flow,
            # 'avg_packets_per_flow': avg_packets_per_flow,
            'avg_fct_ms': avg_fct * 1000  # Convert to milliseconds
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(deflection_summary)
    
    print("Summary by Threshold:")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"Threshold: {row['threshold_norm']} ({row['threshold_bytes']} bytes)")
        print(f"  Total packets: {row['total_packets']}")
        print(f"  Unique flows: {row['unique_flows']}")
        print(f"  Deflected packets: {row['deflected_packets']} ({row['deflection_rate']:.1%})")
        print(f"  Flows with deflections: {row['flows_with_deflections']} ({row['flow_deflection_rate']:.1%})")
        print(f"  Forwarded packets: {row['forwarded_packets']}")
        print(f"  Total OOO packets: {int(row['total_ooo_packets'])} ({row['avg_ooo_per_flow']:.2f} avg per flow)")
        # print(f"  Avg packets/flow: {row['avg_packets_per_flow']:.1f}")
        print(f"  Avg FCT: {row['avg_fct_ms']:.3f} ms")
        print()
    
    # Hypothesis verification
    print("=" * 60)
    print("HYPOTHESIS VERIFICATION")
    print("=" * 60)
    print("Expected: Lower thresholds → Higher deflection counts")
    print()
    
    # Check trend in deflection rates
    sorted_summary = summary_df.sort_values('threshold_norm')
    
    print("Deflection trends (threshold increasing):")
    print("-" * 50)
    
    prev_rate = None
    trend_violations = []
    
    for _, row in sorted_summary.iterrows():
        rate = row['deflection_rate']
        deflected_count = int(row['deflected_packets'])
        
        if prev_rate is not None:
            if rate > prev_rate:
                trend_violations.append((row['threshold_bytes'], rate, prev_rate))
                trend_indicator = "⬆️ INCREASE (unexpected)"
            else:
                trend_indicator = "⬇️ decrease (expected)"
        else:
            trend_indicator = "baseline"
        
        print(f"  {row['threshold_bytes']:>6} bytes: {rate:>6.1%} deflection rate, {deflected_count:>3} deflected packets {trend_indicator}")
        prev_rate = rate
    
    print()
    
    # Final verdict
    if not trend_violations:
        print("✅ HYPOTHESIS CONFIRMED!")
        print("   Deflection rates consistently decrease as thresholds increase")
    else:
        print("❌ HYPOTHESIS PARTIALLY VIOLATED!")
        print(f"   Found {len(trend_violations)} trend violations:")
        for threshold_bytes, curr_rate, prev_rate in trend_violations:
            print(f"   - {threshold_bytes} bytes: {curr_rate:.1%} > previous {prev_rate:.1%}")
    
    # Additional insights
    print()
    print("=" * 60)
    print("ADDITIONAL INSIGHTS")
    print("=" * 60)
    
    # Calculate correlation between threshold and deflection metrics
    corr_deflection_rate = summary_df['threshold_norm'].corr(summary_df['deflection_rate'])
    corr_flow_deflection = summary_df['threshold_norm'].corr(summary_df['flow_deflection_rate'])
    corr_total_ooo = summary_df['threshold_norm'].corr(summary_df['total_ooo_packets'])
    corr_avg_ooo = summary_df['threshold_norm'].corr(summary_df['avg_ooo_per_flow'])
    
    print(f"Correlations with threshold:")
    print(f"  Packet deflection rate: {corr_deflection_rate:.3f}")
    print(f"  Flow deflection rate: {corr_flow_deflection:.3f}")
    print(f"  Total OOO packets: {corr_total_ooo:.3f}")
    print(f"  Avg OOO per flow: {corr_avg_ooo:.3f}")
    print()
    
    # Performance impact analysis
    print("Performance Impact Analysis:")
    print("-" * 40)
    min_fct = summary_df['avg_fct_ms'].min()
    max_fct = summary_df['avg_fct_ms'].max()
    fct_increase = (max_fct - min_fct) / min_fct * 100
    
    print(f"FCT range: {min_fct:.3f} - {max_fct:.3f} ms")
    print(f"FCT increase: {fct_increase:.1f}% from lowest to highest threshold")
    
    # Find threshold with best trade-off (low deflection, good FCT)
    summary_df['fct_rank'] = summary_df['avg_fct_ms'].rank()
    summary_df['deflection_rank'] = summary_df['deflection_rate'].rank(ascending=False)  # Lower deflection is better
    summary_df['combined_score'] = summary_df['fct_rank'] + summary_df['deflection_rank']
    
    best_threshold = summary_df.loc[summary_df['combined_score'].idxmin()]
    print()
    print(f"Best trade-off threshold: {best_threshold['threshold_bytes']} bytes")
    print(f"  Packet deflection rate: {best_threshold['deflection_rate']:.1%}")
    print(f"  Flow deflection rate: {best_threshold['flow_deflection_rate']:.1%}")
    print(f"  Avg FCT: {best_threshold['avg_fct_ms']:.3f} ms")

if __name__ == "__main__":
    main()
