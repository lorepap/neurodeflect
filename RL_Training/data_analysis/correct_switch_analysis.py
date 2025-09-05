#!/usr/bin/env python3
"""
Correct analysis of switch-level packet observations with flow statistics.
Each row = one packet traversing a switch, with flow-level stats merged in.
"""

import pandas as pd
import numpy as np

def main():
    # Load the dataset
    print("SWITCH-LEVEL PACKET OBSERVATION ANALYSIS")
    print("=" * 70)
    print("Data structure: Each row = packet traversing switch + flow stats")
    print()
    
    df = pd.read_csv('combined_threshold_dataset.csv')
    
    print(f"Total switch-level packet observations: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Check unique thresholds
    thresholds = sorted(df['deflection_threshold'].unique())
    print(f"Unique thresholds: {thresholds}")
    
    # Map to actual byte values
    threshold_mapping = {0.3: 15000, 0.5: 25000, 0.75: 37500, 1.0: 50000}
    print("Threshold mapping:")
    for norm, bytes_val in threshold_mapping.items():
        if norm in thresholds:
            print(f"  {norm} → {bytes_val} bytes")
    print()
    
    print("=" * 70)
    print("DETAILED ANALYSIS BY THRESHOLD")
    print("=" * 70)
    
    analysis_results = []
    
    for threshold in sorted(thresholds):
        subset = df[df['deflection_threshold'] == threshold]
        threshold_bytes = threshold_mapping.get(threshold, int(threshold * 50000))
        
        print(f"\nThreshold: {threshold} ({threshold_bytes} bytes)")
        print("-" * 50)
        
        # Switch-level packet observations
        total_observations = len(subset)
        deflected_observations = len(subset[subset['ooo'] > 0])
        total_ooo_events = subset['ooo'].sum()
        
        print(f"Switch-level packet observations: {total_observations}")
        print(f"Observations with deflection (ooo > 0): {deflected_observations}")
        print(f"Total OOO events across all observations: {int(total_ooo_events)}")
        print(f"Deflection rate: {deflected_observations/total_observations:.1%}")
        
        # Flow-level insights
        unique_flows = subset['FlowID'].nunique()
        flows_with_deflection = subset[subset['ooo'] > 0]['FlowID'].nunique()
        
        print(f"\nFlow-level insights:")
        print(f"  Unique flows represented: {unique_flows}")
        print(f"  Flows experiencing deflection: {flows_with_deflection}")
        print(f"  Flow deflection rate: {flows_with_deflection/unique_flows:.1%}")
        
        # Switch-level insights
        unique_switches = subset['SWITCH_ID'].nunique()
        switches_with_deflection = subset[subset['ooo'] > 0]['SWITCH_ID'].nunique()
        
        print(f"\nSwitch-level insights:")
        print(f"  Unique switches: {unique_switches}")
        print(f"  Switches observing deflection: {switches_with_deflection}")
        
        # Packet size distribution (from flow stats)
        avg_packets_per_flow = subset['packet_count'].mean()
        print(f"\nFlow size distribution:")
        print(f"  Average packets per flow: {avg_packets_per_flow:.1f}")
        print(f"  Flow size range: {subset['packet_count'].min()}-{subset['packet_count'].max()} packets")
        
        # Performance metrics
        avg_fct = subset['FCT'].mean() * 1000  # Convert to ms
        print(f"\nPerformance:")
        print(f"  Average FCT: {avg_fct:.3f} ms")
        
        # Deflection details
        if deflected_observations > 0:
            print(f"\nDeflection details:")
            deflected_subset = subset[subset['ooo'] > 0]
            
            # Group by flow to see deflection patterns
            deflection_by_flow = deflected_subset.groupby('FlowID').agg({
                'ooo': ['count', 'sum'],
                'SWITCH_ID': lambda x: len(set(x)),  # unique switches
                'packet_count': 'first',  # flow size (should be same for all rows of same flow)
                'FCT': 'first'
            }).round(3)
            
            deflection_by_flow.columns = ['deflected_observations', 'total_ooo_events', 'switches_hit', 'flow_size', 'fct']
            
            print(f"  Deflected flows breakdown:")
            for flow_id, row in deflection_by_flow.iterrows():
                fct_ms = row['fct'] * 1000
                print(f"    FlowID {flow_id}: {row['deflected_observations']} deflected observations, "
                      f"{row['total_ooo_events']} OOO events, {row['switches_hit']} switches, "
                      f"{row['flow_size']} packets total, FCT={fct_ms:.3f}ms")
        
        analysis_results.append({
            'threshold': threshold,
            'threshold_bytes': threshold_bytes,
            'total_observations': total_observations,
            'deflected_observations': deflected_observations,
            'deflection_rate': deflected_observations/total_observations,
            'total_ooo_events': int(total_ooo_events),
            'unique_flows': unique_flows,
            'flows_with_deflection': flows_with_deflection,
            'avg_fct_ms': avg_fct
        })
    
    print("\n" + "=" * 70)
    print("HYPOTHESIS VERIFICATION")
    print("=" * 70)
    print("Expected: Lower thresholds → Higher deflection rates")
    print("(More packets should be deflected when buffers are smaller)")
    print()
    
    # Check trend in deflection rates
    print("Deflection trends (threshold increasing):")
    print("-" * 60)
    
    prev_rate = None
    trend_violations = []
    
    for result in analysis_results:
        rate = result['deflection_rate']
        
        if prev_rate is not None:
            if rate > prev_rate:
                trend_violations.append((result['threshold_bytes'], rate, prev_rate))
                trend_indicator = "⬆️ INCREASE (violation)"
            else:
                trend_indicator = "⬇️ decrease (expected)"
        else:
            trend_indicator = "baseline"
        
        print(f"  {result['threshold_bytes']:>6} bytes: {rate:>6.1%} deflection rate "
              f"({result['deflected_observations']:>2}/{result['total_observations']:>2} observations) {trend_indicator}")
        prev_rate = rate
    
    print()
    
    # Final verdict
    if not trend_violations:
        print("✅ HYPOTHESIS CONFIRMED!")
        print("   Deflection rates consistently decrease as thresholds increase")
        print("   Smaller buffers (lower thresholds) → More deflections")
    else:
        print("❌ HYPOTHESIS PARTIALLY VIOLATED!")
        print(f"   Found {len(trend_violations)} trend violations:")
        for threshold_bytes, curr_rate, prev_rate in trend_violations:
            print(f"   - {threshold_bytes} bytes: {curr_rate:.1%} > previous {prev_rate:.1%}")
    
    print()
    print("=" * 70)
    print("NETWORK BEHAVIOR ANALYSIS")
    print("=" * 70)
    
    # Analyze deflection patterns across switches and flows
    print("Switch utilization analysis:")
    switch_analysis = df.groupby(['deflection_threshold', 'SWITCH_ID']).agg({
        'ooo': ['count', 'sum'],
        'FlowID': 'nunique'
    }).round(3)
    
    for threshold in thresholds:
        threshold_bytes = threshold_mapping.get(threshold, int(threshold * 50000))
        print(f"\nThreshold {threshold_bytes} bytes:")
        
        subset = df[df['deflection_threshold'] == threshold]
        switch_stats = subset.groupby('SWITCH_ID').agg({
            'ooo': ['count', 'sum'],
            'FlowID': 'nunique'
        })
        switch_stats.columns = ['observations', 'total_ooo', 'unique_flows']
        
        for switch_id, row in switch_stats.iterrows():
            deflection_rate = row['total_ooo'] / row['observations'] if row['observations'] > 0 else 0
            print(f"  Switch {switch_id}: {row['observations']} observations, "
                  f"{row['total_ooo']} OOO events ({deflection_rate:.1%}), {row['unique_flows']} flows")
    
    print()
    print("=" * 70)
    print("SUMMARY INSIGHTS")
    print("=" * 70)
    
    # Calculate correlations
    results_df = pd.DataFrame(analysis_results)
    corr_deflection = results_df['threshold'].corr(results_df['deflection_rate'])
    corr_ooo = results_df['threshold'].corr(results_df['total_ooo_events'])
    
    print(f"Correlations with threshold:")
    print(f"  Deflection rate: {corr_deflection:.3f}")
    print(f"  Total OOO events: {corr_ooo:.3f}")
    print()
    
    if corr_deflection < 0:
        print("✅ Overall correlation supports hypothesis")
        print("   Higher thresholds correlate with lower deflection rates")
    else:
        print("❌ Overall correlation contradicts hypothesis")
        print("   Higher thresholds correlate with higher deflection rates")
    
    print()
    print("Key observations:")
    print("- Each row represents a packet's journey through a switch")
    print("- Flow-level statistics (FCT, packet_count) are merged for context")
    print("- OOO events indicate deflection routing decisions")
    print("- Switch-level deflection patterns reveal network congestion behavior")

if __name__ == "__main__":
    main()
