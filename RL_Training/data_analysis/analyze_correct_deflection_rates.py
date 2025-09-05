#!/usr/bin/env python3
"""
Deflection rate analysis using correct PacketAction data

This script analyzes deflection rates across different thresholds using the
correct PacketAction signal (0=forward, 1=deflection) rather than out-of-order signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_deflection_rates():
    """Analyze deflection rates across all thresholds using PacketAction data"""
    
    thresholds = [15000, 25000, 37500, 50000]
    results = []
    
    print("=== DEFLECTION RATE ANALYSIS USING PACKET ACTION DATA ===\n")
    
    for threshold in thresholds:
        print(f"Analyzing threshold {threshold}:")
        
        # Load the merged data for this threshold (updated for new experiment)
        file_path = f"results_1G_thr_{threshold}/merged_final.csv"
        try:
            df = pd.read_csv(file_path)
            print(f"  Loaded {len(df)} observations")
            
            # Count deflections (action=1) vs forwards (action=0)
            deflection_count = (df['action'] == 1).sum()
            forward_count = (df['action'] == 0).sum() 
            total_count = len(df)
            deflection_rate = deflection_count / total_count if total_count > 0 else 0
            
            print(f"  Deflections: {deflection_count}")
            print(f"  Forwards: {forward_count}")
            print(f"  Total: {total_count}")
            print(f"  Deflection Rate: {deflection_rate:.4f} ({deflection_rate*100:.2f}%)")
            
            # Get unique flows (seq_num groups)
            unique_flows = df['seq_num'].nunique()
            print(f"  Unique flows: {unique_flows}")
            
            # Analyze deflection per flow
            flow_deflections = df.groupby('seq_num')['action'].agg(['sum', 'count']).reset_index()
            flow_deflections['deflection_rate'] = flow_deflections['sum'] / flow_deflections['count']
            flows_with_deflections = (flow_deflections['sum'] > 0).sum()
            
            print(f"  Flows with deflections: {flows_with_deflections}/{unique_flows}")
            
            results.append({
                'threshold': threshold,
                'deflection_count': deflection_count,
                'forward_count': forward_count,
                'total_count': total_count,
                'deflection_rate': deflection_rate,
                'unique_flows': unique_flows,
                'flows_with_deflections': flows_with_deflections
            })
            
            print()
            
        except FileNotFoundError:
            print(f"  File not found: {file_path}")
            print()
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    if not summary_df.empty:
        print("=== SUMMARY ACROSS ALL THRESHOLDS ===")
        print(summary_df.to_string(index=False))
        print()
        
        # Test hypothesis: lower thresholds should have higher deflection rates
        print("=== HYPOTHESIS TESTING ===")
        print("Expected: Lower thresholds → Higher deflection rates")
        print("Actual deflection rates:")
        for _, row in summary_df.iterrows():
            print(f"  Threshold {int(row['threshold']):5d}: {row['deflection_rate']:.4f} ({row['deflection_rate']*100:.2f}%)")
        
        # Check if pattern matches hypothesis
        deflection_rates = summary_df['deflection_rate'].tolist()
        is_decreasing = all(deflection_rates[i] >= deflection_rates[i+1] for i in range(len(deflection_rates)-1))
        
        print(f"\nHypothesis verification:")
        print(f"  Deflection rates decreasing with threshold: {'✓ YES' if is_decreasing else '✗ NO'}")
        
        # Calculate correlation
        correlation = np.corrcoef(summary_df['threshold'], summary_df['deflection_rate'])[0,1]
        print(f"  Correlation (threshold vs deflection_rate): {correlation:.3f}")
        
        if correlation > 0:
            print("  ⚠️  UNEXPECTED: Positive correlation (higher thresholds → more deflections)")
        else:
            print("  ✓ EXPECTED: Negative correlation (higher thresholds → fewer deflections)")
        
        # Compare with raw OMNeT++ data (updated for new experiment)
        print(f"\n=== COMPARISON WITH RAW OMNET++ DATA ===")
        print("Raw OMNeT++ deflection counts (from PacketAction files):")
        print("  Threshold 15000: 34 deflections")
        print("  Threshold 25000: 28 deflections") 
        print("  Threshold 37500: 19 deflections")
        print("  Threshold 50000: 12 deflections")
        print("\nProcessed dataset deflection counts:")
        for _, row in summary_df.iterrows():
            print(f"  Threshold {int(row['threshold']):5d}: {int(row['deflection_count']):4d} deflections")
        
        # Check data processing accuracy
        raw_counts = [34, 28, 19, 12]
        processed_counts = summary_df['deflection_count'].tolist()
        
        print(f"\nData processing accuracy:")
        for i, (raw, processed) in enumerate(zip(raw_counts, processed_counts)):
            threshold = int(summary_df.iloc[i]['threshold'])
            accuracy = processed / raw if raw > 0 else 0
            print(f"  Threshold {threshold}: {processed}/{raw} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return summary_df

if __name__ == "__main__":
    results = analyze_deflection_rates()
