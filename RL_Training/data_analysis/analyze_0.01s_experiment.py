#!/usr/bin/env python3
"""
Deflection Rate Analysis for 0.01s Experiment

Analyze deflection rates using the 4 threshold levels (0.3, 0.5, 0.75, 1.0) 
relative to 50KB buffer capacity from the combined threshold dataset.
"""

import pandas as pd
import numpy as np

def analyze_deflection_rates_combined():
    """Analyze deflection rates from the combined threshold dataset"""
    
    print("=== DEFLECTION RATE ANALYSIS FOR 0.01s EXPERIMENT ===")
    print("Buffer capacity: 50KB (50000 bytes)")
    print("Thresholds: 0.3, 0.5, 0.75, 1.0 of buffer capacity\n")
    
    # Load the combined dataset
    df = pd.read_csv("combined_threshold_dataset_corrected.csv")
    
    print(f"Total observations loaded: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Thresholds in dataset: {sorted(df['deflection_threshold'].unique())}\n")
    
    # Analyze by threshold
    results = []
    
    print("=== THRESHOLD ANALYSIS ===")
    for threshold in sorted(df['deflection_threshold'].unique()):
        threshold_data = df[df['deflection_threshold'] == threshold]
        
        # Calculate metrics
        total_obs = len(threshold_data)
        deflections = (threshold_data['deflection'] == 1).sum()
        forwards = (threshold_data['deflection'] == 0).sum()
        deflection_rate = deflections / total_obs if total_obs > 0 else 0
        unique_flows = threshold_data['seq_num'].nunique()
        
        # Calculate threshold as fraction of buffer
        threshold_fraction = threshold / 50000
        
        print(f"Threshold {threshold} bytes ({threshold_fraction:.2f} of buffer):")
        print(f"  Total observations: {total_obs}")
        print(f"  Deflections: {deflections}")
        print(f"  Forwards: {forwards}")
        print(f"  Deflection rate: {deflection_rate:.4f} ({deflection_rate*100:.2f}%)")
        print(f"  Unique flows: {unique_flows}")
        
        # Flow-level analysis
        flow_stats = threshold_data.groupby('seq_num')['deflection'].agg(['sum', 'count']).reset_index()
        flows_with_deflections = (flow_stats['sum'] > 0).sum()
        print(f"  Flows with deflections: {flows_with_deflections}/{unique_flows} ({flows_with_deflections/unique_flows*100:.1f}%)")
        print()
        
        results.append({
            'threshold_bytes': threshold,
            'threshold_fraction': threshold_fraction,
            'total_obs': total_obs,
            'deflections': deflections,
            'deflection_rate': deflection_rate,
            'unique_flows': unique_flows,
            'flows_with_deflections': flows_with_deflections
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results)
    
    print("=== SUMMARY TABLE ===")
    print("Threshold  Fraction  Observations  Deflections  Rate      Flows  Flows_w_Deflections")
    print("-" * 85)
    for _, row in summary_df.iterrows():
        print(f"{int(row['threshold_bytes']):8d}  {row['threshold_fraction']:8.2f}  {int(row['total_obs']):11d}  {int(row['deflections']):10d}  {row['deflection_rate']:8.4f}  {int(row['unique_flows']):5d}  {int(row['flows_with_deflections']):15d}")
    
    # Hypothesis testing
    print("\n=== HYPOTHESIS VALIDATION ===")
    print("Expected: Lower thresholds → Higher deflection rates")
    print("(Lower fraction of buffer capacity should lead to more deflections)")
    print()
    
    print("Results by threshold fraction:")
    for _, row in summary_df.iterrows():
        print(f"  {row['threshold_fraction']:.2f} buffer: {row['deflection_rate']:.4f} ({row['deflection_rate']*100:.2f}%)")
    
    # Calculate correlation
    correlation = np.corrcoef(summary_df['threshold_fraction'], summary_df['deflection_rate'])[0,1]
    print(f"\nCorrelation (threshold_fraction vs deflection_rate): {correlation:.4f}")
    
    # Validate hypothesis
    rates = summary_df['deflection_rate'].tolist()
    is_decreasing = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))
    
    print(f"Deflection rates decreasing with threshold: {'✓ YES' if is_decreasing else '✗ NO'}")
    print(f"Expected correlation: Negative (higher threshold → fewer deflections)")
    print(f"Actual correlation: {correlation:.4f} ({'✓ CONFIRMED' if correlation < -0.5 else '? WEAK' if correlation < 0 else '✗ OPPOSITE'})")
    
    # Statistical significance
    if len(summary_df) >= 3:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            summary_df['threshold_fraction'], summary_df['deflection_rate']
        )
        print(f"Linear regression: R² = {r_value**2:.4f}, p-value = {p_value:.4f}")
        if p_value < 0.05:
            print("✓ Statistically significant relationship (p < 0.05)")
        else:
            print("? Not statistically significant (p ≥ 0.05)")
    
    # Deflection distribution analysis
    print("\n=== DEFLECTION DISTRIBUTION ===")
    total_deflections = summary_df['deflections'].sum()
    print(f"Total deflections across all thresholds: {total_deflections}")
    
    for _, row in summary_df.iterrows():
        pct = (row['deflections'] / total_deflections * 100) if total_deflections > 0 else 0
        print(f"  {row['threshold_fraction']:.2f} buffer: {int(row['deflections']):3d} deflections ({pct:.1f}%)")
    
    return summary_df

if __name__ == "__main__":
    try:
        results = analyze_deflection_rates_combined()
        print("\n=== ANALYSIS COMPLETE ===")
        print("✓ Successfully analyzed deflection rates for 0.01s experiment")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
