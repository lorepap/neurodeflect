#!/usr/bin/env python3
"""
Final Data Discrepancy Analysis

This script provides a comprehensive analysis of the data discrepancy discovered
during deflection threshold experiments, showing both the processed dataset
and the raw extracted data to identify simulation version differences.
"""

import pandas as pd
import os

base_path = '/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims'

def analyze_data_versions():
    """Analyze both processed and raw data versions"""
    
    print("=" * 80)
    print("FINAL DATA DISCREPANCY ANALYSIS")
    print("=" * 80)
    
    # Analyze processed data (used in combined dataset)
    print("\n1. PROCESSED DATA ANALYSIS (used in combined dataset)")
    print("-" * 60)
    
    # Use normalized threshold values that actually exist in the dataset
    thresholds = [0.3, 0.5, 0.75, 1.0]
    # Map to byte values for display
    threshold_byte_mapping = {
        0.3: 15000,
        0.5: 25000, 
        0.75: 37500,
        1.0: 50000
    }
    processed_results = {}
    
    # Load the combined dataset once
    combined_file_path = os.path.join(base_path, "combined_threshold_dataset.csv")
    if not os.path.exists(combined_file_path):
        raise FileNotFoundError(f"Combined threshold dataset not found: {combined_file_path}")
    
    try:
        combined_df = pd.read_csv(combined_file_path)
        
        for threshold in thresholds:
            # Filter data for current threshold using normalized values
            df = combined_df[combined_df['deflection_threshold'] == threshold]
            
            if len(df) > 0:
                deflections = (df['action'] == 1).sum()
                total = len(df)
                rate = deflections / total
                
                processed_results[threshold] = {
                    'total': total,
                    'deflections': deflections,
                    'rate': rate
                }
                
                threshold_bytes = threshold_byte_mapping[threshold]
                print(f"Threshold {threshold} ({threshold_bytes:5d} bytes): {total:6d} obs, {deflections:4d} deflections ({rate:.4f})")
            else:
                print(f"Threshold {threshold}: NO DATA FOUND in combined dataset")
                processed_results[threshold] = {'total': 0, 'deflections': 0, 'rate': 0}
    
    except FileNotFoundError:
        print(f"Combined dataset file not found")
        for threshold in thresholds:
            processed_results[threshold] = {'total': 0, 'deflections': 0, 'rate': 0}
    
    # Analyze raw extracted data
    print("\n2. RAW EXTRACTED DATA ANALYSIS")
    print("-" * 60)
    
    raw_results = {}
    
    # Map normalized thresholds to their simulation IDs and byte values
    sim_ids = {
        0.3: ("688b2599", 15000),
        0.5: ("b7ca6d59", 25000), 
        0.75: ("99ffc3ea", 37500),
        1.0: ("c2a2613f", 50000)
    }
    
    for threshold in thresholds:
        sim_id, threshold_bytes = sim_ids.get(threshold, (None, None))
        if sim_id is None:
            continue
            
        raw_file = os.path.join(base_path, f"extracted_results/PACKET_ACTION/sim_{sim_id}_{threshold_bytes}_deflection_threshold_{threshold_bytes}.csv")
        
        try:
            # Count total lines and deflections in raw CSV-R format
            with open(raw_file, 'r') as f:
                lines = f.readlines()
                total = len(lines) - 1  # Exclude header
                deflections = sum(1 for line in lines[1:] if line.strip().split(',')[1] == '1')
                rate = deflections / total if total > 0 else 0
                
                raw_results[threshold] = {
                    'total': total,
                    'deflections': deflections,
                    'rate': rate
                }
                
                print(f"Threshold {threshold} ({threshold_bytes:5d} bytes): {total:6d} obs, {deflections:4d} deflections ({rate:.4f})")
                
        except FileNotFoundError:
            threshold_bytes = threshold_byte_mapping[threshold]
            print(f"Threshold {threshold} ({threshold_bytes:5d} bytes): FILE NOT FOUND")
            raw_results[threshold] = {'total': 0, 'deflections': 0, 'rate': 0}
    
    # Compare the two datasets
    print("\n3. DATA VERSION COMPARISON")
    print("-" * 60)
    print(f"{'Threshold':<15} {'Processed':<15} {'Raw':<15} {'Match':<10} {'Data Loss':<15}")
    print("-" * 70)
    
    for threshold in thresholds:
        p_def = processed_results[threshold]['deflections']
        r_def = raw_results[threshold]['deflections']
        p_tot = processed_results[threshold]['total']
        r_tot = raw_results[threshold]['total']
        
        match = "YES" if p_def == r_def and p_tot == r_tot else "NO"
        
        if r_tot > 0:
            data_loss = f"{(1 - p_tot/r_tot)*100:.1f}%"
        else:
            data_loss = "N/A"
        
        threshold_bytes = threshold_byte_mapping[threshold]
        threshold_str = f"{threshold} ({threshold_bytes})"
        print(f"{threshold_str:<15} {p_def}/{p_tot:<15} {r_def}/{r_tot:<15} {match:<10} {data_loss:<15}")
    
    # Show deflection rate trend
    print("\n4. DEFLECTION RATE TRENDS")
    print("-" * 60)
    
    print("Processed data deflection rates:")
    for threshold in sorted(thresholds):
        rate = processed_results[threshold]['rate']
        threshold_bytes = threshold_byte_mapping[threshold]
        print(f"  {threshold} ({threshold_bytes:5d}): {rate:.4f} ({rate*100:.2f}%)")
    
    print("\nRaw data deflection rates:")
    for threshold in sorted(thresholds):
        rate = raw_results[threshold]['rate']
        threshold_bytes = threshold_byte_mapping[threshold]
        print(f"  {threshold} ({threshold_bytes:5d}): {rate:.4f} ({rate*100:.2f}%)")
    
    # Calculate correlations
    print("\n5. HYPOTHESIS VALIDATION")
    print("-" * 60)
    
    # Processed data correlation
    proc_thresholds = [t for t in thresholds if processed_results[t]['total'] > 0]
    proc_rates = [processed_results[t]['rate'] for t in proc_thresholds]
    
    if len(proc_rates) >= 2:
        proc_corr = pd.Series(proc_thresholds).corr(pd.Series(proc_rates))
        print(f"Processed data correlation (threshold vs deflection rate): {proc_corr:.4f}")
    
    # Raw data correlation
    raw_thresholds = [t for t in thresholds if raw_results[t]['total'] > 0]
    raw_rates = [raw_results[t]['rate'] for t in raw_thresholds]
    
    if len(raw_rates) >= 2:
        raw_corr = pd.Series(raw_thresholds).corr(pd.Series(raw_rates))
        print(f"Raw data correlation (threshold vs deflection rate): {raw_corr:.4f}")
    
    print("\nExpected: Negative correlation (higher threshold → lower deflection rate)")

if __name__ == "__main__":
    analyze_data_versions()
