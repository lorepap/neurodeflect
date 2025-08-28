#!/usr/bin/env python3
"""
Combine threshold datasets following the same pattern as create_dataset.py
This script combines the final processed datasets from each threshold value.
"""

import sys
import pandas as pd
import glob
import os

def combine_threshold_datasets(thresholds_str):
    """
    Combine all threshold datasets into a single dataset with deflection_threshold feature
    """
    # Parse thresholds (handle both comma and space separated)
    if ',' in thresholds_str:
        thresholds = [t.strip() for t in thresholds_str.split(',')]
    else:
        thresholds = [t.strip() for t in thresholds_str.split()]
    
    print(f"Combining datasets for thresholds: {thresholds}")
    
    # Buffer capacity in bytes (from omnetpp_1G.ini configuration)
    BUFFER_CAPACITY_BYTES = 50000
    
    # Collect all threshold datasets
    all_datasets = []
    successful_thresholds = []
    
    for threshold in thresholds:
        dataset_file = f"threshold_dataset_{threshold}.csv"
        
        if os.path.exists(dataset_file):
            print(f"Loading dataset for threshold {threshold}...")
            df = pd.read_csv(dataset_file)
            
            # Convert byte threshold to percentage of buffer capacity
            threshold_bytes = float(threshold)
            threshold_percentage = threshold_bytes / BUFFER_CAPACITY_BYTES
            
            # Add deflection_threshold column as percentage
            df['deflection_threshold'] = threshold_percentage
            
            all_datasets.append(df)
            successful_thresholds.append(threshold)
            
            print(f"✓ Loaded {len(df):,} rows for threshold {threshold} bytes ({threshold_percentage:.1%} of buffer)")
        else:
            print(f"✗ Dataset file {dataset_file} not found for threshold {threshold}")
    
    if not all_datasets:
        print("✗ No threshold datasets found to combine")
        return False
    
    # Combine all datasets
    print(f"\nCombining datasets from thresholds: {successful_thresholds}")
    combined_df = pd.concat(all_datasets, ignore_index=True)
    
    # Reorder columns to put deflection_threshold after timestamp
    cols = combined_df.columns.tolist()
    if 'deflection_threshold' in cols:
        cols.remove('deflection_threshold')
        # Insert deflection_threshold after timestamp if it exists
        if 'timestamp' in cols:
            timestamp_idx = cols.index('timestamp')
            cols.insert(timestamp_idx + 1, 'deflection_threshold')
        else:
            cols.insert(0, 'deflection_threshold')
        combined_df = combined_df[cols]
    
    # Save combined dataset
    output_file = "combined_threshold_dataset.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ Combined dataset saved: {output_file}")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Total columns: {len(combined_df.columns)}")
    print(f"Columns: {list(combined_df.columns)}")
    
    print(f"\nThreshold distribution:")
    threshold_counts = combined_df['deflection_threshold'].value_counts().sort_index()
    
    # Create a mapping from percentage back to bytes for display
    BUFFER_CAPACITY_BYTES = 50000
    for threshold_pct, count in threshold_counts.items():
        threshold_bytes = int(threshold_pct * BUFFER_CAPACITY_BYTES)
        print(f"  Threshold {threshold_pct:.1f} ({threshold_bytes} bytes, {threshold_pct:.1%}): {count:,} rows")
    
    print(f"\nSample data (first 5 rows):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(combined_df.head())
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 combine_threshold_datasets.py <space_or_comma_separated_thresholds_in_bytes>")
        print("Example: python3 combine_threshold_datasets.py \"15000 25000 35000 45000 50000\"")
        print("Example: python3 combine_threshold_datasets.py \"15000,25000,35000,45000,50000\"")
        print("Note: Byte values will be converted to percentages based on 50,000 byte buffer capacity")
        sys.exit(1)
    
    thresholds_str = sys.argv[1]
    success = combine_threshold_datasets(thresholds_str)
    sys.exit(0 if success else 1)
