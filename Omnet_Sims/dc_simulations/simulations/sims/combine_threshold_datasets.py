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
    # Parse thresholds
    thresholds = [t.strip() for t in thresholds_str.split(',')]
    
    print(f"Combining datasets for thresholds: {thresholds}")
    
    # Collect all threshold datasets
    all_datasets = []
    successful_thresholds = []
    
    for threshold in thresholds:
        dataset_file = f"threshold_dataset_{threshold}.csv"
        
        if os.path.exists(dataset_file):
            print(f"Loading dataset for threshold {threshold}...")
            df = pd.read_csv(dataset_file)
            
            # Add deflection_threshold column
            df['deflection_threshold'] = float(threshold)
            
            all_datasets.append(df)
            successful_thresholds.append(threshold)
            
            print(f"✓ Loaded {len(df):,} rows for threshold {threshold}")
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
    for threshold, count in threshold_counts.items():
        print(f"  Threshold {threshold}: {count:,} rows")
    
    print(f"\nSample data (first 5 rows):")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(combined_df.head())
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 combine_threshold_datasets.py <comma_separated_thresholds>")
        print("Example: python3 combine_threshold_datasets.py 0.3,0.5,0.7,0.9")
        sys.exit(1)
    
    thresholds_str = sys.argv[1]
    success = combine_threshold_datasets(thresholds_str)
    sys.exit(0 if success else 1)
