#!/usr/bin/env python3
"""
Rename Occupancy Columns for Clarity

This script renames the occupancy-related columns in the dataset to make their meaning clearer:
- occupancy -> forward_port_occupancy
- total_occupancy -> switch_total_occupancy
- capacity -> forward_port_capacity
- total_capacity -> switch_total_capacity

This clarifies that:
- forward_port_* refers to the intended destination interface (where packet should go)
- switch_total_* refers to the aggregate across all interfaces of the current switch
"""

import pandas as pd
import os

def rename_occupancy_columns(input_file, output_file=None):
    """
    Rename occupancy columns for clarity
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input)
    """
    
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original dataset shape: {df.shape}")
    print("Original columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    print()
    
    # Define column renaming mapping
    column_mapping = {
        'occupancy': 'forward_port_occupancy',
        'total_occupancy': 'switch_total_occupancy', 
        'capacity': 'forward_port_capacity',
        'total_capacity': 'switch_total_capacity'
    }
    
    # Apply renaming
    df_renamed = df.rename(columns=column_mapping)
    
    print("Column renaming applied:")
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            print(f"  ✓ {old_name} -> {new_name}")
        else:
            print(f"  ⚠ {old_name} not found in dataset")
    print()
    
    print("New columns:")
    for i, col in enumerate(df_renamed.columns, 1):
        print(f"  {i:2d}. {col}")
    print()
    
    # Determine output file
    if output_file is None:
        output_file = input_file
        print(f"Overwriting original file: {output_file}")
    else:
        print(f"Saving to new file: {output_file}")
    
    # Save the renamed dataset
    df_renamed.to_csv(output_file, index=False)
    
    print(f"✅ Dataset saved with renamed columns!")
    print(f"   Rows: {len(df_renamed):,}")
    print(f"   Columns: {len(df_renamed.columns)}")
    
    # Show sample of renamed data
    print("\nSample of renamed dataset:")
    print(df_renamed.head(3)[['timestamp', 'forward_port_occupancy', 'switch_total_occupancy', 
                              'forward_port_capacity', 'switch_total_capacity', 'action']].to_string())
    
    return df_renamed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rename occupancy columns for clarity')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', help='Output CSV file path (default: overwrite input)')
    parser.add_argument('--backup', '-b', action='store_true', 
                        help='Create backup of original file before overwriting')
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup and args.output is None:
        backup_file = args.input_file + '.backup'
        print(f"Creating backup: {backup_file}")
        import shutil
        shutil.copy2(args.input_file, backup_file)
    
    # Rename columns
    rename_occupancy_columns(args.input_file, args.output)
