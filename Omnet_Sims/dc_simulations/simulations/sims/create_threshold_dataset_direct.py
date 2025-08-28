#!/usr/bin/env python3
"""
Direct threshold dataset creation following the pattern from create_dataset.py
Uses scavetool directly on the threshold simulation files to create combined dataset.
"""

import pandas as pd
import os
import subprocess
import tempfile
import shutil
import sys

def extract_data_direct(threshold, output_dir):
    """
    Extract data directly from threshold simulation files using scavetool
    """
    threshold_dir = f"results_backup/threshold_{threshold}"
    
    if not os.path.exists(threshold_dir):
        print(f"✗ Threshold directory {threshold_dir} not found")
        return None
    
    # Find the .vec file for this threshold
    vec_files = [f for f in os.listdir(threshold_dir) if f.endswith('.vec')]
    if not vec_files:
        print(f"✗ No .vec files found in {threshold_dir}")
        return None
    
    vec_file = os.path.join(threshold_dir, vec_files[0])
    
    print(f"Processing {vec_file}...")
    
    # Create temporary directory for extraction
    temp_dir = os.path.join(output_dir, f"temp_threshold_{threshold}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define the data we need to extract (following create_dataset.py pattern)
    extractions = [
        ("QueueLen:vector", "queue_len.csv"),
        ("QueuesTotLen:vector", "total_queue_len.csv"), 
        ("QueueCapacity:vector", "queue_capacity.csv"),
        ("QueuesTotCapacity:vector", "total_queue_capacity.csv"),
        ("switchSeqNum:vector", "seq_num.csv"),
        ("switchTtl:vector", "ttl.csv"),
        ("actionSeqNum:vector", "action_seq_num.csv"),
        ("PacketAction:vector", "packet_action.csv")
    ]
    
    extracted_files = {}
    
    for vector_name, output_file in extractions:
        output_path = os.path.join(temp_dir, output_file)
        
        # Use scavetool to extract the data
        cmd = [
            'scavetool', 'x', '--type', 'v', 
            '--filter', f'module(LeafSpine1G) AND "{vector_name}"',
            '-o', output_path, '-F', 'CSV-S', vec_file
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                extracted_files[vector_name] = output_path
                print(f"  ✓ Extracted {vector_name}")
            else:
                print(f"  ✗ No data for {vector_name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to extract {vector_name}: {e}")
    
    return extracted_files, temp_dir

def create_threshold_dataset(threshold, extracted_files):
    """
    Create dataset for a single threshold following create_dataset.py pattern
    """
    try:
        # Load the main data files following the create_dataset.py pattern
        datasets = {}
        
        # Load each extracted file
        for vector_name, file_path in extracted_files.items():
            if os.path.exists(file_path):
                df = pd.read_csv(file_path, header=None, names=['timestamp', 'value'])
                datasets[vector_name] = df
                print(f"  Loaded {len(df)} rows from {vector_name}")
        
        if not datasets:
            print(f"✗ No data loaded for threshold {threshold}")
            return None
        
        # Create the combined dataset following create_dataset.py pattern
        # Start with the first dataset (typically queue length)
        base_key = list(datasets.keys())[0]
        combined_df = datasets[base_key].copy()
        combined_df = combined_df.rename(columns={'value': base_key.replace(':vector', '')})
        
        # Add other columns by timestamp alignment (simplified approach)
        for vector_name, df in datasets.items():
            if vector_name != base_key:
                col_name = vector_name.replace(':vector', '')
                # Simple join by index (assuming similar timestamps)
                if len(df) == len(combined_df):
                    combined_df[col_name] = df['value'].values
                else:
                    print(f"  Warning: Size mismatch for {vector_name}")
        
        # Add threshold column
        combined_df['deflection_threshold'] = float(threshold)
        
        print(f"  ✓ Created dataset with {len(combined_df)} rows and {len(combined_df.columns)} columns")
        return combined_df
        
    except Exception as e:
        print(f"✗ Error creating dataset for threshold {threshold}: {e}")
        return None

def main():
    """
    Main function to create combined threshold dataset
    """
    print("="*70)
    print("Direct Threshold Dataset Creation")
    print("Following create_dataset.py pattern with direct scavetool extraction")
    print("="*70)
    
    thresholds = ['0.3', '0.5', '0.7', '0.9']
    all_datasets = []
    successful_thresholds = []
    
    # Create working directory
    work_dir = "threshold_work"
    os.makedirs(work_dir, exist_ok=True)
    
    try:
        for threshold in thresholds:
            print(f"\nProcessing threshold {threshold}...")
            
            # Extract data for this threshold
            result = extract_data_direct(threshold, work_dir)
            if result is None:
                continue
                
            extracted_files, temp_dir = result
            
            if not extracted_files:
                print(f"✗ No data extracted for threshold {threshold}")
                continue
            
            # Create dataset for this threshold
            df = create_threshold_dataset(threshold, extracted_files)
            if df is not None:
                all_datasets.append(df)
                successful_thresholds.append(threshold)
            
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        if not all_datasets:
            print("✗ No threshold datasets were successfully created")
            return 1
        
        print(f"\n✓ Successfully processed thresholds: {successful_thresholds}")
        
        # Combine all datasets
        print("\nCombining all threshold datasets...")
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Reorder columns to put deflection_threshold near the front
        cols = combined_df.columns.tolist()
        if 'deflection_threshold' in cols:
            cols.remove('deflection_threshold')
            # Insert after timestamp
            if 'timestamp' in cols:
                timestamp_idx = cols.index('timestamp')
                cols.insert(timestamp_idx + 1, 'deflection_threshold')
            else:
                cols.insert(0, 'deflection_threshold')
            combined_df = combined_df[cols]
        
        # Save the combined dataset
        output_file = "combined_threshold_dataset.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Combined dataset created: {output_file}")
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
        
        return 0
        
    finally:
        # Clean up working directory
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
    
    print("\n" + "="*70)
    print("Dataset creation completed!")
    print("="*70)

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
