#!/usr/bin/env python3
"""
Enhanced dataset creation script for deflection threshold variation experiments.
This script extends the original create_dataset.py to handle threshold-specific data collection.

Usage:
    python3 create_threshold_dataset.py --threshold 0.5 --base_folders results/threshold_0.5
    
Or to combine all thresholds:
    python3 create_threshold_dataset.py --combine_all_thresholds
"""

import argparse
import glob
import os
import pandas as pd
import sys


def first_csv_in(folder: str) -> str:
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV file found in {folder}")
    return files[0]


def zip_first_six(folders, names):
    """
    Returns a DataFrame with 1 + len(folders) columns:
        timestamp, <name1>, <name2>, ..., <nameN>
    """
    # csv0: timestamp + first value
    csv0 = first_csv_in(folders[0])
    df = pd.read_csv(
        csv0,
        skiprows=1,
        header=None,
        names=['timestamp', names[0]],
        dtype={'timestamp': 'float64'}
    )

    # subsequent files: only value column
    for folder, colname in zip(folders[1:], names[1:]):
        csv_file = first_csv_in(folder)
        df_temp = pd.read_csv(
            csv_file,
            skiprows=1,
            header=None,
            names=['timestamp', colname],
            dtype={'timestamp': 'float64'}
        )
        # Add only the value column (not timestamp)
        df[colname] = df_temp[colname]

    return df


def zip_last_two(folder1, name1, folder2, name2):
    """
    Returns a DataFrame with columns:
        timestamp, <name1>, <name2>
    """
    csv1 = first_csv_in(folder1)
    csv2 = first_csv_in(folder2)
    
    df1 = pd.read_csv(
        csv1,
        skiprows=1,
        header=None,
        names=['timestamp', name1],
        dtype={'timestamp': 'float64'}
    )
    
    df2 = pd.read_csv(
        csv2,
        skiprows=1,
        header=None,
        names=['timestamp', name2],
        dtype={'timestamp': 'float64'}
    )
    
    # Add the second column from df2
    df1[name2] = df2[name2]
    
    return df1


def create_dataset_for_threshold(threshold, base_path, output_path):
    """
    Create a dataset for a specific threshold value.
    """
    print(f"Creating dataset for threshold {threshold}...")
    
    # Define folder paths for this threshold
    threshold_path = os.path.join(base_path, f"threshold_{threshold}")
    
    if not os.path.exists(threshold_path):
        print(f"Warning: Path {threshold_path} does not exist. Skipping threshold {threshold}")
        return None
    
    # Default folder structure (update paths as needed)
    first_folders = [
        os.path.join(threshold_path, "QUEUE_CAPACITY"),
        os.path.join(threshold_path, "QUEUES_TOT_CAPACITY"), 
        os.path.join(threshold_path, "QUEUE_LEN"),
        os.path.join(threshold_path, "QUEUES_TOT_LEN"),
        os.path.join(threshold_path, "SWITCH_SEQ_NUM"),
        os.path.join(threshold_path, "TTL")
    ]
    
    first_names = ["capacity", "total_capacity", "occupancy", "total_occupancy", "seq_num", "ttl"]
    
    # Check if all required folders exist
    missing_folders = [f for f in first_folders if not os.path.exists(f)]
    if missing_folders:
        print(f"Warning: Missing folders for threshold {threshold}: {missing_folders}")
        return None
    
    try:
        # Process first six folders
        df_first = zip_first_six(first_folders, first_names)
        
        # Process last two folders
        action_seq_folder = os.path.join(threshold_path, "ACTION_SEQ_NUM")
        packet_action_folder = os.path.join(threshold_path, "PACKET_ACTION")
        
        if os.path.exists(action_seq_folder) and os.path.exists(packet_action_folder):
            df_second = zip_last_two(action_seq_folder, "seq_num", packet_action_folder, "action")
            
            # Join on timestamp and seq_num
            join_keys = ['timestamp', 'seq_num']
            merged = pd.merge(df_first, df_second[['timestamp', 'seq_num', 'action']], 
                            on=join_keys, how='left')
            merged['action'] = merged['action'].fillna(2).astype('int32')
        else:
            # If action data is not available, just use the first dataset
            merged = df_first
            merged['action'] = 2  # Default action value
        
        # Add threshold information to the dataset
        merged['deflection_threshold'] = threshold
        
        # Save the dataset
        threshold_output = f"{output_path}_threshold_{threshold}.csv"
        merged.to_csv(threshold_output, index=False)
        print(f"Dataset for threshold {threshold} saved to {threshold_output}")
        
        return merged
        
    except Exception as e:
        print(f"Error creating dataset for threshold {threshold}: {e}")
        return None


def combine_all_thresholds(base_path, output_path):
    """
    Combine datasets from all available thresholds into a single dataset.
    """
    print("Combining datasets from all thresholds...")
    
    # Find all threshold directories
    threshold_dirs = glob.glob(os.path.join(base_path, "threshold_*"))
    thresholds = []
    
    for dir_path in threshold_dirs:
        dir_name = os.path.basename(dir_path)
        if dir_name.startswith("threshold_"):
            threshold_value = dir_name.replace("threshold_", "")
            thresholds.append(threshold_value)
    
    if not thresholds:
        print(f"No threshold directories found in {base_path}")
        return
    
    print(f"Found thresholds: {sorted(thresholds)}")
    
    combined_dfs = []
    
    for threshold in sorted(thresholds):
        df = create_dataset_for_threshold(threshold, base_path, output_path)
        if df is not None:
            combined_dfs.append(df)
    
    if combined_dfs:
        # Combine all dataframes
        final_df = pd.concat(combined_dfs, ignore_index=True)
        
        # Sort by threshold and timestamp for better organization
        final_df = final_df.sort_values(['deflection_threshold', 'timestamp']).reset_index(drop=True)
        
        # Save combined dataset
        combined_output = f"{output_path}_all_thresholds_combined.csv"
        final_df.to_csv(combined_output, index=False)
        print(f"Combined dataset saved to {combined_output}")
        
        # Print summary statistics
        print("\nDataset Summary:")
        print(f"Total records: {len(final_df)}")
        print("Records per threshold:")
        print(final_df.groupby('deflection_threshold').size())
        
    else:
        print("No datasets were successfully created.")


def main():
    parser = argparse.ArgumentParser(description="Create datasets for deflection threshold experiments")
    
    parser.add_argument("--threshold", type=float, help="Specific threshold to process")
    parser.add_argument("--base_path", default="results", help="Base path containing threshold directories")
    parser.add_argument("--output", default="deflection_threshold_dataset", help="Output file prefix")
    parser.add_argument("--combine_all_thresholds", action="store_true", 
                       help="Combine all thresholds into a single dataset")
    
    args = parser.parse_args()
    
    if args.combine_all_thresholds:
        combine_all_thresholds(args.base_path, args.output)
    elif args.threshold is not None:
        create_dataset_for_threshold(args.threshold, args.base_path, args.output)
    else:
        print("Error: Must specify either --threshold or --combine_all_thresholds")
        sys.exit(1)


if __name__ == '__main__':
    main()
