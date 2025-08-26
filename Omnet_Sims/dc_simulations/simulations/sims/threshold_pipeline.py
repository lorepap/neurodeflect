#!/usr/bin/env python3
"""
Complete Threshold Dataset Pipeline Script

This script automates the entire process of:
1. Extracting data from simulation results (.vec and .sca files)
2. Creating the necessary directory structure for the extracted data
3. Processing the raw data into a dataset suitable for ML training

Usage:
    python3 threshold_pipeline.py --thresholds 0.3,0.5,0.7,0.9 --results_dir results

The script will handle all necessary steps to generate the dataset.
"""

import argparse
import glob
import os
import pandas as pd
import subprocess
import sys
import time
from pathlib import Path


# Constants
DEFAULT_OUTPUT_DIR = "dataset_output"


def create_extraction_script(threshold, results_dir, output_dir):
    """
    Creates and returns a shell script to extract data from simulation results for a specific threshold.
    """
    # Find all .vec and .sca files for the given threshold
    pattern = os.path.join(results_dir, f"threshold_{threshold}", "*.vec")
    vec_files = glob.glob(pattern)
    
    if not vec_files:
        print(f"No .vec files found for threshold {threshold} in {pattern}")
        return None
    
    # Create the threshold output directory
    threshold_dir = os.path.join(output_dir, f"threshold_{threshold}")
    os.makedirs(threshold_dir, exist_ok=True)
    
    # Create subdirectories for different data types
    data_types = [
        "QUEUE_CAPACITY", "QUEUES_TOT_CAPACITY", "QUEUE_LEN", "QUEUES_TOT_LEN",
        "SWITCH_SEQ_NUM", "TTL", "ACTION_SEQ_NUM", "PACKET_ACTION"
    ]
    
    for data_type in data_types:
        os.makedirs(os.path.join(threshold_dir, data_type), exist_ok=True)
    
    # Create extraction script content
    script_content = "#!/bin/bash\n\n"
    script_content += f"# Extraction script for threshold {threshold}\n\n"
    
    for vec_file in vec_files:
        # Get corresponding .sca file
        sca_file = vec_file.replace(".vec", ".sca")
        if not os.path.exists(sca_file):
            print(f"Warning: Scalar file not found for {vec_file}")
            continue
            
        base_name = os.path.basename(vec_file).replace(".vec", "")
        # Use a simpler output filename to avoid path length issues
        output_base = f"data_{threshold}"
        
        # Add extraction commands for vector data
        script_content += f"# Processing {base_name}\n"
        
        # QUEUE_CAPACITY extraction
        script_content += f'echo "Extracting QUEUE_CAPACITY"\n'
        script_content += f'scavetool x --type v --filter "\\\"QueueCapacity:vector\\\"" -o {threshold_dir}/QUEUE_CAPACITY/{output_base}.csv -F CSV-S {vec_file}\n\n'
        
        # QUEUES_TOT_CAPACITY extraction
        script_content += f'echo "Extracting QUEUES_TOT_CAPACITY"\n'
        script_content += f'scavetool x --type v --filter "\\\"QueuesTotCapacity:vector\\\"" -o {threshold_dir}/QUEUES_TOT_CAPACITY/{output_base}.csv -F CSV-S {vec_file}\n\n'
        
        # QUEUE_LEN extraction
        script_content += f'echo "Extracting QUEUE_LEN"\n'
        script_content += f'scavetool x --type v --filter "\\\"QueueLen:vector\\\"" -o {threshold_dir}/QUEUE_LEN/{output_base}.csv -F CSV-S {vec_file}\n\n'
        
        # QUEUES_TOT_LEN extraction
        script_content += f'echo "Extracting QUEUES_TOT_LEN"\n'
        script_content += f'scavetool x --type v --filter "\\\"QueuesTotLen:vector\\\"" -o {threshold_dir}/QUEUES_TOT_LEN/{output_base}.csv -F CSV-S {vec_file}\n\n'
        
        # SWITCH_SEQ_NUM extraction
        script_content += f'echo "Extracting SWITCH_SEQ_NUM"\n'
        script_content += f'scavetool x --type v --filter "\\\"switchSeqNum:vector\\\"" -o {threshold_dir}/SWITCH_SEQ_NUM/{output_base}.csv -F CSV-S {vec_file}\n\n'
        
        # TTL extraction
        script_content += f'echo "Extracting TTL"\n'
        script_content += f'scavetool x --type v --filter "\\\"switchTtl:vector\\\"" -o {threshold_dir}/TTL/{output_base}.csv -F CSV-S {vec_file}\n\n'
        
        # ACTION_SEQ_NUM extraction
        script_content += f'echo "Extracting ACTION_SEQ_NUM"\n'
        script_content += f'scavetool x --type v --filter "\\\"actionSeqNum:vector\\\"" -o {threshold_dir}/ACTION_SEQ_NUM/{output_base}.csv -F CSV-S {vec_file}\n\n'
        
        # PACKET_ACTION extraction
        script_content += f'echo "Extracting PACKET_ACTION"\n'
        script_content += f'scavetool x --type v --filter "\\\"PacketAction:vector\\\"" -o {threshold_dir}/PACKET_ACTION/{output_base}.csv -F CSV-S {vec_file}\n\n'
    
    # Create script file
    script_path = os.path.join(output_dir, f"extract_threshold_{threshold}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return script_path


def first_csv_in(folder):
    """
    Returns the path to the first CSV file in the given folder.
    """
    pattern = os.path.join(folder, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSV file found in {folder}")
    return files[0]


def zip_datasets(threshold_dir, output_path):
    """
    Combines data from multiple CSV files into a single dataset for a specific threshold.
    """
    try:
        # Define folder paths
        folders = [
            os.path.join(threshold_dir, "QUEUE_CAPACITY"),
            os.path.join(threshold_dir, "QUEUES_TOT_CAPACITY"),
            os.path.join(threshold_dir, "QUEUE_LEN"),
            os.path.join(threshold_dir, "QUEUES_TOT_LEN"),
            os.path.join(threshold_dir, "SWITCH_SEQ_NUM"),
            os.path.join(threshold_dir, "TTL")
        ]
        
        column_names = [
            "capacity", "total_capacity", "occupancy", 
            "total_occupancy", "seq_num", "ttl"
        ]
        
        # Check if all required folders exist and have CSV files
        missing_folders = []
        for folder in folders:
            if not os.path.exists(folder):
                missing_folders.append(folder)
            else:
                try:
                    first_csv_in(folder)
                except FileNotFoundError:
                    missing_folders.append(f"{folder} (no CSV files)")
        
        if missing_folders:
            print(f"Warning: Missing folders or CSV files: {missing_folders}")
            return None
        
        # Process first six columns from the main folders
        # Read the first CSV with timestamp
        csv0 = first_csv_in(folders[0])
        df = pd.read_csv(
            csv0,
            skiprows=1,
            header=None,
            names=['timestamp', column_names[0]],
            dtype={'timestamp': 'float64'}
        )
        
        # Add remaining columns
        for folder, colname in zip(folders[1:], column_names[1:]):
            csv_file = first_csv_in(folder)
            df_temp = pd.read_csv(
                csv_file,
                skiprows=1,
                header=None,
                names=['timestamp', colname],
                dtype={'timestamp': 'float64'}
            )
            df[colname] = df_temp[colname]
        
        # Process action data if available
        action_seq_folder = os.path.join(threshold_dir, "ACTION_SEQ_NUM")
        packet_action_folder = os.path.join(threshold_dir, "PACKET_ACTION")
        
        if (os.path.exists(action_seq_folder) and os.path.exists(packet_action_folder) and
            len(glob.glob(os.path.join(action_seq_folder, "*.csv"))) > 0 and
            len(glob.glob(os.path.join(packet_action_folder, "*.csv"))) > 0):
            
            # Read action sequence data
            action_csv = first_csv_in(action_seq_folder)
            df_action_seq = pd.read_csv(
                action_csv,
                skiprows=1,
                header=None,
                names=['timestamp', 'action_seq_num'],
                dtype={'timestamp': 'float64'}
            )
            
            # Read packet action data
            action_csv = first_csv_in(packet_action_folder)
            df_action = pd.read_csv(
                action_csv,
                skiprows=1,
                header=None,
                names=['timestamp', 'action'],
                dtype={'timestamp': 'float64'}
            )
            
            # Join action data on timestamp
            df = pd.merge_asof(
                df.sort_values('timestamp'),
                df_action.sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
        else:
            # If action data is not available, add a default action column
            print(f"Warning: Action data not available for threshold directory {threshold_dir}")
            df['action'] = 2  # Default action value
        
        # Add threshold value as a column
        threshold_value = os.path.basename(threshold_dir).replace("threshold_", "")
        df['deflection_threshold'] = float(threshold_value)
        
        # Save dataset
        df.to_csv(output_path, index=False)
        print(f"Dataset created with {len(df)} rows and saved to {output_path}")
        
        return df
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_extraction_script(script_path):
    """
    Runs an extraction script and returns True if successful.
    """
    try:
        print(f"Running extraction script: {script_path}")
        result = subprocess.run(['bash', script_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running extraction script: {result.stderr}")
            return False
            
        print(f"Extraction completed successfully")
        return True
    except Exception as e:
        print(f"Exception running extraction script: {e}")
        return False


def combine_threshold_datasets(datasets, output_path):
    """
    Combines datasets from multiple thresholds into a single dataset.
    """
    if not datasets:
        print("No datasets to combine")
        return
    
    try:
        # Combine all dataframes
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Sort by threshold and timestamp
        combined_df = combined_df.sort_values(['deflection_threshold', 'timestamp']).reset_index(drop=True)
        
        # Save combined dataset
        combined_df.to_csv(output_path, index=False)
        
        # Print summary statistics
        print("\nCombined Dataset Summary:")
        print(f"Total records: {len(combined_df)}")
        print("Records per threshold:")
        print(combined_df.groupby('deflection_threshold').size())
        
        print(f"Combined dataset saved to {output_path}")
        
    except Exception as e:
        print(f"Error combining datasets: {e}")


def main():
    parser = argparse.ArgumentParser(description="Complete Threshold Dataset Pipeline")
    
    parser.add_argument("--thresholds", default="0.3,0.5,0.7,0.9", 
                        help="Comma-separated list of threshold values to process")
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing simulation results")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to store extracted data and datasets")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip data extraction step (use if data is already extracted)")
    parser.add_argument("--combine_only", action="store_true",
                        help="Only combine existing datasets, don't create new ones")
    
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [t.strip() for t in args.thresholds.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting threshold dataset pipeline with thresholds: {thresholds}")
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    
    all_datasets = []
    
    for threshold in thresholds:
        print(f"\nProcessing threshold {threshold}...")
        threshold_dir = os.path.join(args.output_dir, f"threshold_{threshold}")
        
        if not args.skip_extraction and not args.combine_only:
            # Step 1: Create and run extraction script
            script_path = create_extraction_script(threshold, args.results_dir, args.output_dir)
            if script_path and run_extraction_script(script_path):
                print(f"Data extraction completed for threshold {threshold}")
            else:
                print(f"Data extraction failed for threshold {threshold}, skipping dataset creation")
                continue
        
        if not args.combine_only:
            # Step 2: Create dataset for this threshold
            dataset_path = os.path.join(args.output_dir, f"threshold_dataset_{threshold}.csv")
            df = zip_datasets(threshold_dir, dataset_path)
            
            if df is not None:
                all_datasets.append(df)
                print(f"Dataset for threshold {threshold} created successfully")
            else:
                print(f"Failed to create dataset for threshold {threshold}")
        else:
            # If combine_only, try to load existing dataset
            dataset_path = os.path.join(args.output_dir, f"threshold_dataset_{threshold}.csv")
            if os.path.exists(dataset_path):
                try:
                    df = pd.read_csv(dataset_path)
                    all_datasets.append(df)
                    print(f"Loaded existing dataset for threshold {threshold}")
                except Exception as e:
                    print(f"Error loading dataset for threshold {threshold}: {e}")
            else:
                print(f"Dataset file not found for threshold {threshold}")
    
    # Step 3: Combine all threshold datasets
    if all_datasets:
        combined_path = os.path.join(args.output_dir, "combined_threshold_dataset.csv")
        combine_threshold_datasets(all_datasets, combined_path)
    else:
        print("No datasets were successfully created or loaded, cannot create combined dataset")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"\nPipeline completed in {elapsed_time:.2f} seconds")
