#!/usr/bin/env python3
"""
Simple threshold dataset combination script following the exact pattern 
from create_dataset.py and run_1G_dataset.sh.

This script processes the existing threshold data in results_backup/ and 
combines all thresholds into a single dataset with deflection_threshold feature.
"""

import pandas as pd
import glob
import os
import shutil
import sys

def process_single_threshold(threshold):
    """
    Process a single threshold using the exact same pattern as run_1G_dataset.sh
    """
    threshold_dir = f"results_backup/threshold_{threshold}"
    
    if not os.path.exists(threshold_dir):
        print(f"✗ Threshold directory {threshold_dir} not found")
        return None
        
    print(f"Processing threshold {threshold}...")
    
    # Clean up any previous processing
    if os.path.exists("results"):
        shutil.rmtree("results")
    if os.path.exists("extracted_results"):
        shutil.rmtree("extracted_results")
    if os.path.exists("results_1G"):
        shutil.rmtree("results_1G")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Copy threshold files to results directory, removing threshold suffix
    # This matches the expected naming pattern for the extraction pipeline
    for file_path in glob.glob(f"{threshold_dir}/*"):
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            # Remove threshold suffix to match expected naming pattern
            new_filename = filename.replace(f"_threshold_{threshold}", "")
            shutil.copy2(file_path, f"results/{new_filename}")
    
    print(f"Extracting data for threshold {threshold}...")
    
    # Follow exact same pattern as run_1G_dataset.sh
    # Extract data using the existing extraction pipeline
    exit_code = os.system("python3 ./extractor_shell_creator.py dctcp_sd")
    if exit_code != 0:
        print(f"✗ Extraction script failed for threshold {threshold}")
        return None
    
    # Run the extractor
    if os.path.exists("results/extractor.sh"):
        os.chdir("results")
        exit_code = os.system("bash extractor.sh")
        os.chdir("..")
        if exit_code != 0:
            print(f"✗ Extractor failed for threshold {threshold}")
            return None
    else:
        print(f"✗ extractor.sh not found for threshold {threshold}")
        return None
    
    # Move extracted results following the exact same pattern
    if os.path.exists("extracted_results"):
        shutil.move("extracted_results", "results_1G")
    else:
        print(f"✗ extracted_results not found for threshold {threshold}")
        return None
    
    print(f"Creating dataset for threshold {threshold}...")
    
    # Use the exact same create_dataset.py script
    exit_code = os.system("python3 create_dataset.py")
    if exit_code != 0:
        print(f"✗ Dataset creation failed for threshold {threshold}")
        return None
    
    # Check if dataset was created
    dataset_path = "results_1G/merged_final.csv"
    if not os.path.exists(dataset_path):
        print(f"✗ Dataset file not found at {dataset_path} for threshold {threshold}")
        return None
    
    # Load the dataset and add threshold column
    df = pd.read_csv(dataset_path)
    df['deflection_threshold'] = float(threshold)
    
    print(f"✓ Successfully processed {len(df)} rows for threshold {threshold}")
    
    return df

def main():
    """
    Main function to process all thresholds and combine datasets
    """
    print("="*70)
    print("Simple Threshold Dataset Creation")
    print("Following the exact pattern from create_dataset.py and run_1G_dataset.sh")
    print("="*70)
    
    # Define threshold values (same as in DCTCP_SD_THRESHOLD_VARIATION)
    thresholds = ['0.3', '0.5', '0.7', '0.9']
    
    # Store all datasets
    all_datasets = []
    successful_thresholds = []
    
    # Process each threshold
    for threshold in thresholds:
        try:
            df = process_single_threshold(threshold)
            if df is not None:
                all_datasets.append(df)
                successful_thresholds.append(threshold)
            else:
                print(f"✗ Failed to process threshold {threshold}")
        except Exception as e:
            print(f"✗ Error processing threshold {threshold}: {e}")
    
    if not all_datasets:
        print("✗ No datasets were successfully processed")
        return 1
    
    print(f"\n✓ Successfully processed thresholds: {successful_thresholds}")
    
    # Combine all datasets
    print("\nCombining all threshold datasets...")
    combined_df = pd.concat(all_datasets, ignore_index=True)
    
    # Reorder columns to put deflection_threshold near the front
    # Following the same logic as the existing scripts
    cols = combined_df.columns.tolist()
    if 'deflection_threshold' in cols:
        cols.remove('deflection_threshold')
        # Insert after timestamp if it exists, otherwise at the beginning
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
    
    # Clean up temporary files
    print("\nCleaning up temporary files...")
    for dir_name in ["results", "extracted_results", "results_1G"]:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
    
    print("\n" + "="*70)
    print("Dataset creation completed successfully!")
    print(f"Final dataset: {output_file}")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
