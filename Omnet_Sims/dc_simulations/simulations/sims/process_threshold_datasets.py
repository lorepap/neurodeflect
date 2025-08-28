#!/usr/bin/env python3

import os
import pandas as pd
import glob
import sys

def process_threshold_datasets():
    """
    Process datasets from multiple threshold directories and combine them
    with deflection_threshold as a feature.
    """
    
    # List of threshold values
    thresholds = ['0.3', '0.5', '0.7', '0.9']
    
    # Store all datasets
    all_datasets = []
    
    print("Processing threshold datasets...")
    
    for threshold in thresholds:
        threshold_dir = f"results/threshold_{threshold}"
        
        if not os.path.exists(threshold_dir):
            print(f"Warning: Threshold directory {threshold_dir} not found")
            continue
            
        print(f"Processing threshold {threshold}...")
        
        # Temporarily move files for processing
        temp_results_dir = "temp_results"
        os.makedirs(temp_results_dir, exist_ok=True)
        
        # Copy files to temp directory
        vec_files = glob.glob(f"{threshold_dir}/*.vec")
        sca_files = glob.glob(f"{threshold_dir}/*.sca")
        out_files = glob.glob(f"{threshold_dir}/*.out")
        
        if not vec_files:
            print(f"Warning: No .vec files found for threshold {threshold}")
            continue
            
        print(f"Found {len(vec_files)} .vec files for threshold {threshold}")
        
        # Create symbolic links to temp directory
        for file_path in vec_files + sca_files + out_files:
            filename = os.path.basename(file_path)
            # Remove threshold suffix to match expected naming
            new_name = filename.replace(f"_threshold_{threshold}", "")
            temp_file = os.path.join(temp_results_dir, new_name)
            
            # Remove existing link if it exists
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # Create symbolic link
            os.symlink(os.path.abspath(file_path), temp_file)
        
        # Extract data for this threshold
        print(f"Extracting data for threshold {threshold}...")
        os.system(f"python3 ./extractor_shell_creator.py dctcp_sd")
        
        # Move temp results to extraction directory
        if os.path.exists("extracted_results"):
            os.system("rm -rf extracted_results")
        os.system(f"mv {temp_results_dir} extracted_results")
        
        # Process extracted results
        os.chdir("extracted_results")
        os.system("bash extractor.sh")
        os.chdir("..")
        
        # Move to results_1G for dataset creation
        if os.path.exists("results_1G"):
            os.system("rm -rf results_1G")
        os.system("mv extracted_results results_1G")
        
        # Create dataset for this threshold
        os.system("python3 create_dataset.py")
        
        # Load the created dataset
        if os.path.exists("dataset.csv"):
            df = pd.read_csv("dataset.csv")
            # Add threshold column
            df['deflection_threshold'] = float(threshold)
            all_datasets.append(df)
            print(f"✓ Processed {len(df)} rows for threshold {threshold}")
            
            # Clean up for next iteration
            os.system("rm -f dataset.csv")
        else:
            print(f"Warning: dataset.csv not created for threshold {threshold}")
    
    # Combine all datasets
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Reorder columns to put deflection_threshold near the front
        cols = combined_df.columns.tolist()
        if 'deflection_threshold' in cols:
            cols.remove('deflection_threshold')
            # Insert after time if it exists, otherwise at the beginning
            if 'time' in cols:
                time_idx = cols.index('time')
                cols.insert(time_idx + 1, 'deflection_threshold')
            else:
                cols.insert(0, 'deflection_threshold')
            combined_df = combined_df[cols]
        
        # Save combined dataset
        combined_df.to_csv("threshold_dataset_combined.csv", index=False)
        
        print(f"\n✓ Combined dataset created: threshold_dataset_combined.csv")
        print(f"Total rows: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"Threshold distribution:")
        print(combined_df['deflection_threshold'].value_counts().sort_index())
        
        # Show sample data
        print(f"\nSample data:")
        print(combined_df.head())
        
        return True
    else:
        print("No datasets were successfully processed")
        return False

if __name__ == "__main__":
    success = process_threshold_datasets()
    sys.exit(0 if success else 1)
