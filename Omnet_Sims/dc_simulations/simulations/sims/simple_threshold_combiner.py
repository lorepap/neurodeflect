#!/usr/bin/env python3
"""
Simple threshold combiner - manually extract and combine threshold datasets
"""

import os
import shutil
import subprocess
import pandas as pd

def manually_extract_threshold(threshold):
    """Manually extract threshold data using working filename patterns"""
    print(f"\n=== Manually processing threshold {threshold} ===")
    
    # Clean up
    if os.path.exists("results"):
        shutil.rmtree("results")
    if os.path.exists("extracted_results"):
        shutil.rmtree("extracted_results")
    
    # Copy threshold data
    threshold_dir = f"results_backup/threshold_{threshold}"
    if not os.path.exists(threshold_dir):
        print(f"Threshold directory {threshold_dir} not found!")
        return None
        
    print(f"Copying threshold data from {threshold_dir}")
    shutil.copytree(threshold_dir, "results")
    
    # Get the actual filename pattern from threshold files
    threshold_files = [f for f in os.listdir("results") if f.endswith(f"_threshold_{threshold}.vec")]
    if not threshold_files:
        print(f"No .vec files found for threshold {threshold}")
        return None
        
    original_file = threshold_files[0]
    base_name = original_file.replace(f"_threshold_{threshold}", "")
    
    print(f"Original file: {original_file}")
    print(f"Base name: {base_name}")
    
    # Create renamed copies for extraction
    for ext in ['.vec', '.out', '.sca', '.vci']:
        threshold_file = base_name.replace('.vec', '') + f"_threshold_{threshold}" + ext
        target_file = base_name.replace('.vec', '') + "_0_rep" + ext
        
        if os.path.exists(f"results/{threshold_file}"):
            print(f"Copying {threshold_file} -> {target_file}")
            shutil.copy2(f"results/{threshold_file}", f"results/{target_file}")
    
    # Now try to use the existing extraction tools
    try:
        print("Running extractor_shell_creator...")
        result = subprocess.run([
            'python3', './extractor_shell_creator.py', f'threshold_{threshold}'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Extractor shell creator failed: {result.stderr}")
            return None
            
        print("Running extraction shell script...")
        original_dir = os.getcwd()
        os.chdir("results")
        result = subprocess.run(['bash', 'extractor.sh'], capture_output=True, text=True)
        os.chdir(original_dir)
        
        if result.returncode != 0:
            print(f"Extraction failed: {result.stderr}")
            return None
            
        # Check if extracted_results exist
        if os.path.exists("extracted_results"):
            print("Extraction successful, creating dataset...")
            # Run create_dataset.py
            result = subprocess.run([
                'python3', './create_dataset.py', f'threshold_{threshold}'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Dataset creation failed: {result.stderr}")
                return None
                
            # Find the generated dataset
            dataset_files = [f for f in os.listdir('.') if f.startswith('dataset_') and f.endswith('.csv')]
            if dataset_files:
                dataset_file = dataset_files[0]
                threshold_dataset = f"dataset_threshold_{threshold}.csv"
                shutil.move(dataset_file, threshold_dataset)
                print(f"Dataset created: {threshold_dataset}")
                return threshold_dataset
            else:
                print("No dataset file was created")
                return None
        else:
            print("No extracted_results directory found")
            return None
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        return None

def combine_threshold_datasets():
    """Combine all threshold datasets"""
    print("=== Combining Threshold Datasets ===")
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    combined_dfs = []
    
    for threshold in thresholds:
        dataset_file = manually_extract_threshold(threshold)
        if dataset_file and os.path.exists(dataset_file):
            print(f"Loading dataset for threshold {threshold}")
            df = pd.read_csv(dataset_file)
            df['deflection_threshold'] = threshold
            combined_dfs.append(df)
            print(f"Loaded {len(df)} rows for threshold {threshold}")
        else:
            print(f"Failed to create dataset for threshold {threshold}")
    
    if combined_dfs:
        print("Combining all datasets...")
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        output_file = "combined_threshold_dataset.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"Combined dataset saved: {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Threshold distribution:")
        print(combined_df['deflection_threshold'].value_counts().sort_index())
        return output_file
    else:
        print("No datasets were successfully created!")
        return None

if __name__ == "__main__":
    combine_threshold_datasets()
