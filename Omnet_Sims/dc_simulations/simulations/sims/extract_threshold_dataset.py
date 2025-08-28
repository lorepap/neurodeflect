#!/usr/bin/env python3

import os
import subprocess
import pandas as pd
import glob

def extract_basic_dataset_for_threshold(threshold):
    """Extract basic dataset for a specific threshold"""
    print(f"Processing threshold {threshold}...")
    
    result_dir = f"results/threshold_{threshold}"
    
    # Check if directory exists
    if not os.path.exists(result_dir):
        print(f"Directory {result_dir} not found!")
        return None
    
    # Find actual .vec files in the directory
    import glob
    vec_files = glob.glob(f"{result_dir}/*.vec")
    if not vec_files:
        print(f"No .vec files found in {result_dir}")
        return None
    
    print(f"  Found {len(vec_files)} .vec files")
    
    # Initialize output data
    combined_data = []
    
    # Define the metrics we want to extract
    metrics = {
        'switchSeqNum': 'seq_num',
        'switchTtl': 'ttl', 
        'actionSeqNum': 'action'
    }
    
    # Extract each metric
    for metric_name, col_name in metrics.items():
        try:
            print(f"  Extracting {metric_name}...")
            cmd = [
                'scavetool', 'x',
                '--type', 'v',
                '--filter', f'module(LeafSpine1G) AND "{metric_name}:vector"',
                '-o', 'temp_extract.csv',
                '-F', 'CSV-R'  # Use CSV-R format for proper headers
            ] + vec_files  # Add actual file paths
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
            
            if result.returncode != 0:
                print(f"    Warning: Failed to extract {metric_name}: {result.stderr}")
                continue
                
            # Check if file was created and has data
            if not os.path.exists('temp_extract.csv'):
                print(f"    Warning: No output file for {metric_name}")
                continue
                
            if os.path.getsize('temp_extract.csv') == 0:
                print(f"    Warning: Empty output file for {metric_name}")
                continue
                
            # Read the CSV and look for the actual vector data
            data_rows = []
            with open('temp_extract.csv', 'r') as f:
                lines = f.readlines()
                
            # Find the vector data section (after the metadata)
            in_vector_section = False
            for i, line in enumerate(lines):
                line = line.strip()
                if 'vectime,vecvalue' in line:
                    in_vector_section = True
                    continue
                elif in_vector_section and line and not line.startswith('DCTCP_SD_THRESHOLD_VARIATION'):
                    # This should be the vector data line with time,value pairs
                    # Split by commas and parse as pairs
                    values = line.split(',')
                    for j in range(0, len(values)-1, 2):  # Process pairs
                        try:
                            if j+1 < len(values):
                                time_val = float(values[j])
                                metric_val = float(values[j+1])
                                data_rows.append({'time': time_val, col_name: metric_val})
                        except (ValueError, IndexError):
                            continue
                    break  # Usually just one data line per vector
            
            if not data_rows:
                print(f"    Warning: No valid data rows for {metric_name}")
                continue
                
            df = pd.DataFrame(data_rows)
            print(f"    Extracted {len(df)} rows for {metric_name}")
            
            # Add the metric data to combined data
            if not combined_data:
                # First metric - create the base structure
                combined_data = df.copy()
            else:
                # Merge with existing data on time
                combined_data = pd.merge(combined_data, df, on='time', how='outer')
            
        except Exception as e:
            print(f"    Error extracting {metric_name}: {e}")
            continue
        finally:
            # Clean up temp file
            if os.path.exists('temp_extract.csv'):
                os.remove('temp_extract.csv')
    
    if not combined_data or len(combined_data) == 0:
        print(f"    No data extracted for threshold {threshold}")
        return None
        
    # Add threshold column
    combined_data['deflection_threshold'] = threshold
    
    # Fill missing columns with default values for ML compatibility
    if 'capacity' not in combined_data.columns:
        combined_data['capacity'] = 0.0
    if 'total_capacity' not in combined_data.columns:
        combined_data['total_capacity'] = 0.0
    if 'occupancy' not in combined_data.columns:
        combined_data['occupancy'] = 0.0
    if 'total_occupancy' not in combined_data.columns:
        combined_data['total_occupancy'] = 0.0
    if 'start_time' not in combined_data.columns:
        combined_data['start_time'] = combined_data['time'] - 0.001  # Estimate
    if 'end_time' not in combined_data.columns:
        combined_data['end_time'] = combined_data['time'] + 0.001  # Estimate
    if 'ooo' not in combined_data.columns:
        combined_data['ooo'] = 0  # Out of order flag
    
    print(f"  Final dataset shape: {combined_data.shape}")
    return combined_data

def main():
    """Extract and combine threshold datasets"""
    
    print("==================================================================")
    print("Direct Threshold Dataset Extraction")
    print("==================================================================")
    
    # Change to simulation directory
    os.chdir('/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims')
    
    thresholds = ['0.3', '0.5', '0.7', '0.9']
    all_datasets = []
    
    for threshold in thresholds:
        print(f"\nProcessing threshold {threshold}...")
        df = extract_basic_dataset_for_threshold(threshold)
        if df is not None and not df.empty:
            all_datasets.append(df)
            print(f"✓ Successfully processed threshold {threshold}: {len(df)} rows")
        else:
            print(f"✗ Failed to process threshold {threshold}")
    
    if all_datasets:
        # Combine all datasets
        print(f"\nCombining {len(all_datasets)} datasets...")
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Save the final dataset
        output_file = 'threshold_dataset_simple.csv'
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Final dataset saved: {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"\nThreshold distribution:")
        print(combined_df['deflection_threshold'].value_counts().sort_index())
        
        # Show sample data
        print(f"\nSample data (first 5 rows):")
        print(combined_df.head())
        
        print(f"\nDataset info:")
        print(combined_df.info())
        
        return True
    else:
        print("\n✗ No datasets were successfully processed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
