#!/usr/bin/env python3
"""
Convert OMNeT++ vector format files to simple CSV format expected by create_dataset.py

The input files are in vector format with headers like:
"QueueCapacity:vector ...", timestamp1,value1
timestamp2,value2

The output should be simple CSV with:
timestamp,value
timestamp1,value1
timestamp2,value2
"""

import os
import glob
import pandas as pd
import argparse

def convert_vector_file(input_path, output_path):
    """Convert a single vector file to CSV format"""
    data = []
    
    with open(input_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
                
            # Skip the header line (contains "vector" and long config string)
            if line_num == 0 and '"' in line and 'vector' in line:
                continue
                
            # Parse timestamp,value pairs
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    timestamp = float(parts[0])
                    value = float(parts[1])
                    data.append([timestamp, value])
                except ValueError:
                    continue
    
    if data:
        df = pd.DataFrame(data, columns=['timestamp', 'value'])
        df.to_csv(output_path, index=False)
        print(f"Converted {len(data)} rows: {input_path} -> {output_path}")
        return len(data)
    else:
        print(f"No data found in {input_path}")
        return 0

def convert_directory(input_dir, output_dir):
    """Convert all vector files in a directory"""
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist")
        return 0
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files (which are actually vector format)
    pattern = os.path.join(input_dir, "*.csv")
    files = glob.glob(pattern)
    
    total_converted = 0
    for file_path in files:
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        rows = convert_vector_file(file_path, output_path)
        total_converted += rows
    
    return total_converted

def main():
    parser = argparse.ArgumentParser(description='Convert vector format files to CSV')
    parser.add_argument('results_dir', help='Results directory (e.g., results_1G_thr_37500)')
    
    args = parser.parse_args()
    
    # Directories that need conversion
    vector_dirs = [
        'QUEUE_CAPACITY', 'QUEUES_TOT_CAPACITY', 'QUEUE_LEN', 'QUEUES_TOT_LEN',
        'SWITCH_SEQ_NUM', 'TTL', 'PACKET_SIZE', 'ACTION_SEQ_NUM', 'PACKET_ACTION'
    ]
    
    print(f"Converting vector files in {args.results_dir}")
    print("=" * 50)
    
    total_files = 0
    total_rows = 0
    
    for dir_name in vector_dirs:
        input_dir = os.path.join(args.results_dir, dir_name)
        output_dir = os.path.join(args.results_dir, f"{dir_name}_csv")
        
        if os.path.exists(input_dir):
            rows = convert_directory(input_dir, output_dir)
            if rows > 0:
                total_files += 1
                total_rows += rows
        else:
            print(f"Directory {input_dir} not found")
    
    print("=" * 50)
    print(f"Conversion complete: {total_files} directories, {total_rows} total rows")
    
    # Now update the paths in create_dataset.py arguments
    print("\nTo use the converted files, run create_dataset.py with these arguments:")
    for i, dir_name in enumerate(vector_dirs[:7], 1):  # First 7 are for the main merge
        print(f"--folder{i} {args.results_dir}/{dir_name}_csv")
    
    # Last 2 are for action data
    print(f"--folder8 {args.results_dir}/ACTION_SEQ_NUM_csv")
    print(f"--folder9 {args.results_dir}/PACKET_ACTION_csv")

if __name__ == '__main__':
    main()
