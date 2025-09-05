#!/usr/bin/env python3
import os
import sys
import glob
import pandas as pd
import re

def convert_vector_to_csv(input_file, output_file):
    """
    Convert the vector format from scavetool to a simple timestamp,value CSV format
    that can be easily processed by our data pipeline.
    """
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Find vector data rows
    vector_matches = re.findall(r'vector.*switchId:vector.*,"(.*?)","(.*?)"', content)
    
    if not vector_matches:
        print(f"No vector data found in {input_file}")
        return False
    
    times = []
    values = []
    
    # Process each vector match
    for time_str, value_str in vector_matches:
        # Split space-separated values
        times.extend([float(t) for t in time_str.split()])
        values.extend([int(v) for v in value_str.split()])
    
    # Create dataframe
    df = pd.DataFrame({
        'timestamp': times,
        'switch_id': values
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Write to CSV
    df.to_csv(output_file, index=False)
    print(f"Converted {len(df)} rows from {input_file} to {output_file}")
    return True

def process_switch_id_directory(switch_id_dir):
    """
    Process all files in the SWITCH_ID directory and convert them to the right format.
    """
    os.makedirs(switch_id_dir + '_fixed', exist_ok=True)
    
    files = glob.glob(os.path.join(switch_id_dir, "*.csv"))
    total_converted = 0
    
    for file in files:
        base_name = os.path.basename(file)
        output_file = os.path.join(switch_id_dir + '_fixed', base_name)
        
        if convert_vector_to_csv(file, output_file):
            total_converted += 1
    
    print(f"Converted {total_converted} out of {len(files)} files")
    
    # Create a symlink to the fixed directory
    os.system(f"rm -f {switch_id_dir}.backup")
    os.system(f"mv {switch_id_dir} {switch_id_dir}.backup")
    os.system(f"ln -sf {switch_id_dir}_fixed {switch_id_dir}")
    print(f"Created symlink from {switch_id_dir} to {switch_id_dir}_fixed")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        switch_id_dir = sys.argv[1]
    else:
        switch_id_dir = "extracted_results/SWITCH_ID"
    
    process_switch_id_directory(switch_id_dir)
