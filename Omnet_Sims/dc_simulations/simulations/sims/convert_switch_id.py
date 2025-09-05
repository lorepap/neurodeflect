#!/usr/bin/env python3
import re
import sys
import pandas as pd

def convert_switch_id_simple(input_file, output_file):
    """
    Convert scavetool vector format to simple timestamp,switch_id CSV.
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_switch_id.py input_file output_file")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_switch_id_simple(input_file, output_file)
