#!/usr/bin/env python3
import pandas as pd
import sys
import os

def convert_csvr_to_simple_csv(input_file, output_file):
    """Convert CSV-R format SWITCH_ID data to simple timestamp,switch_id format"""
    print(f"Converting {input_file} to {output_file}")
    
    # Read the CSV-R file
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return False
    
    # Find vector rows (type == 'vector' and contains vectime and vecvalue data)
    vector_rows = df[(df['type'] == 'vector') & 
                     (df['name'] == 'switchId:vector') & 
                     (df['vectime'].notna()) & 
                     (df['vecvalue'].notna())]
    
    if vector_rows.empty:
        print(f"No vector data found in {input_file}")
        return False
    
    # Collect all timestamp,switch_id pairs
    all_timestamps = []
    all_switch_ids = []
    
    for _, row in vector_rows.iterrows():
        vectime = str(row['vectime']).strip()
        vecvalue = str(row['vecvalue']).strip()
        
        if vectime and vecvalue and vectime != 'nan' and vecvalue != 'nan':
            timestamps = vectime.split()
            switch_ids = vecvalue.split()
            
            if len(timestamps) == len(switch_ids):
                all_timestamps.extend([float(t) for t in timestamps])
                all_switch_ids.extend([int(s) for s in switch_ids])
            else:
                print(f"Warning: timestamp and switch_id length mismatch in {input_file}")
    
    if not all_timestamps:
        print(f"No valid vector data found in {input_file}")
        return False
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'timestamp': all_timestamps,
        'switch_id': all_switch_ids
    })
    
    # Sort by timestamp
    output_df = output_df.sort_values('timestamp')
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"✓ Converted {len(output_df)} switch_id records to {output_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_csvr_switch_id.py <input_csvr_file> <output_csv_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    success = convert_csvr_to_simple_csv(input_file, output_file)
    if not success:
        sys.exit(1)
