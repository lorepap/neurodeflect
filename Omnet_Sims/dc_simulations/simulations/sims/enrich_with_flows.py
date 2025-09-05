#!/usr/bin/env python3
"""
Enhanced flow enrichment script with FlowID-based joining

This script implements the new approach that:
1. Uses true flow_id (5-tuple hash) from the FLOW_ID signal for proper flow grouping
2. Creates flow-level observations instead of packet-level groupings
3. Provides accurate flow completion times without artificial constraints
4. Falls back to seq_num grouping when FlowID data is unavailable
"""

import os
import glob
import csv
import pandas as pd
import argparse
from scipy.spatial import cKDTree

# Helper to load CSV-S format files (timestamp,value pairs)
def load_csvs_vecs(folder: str, name: str):
    pattern = os.path.join(folder, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['timestamp', name])
    dfs = []
    for fp in files:
        try:
            # CSV-S format: simple timestamp,value pairs
            dfi = pd.read_csv(fp)
            # Rename columns to standard format
            if len(dfi.columns) >= 2:
                dfi.columns = ['timestamp', name]
                dfi['timestamp'] = pd.to_numeric(dfi['timestamp'], errors='coerce')
                dfi[name] = pd.to_numeric(dfi[name], errors='coerce')
                dfi = dfi.dropna(subset=['timestamp', name])
                print(f"Loaded {len(dfi)} rows from {os.path.basename(fp)} for {name}")
                dfs.append(dfi)
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', name])
    out = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in out.columns:
        out = out.sort_values('timestamp').reset_index(drop=True)
    return out

# Helper to load CSV-R format files with FlowID extraction
def load_csvr_flowid_vecs(folder: str, name: str):
    """Load CSV-R format files containing FlowID data"""
    pattern = os.path.join(folder, '*')
    files = sorted(glob.glob(pattern))
    files = [f for f in files if os.path.isfile(f)]  # Filter only files
    if not files:
        return pd.DataFrame(columns=['timestamp', name])
    dfs = []
    for fp in files:
        try:
            vector_data = []
            with open(fp, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Look for FlowID vector data rows
                    if len(row) >= 8 and row[1] == 'vector' and 'FlowID:vector' in row[3]:
                        # Extract timestamps and flow IDs
                        if len(row) >= 9:
                            timestamps_str = row[7].strip('"')  # Column 7 has timestamps
                            flowids_str = row[8].strip('"')     # Column 8 has flow IDs
                            
                            if timestamps_str and flowids_str:
                                timestamps = timestamps_str.split()
                                flowids = flowids_str.split()
                                
                                if len(timestamps) == len(flowids):
                                    for ts, fid in zip(timestamps, flowids):
                                        try:
                                            vector_data.append([float(ts), int(fid)])
                                        except ValueError:
                                            continue
            
            if vector_data:
                dfi = pd.DataFrame(vector_data, columns=['timestamp', name])
                print(f"Loaded {len(dfi)} FlowID entries from {os.path.basename(fp)}")
                dfs.append(dfi)
            else:
                print(f"No FlowID vector data found in {os.path.basename(fp)}")
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', name])
    out = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in out.columns:
        out = out.sort_values('timestamp').reset_index(drop=True)
    return out

# Helper to load CSV-R format files with switch ID extraction
def load_csvr_vecs_with_seqnum(folder: str, name: str):
    """Load CSV-R format files with switch ID extraction"""
    pattern = os.path.join(folder, '*')
    files = sorted(glob.glob(pattern))
    files = [f for f in files if os.path.isfile(f)]  # Filter only files
    if not files:
        return pd.DataFrame(columns=['timestamp', 'seq_num', name])
    dfs = []
    for fp in files:
        try:
            vector_data = []
            with open(fp, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Look for switchId vector data rows
                    if len(row) >= 8 and row[1] == 'vector' and 'switchId:vector' in row[3]:
                        # Extract timestamps and switch IDs
                        if len(row) >= 9:
                            timestamps_str = row[7].strip('"')  # Column 7 has timestamps
                            switchids_str = row[8].strip('"')   # Column 8 has switch IDs
                            
                            if timestamps_str and switchids_str:
                                timestamps = timestamps_str.split()
                                switchids = switchids_str.split()
                                
                                if len(timestamps) == len(switchids):
                                    for ts, sid in zip(timestamps, switchids):
                                        try:
                                            vector_data.append([float(ts), 0, float(sid)])  # seq_num=0 for now
                                        except ValueError:
                                            continue
            
            if vector_data:
                dfi = pd.DataFrame(vector_data, columns=['timestamp', 'seq_num', name])
                print(f"Loaded {len(dfi)} SWITCH_ID entries from {os.path.basename(fp)}")
                dfs.append(dfi)
            else:
                print(f"No SWITCH_ID vector data found in {os.path.basename(fp)}")
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', 'seq_num', name])
    out = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in out.columns:
        out = out.sort_values('timestamp').reset_index(drop=True)
    return out

def join_with_flowid(dataset: pd.DataFrame, flowid_data: pd.DataFrame, max_dist=0.0001):
    """Join dataset with FlowID data using nearest neighbor matching"""
    if len(flowid_data) == 0:
        print("No FlowID data available, returning dataset unchanged")
        return dataset
    
    print(f"Joining {len(dataset)} dataset rows with {len(flowid_data)} FlowID rows")
    
    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(flowid_data[['timestamp']].values)
    
    # Find nearest neighbors for each dataset timestamp
    distances, indices = tree.query(dataset[['timestamp']].values, k=1)
    
    # Only keep matches within max_dist
    valid_matches = distances.flatten() <= max_dist
    matched_flowids = []
    
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= max_dist:
            matched_flowids.append(flowid_data.iloc[idx]['FlowID'])
        else:
            matched_flowids.append(None)
    
    dataset = dataset.copy()
    dataset['FlowID'] = matched_flowids
    
    print(f"Successfully matched {sum(pd.notna(matched_flowids))} FlowID values")
    return dataset

def join_with_switch_id(dataset: pd.DataFrame, switch_data: pd.DataFrame, max_dist=0.0001):
    """Join dataset with SWITCH_ID data using nearest neighbor matching"""
    if len(switch_data) == 0:
        print("No SWITCH_ID data available, returning dataset unchanged")
        return dataset
    
    print(f"Joining {len(dataset)} dataset rows with {len(switch_data)} SWITCH_ID rows")
    
    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(switch_data[['timestamp']].values)
    
    # Find nearest neighbors for each dataset timestamp
    distances, indices = tree.query(dataset[['timestamp']].values, k=1)
    
    # Only keep matches within max_dist
    valid_matches = distances.flatten() <= max_dist
    matched_switch_ids = []
    
    for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
        if dist <= max_dist:
            matched_switch_ids.append(switch_data.iloc[idx]['SWITCH_ID'])
        else:
            matched_switch_ids.append(None)
    
    dataset = dataset.copy()
    dataset['SWITCH_ID'] = matched_switch_ids
    
    print(f"Successfully matched {sum(pd.notna(matched_switch_ids))} SWITCH_ID values")
    return dataset

def calculate_fct_from_timing(dataset: pd.DataFrame):
    """Calculate FCT from existing start_time and end_time columns"""
    if 'start_time' in dataset.columns and 'end_time' in dataset.columns:
        dataset = dataset.copy()
        # FCT = end_time - start_time
        dataset['FCT'] = dataset['end_time'] - dataset['start_time']
        print(f"Calculated FCT for {len(dataset)} rows from start_time and end_time")
        return dataset
    else:
        print("Cannot calculate FCT: start_time and/or end_time columns missing")
        return dataset

def create_flow_level_data(dataset: pd.DataFrame):
    """Create flow-level observations from packet-level data"""
    if 'FlowID' not in dataset.columns:
        print("No FlowID column found, falling back to seq_num grouping")
        # Group by seq_num instead
        flow_groups = dataset.groupby('seq_num')
    else:
        # Group by FlowID for true flow-level aggregation
        flow_groups = dataset.groupby('FlowID')
    
    flow_data = []
    
    for flow_id, group in flow_groups:
        if len(group) == 0:
            continue
            
        # Calculate flow-level metrics
        flow_start = group['timestamp'].min()
        flow_end = group['timestamp'].max()
        flow_fct = flow_end - flow_start
        
        # Get other flow characteristics
        flow_info = {
            'FlowID': flow_id if 'FlowID' in dataset.columns else None,
            'seq_num': flow_id if 'FlowID' not in dataset.columns else group['seq_num'].iloc[0],
            'start_time': flow_start,
            'end_time': flow_end,
            'FCT': flow_fct,
            'packet_count': len(group),
            'deflection_threshold': group['deflection_threshold'].iloc[0] if 'deflection_threshold' in group.columns else None
        }
        
        # Add SWITCH_ID if available (take mode)
        if 'SWITCH_ID' in group.columns:
            switch_counts = group['SWITCH_ID'].value_counts()
            if len(switch_counts) > 0:
                flow_info['SWITCH_ID'] = switch_counts.index[0]
        
        # Add other columns that should be preserved at flow level
        for col in ['ooo', 'timestamp']:
            if col in group.columns:
                if col == 'timestamp':
                    flow_info[col] = flow_start  # Use flow start time
                else:
                    flow_info[col] = group[col].iloc[0]  # Take first value
        
        flow_data.append(flow_info)
    
    result = pd.DataFrame(flow_data)
    print(f"Created {len(result)} flow-level observations from {len(dataset)} packet-level records")
    return result

def main():
    parser = argparse.ArgumentParser(description='Enrich dataset with FlowID and SWITCH_ID')
    parser.add_argument('input_file', help='Input CSV file (output from filter_overlapping_timestamps.py)')
    parser.add_argument('results_dir', help='Results directory containing FLOW_ID and SWITCH_ID subdirectories')
    parser.add_argument('output_file', help='Output CSV file with enriched data')
    parser.add_argument('--log-file', help='Log file for debugging output')
    
    args = parser.parse_args()
    
    # Setup logging if specified
    if args.log_file:
        import sys
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(args.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
    
    print(f"Loading input dataset from: {args.input_file}")
    
    # Load the main dataset
    try:
        dataset = pd.read_csv(args.input_file)
        print(f"Loaded dataset with {len(dataset)} rows and columns: {list(dataset.columns)}")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return
    
    # Load FlowID data
    flowid_dir = os.path.join(args.results_dir, 'FLOW_ID')
    print(f"Loading FlowID data from: {flowid_dir}")
    flowid_data = load_csvr_flowid_vecs(flowid_dir, 'FlowID')
    print(f"FlowID data shape: {flowid_data.shape}")
    
    # Load SWITCH_ID data  
    switch_dir = os.path.join(args.results_dir, 'SWITCH_ID')
    print(f"Loading SWITCH_ID data from: {switch_dir}")
    switch_data = load_csvr_vecs_with_seqnum(switch_dir, 'SWITCH_ID')
    print(f"SWITCH_ID data shape: {switch_data.shape}")
    
    # Join with FlowID data
    dataset = join_with_flowid(dataset, flowid_data)
    
    # Join with SWITCH_ID data
    dataset = join_with_switch_id(dataset, switch_data)
    
    # Calculate FCT from timing columns if available
    dataset = calculate_fct_from_timing(dataset)
    
    # Create flow-level observations
    flow_dataset = create_flow_level_data(dataset)
    
    # Convert FlowID to integer if it exists and is not null
    if 'FlowID' in flow_dataset.columns:
        flow_dataset['FlowID'] = flow_dataset['FlowID'].astype('Int64')  # Use nullable integer type
    
    # Save the enriched dataset
    print(f"Saving enriched dataset to: {args.output_file}")
    flow_dataset.to_csv(args.output_file, index=False)
    print(f"Saved {len(flow_dataset)} flow-level observations")
    
    # Print summary
    print("\nEnrichment Summary:")
    print(f"Input rows: {len(dataset)}")
    print(f"Output flows: {len(flow_dataset)}")
    if 'FlowID' in flow_dataset.columns:
        flowid_count = flow_dataset['FlowID'].notna().sum()
        print(f"Flows with FlowID: {flowid_count}")
    if 'SWITCH_ID' in flow_dataset.columns:
        switch_count = flow_dataset['SWITCH_ID'].notna().sum()
        print(f"Flows with SWITCH_ID: {switch_count}")
    if 'FCT' in flow_dataset.columns:
        fct_count = flow_dataset['FCT'].notna().sum()
        print(f"Flows with FCT: {fct_count}")

if __name__ == "__main__":
    main()
