#!/usr/bin/env python3
"""
Enhanced flow enrichment script with FlowID-based joining and PacketAction integration

This script implements the new approach that:
1. Uses true flow_id (5-tuple hash) from the FLOW_ID signal for proper flow grouping
2. Creates flow-level observations instead of packet-level groupings
3. Provides accurate flow completion times without artificial constraints
4. Includes PacketAction data (0=forward, 1=deflection) for proper deflection analysis
5. Falls back to seq_num grouping when FlowID data is unavailable
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
    """Load CSV-R format files with FlowID extraction"""
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
                    # Look for flowId vector data rows
                    if len(row) >= 8 and row[1] == 'vector' and 'flowId:vector' in row[3]:
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
                                            vector_data.append([float(ts), 0, int(fid)])  # seq_num=0 for now
                                        except ValueError:
                                            continue
            
            if vector_data:
                dfi = pd.DataFrame(vector_data, columns=['timestamp', 'seq_num', name])
                print(f"Loaded {len(dfi)} FlowID entries from {os.path.basename(fp)}")
                dfs.append(dfi)
            else:
                print(f"No FlowID vector data found in {os.path.basename(fp)}")
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', 'seq_num', name])
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

# Helper to load CSV-R format files with PacketAction extraction
def load_csvr_packet_action_vecs(folder: str, name: str):
    """Load CSV-R format files with PacketAction extraction (0=forward, 1=deflection)"""
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
                    # Look for PacketAction vector data rows
                    if len(row) >= 8 and row[1] == 'vector' and 'PacketAction:vector' in row[3]:
                        # Extract timestamps and packet actions
                        if len(row) >= 9:
                            timestamps_str = row[7].strip('"')  # Column 7 has timestamps
                            actions_str = row[8].strip('"')     # Column 8 has actions
                            
                            if timestamps_str and actions_str:
                                timestamps = timestamps_str.split()
                                actions = actions_str.split()
                                
                                if len(timestamps) == len(actions):
                                    for ts, action in zip(timestamps, actions):
                                        try:
                                            vector_data.append([float(ts), 0, int(action)])  # seq_num=0 for now
                                        except ValueError:
                                            continue
            
            if vector_data:
                dfi = pd.DataFrame(vector_data, columns=['timestamp', 'seq_num', name])
                print(f"Loaded {len(dfi)} PacketAction entries from {os.path.basename(fp)}")
                dfs.append(dfi)
            else:
                print(f"No PacketAction vector data found in {os.path.basename(fp)}")
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', 'seq_num', name])
    out = pd.concat(dfs, ignore_index=True)
    if 'timestamp' in out.columns:
        out = out.sort_values('timestamp').reset_index(drop=True)
    return out

def join_with_flowid(dataset, flowid_data):
    """Join dataset with FlowID data using timestamp matching"""
    if flowid_data.empty:
        print("No FlowID data available")
        return dataset
    
    print(f"Joining with FlowID data: {len(dataset)} dataset rows, {len(flowid_data)} FlowID rows")
    
    # Create KDTree for fast nearest neighbor search
    flowid_tree = cKDTree(flowid_data[['timestamp']].values)
    
    # Find nearest timestamp matches
    distances, indices = flowid_tree.query(dataset[['timestamp']].values, k=1)
    
    # Set distance threshold (e.g., 0.1 seconds)
    threshold = 0.1
    valid_matches = distances < threshold
    
    # Add FlowID column
    dataset['FlowID'] = None
    dataset.loc[valid_matches, 'FlowID'] = flowid_data.iloc[indices[valid_matches]]['FlowID'].values
    
    matched_count = valid_matches.sum()
    print(f"FlowID matching: {matched_count}/{len(dataset)} rows matched within threshold {threshold}")
    
    return dataset

def join_with_switch_id(dataset, switch_data):
    """Join dataset with SWITCH_ID data using timestamp matching"""
    if switch_data.empty:
        print("No SWITCH_ID data available")
        return dataset
    
    print(f"Joining with SWITCH_ID data: {len(dataset)} dataset rows, {len(switch_data)} SWITCH_ID rows")
    
    # Create KDTree for fast nearest neighbor search
    switch_tree = cKDTree(switch_data[['timestamp']].values)
    
    # Find nearest timestamp matches
    distances, indices = switch_tree.query(dataset[['timestamp']].values, k=1)
    
    # Set distance threshold (e.g., 0.1 seconds)
    threshold = 0.1
    valid_matches = distances < threshold
    
    # Add SWITCH_ID column
    dataset['SWITCH_ID'] = None
    dataset.loc[valid_matches, 'SWITCH_ID'] = switch_data.iloc[indices[valid_matches]]['SWITCH_ID'].values
    
    matched_count = valid_matches.sum()
    print(f"SWITCH_ID matching: {matched_count}/{len(dataset)} rows matched within threshold {threshold}")
    
    return dataset

def join_with_packet_action(dataset, packet_action_data):
    """Join dataset with PacketAction data using timestamp matching"""
    if packet_action_data.empty:
        print("No PacketAction data available")
        return dataset
    
    print(f"Joining with PacketAction data: {len(dataset)} dataset rows, {len(packet_action_data)} PacketAction rows")
    
    # Create KDTree for fast nearest neighbor search
    action_tree = cKDTree(packet_action_data[['timestamp']].values)
    
    # Find nearest timestamp matches
    distances, indices = action_tree.query(dataset[['timestamp']].values, k=1)
    
    # Set distance threshold (e.g., 0.1 seconds)
    threshold = 0.1
    valid_matches = distances < threshold
    
    # Add PacketAction column (0=forward, 1=deflection)
    dataset['PacketAction'] = 0  # Default to forward
    dataset.loc[valid_matches, 'PacketAction'] = packet_action_data.iloc[indices[valid_matches]]['PacketAction'].values
    
    matched_count = valid_matches.sum()
    deflection_count = (dataset['PacketAction'] == 1).sum()
    print(f"PacketAction matching: {matched_count}/{len(dataset)} rows matched within threshold {threshold}")
    print(f"Deflection events found: {deflection_count} out of {len(dataset)} observations")
    
    return dataset

def calculate_fct_from_timing(dataset):
    """Calculate FCT from start_time and end_time columns if available"""
    if 'start_time' in dataset.columns and 'end_time' in dataset.columns:
        print("Calculating FCT from start_time and end_time columns")
        dataset['FCT'] = dataset['end_time'] - dataset['start_time']
        fct_count = dataset['FCT'].notna().sum()
        print(f"Calculated FCT for {fct_count} rows")
    else:
        print("No start_time/end_time columns found, skipping FCT calculation")
    return dataset

def create_flow_level_data(dataset):
    """Create flow-level observations by grouping switch-level data by FlowID or seq_num"""
    print("Creating flow-level observations...")
    
    # Use FlowID if available, otherwise fall back to seq_num
    if 'FlowID' in dataset.columns and dataset['FlowID'].notna().any():
        group_col = 'FlowID'
        print(f"Grouping by FlowID: {dataset['FlowID'].notna().sum()} rows have FlowID")
    else:
        group_col = 'seq_num'
        print(f"Grouping by seq_num: {len(dataset)} rows")
    
    # Group by flow identifier
    flow_data = []
    grouped = dataset.groupby(group_col)
    
    for flow_id, group in grouped:
        if pd.isna(flow_id):
            continue
            
        # Calculate flow timing
        flow_start = group['timestamp'].min()
        flow_end = group['timestamp'].max()
        flow_fct = flow_end - flow_start
        
        # Calculate deflection statistics
        deflection_count = (group['PacketAction'] == 1).sum() if 'PacketAction' in group.columns else 0
        total_observations = len(group)
        deflection_rate = deflection_count / total_observations if total_observations > 0 else 0
        
        # Get other flow characteristics
        flow_info = {
            'FlowID': flow_id if 'FlowID' in dataset.columns else None,
            'seq_num': flow_id if 'FlowID' not in dataset.columns else group['seq_num'].iloc[0],
            'start_time': flow_start,
            'end_time': flow_end,
            'FCT': flow_fct,
            'packet_count': len(group),
            'deflection_count': deflection_count,
            'deflection_rate': deflection_rate,
            'deflection_threshold': group['deflection_threshold'].iloc[0] if 'deflection_threshold' in group.columns else None
        }
        
        # Add SWITCH_ID if available (take mode)
        if 'SWITCH_ID' in group.columns:
            switch_counts = group['SWITCH_ID'].value_counts()
            if len(switch_counts) > 0:
                flow_info['SWITCH_ID'] = switch_counts.index[0]
        
        # Add other columns that should be preserved at flow level
        for col in ['timestamp']:
            if col in group.columns:
                flow_info[col] = flow_start  # Use flow start time
        
        flow_data.append(flow_info)
    
    result = pd.DataFrame(flow_data)
    print(f"Created {len(result)} flow-level observations from {len(dataset)} packet-level records")
    return result

def main():
    parser = argparse.ArgumentParser(description='Enrich dataset with FlowID, SWITCH_ID, and PacketAction')
    parser.add_argument('input_file', help='Input CSV file (output from filter_overlapping_timestamps.py)')
    parser.add_argument('results_dir', help='Results directory containing FLOW_ID, SWITCH_ID, and PACKET_ACTION subdirectories')
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
    
    # Load PacketAction data
    packet_action_dir = os.path.join(args.results_dir, 'PACKET_ACTION')
    print(f"Loading PacketAction data from: {packet_action_dir}")
    packet_action_data = load_csvr_packet_action_vecs(packet_action_dir, 'PacketAction')
    print(f"PacketAction data shape: {packet_action_data.shape}")
    
    # Join with FlowID data
    dataset = join_with_flowid(dataset, flowid_data)
    
    # Join with SWITCH_ID data
    dataset = join_with_switch_id(dataset, switch_data)
    
    # Join with PacketAction data
    dataset = join_with_packet_action(dataset, packet_action_data)
    
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
    if 'deflection_count' in flow_dataset.columns:
        total_deflections = flow_dataset['deflection_count'].sum()
        total_packets = flow_dataset['packet_count'].sum()
        overall_deflection_rate = total_deflections / total_packets if total_packets > 0 else 0
        print(f"Total deflections: {total_deflections} out of {total_packets} packets ({overall_deflection_rate:.4f} rate)")

if __name__ == "__main__":
    main()
