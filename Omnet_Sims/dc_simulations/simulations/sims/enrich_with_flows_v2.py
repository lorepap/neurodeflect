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
    pattern = os.path.join(folder, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['timestamp', 'seq_num', name])
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
                                            vector_data.append([float(ts), float(fid)])
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
    pattern = os.path.join(folder, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['timestamp', name])
    dfs = []
    for fp in files:
        try:
            vector_data = []
            with open(fp, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    # Look for switchId vector data rows
                    if len(row) >= 8 and row[1] == 'vector' and 'switchId:vector' in row[3]:
                        # Parse module name to determine switch type and ID
                        module_name = row[2] if len(row) >= 3 else ""
                        switch_index = None
                        
                        if 'agg[' in module_name:
                            # Extract aggregate switch index
                            start = module_name.find('agg[') + 4
                            end = module_name.find(']', start)
                            switch_index = int(module_name[start:end])
                        elif 'spine[' in module_name:
                            # Extract spine switch index and add offset of 10
                            start = module_name.find('spine[') + 6
                            end = module_name.find(']', start)
                            switch_index = int(module_name[start:end]) + 10  # Add offset to avoid overlap
                        else:
                            continue
                        
                        # Extract timestamps and use switch_index as value
                        if len(row) >= 9:
                            timestamps_str = row[7].strip('"')  # Column 7 has timestamps
                            values_str = row[8].strip('"')      # Column 8 has values (ignored)
                            
                            if timestamps_str and values_str:
                                timestamps = timestamps_str.split()
                                values = values_str.split()
                                
                                if len(timestamps) == len(values):
                                    for ts, val in zip(timestamps, values):
                                        try:
                                            vector_data.append([float(ts), float(switch_index)])
                                        except ValueError:
                                            continue
            
            if vector_data:
                dfi = pd.DataFrame(vector_data, columns=['timestamp', name])
                print(f"Loaded {len(dfi)} switch ID entries from {os.path.basename(fp)}")
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

def enrich_merged_csv_with_flows(merged_file: str, output_file: str):
    """Main function to enrich merged_final.csv with flow information using FlowID-based joining"""
    
    # Load base data (merged_final.csv)
    print(f"Loading base data from {merged_file}")
    base_df = pd.read_csv(merged_file)
    print(f"Base data shape: {base_df.shape}")
    
    # Verify seq_num column exists for fallback
    if 'seq_num' not in base_df.columns:
        print("ERROR: No seq_num column found in base data - cannot proceed")
        return None
    
    print(f"Base data seq_num range: {base_df['seq_num'].min()} to {base_df['seq_num'].max()}")
    print(f"Base data unique packets: {base_df['seq_num'].nunique()}")
    
    # Load FlowID data (primary approach)
    print("Loading FlowID data for true flow identification...")
    flowid_df = load_csvr_flowid_vecs('./FLOW_ID', 'FLOW_ID')
    print(f"FlowID data shape: {flowid_df.shape}")
    
    # Load SWITCH_ID data for switch routing information
    print("Loading SWITCH_ID data...")
    switch_df = load_csvr_vecs_with_seqnum('./SWITCH_ID', 'SWITCH_ID')
    print(f"SWITCH_ID data shape: {switch_df.shape}")
    
    # Load sequence number data for joining
    print("Loading SWITCH_SEQ_NUM data for packet identification...")
    seq_df = load_csvs_vecs('./SWITCH_SEQ_NUM', 'SWITCH_SEQ_NUM')
    print(f"SWITCH_SEQ_NUM data shape: {seq_df.shape}")
    
    enriched_df = base_df.copy()
    
    # Strategy 1: Use FlowID data for true flow grouping (preferred)
    if len(flowid_df) > 0 and len(seq_df) > 0 and len(switch_df) > 0:
        print("\nUsing FlowID-based approach for flow identification...")
        
        # Step 1: Create timestamp -> seq_num mapping
        tolerance = 0.001
        seq_timestamps = seq_df['timestamp'].values.reshape(-1, 1)
        seq_tree = cKDTree(seq_timestamps)
        
        # Step 2: Join FlowID with seq_num using timestamps
        flowid_seqnum_mapping = []
        for _, flowid_row in flowid_df.iterrows():
            flowid_ts = flowid_row['timestamp']
            flow_id_val = flowid_row['FLOW_ID']
            
            # Find closest seq_num data
            distances, indices = seq_tree.query([[flowid_ts]], k=1)
            closest_distance = distances[0]
            closest_idx = indices[0]
            
            if closest_distance <= tolerance:
                seq_row = seq_df.iloc[closest_idx]
                seq_num_val = seq_row['SWITCH_SEQ_NUM']
                
                flowid_seqnum_mapping.append({
                    'seq_num': seq_num_val,
                    'flow_id': flow_id_val,
                    'timestamp': flowid_ts
                })
        
        flowid_mapping_df = pd.DataFrame(flowid_seqnum_mapping)
        print(f"Created {len(flowid_mapping_df)} seq_num -> flow_id mappings")
        
        if len(flowid_mapping_df) > 0:
            print(f"FlowID range: {flowid_mapping_df['flow_id'].min()} to {flowid_mapping_df['flow_id'].max()}")
            print(f"Unique flows: {flowid_mapping_df['flow_id'].nunique()}")
            
            # Step 3: Join base data with FlowID mapping
            enriched_df = enriched_df.merge(
                flowid_mapping_df[['seq_num', 'flow_id']], 
                on='seq_num', 
                how='left'
            )
            
            flow_coverage = enriched_df['flow_id'].notna().sum() / len(enriched_df) * 100
            print(f"FlowID coverage: {flow_coverage:.1f}% of packets")
            
            # Step 4: Add switch routing information
            # Create timestamp -> switch_id mapping
            switch_timestamps = switch_df['timestamp'].values.reshape(-1, 1)
            switch_tree = cKDTree(switch_timestamps)
            
            # Join FlowID timestamps with switch data
            flowid_switch_mapping = []
            for _, mapping_row in flowid_mapping_df.iterrows():
                mapping_ts = mapping_row['timestamp']
                seq_num_val = mapping_row['seq_num']
                flow_id_val = mapping_row['flow_id']
                
                # Find all switch observations for this timestamp
                distances, indices = switch_tree.query([[mapping_ts]], k=10)  # Get up to 10 closest
                
                for distance, idx in zip(distances, indices):
                    if distance <= tolerance:
                        switch_row = switch_df.iloc[idx]
                        switch_id_val = switch_row['SWITCH_ID']
                        
                        flowid_switch_mapping.append({
                            'seq_num': seq_num_val,
                            'flow_id': flow_id_val,
                            'switch_id': switch_id_val
                        })
            
            if flowid_switch_mapping:
                switch_mapping_df = pd.DataFrame(flowid_switch_mapping)
                print(f"Created {len(switch_mapping_df)} flow -> switch mappings")
                print(f"Unique switches: {switch_mapping_df['switch_id'].nunique()}")
                
                # Create expanded dataset with one row per (packet, switch) combination
                # This represents each packet's journey through multiple switches
                expanded_data = []
                for _, base_row in enriched_df.iterrows():
                    if pd.notna(base_row['flow_id']):
                        # Find all switch observations for this packet
                        packet_switches = switch_mapping_df[
                            (switch_mapping_df['seq_num'] == base_row['seq_num'])
                        ]
                        
                        if len(packet_switches) > 0:
                            # Create one row per switch
                            for _, switch_row in packet_switches.iterrows():
                                expanded_row = base_row.copy()
                                expanded_row['SWITCH_ID'] = switch_row['switch_id']
                                expanded_data.append(expanded_row)
                        else:
                            # No switch data for this packet
                            expanded_row = base_row.copy()
                            expanded_row['SWITCH_ID'] = None
                            expanded_data.append(expanded_row)
                    else:
                        # No flow ID for this packet
                        expanded_row = base_row.copy()
                        expanded_row['SWITCH_ID'] = None
                        expanded_data.append(expanded_row)
                
                enriched_df = pd.DataFrame(expanded_data)
                switch_coverage = enriched_df['SWITCH_ID'].notna().sum() / len(enriched_df) * 100
                print(f"Switch coverage: {switch_coverage:.1f}% of observations")
            else:
                enriched_df['SWITCH_ID'] = None
                print("No switch routing data available")
        else:
            print("No FlowID mappings created - using fallback approach")
            enriched_df['flow_id'] = enriched_df['seq_num'].apply(lambda x: int(x) // 1000)
            enriched_df['SWITCH_ID'] = None
    
    else:
        print("\nFalling back to seq_num-based flow grouping...")
        # Fallback: Use seq_num grouping (remove sequence number variation)
        enriched_df['flow_id'] = enriched_df['seq_num'].apply(lambda x: int(x) // 1000)
        enriched_df['SWITCH_ID'] = None
        print(f"Created {enriched_df['flow_id'].nunique()} flows from seq_num grouping")
    
    # Final summary
    print(f"\nFinal enriched data shape: {enriched_df.shape}")
    print("Column summary:")
    if 'flow_id' in enriched_df.columns:
        print(f"  Unique flows: {enriched_df['flow_id'].nunique()}")
        print(f"  Flow coverage: {enriched_df['flow_id'].notna().sum()} / {len(enriched_df)} observations")
    if 'SWITCH_ID' in enriched_df.columns:
        print(f"  Unique switches: {enriched_df['SWITCH_ID'].nunique()}")
        print(f"  Switch coverage: {enriched_df['SWITCH_ID'].notna().sum()} / {len(enriched_df)} observations")
    
    # Save enriched data
    print(f"Saving enriched data to {output_file}")
    enriched_df.to_csv(output_file, index=False)
    print("FlowID-based enrichment complete!")
    
    return enriched_df

def main():
    """Run the FlowID-based enrichment"""
    ap = argparse.ArgumentParser(description="Enrich merged CSV with flow information using FlowID")
    ap.add_argument('--base', required=True, help='Input merged CSV file')
    ap.add_argument('--out', required=True, help='Output enriched CSV file')
    ap.add_argument('--dir', default='extracted_results', help='Results directory')
    args = ap.parse_args()
    
    # Change to the results directory if provided
    if args.dir and os.path.exists(args.dir):
        os.chdir(args.dir)
        print(f"Changed working directory to: {args.dir}")
    
    # Run enrichment
    result = enrich_merged_csv_with_flows(args.base, args.out)
    
    if result is not None:
        print(f"Successfully created enriched dataset: {args.out}")
        print(f"Final dataset has {len(result)} rows")
        if 'flow_id' in result.columns:
            print(f"Identified {result['flow_id'].nunique()} unique flows")
        if 'SWITCH_ID' in result.columns:
            unique_switches = result['SWITCH_ID'].nunique()
            switch_coverage = result['SWITCH_ID'].notna().sum() / len(result) * 100
            print(f"Switch coverage: {switch_coverage:.1f}% with {unique_switches} unique switches")
    else:
        print("Enrichment failed!")

if __name__ == '__main__':
    main()
