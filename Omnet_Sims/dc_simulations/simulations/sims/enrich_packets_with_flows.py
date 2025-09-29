#!/usr/bin/env python3
"""
Packet-Level Flow Enrichment Script

This script enriches packet-level data with flow-level information.
Each row remains a packet, but gets additional flow context:
- RequesterID: Flow identifier from server that initiated the flow
- FCT: Flow Completion Time (calculated from server-side flow start/end signals)
- flow_start_time: When the flow started (from server logs)
- flow_end_time: When the flow ended (from server logs) 
- flow_packet_count: Total packets in this flow
- packet_position: Position of this packet within the flow (1, 2, 3, ...)
- SWITCH_ID: Switch where the packet was processed 

Input: Packet-level dataset with columns: timestamp, capacity, total_capacity, occupancy, total_occupancy, seq_num, ttl, packet_size, action
Output: Same packet-level data enriched with flow information
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
import traceback
from collections import defaultdict
import sys

def load_csvs_vecs(folder: str, name: str):
    """Load CSV-S format files (timestamp,value pairs)"""
    pattern = os.path.join(folder, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['timestamp', name])
    dfs = []
    for fp in files:
        try:
            dfi = pd.read_csv(fp)
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

def load_wide_csv_requester_id(folder: str, name: str):
    """Load RequesterID data from wide CSV format with multiple switches per file"""
    pattern = os.path.join(folder, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['timestamp', name])
    
    dfs = []
    for fp in files:
        try:
            print(f"Processing {os.path.basename(fp)} for {name}")
            
            # Read the CSV file
            df = pd.read_csv(fp)
            
            # The file has multiple columns representing different switches
            # Each pair of columns is (timestamp, RequesterID) for a specific switch
            data_rows = []
            
            # Process each row
            for idx, row in df.iterrows():
                # Extract timestamp,value pairs from each row
                # Skip first column (header info) and process in pairs
                row_values = row.values
                
                # Process pairs of columns: timestamp, value
                for i in range(0, len(row_values), 2):
                    if i + 1 < len(row_values):
                        try:
                            timestamp = pd.to_numeric(row_values[i], errors='coerce')
                            value = pd.to_numeric(row_values[i + 1], errors='coerce')
                            
                            if pd.notna(timestamp) and pd.notna(value):
                                data_rows.append([timestamp, value])
                        except:
                            continue
            
            if data_rows:
                dfi = pd.DataFrame(data_rows, columns=['timestamp', name])
                dfi = dfi.drop_duplicates().sort_values('timestamp').reset_index(drop=True)
                print(f"  Loaded {len(dfi)} {name} records")
                print(f"  Sample values: {sorted(dfi[name].unique())[:10]}")
                dfs.append(dfi)
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', name])
    
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates().sort_values('timestamp').reset_index(drop=True)
    return out

def load_csvr_flowid_vecs(folder: str, name: str):
    """Load CSV-R format files with vector data extraction, preserving switch identity"""
    pattern = os.path.join(folder, '*.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame(columns=['timestamp', name])
    
    dfs = []
    for fp in files:
        try:
            with open(fp, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                continue
                
            # Parse CSV-R format for vector data
            data_lines = []
            switch_counts = defaultdict(int)
            
            for line in lines:
                line = line.strip()
                if not line or not line.startswith('DCTCP_SD_THRESHOLD_VARIATION') or ',vector,' not in line:
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 9:
                    # Check if this is a vector line with vectime and vecvalue
                    if parts[1] == 'vector' and len(parts) >= 9:
                        try:
                            # Extract module name to identify the switch type and ID
                            module = parts[2] if len(parts) > 2 else 'unknown'
                            
                            # Extract vectime and vecvalue (last two columns)
                            vectime_str = parts[-2].strip('"')
                            vecvalue_str = parts[-1].strip('"')
                            
                            if vectime_str and vecvalue_str and module != 'unknown':
                                # Determine switch type and create unique identifier
                                if '.agg[' in module:
                                    # Extract agg switch number and create unique ID
                                    agg_num = module.split('.agg[')[1].split(']')[0]
                                    switch_unique_id = f"agg{agg_num}"
                                elif '.spine[' in module:
                                    # Extract spine switch number and create unique ID  
                                    spine_num = module.split('.spine[')[1].split(']')[0]
                                    switch_unique_id = f"spine{spine_num}"
                                else:
                                    continue  # Skip unknown switch types
                                
                                # Split the space-separated values
                                timestamps = [float(t) for t in vectime_str.split()]
                                values = [float(v) for v in vecvalue_str.split()]
                                
                                # Create timestamp-value pairs with unique switch ID
                                for timestamp, value in zip(timestamps, values):
                                    if name == 'SWITCH_ID':
                                        # For SWITCH_ID, use our unique identifier instead of raw value
                                        data_lines.append([timestamp, switch_unique_id])
                                        switch_counts[switch_unique_id] += 1
                                    else:
                                        # For other signals (like FlowID), keep original value
                                        data_lines.append([timestamp, value])
                                        switch_counts[value] += 1
                        except (ValueError, IndexError):
                            continue
            
            if data_lines:
                dfi = pd.DataFrame(data_lines, columns=['timestamp', name])
                print(f"Loaded {len(dfi)} rows from {os.path.basename(fp)} for {name}")
                if name == 'SWITCH_ID':
                    print(f"  Switch distribution: {dict(switch_counts)}")
                else:
                    print(f"  Value distribution: {len(switch_counts)} unique values")
                dfs.append(dfi)
                
        except Exception as e:
            print(f"Error processing {fp}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame(columns=['timestamp', name])
    
    out = pd.concat(dfs, ignore_index=True)
    out = out.sort_values('timestamp').reset_index(drop=True)
    return out

def enrich_packets_with_flows(input_file, results_dir, output_file, log_file=None):
    """Enrich packet-level data with flow information"""
    
    if log_file:
        log_f = open(log_file, 'w')
        def log(msg):
            print(msg)
            log_f.write(msg + '\n')
            log_f.flush()
    else:
        def log(msg):
            print(msg)
    
    try:
        log("Loading packet-level input dataset...")
        df = pd.read_csv(input_file)
        log(f"Loaded {len(df)} packet records")
        
        if len(df) == 0:
            log("No data to process")
            return False
        
        # Load RequesterID data from switch logs
        log("Loading RequesterID data from switch logs...")
        requester_id_df = load_wide_csv_requester_id(os.path.join(results_dir, 'REQUESTER_ID'), 'RequesterID')
        
        if len(requester_id_df) == 0:
            log("Warning: No RequesterID data found")
            df['RequesterID'] = np.nan
        else:
            log(f"Found {len(requester_id_df)} RequesterID records")
            
            # OPTIMIZED: Use pandas merge for fast timestamp-based matching
            log("Performing optimized RequesterID assignment...")
            
            # Round timestamps to avoid floating point precision issues
            df['timestamp_rounded'] = df['timestamp'].round(12)
            requester_id_df['timestamp_rounded'] = requester_id_df['timestamp'].round(12)
            
            # Use pandas merge for O(n log n) performance instead of O(n*m)
            df_with_requester = df.merge(
                requester_id_df[['timestamp_rounded', 'RequesterID']], 
                on='timestamp_rounded', 
                how='left'
            )
            
            # Count successful matches
            matched_count = df_with_requester['RequesterID'].notna().sum()
            unmatched_count = df_with_requester['RequesterID'].isna().sum()
            
            log(f"RequesterID assignment: {matched_count} matched, {unmatched_count} unmatched")
            
            # Update the main dataframe
            df = df_with_requester.drop(columns=['timestamp_rounded'])
        
        # Load switch ID data
        log("Loading switch ID data...")
        switch_id_df = load_csvr_flowid_vecs(os.path.join(results_dir, 'SWITCH_ID'), 'SWITCH_ID')
        
        if len(switch_id_df) == 0:
            log("Warning: No SWITCH_ID data found")
            df['SWITCH_ID'] = np.nan
        else:
            log(f"Found {len(switch_id_df)} SWITCH_ID records")
            
            # OPTIMIZED: Use pandas merge for fast timestamp-based matching
            log("Performing optimized SWITCH_ID assignment...")
            
            # Round timestamps to avoid floating point precision issues
            df['timestamp_rounded'] = df['timestamp'].round(12)
            switch_id_df['timestamp_rounded'] = switch_id_df['timestamp'].round(12)
            
            # Use pandas merge for O(n log n) performance
            df_with_switch = df.merge(
                switch_id_df[['timestamp_rounded', 'SWITCH_ID']], 
                on='timestamp_rounded', 
                how='left'
            )
            
            # Count successful matches
            matched_count = df_with_switch['SWITCH_ID'].notna().sum()
            unmatched_count = df_with_switch['SWITCH_ID'].isna().sum()
            
            log(f"SWITCH_ID assignment: {matched_count} matched, {unmatched_count} unmatched")
            
            # Update the main dataframe
            df = df_with_switch.drop(columns=['timestamp_rounded'])

        # Load flow start and end times from server logs
        log("Loading flow start times...")
        flow_started_df = load_wide_csv_requester_id(os.path.join(results_dir, 'FLOW_STARTED'), 'RequesterID')
        flow_started_df.rename(columns={'timestamp': 'flow_start_time'}, inplace=True)
        
        log("Loading flow end times...")
        flow_ended_df = load_wide_csv_requester_id(os.path.join(results_dir, 'FLOW_ENDED'), 'RequesterID')
        flow_ended_df.rename(columns={'timestamp': 'flow_end_time'}, inplace=True)
        
        # Merge flow timing data with packet data using RequesterID
        if len(flow_started_df) > 0:
            log(f"Found {len(flow_started_df)} flow start records")
            df = df.merge(flow_started_df, on='RequesterID', how='left')
        else:
            log("Warning: No flow start data found")
            df['flow_start_time'] = np.nan
            
        if len(flow_ended_df) > 0:
            log(f"Found {len(flow_ended_df)} flow end records")
            df = df.merge(flow_ended_df, on='RequesterID', how='left')
        else:
            log("Warning: No flow end data found")
            df['flow_end_time'] = np.nan
        
        # Calculate FCT (Flow Completion Time) for each packet row
        log("Calculating Flow Completion Time (FCT) for each packet...")
        df['FCT'] = df['flow_end_time'] - df['flow_start_time']
        
        # Add packet position within flow based on RequesterID
        log("Adding packet positions within flows...")
        df['packet_position'] = df.groupby('RequesterID').cumcount() + 1
        
        # Calculate deflection statistics per flow based on RequesterID
        # log("Calculating deflection statistics per flow...")
        # deflection_stats = df.groupby('RequesterID')['action'].agg(['sum', 'count']).reset_index()
        # deflection_stats.columns = ['RequesterID', 'deflection_count', 'total_packets']
        # deflection_stats['deflection_rate'] = deflection_stats['deflection_count'] / deflection_stats['total_packets']
        
        # Add deflection statistics to each packet row
        # df_enriched = df.merge(deflection_stats[['RequesterID', 'deflection_count', 'deflection_rate']], 
        #                       on='RequesterID', how='left')

        df_enriched = df.copy() # just removed the deflection statistics (not needed)
        
        # Reorder columns for better readability
        column_order = [
            'timestamp', 'RequesterID', 'seq_num', 'packet_position',
            'flow_start_time', 'flow_end_time', 'FCT',
            'action', 'capacity', 'total_capacity', 'occupancy', 'total_occupancy', 
            'ttl', 'packet_size', 'SWITCH_ID', 'ooo'  # Added OOO feature
        ]
        
        # Only include columns that exist - this preserves any additional columns like 'ooo'
        available_columns = [col for col in column_order if col in df_enriched.columns]
        
        # Add any remaining columns not in the predefined order (to preserve unexpected features)
        remaining_columns = [col for col in df_enriched.columns if col not in available_columns]
        available_columns.extend(remaining_columns)
        
        df_final = df_enriched[available_columns]

        # Sort by timestamp for chronological order
        df_final = df_final.sort_values('timestamp').reset_index(drop=True)

        log(f"Final dataset: {len(df_final)} packets across {df_final['RequesterID'].nunique()} flows")

        # Save the enriched dataset
        df_final.to_csv(output_file, index=False)
        log(f"Saved enriched packet-level dataset to {output_file}")

        if log_file:
            log_f.close()

        return True
        
    except Exception as e:
        log(f"Error during packet enrichment: {e}")
        import traceback
        log(traceback.format_exc())
        if log_file:
            log_f.close()
        return False

def main():
    parser = argparse.ArgumentParser(description="Enrich packet-level data with flow information")
    parser.add_argument("input_file", help="Input packet-level CSV file")
    parser.add_argument("results_dir", help="Directory containing extracted results")
    parser.add_argument("output_file", help="Output enriched CSV file")
    parser.add_argument("--log-file", help="Log file for detailed output")
    
    args = parser.parse_args()
    
    success = enrich_packets_with_flows(args.input_file, args.results_dir, args.output_file, args.log_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
