#!/usr/bin/env python3
"""
Packet-Level Flow Enrichment Script

This script enriches packet-level data with flow-level information.
Each row remains a packet, but gets additional flow context:
- FlowID: Flow identifier for grouping packets
- FCT: Flow Completion Time (same for all packets in a flow)
- flow_start_time: When the flow started
- flow_end_time: When the flow ended
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
        
        # Load flow-related data
        log("Loading flow identification data...")
        flow_id_df = load_csvr_flowid_vecs(os.path.join(results_dir, 'FLOW_ID'), 'FlowID')
        
        if len(flow_id_df) == 0:
            log("Warning: No FlowID data found, will use seq_num for grouping")
            # Fallback: group by seq_num
            df['FlowID'] = df['seq_num']
        else:
            log(f"Found {len(flow_id_df)} FlowID records")
            
            # OPTIMIZED: Use pandas merge for fast timestamp-based matching
            log("Performing optimized FlowID assignment...")
            
            # Round timestamps to avoid floating point precision issues
            df['timestamp_rounded'] = df['timestamp'].round(12)
            flow_id_df['timestamp_rounded'] = flow_id_df['timestamp'].round(12)
            
            # Use pandas merge for O(n log n) performance instead of O(n*m)
            df_with_flow = df.merge(
                flow_id_df[['timestamp_rounded', 'FlowID']], 
                on='timestamp_rounded', 
                how='left'
            )
            
            # Count successful matches
            matched_count = df_with_flow['FlowID'].notna().sum()
            unmatched_count = df_with_flow['FlowID'].isna().sum()
            
            log(f"FlowID assignment: {matched_count} matched, {unmatched_count} unmatched")
            
            # For unmatched packets, use seq_num as fallback
            if unmatched_count > 0:
                log(f"Using seq_num fallback for {unmatched_count} unmatched packets")
                df_with_flow['FlowID'] = df_with_flow['FlowID'].fillna(df_with_flow['seq_num'])
            
            # Update the main dataframe
            df = df_with_flow.drop(columns=['timestamp_rounded'])
        
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
        
        # Calculate flow-level statistics
        log("Calculating flow-level statistics...")
        flow_stats = df.groupby('FlowID').agg({
            'timestamp': ['min', 'max', 'count'],
            'seq_num': 'first'
        }).round(12)
        
        # Flatten column names
        flow_stats.columns = ['flow_start_time', 'flow_end_time', 'flow_packet_count', 'first_seq_num']
        flow_stats['FCT'] = flow_stats['flow_end_time'] - flow_stats['flow_start_time']
        flow_stats = flow_stats.reset_index()
        
        log(f"Identified {len(flow_stats)} unique flows")
        log(f"Flow packet counts: min={flow_stats['flow_packet_count'].min()}, max={flow_stats['flow_packet_count'].max()}, mean={flow_stats['flow_packet_count'].mean():.1f}")
        
        # Add packet position within flow - PRESERVE ORIGINAL ORDER
        log("Adding packet positions within flows...")
        # IMPORTANT: Do NOT sort by timestamp here! This destroys the original network order
        # Instead, use the original data order which represents when packets arrived at the switch
        df['packet_position'] = df.groupby('FlowID').cumcount() + 1
        
        # Now we can create a sorted version for flow statistics calculations
        df_sorted = df.sort_values(['FlowID', 'timestamp']).reset_index(drop=True)
        
        # Merge flow statistics back to packet data
        log("Merging flow statistics with packet data...")
        df_enriched = df.merge(flow_stats[['FlowID', 'flow_start_time', 'flow_end_time', 'flow_packet_count', 'FCT']], 
                              on='FlowID', how='left')
        
        # Check for deflection statistics per flow
        deflection_stats = df_enriched.groupby('FlowID')['action'].agg(['sum', 'count']).reset_index()
        deflection_stats.columns = ['FlowID', 'deflection_count', 'total_packets']
        deflection_stats['deflection_rate'] = deflection_stats['deflection_count'] / deflection_stats['total_packets']
        
        # Add deflection statistics
        df_enriched = df_enriched.merge(deflection_stats[['FlowID', 'deflection_count', 'deflection_rate']], 
                                       on='FlowID', how='left')
        
        # Reorder columns for better readability
        column_order = [
            'timestamp', 'FlowID', 'seq_num', 'packet_position',
            'flow_start_time', 'flow_end_time', 'FCT', 'flow_packet_count',
            'deflection_count', 'deflection_rate',
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
        
        log(f"Final dataset: {len(df_final)} packets across {df_final['FlowID'].nunique()} flows")
        log(f"Deflection statistics: {df_final['action'].sum()} deflections out of {len(df_final)} packets ({df_final['action'].mean()*100:.1f}% deflection rate)")
        
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
