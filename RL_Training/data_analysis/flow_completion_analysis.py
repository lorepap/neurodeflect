#!/usr/bin/env python3
"""
Flow Completion Analysis Script

Analyzes flow completion statistics from OMNeT++ simulation data:
- Number of flows started vs. completed
- Flow completion times (FCT)
- Flow sizes in packets
- Completion rates and statistics

Reads data from:
- REQUESTER_ID: All deflection decisions (packet-level)
- FLOW_STARTED: Flow start events (server-side)
- FLOW_ENDED: Flow completion events (server-side)
"""

import os
import pandas as pd
import numpy as np
import argparse
from enrich_packets_with_flows import load_wide_csv_requester_id
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_flow_completion(base_path, output_dir=None):
    """
    Analyze flow completion statistics for a given experiment directory
    
    Args:
        base_path: Path to experiment directory (e.g., results_1G_thr_15000)
        output_dir: Optional output directory for results
    """
    print(f"Analyzing flow completion for: {base_path}")
    
    # Load all three datasets
    print("\n=== Loading Data ===")
    requester_df = load_wide_csv_requester_id(os.path.join(base_path, 'REQUESTER_ID'), 'RequesterID')
    flow_started_df = load_wide_csv_requester_id(os.path.join(base_path, 'FLOW_STARTED'), 'flowStarted')
    flow_ended_df = load_wide_csv_requester_id(os.path.join(base_path, 'FLOW_ENDED'), 'flowEnded')
    
    print(f"RequesterID records: {len(requester_df):,}")
    print(f"Flow started records: {len(flow_started_df):,}")
    print(f"Flow ended records: {len(flow_ended_df):,}")
    
    # Analyze packet-level flow activity
    print("\n=== Packet-Level Flow Analysis ===")
    unique_flows_in_packets = requester_df['RequesterID'].nunique()
    total_packets = len(requester_df)
    print(f"Unique flows observed in packet traces: {unique_flows_in_packets:,}")
    print(f"Total packets processed: {total_packets:,}")
    
    # Count packets per flow
    packets_per_flow = requester_df['RequesterID'].value_counts().sort_index()
    print(f"Average packets per flow: {packets_per_flow.mean():.2f}")
    print(f"Median packets per flow: {packets_per_flow.median():.2f}")
    print(f"Max packets in a flow: {packets_per_flow.max():,}")
    print(f"Min packets in a flow: {packets_per_flow.min():,}")
    
    # Analyze flow start/end timing
    print("\n=== Flow Timing Analysis ===")
    unique_flows_started = flow_started_df['flowStarted'].nunique()
    unique_flows_ended = flow_ended_df['flowEnded'].nunique()
    
    print(f"Unique flows started: {unique_flows_started:,}")
    print(f"Unique flows ended: {unique_flows_ended:,}")
    
    # Find flows that both started and ended (completed flows)
    started_flows = set(flow_started_df['flowStarted'].unique())
    ended_flows = set(flow_ended_df['flowEnded'].unique())
    completed_flows = started_flows.intersection(ended_flows)
    incomplete_flows = started_flows - ended_flows
    
    print(f"Flows that completed: {len(completed_flows):,}")
    print(f"Flows that started but didn't complete: {len(incomplete_flows):,}")
    
    completion_rate = len(completed_flows) / len(started_flows) * 100 if started_flows else 0
    print(f"Flow completion rate: {completion_rate:.2f}%")
    
    # Calculate FCT for completed flows
    print("\n=== Flow Completion Time (FCT) Analysis ===")
    fct_data = []
    
    for flow_id in completed_flows:
        # Get start time (earliest start event for this flow)
        start_events = flow_started_df[flow_started_df['flowStarted'] == flow_id]
        end_events = flow_ended_df[flow_ended_df['flowEnded'] == flow_id]
        
        if len(start_events) > 0 and len(end_events) > 0:
            start_time = start_events['timestamp'].min()
            end_time = end_events['timestamp'].max()
            fct = end_time - start_time
            
            # Get packet count for this flow
            flow_packets = packets_per_flow.get(flow_id, 0)
            
            fct_data.append({
                'flow_id': flow_id,
                'start_time': start_time,
                'end_time': end_time,
                'fct': fct,
                'packet_count': flow_packets
            })
    
    if fct_data:
        fct_df = pd.DataFrame(fct_data)
        
        print(f"FCT analysis for {len(fct_df)} completed flows:")
        print(f"Average FCT: {fct_df['fct'].mean():.6f} seconds")
        print(f"Median FCT: {fct_df['fct'].median():.6f} seconds")
        print(f"Min FCT: {fct_df['fct'].min():.6f} seconds")
        print(f"Max FCT: {fct_df['fct'].max():.6f} seconds")
        print(f"FCT std dev: {fct_df['fct'].std():.6f} seconds")
        
        # FCT percentiles
        percentiles = [50, 90, 95, 99]
        print("\nFCT Percentiles:")
        for p in percentiles:
            pct_value = np.percentile(fct_df['fct'], p)
            print(f"  {p}th percentile: {pct_value:.6f} seconds")
        
        # Packet count analysis for completed flows
        print(f"\nPacket counts for completed flows:")
        print(f"Average packets: {fct_df['packet_count'].mean():.2f}")
        print(f"Median packets: {fct_df['packet_count'].median():.2f}")
        print(f"Total packets in completed flows: {fct_df['packet_count'].sum():,}")
        
        # Flow size distribution
        print(f"\nFlow size distribution (packet counts):")
        size_bins = [1, 5, 10, 20, 50, 100, float('inf')]
        size_labels = ['1', '2-5', '6-10', '11-20', '21-50', '51-100', '100+']
        
        for i in range(len(size_bins)-1):
            min_size = size_bins[i] if i > 0 else 1
            max_size = size_bins[i+1]
            
            if max_size == float('inf'):
                count = len(fct_df[fct_df['packet_count'] > min_size])
            else:
                count = len(fct_df[(fct_df['packet_count'] >= min_size) & 
                                 (fct_df['packet_count'] < max_size)])
            
            percentage = count / len(fct_df) * 100
            print(f"  {size_labels[i]} packets: {count} flows ({percentage:.1f}%)")
    
    else:
        print("No completed flows found for FCT analysis")
    
    # Analyze flows with packets but no start/end events
    print("\n=== Flow Coverage Analysis ===")
    flows_with_packets = set(requester_df['RequesterID'].unique())
    flows_only_packets = flows_with_packets - started_flows
    flows_started_no_packets = started_flows - flows_with_packets
    
    print(f"Flows with packets but no start event: {len(flows_only_packets):,}")
    print(f"Flows with start event but no packets: {len(flows_started_no_packets):,}")
    
    if len(flows_only_packets) > 0:
        print(f"  Sample flows with packets only: {sorted(list(flows_only_packets))[:10]}")
    
    if len(flows_started_no_packets) > 0:
        print(f"  Sample flows started without packets: {sorted(list(flows_started_no_packets))[:10]}")
    
    # Summary statistics
    print("\n=== SUMMARY ===")
    print(f"Total unique flows observed: {len(flows_with_packets | started_flows):,}")
    print(f"Flows started: {len(started_flows):,}")
    print(f"Flows completed: {len(completed_flows):,}")
    print(f"Flow completion rate: {completion_rate:.2f}%")
    print(f"Total packets: {total_packets:,}")
    print(f"Packets in completed flows: {fct_df['packet_count'].sum() if fct_data else 0:,}")
    
    if fct_data:
        packets_completed_rate = fct_df['packet_count'].sum() / total_packets * 100
        print(f"Packet completion rate: {packets_completed_rate:.2f}%")
        print(f"Average FCT: {fct_df['fct'].mean():.6f} seconds")
        print(f"Average flow size: {fct_df['packet_count'].mean():.2f} packets")
    
    # Save results if output directory specified
    if output_dir and fct_data:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save FCT data
        fct_df.to_csv(os.path.join(output_dir, 'flow_completion_times.csv'), index=False)
        
        # Save summary stats
        summary_stats = {
            'total_flows_observed': len(flows_with_packets | started_flows),
            'flows_started': len(started_flows),
            'flows_completed': len(completed_flows),
            'completion_rate_percent': completion_rate,
            'total_packets': total_packets,
            'packets_in_completed_flows': fct_df['packet_count'].sum(),
            'packet_completion_rate_percent': fct_df['packet_count'].sum() / total_packets * 100,
            'avg_fct_seconds': fct_df['fct'].mean(),
            'median_fct_seconds': fct_df['fct'].median(),
            'avg_flow_size_packets': fct_df['packet_count'].mean(),
            'median_flow_size_packets': fct_df['packet_count'].median()
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(os.path.join(output_dir, 'flow_analysis_summary.csv'), index=False)
        
        print(f"\nResults saved to: {output_dir}")
    
    return fct_df if fct_data else None

def main():
    parser = argparse.ArgumentParser(description='Analyze flow completion statistics')
    parser.add_argument('experiment_dir', help='Path to experiment directory (e.g., results_1G_thr_15000)', 
                        default='/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/results_1G_thr_15000')
    parser.add_argument('--output', '-o', help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory not found: {args.experiment_dir}")
        return 1
    
    analyze_flow_completion(args.experiment_dir, args.output)
    return 0

if __name__ == '__main__':
    exit(main())
