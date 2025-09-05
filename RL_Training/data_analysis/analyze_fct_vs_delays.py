#!/usr/bin/env python3
"""
FCT vs Packet One-Way Delay Analysis

This script analyzes the relationship between Flow Completion Time (FCT) 
and individual packet one-way delays for a specific flow.

Analysis includes:
1. FCT computation verification
2. Packet one-way delay distribution
3. Comparison between FCT and packet delays
4. Visualization of packet timing within the flow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def analyze_flow_timing(dataset_file, flow_id=None, threshold=None, output_dir="flow_analysis"):
    """
    Analyze FCT vs packet delays for a specific flow
    
    Args:
        dataset_file: Path to the threshold dataset CSV
        flow_id: Specific flow ID to analyze (if None, picks first flow)
        threshold: Threshold value for context
        output_dir: Directory to save analysis results
    """
    
    print(f"Loading dataset: {dataset_file}")
    df = pd.read_csv(dataset_file)
    
    print(f"Dataset loaded: {len(df)} packets")
    print(f"Unique flows: {df['FlowID'].nunique()}")
    print(f"Threshold: {threshold}")
    
    # Select a flow to analyze
    if flow_id is None:
        # Pick the first flow with multiple packets
        flow_counts = df['FlowID'].value_counts()
        multi_packet_flows = flow_counts[flow_counts > 1]
        if len(multi_packet_flows) == 0:
            print("No flows with multiple packets found!")
            return
        flow_id = multi_packet_flows.index[0]
        print(f"Auto-selected FlowID: {flow_id} ({multi_packet_flows.iloc[0]} packets)")
    else:
        print(f"Analyzing FlowID: {flow_id}")
    
    # Extract the specific flow
    flow_df = df[df['FlowID'] == flow_id].copy()
    
    if len(flow_df) == 0:
        print(f"Flow {flow_id} not found!")
        return
    
    # Sort by packet position
    flow_df = flow_df.sort_values('packet_position').reset_index(drop=True)
    
    print(f"\nFlow Analysis for FlowID: {flow_id}")
    print(f"Total packets: {len(flow_df)}")
    print(f"FCT: {flow_df['FCT'].iloc[0]:.6f} seconds")
    print(f"Flow start time: {flow_df['flow_start_time'].iloc[0]:.6f}")
    print(f"Flow end time: {flow_df['flow_end_time'].iloc[0]:.6f}")
    print(f"Deflection count: {flow_df['deflection_count'].iloc[0]}")
    print(f"Deflection rate: {flow_df['deflection_rate'].iloc[0]:.3f}")
    
    # Calculate packet one-way delays
    flow_start = flow_df['flow_start_time'].iloc[0]
    flow_df['packet_delay'] = flow_df['timestamp'] - flow_start
    flow_df['packet_relative_time'] = flow_df['timestamp'] - flow_start
    
    # FCT verification
    computed_fct = flow_df['flow_end_time'].iloc[0] - flow_df['flow_start_time'].iloc[0]
    stored_fct = flow_df['FCT'].iloc[0]
    
    print(f"\nFCT Verification:")
    print(f"Stored FCT: {stored_fct:.6f} seconds")
    print(f"Computed FCT (end - start): {computed_fct:.6f} seconds")
    print(f"FCT difference: {abs(stored_fct - computed_fct):.9f} seconds")
    print(f"FCT matches: {'✓' if abs(stored_fct - computed_fct) < 1e-9 else '✗'}")
    
    # Packet delay statistics
    print(f"\nPacket Delay Distribution:")
    print(f"Min delay: {flow_df['packet_delay'].min():.6f} seconds")
    print(f"Max delay: {flow_df['packet_delay'].max():.6f} seconds")
    print(f"Mean delay: {flow_df['packet_delay'].mean():.6f} seconds")
    print(f"Std delay: {flow_df['packet_delay'].std():.6f} seconds")
    
    # Check if FCT equals max packet delay
    max_packet_delay = flow_df['packet_delay'].max()
    print(f"\nFCT vs Max Packet Delay:")
    print(f"FCT: {stored_fct:.6f} seconds")
    print(f"Max packet delay: {max_packet_delay:.6f} seconds")
    print(f"Difference: {abs(stored_fct - max_packet_delay):.9f} seconds")
    print(f"FCT = Max delay: {'✓' if abs(stored_fct - max_packet_delay) < 1e-9 else '✗'}")
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Flow Analysis - FlowID: {flow_id}, Threshold: {threshold}', fontsize=16)
    
    # 1. Packet delays over packet position
    axes[0,0].plot(flow_df['packet_position'].values, flow_df['packet_delay'].values, 'bo-', markersize=6)
    axes[0,0].axhline(y=stored_fct, color='r', linestyle='--', label=f'FCT: {stored_fct:.6f}s')
    axes[0,0].set_xlabel('Packet Position')
    axes[0,0].set_ylabel('Packet One-Way Delay (s)')
    axes[0,0].set_title('Packet Delays vs Position')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Packet delay distribution
    axes[0,1].hist(flow_df['packet_delay'].values, bins=min(20, len(flow_df)), alpha=0.7, edgecolor='black')
    axes[0,1].axvline(x=stored_fct, color='r', linestyle='--', label=f'FCT: {stored_fct:.6f}s')
    axes[0,1].axvline(x=max_packet_delay, color='g', linestyle='--', label=f'Max delay: {max_packet_delay:.6f}s')
    axes[0,1].set_xlabel('Packet One-Way Delay (s)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Packet Delay Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Deflection pattern
    deflection_colors = ['green' if x == 0 else 'red' for x in flow_df['action'].values]
    axes[1,0].scatter(flow_df['packet_position'].values, flow_df['packet_delay'].values, 
                     c=deflection_colors, s=60, alpha=0.7)
    axes[1,0].axhline(y=stored_fct, color='r', linestyle='--', alpha=0.5)
    axes[1,0].set_xlabel('Packet Position')
    axes[1,0].set_ylabel('Packet One-Way Delay (s)')
    axes[1,0].set_title('Deflection Pattern (Green=Normal, Red=Deflected)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Cumulative delay
    flow_df['cumulative_time'] = flow_df['packet_delay']
    axes[1,1].plot(flow_df['packet_position'].values, flow_df['cumulative_time'].values, 'bo-', markersize=4)
    axes[1,1].axhline(y=stored_fct, color='r', linestyle='--', label=f'FCT: {stored_fct:.6f}s')
    axes[1,1].set_xlabel('Packet Position')
    axes[1,1].set_ylabel('Packet Timestamp Relative to Flow Start (s)')
    axes[1,1].set_title('Packet Arrival Timeline')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = f"{output_dir}/flow_{flow_id}_analysis_threshold_{threshold}.png"
    try:
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved: {plot_file}")
    except Exception as e:
        print(f"\nPlot creation failed: {e}")
        print("Continuing with analysis...")
    
    # Save detailed packet data
    output_file = f"{output_dir}/flow_{flow_id}_packets_threshold_{threshold}.csv"
    flow_analysis = flow_df[['packet_position', 'timestamp', 'packet_delay', 'action', 
                            'deflection_count', 'deflection_rate', 'FCT', 'occupancy', 
                            'total_occupancy', 'capacity', 'total_capacity', 'SWITCH_ID']].copy()
    flow_analysis.to_csv(output_file, index=False)
    print(f"Detailed packet data saved: {output_file}")
    
    # Summary analysis
    print(f"\n" + "="*60)
    print(f"ANALYSIS SUMMARY")
    print(f"="*60)
    print(f"FlowID: {flow_id}")
    print(f"Threshold: {threshold}")
    print(f"Total packets: {len(flow_df)}")
    print(f"FCT: {stored_fct:.6f} seconds")
    print(f"FCT computation is correct: {'✓' if abs(stored_fct - computed_fct) < 1e-9 else '✗'}")
    print(f"FCT equals max packet delay: {'✓' if abs(stored_fct - max_packet_delay) < 1e-9 else '✗'}")
    print(f"Deflections in flow: {flow_df['deflection_count'].iloc[0]} out of {len(flow_df)} packets")
    print(f"Average packet delay: {flow_df['packet_delay'].mean():.6f} seconds")
    print(f"Packet delay range: {flow_df['packet_delay'].min():.6f} - {flow_df['packet_delay'].max():.6f} seconds")
    
    if abs(stored_fct - max_packet_delay) < 1e-9:
        print(f"\n✓ CONCLUSION: FCT correctly represents the time from flow start to last packet arrival")
    else:
        print(f"\n✗ CONCLUSION: FCT does NOT match max packet delay - potential timing issue")
    
    return flow_df

def main():
    parser = argparse.ArgumentParser(description="Analyze FCT vs packet one-way delays")
    parser.add_argument("--dataset", default="/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/threshold_dataset_15000.csv", 
                       help="Path to threshold dataset CSV file")
    parser.add_argument("--flow-id", type=int, help="Specific flow ID to analyze")
    parser.add_argument("--threshold", default="15000", help="Threshold value for context")
    parser.add_argument("--output-dir", default="flow_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    if not Path(args.dataset).exists():
        print(f"Dataset file not found: {args.dataset}")
        return
    
    analyze_flow_timing(args.dataset, args.flow_id, args.threshold, args.output_dir)

if __name__ == "__main__":
    main()
