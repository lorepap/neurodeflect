#!/usr/bin/env python3
"""
Extract normalization parameters from the dataset for use in C++ inference.
"""

import pandas as pd
import numpy as np

def extract_normalization_params():
    """Extract state normalization parameters from the training dataset."""
    
    # Load the dataset
    dataset_path = "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        
        # Map dataset columns to state features based on the IQL training logic
        # From the training data, we need to construct the 6D state space:
        # 0: queue_utilization (occupancy / capacity)
        # 1: total_utilization (total_occupancy / total_capacity) 
        # 2: ttl_priority (normalized TTL)
        # 3: ooo_indicator (out-of-order indicator)
        # 4: packet_delay (flow duration or packet timing)
        # 5: fct_contribution (packet size normalized)
        
        # Calculate derived features
        state_data = []
        
        # Feature 0: Queue utilization
        if 'occupancy' in df.columns and 'capacity' in df.columns:
            queue_util = df['occupancy'] / (df['capacity'] + 1e-8)  # Add small epsilon to avoid division by zero
            state_data.append(queue_util.values)
        else:
            state_data.append(np.random.uniform(0, 1, len(df)))
            
        # Feature 1: Total utilization  
        if 'total_occupancy' in df.columns and 'total_capacity' in df.columns:
            total_util = df['total_occupancy'] / (df['total_capacity'] + 1e-8)
            state_data.append(total_util.values)
        else:
            state_data.append(np.random.uniform(0, 1, len(df)))
            
        # Feature 2: TTL priority (normalized)
        if 'ttl' in df.columns:
            ttl_normalized = df['ttl'] / 64.0  # Assuming max TTL of 64
            ttl_normalized = np.clip(ttl_normalized, 0, 1)
            state_data.append(ttl_normalized.values)
        else:
            state_data.append(np.random.uniform(0, 1, len(df)))
            
        # Feature 3: Out-of-order indicator
        if 'ooo' in df.columns:
            state_data.append(df['ooo'].values)
        else:
            state_data.append(np.random.uniform(0, 1, len(df)))
            
        # Feature 4: Packet delay (use FCT or flow duration)
        if 'FCT' in df.columns:
            fct_normalized = (df['FCT'] - df['FCT'].min()) / (df['FCT'].max() - df['FCT'].min() + 1e-8)
            state_data.append(fct_normalized.values)
        elif 'flow_end_time' in df.columns and 'flow_start_time' in df.columns:
            duration = df['flow_end_time'] - df['flow_start_time']
            duration_normalized = (duration - duration.min()) / (duration.max() - duration.min() + 1e-8)
            state_data.append(duration_normalized.values)
        else:
            state_data.append(np.random.uniform(0, 1, len(df)))
            
        # Feature 5: FCT contribution (packet size normalized)
        if 'packet_size' in df.columns:
            packet_size_normalized = df['packet_size'] / 1500.0  # Normalize by MTU
            packet_size_normalized = np.clip(packet_size_normalized, 0, 2)  # Allow up to 2x MTU
            state_data.append(packet_size_normalized.values)
        else:
            state_data.append(np.random.uniform(0, 1, len(df)))
        
        # Convert to array
        state_array = np.column_stack(state_data)
        
        # Calculate mean and std
        state_mean = np.mean(state_array, axis=0)
        state_std = np.std(state_array, axis=0)
        
        # Handle zero std (constant features)
        state_std = np.where(state_std == 0, 1.0, state_std)
        
        feature_names = [
            'queue_utilization',
            'total_utilization', 
            'ttl_priority',
            'ooo_indicator',
            'packet_delay',
            'fct_contribution'
        ]
        
        print("\nState Feature Statistics:")
        for i, name in enumerate(feature_names):
            print(f"{name}: mean={state_mean[i]:.6f}, std={state_std[i]:.6f}")
        
        print("\nNormalization Parameters:")
        print("State Mean:", state_mean)
        print("State Std:", state_std)
        
        # Format for OMNeT++ configuration
        mean_str = ",".join([f"{x:.6f}" for x in state_mean])
        std_str = ",".join([f"{x:.6f}" for x in state_std])
        
        print("\nOMNeT++ Configuration:")
        print(f'**.agg[*].rl_state_mean = "{mean_str}"')
        print(f'**.spine[*].rl_state_mean = "{mean_str}"')
        print(f'**.agg[*].rl_state_std = "{std_str}"')
        print(f'**.spine[*].rl_state_std = "{std_str}"')
        
        return state_mean, state_std
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using default normalization parameters")
        
        # Default parameters if dataset loading fails
        default_mean = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.5])
        default_std = np.array([0.2, 0.2, 0.3, 0.5, 1.0, 0.3])
        
        mean_str = ",".join([f"{x:.6f}" for x in default_mean])
        std_str = ",".join([f"{x:.6f}" for x in default_std])
        
        print("Default OMNeT++ Configuration:")
        print(f'**.agg[*].rl_state_mean = "{mean_str}"')
        print(f'**.spine[*].rl_state_mean = "{mean_str}"')
        print(f'**.agg[*].rl_state_std = "{std_str}"')
        print(f'**.spine[*].rl_state_std = "{std_str}"')
        
        return default_mean, default_std

if __name__ == "__main__":
    extract_normalization_params()
