#!/usr/bin/env python3
"""
Script to filter merged CSV datasets based on seq_num collision detection.

This script now works with the merged dataset format where all packet information
is in a single CSV file with columns like:
timestamp, capacity, total_capacity, occupancy, total_occupancy, seq_num, ttl, action, packet_size, SWITCH_ID, FlowID, etc.

The script:
- Detects seq_num collisions in the merged dataset 
- Removes problematic sequences that could cause training issues
- Preserves the integrity of packet-level data for RL training
"""
import argparse
import os
import glob
import pandas as pd

def load_merged_dataset(path):
    """Load the merged dataset CSV file with proper error handling."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
        
        # Ensure seq_num column exists and is properly typed
        if 'seq_num' not in df.columns:
            raise ValueError(f"seq_num column not found in {path}. Available columns: {list(df.columns)}")
        
        # Convert seq_num to int64, handling any conversion issues
        df['seq_num'] = pd.to_numeric(df['seq_num'], errors='coerce').astype('Int64')
        
        # Remove rows where seq_num couldn't be converted
        initial_rows = len(df)
        df = df.dropna(subset=['seq_num'])
        if len(df) < initial_rows:
            print(f"Warning: Removed {initial_rows - len(df)} rows with invalid seq_num values")
        
        if df.empty:
            raise ValueError(f"Dataset {path} is empty after cleaning")
            
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        raise


def detect_seq_num_collisions(df):
    """
    Detect problematic seq_num values that appear multiple times.
    
    Returns set of seq_num values to remove based on collision detection logic.
    """
    # Count occurrences of each seq_num
    seq_counts = df['seq_num'].value_counts()
    
    # Find seq_nums that appear more than once (potential collisions)
    collision_candidates = seq_counts[seq_counts > 1].index
    
    print(f"Found {len(collision_candidates)} seq_num values with multiple occurrences")
    
    # For now, use a simple heuristic: remove seq_nums with excessive duplicates (>10)
    # This can be refined based on domain knowledge
    excessive_duplicates = seq_counts[seq_counts > 10].index
    
    print(f"Removing {len(excessive_duplicates)} seq_num values with >10 duplicates")
    
    return set(excessive_duplicates)


def save_df(path, df):
    """Save dataframe to CSV with proper directory creation."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, header=True)
    print(f"Saved {len(df)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter merged CSV dataset based on seq_num collision detection.')
    parser.add_argument('--input-file', default='results_1G/merged_final.csv',
                        help='Input merged dataset file (default: results_1G/merged_final.csv)')
    parser.add_argument('--output-file', default='results_1G/collision_filtered.csv',
                        help='Output filtered dataset file (default: results_1G/collision_filtered.csv)')
    parser.add_argument('--removed-file', default='results_1G/removed_seq_nums.csv',
                        help='File to save removed seq_nums (default: results_1G/removed_seq_nums.csv)')

    args = parser.parse_args()

    print("=" * 60)
    print("COLLISION FILTERING FOR MERGED DATASET")
    print("=" * 60)

    # Load the merged dataset
    print(f"Loading dataset from {args.input_file}...")
    df = load_merged_dataset(args.input_file)
    
    initial_rows = len(df)
    print(f"Initial dataset: {initial_rows} rows")

    # Detect seq_num collisions
    print("\nDetecting seq_num collisions...")
    to_remove = detect_seq_num_collisions(df)

    if not to_remove:
        print("No seq_num collisions detected. Dataset is clean.")
        # Still save output for pipeline consistency
        save_df(args.output_file, df)
        
        # Save empty removed file
        removed_df = pd.DataFrame({'seq_num': []})
        save_df(args.removed_file, removed_df)
        return

    # Filter out problematic seq_nums
    print(f"\nFiltering out {len(to_remove)} problematic seq_num values...")
    df_filtered = df[~df['seq_num'].isin(to_remove)]
    
    final_rows = len(df_filtered)
    removed_rows = initial_rows - final_rows
    
    print(f"Final dataset: {final_rows} rows ({removed_rows} rows removed, {removed_rows/initial_rows*100:.1f}%)")

    # Save filtered dataset
    save_df(args.output_file, df_filtered)

    # Save removed seq_nums for debugging
    removed_df = pd.DataFrame({'seq_num': list(to_remove)})
    save_df(args.removed_file, removed_df)
    
    print(f"\nCollision filtering completed successfully!")
    print(f"Filtered dataset saved to: {args.output_file}")
    print(f"Removed seq_nums saved to: {args.removed_file}")


if __name__ == '__main__':
    main()
