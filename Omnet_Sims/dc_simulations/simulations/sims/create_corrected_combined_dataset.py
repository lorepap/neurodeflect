#!/usr/bin/env python3
"""
Create combined threshold dataset using correct PacketAction data

This script combines the individual threshold datasets using the correct 'action' column
instead of the misleading 'ooo' column, and includes proper deflection analysis.
"""

import pandas as pd
import os

def create_corrected_combined_dataset():
    """Create combined dataset using correct action column from merged_final.csv files"""
    
    thresholds = [15000, 25000, 37500, 50000]
    combined_data = []
    
    print("=== CREATING CORRECTED COMBINED THRESHOLD DATASET ===\n")
    
    for threshold in thresholds:
        # Map to actual file locations (local paths relative to sims directory)
        file_path_map = {
            15000: "results_1G_thr_15000/merged_final.csv",
            25000: "results_1G_thr_25000/merged_final.csv", 
            37500: "results_1G_thr_37500/merged_final.csv",
            50000: "results_1G_thr_50000/merged_final.csv"
        }
        file_path = file_path_map.get(threshold)
        
        try:
            print(f"Processing threshold {threshold}:")
            df = pd.read_csv(file_path)
            
            # Add deflection threshold column
            df['deflection_threshold'] = threshold
            
            # Rename 'action' to 'deflection' for clarity (0=forward, 1=deflection)
            df['deflection'] = df['action']
            
            # Count deflections for verification
            deflection_count = (df['deflection'] == 1).sum()
            total_count = len(df)
            deflection_rate = deflection_count / total_count if total_count > 0 else 0
            
            print(f"  Loaded {total_count} observations")
            print(f"  Deflections: {deflection_count} ({deflection_rate:.4f} rate)")
            
            # Select relevant columns for combined dataset
            selected_columns = ['timestamp', 'capacity', 'total_capacity', 'occupancy', 
                              'total_occupancy', 'seq_num', 'ttl', 'deflection', 'deflection_threshold']
            
            # Only keep columns that exist
            available_columns = [col for col in selected_columns if col in df.columns]
            df_selected = df[available_columns].copy()
            
            combined_data.append(df_selected)
            print(f"  Added to combined dataset with columns: {available_columns}")
            print()
            
        except FileNotFoundError:
            print(f"  File not found: {file_path}")
            print()
            continue
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            print()
            continue
    
    if combined_data:
        # Combine all threshold data
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        print("=== COMBINED DATASET SUMMARY ===")
        print(f"Total observations: {len(combined_df)}")
        print(f"Columns: {list(combined_df.columns)}")
        print(f"Thresholds: {sorted(combined_df['deflection_threshold'].unique())}")
        
        # Summary by threshold
        print("\nDeflection summary by threshold:")
        summary = combined_df.groupby('deflection_threshold').agg({
            'deflection': ['count', 'sum'],
            'seq_num': 'nunique'
        }).round(4)
        
        summary.columns = ['total_obs', 'deflections', 'unique_flows']
        summary['deflection_rate'] = summary['deflections'] / summary['total_obs']
        
        print(summary)
        
        # Save corrected combined dataset
        output_file = "combined_threshold_dataset_corrected.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\nSaved corrected combined dataset to: {output_file}")
        
        # Verify against raw data
        print("\n=== VERIFICATION AGAINST RAW DATA ===")
        print("Raw PacketAction deflection counts:")
        raw_deflections = {
            15000: 34,
            25000: 28, 
            37500: 19,
            50000: 12
        }
        
        for threshold in thresholds:
            if threshold in summary.index:
                processed = int(summary.loc[threshold, 'deflections'])
                raw = raw_deflections.get(threshold, 0)
                accuracy = processed / raw if raw > 0 else 0
                print(f"  Threshold {threshold}: {processed}/{raw} = {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return combined_df
    
    else:
        print("No data files found to combine!")
        return None

if __name__ == "__main__":
    combined_df = create_corrected_combined_dataset()
