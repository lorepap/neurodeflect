#!/usr/bin/env python3
"""
Comprehensive analysis of the deflection threshold experiment results.
This script identifies issues in the dataset and provides debugging information.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

def analyze_combined_dataset():
    """Analyze the combined threshold dataset and identify issues."""
    print("="*60)
    print("COMPREHENSIVE DEFLECTION THRESHOLD ANALYSIS")
    print("="*60)
    
    # Load the combined dataset
    df = pd.read_csv('combined_threshold_dataset.csv')
    
    print(f"\n1. DATASET OVERVIEW:")
    print(f"   - Total samples: {len(df):,}")
    print(f"   - Columns: {list(df.columns)}")
    print(f"   - Deflection thresholds: {sorted(df['deflection_threshold'].unique())}")
    
    # Check if all thresholds have identical data
    print(f"\n2. DATA UNIQUENESS CHECK:")
    thresholds = sorted(df['deflection_threshold'].unique())
    
    # For each pair of thresholds, check if data is identical
    identical_pairs = []
    for i, t1 in enumerate(thresholds):
        for t2 in thresholds[i+1:]:
            data1 = df[df['deflection_threshold'] == t1].drop('deflection_threshold', axis=1).reset_index(drop=True)
            data2 = df[df['deflection_threshold'] == t2].drop('deflection_threshold', axis=1).reset_index(drop=True)
            
            if data1.equals(data2):
                identical_pairs.append((t1, t2))
    
    if identical_pairs:
        print(f"   ❌ CRITICAL ISSUE: Found {len(identical_pairs)} pairs of thresholds with identical data!")
        for t1, t2 in identical_pairs:
            print(f"      - Threshold {t1} is identical to threshold {t2}")
    else:
        print(f"   ✅ All thresholds have unique data")
    
    # Analyze occupancy patterns
    print(f"\n3. OCCUPANCY ANALYSIS:")
    occupancy_stats = df.groupby('deflection_threshold')['occupancy'].agg([
        'count', 'mean', 'std', 'min', 'max',
        lambda x: (x == 0).sum(),  # zero count
        lambda x: (x > 0).sum()    # non-zero count
    ])
    occupancy_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'zero_count', 'nonzero_count']
    occupancy_stats['zero_percentage'] = (occupancy_stats['zero_count'] / occupancy_stats['count'] * 100).round(2)
    
    print(occupancy_stats)
    
    # Analyze action patterns
    print(f"\n4. ACTION ANALYSIS:")
    action_stats = df.groupby(['deflection_threshold', 'action']).size().unstack(fill_value=0)
    print("   Action distribution (0=Forward, 1=Deflect):")
    print(action_stats)
    
    if 1 in action_stats.columns:
        deflection_rate = (action_stats[1] / action_stats.sum(axis=1) * 100).round(2)
        print(f"\n   Deflection rates:")
        for threshold in deflection_rate.index:
            print(f"   - Threshold {threshold}: {deflection_rate[threshold]:.2f}%")
    else:
        print("   ❌ NO DEFLECTION ACTIONS FOUND - This indicates a serious problem!")
    
    return df, identical_pairs

def check_individual_datasets():
    """Check the individual threshold dataset files."""
    print(f"\n5. INDIVIDUAL DATASET FILE ANALYSIS:")
    
    threshold_files = [
        'threshold_dataset_15000.csv',
        'threshold_dataset_25000.csv', 
        'threshold_dataset_35000.csv',
        'threshold_dataset_45000.csv',
        'threshold_dataset_50000.csv'
    ]
    
    file_info = {}
    for filename in threshold_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename)
            df = pd.read_csv(filename)
            file_info[filename] = {
                'size_bytes': file_size,
                'num_rows': len(df),
                'sample_hash': hash(str(df.iloc[0].values) + str(df.iloc[-1].values)) if len(df) > 0 else None
            }
        else:
            file_info[filename] = {'exists': False}
    
    print("   Individual file analysis:")
    for filename, info in file_info.items():
        if 'exists' in info and not info['exists']:
            print(f"   - {filename}: File not found")
        else:
            print(f"   - {filename}: {info['size_bytes']} bytes, {info['num_rows']} rows, hash: {info['sample_hash']}")
    
    # Check if all files have the same hash (indicating identical content)
    hashes = [info.get('sample_hash') for info in file_info.values() if 'sample_hash' in info and info['sample_hash'] is not None]
    if len(set(hashes)) == 1 and len(hashes) > 1:
        print("   ❌ CRITICAL: All individual dataset files appear to have identical content!")
    elif len(set(hashes)) == len(hashes):
        print("   ✅ Individual dataset files have different content")
    else:
        print("   ⚠️  Some files may have identical content")
    
    return file_info

def check_raw_simulation_results():
    """Check the raw simulation results directories."""
    print(f"\n6. RAW SIMULATION RESULTS ANALYSIS:")
    
    threshold_dirs = [
        'results_1G_thr_15000',
        'results_1G_thr_25000',
        'results_1G_thr_35000', 
        'results_1G_thr_45000',
        'results_1G_thr_50000'
    ]
    
    results_info = {}
    for dirname in threshold_dirs:
        if os.path.exists(dirname):
            final_dataset = os.path.join(dirname, 'final_dataset.csv')
            if os.path.exists(final_dataset):
                file_size = os.path.getsize(final_dataset)
                df = pd.read_csv(final_dataset)
                results_info[dirname] = {
                    'size_bytes': file_size,
                    'num_rows': len(df),
                    'sample_hash': hash(str(df.iloc[0].values) + str(df.iloc[-1].values)) if len(df) > 0 else None
                }
            else:
                results_info[dirname] = {'final_dataset_exists': False}
        else:
            results_info[dirname] = {'dir_exists': False}
    
    print("   Raw simulation results analysis:")
    for dirname, info in results_info.items():
        if 'dir_exists' in info and not info['dir_exists']:
            print(f"   - {dirname}: Directory not found")
        elif 'final_dataset_exists' in info and not info['final_dataset_exists']:
            print(f"   - {dirname}: Directory exists but no final_dataset.csv")
        else:
            print(f"   - {dirname}: {info['size_bytes']} bytes, {info['num_rows']} rows, hash: {info['sample_hash']}")
    
    # Check if all results have the same hash
    hashes = [info.get('sample_hash') for info in results_info.values() if 'sample_hash' in info and info['sample_hash'] is not None]
    if len(set(hashes)) == 1 and len(hashes) > 1:
        print("   ❌ CRITICAL: All raw simulation results appear to have identical final datasets!")
        return True  # Indicates problem found
    elif len(set(hashes)) == len(hashes):
        print("   ✅ Raw simulation results have different content")
        return False
    else:
        print("   ⚠️  Some raw simulation results may have identical content")
        return True
    
def diagnose_problem():
    """Provide diagnosis and recommendations."""
    print(f"\n7. PROBLEM DIAGNOSIS AND RECOMMENDATIONS:")
    
    # Check if raw simulation outputs have different sizes
    raw_outputs = {
        '15000': 'results/threshold_15000',
        '25000': 'results/threshold_25000', 
        '35000': 'results/threshold_35000',
        '45000': 'results/threshold_45000',
        '50000': 'results/threshold_50000'
    }
    
    output_sizes = {}
    for threshold, path in raw_outputs.items():
        out_files = [f for f in os.listdir(path) if f.endswith('.out')] if os.path.exists(path) else []
        if out_files:
            out_file = os.path.join(path, out_files[0])
            output_sizes[threshold] = os.path.getsize(out_file)
    
    print("   Raw simulation output file sizes:")
    for threshold, size in output_sizes.items():
        print(f"   - Threshold {threshold}: {size:,} bytes")
    
    # Diagnosis
    if len(set(output_sizes.values())) == 1:
        print("\n   🔍 DIAGNOSIS:")
        print("   The raw simulation output files are identical in size, which suggests:")
        print("   1. The simulations may not actually be using different deflection thresholds")
        print("   2. There may be a bug in the simulation configuration")
        print("   3. The deflection mechanism may not be working as expected")
        
        print("\n   💡 RECOMMENDATIONS:")
        print("   1. Verify the OMNeT++ simulation configuration is correctly parameterized")
        print("   2. Check if the deflection threshold parameter is actually being applied")
        print("   3. Look for deflection-related log messages in the simulation outputs")
        print("   4. Verify that the buffer capacity and deflection threshold relationship is correct")
        
    elif len(set(output_sizes.values())) > 1:
        print("\n   🔍 DIAGNOSIS:")
        print("   The raw simulation outputs have different sizes, which is good!")
        print("   The problem is likely in the data extraction/processing pipeline:")
        print("   1. The extraction scripts may be using the wrong source data")
        print("   2. There may be a bug in the dataset creation process")
        print("   3. The results may be getting overwritten during processing")
        
        print("\n   💡 RECOMMENDATIONS:")
        print("   1. Re-run the data extraction pipeline for each threshold separately")
        print("   2. Check the extraction scripts for bugs")
        print("   3. Verify that results are being saved to the correct threshold-specific directories")
        print("   4. Compare raw simulation data between thresholds manually")

def main():
    """Main analysis function."""
    try:
        # Analyze combined dataset
        df, identical_pairs = analyze_combined_dataset()
        
        # Check individual files
        file_info = check_individual_datasets()
        
        # Check raw results
        raw_problem = check_raw_simulation_results()
        
        # Provide diagnosis
        diagnose_problem()
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print("="*60)
        
        if identical_pairs:
            print("❌ CRITICAL ISSUE FOUND: Dataset contains identical data for different thresholds")
            print("   This means the deflection threshold variation experiment is not working correctly.")
            print("   The problem could be in:")
            print("   - Simulation configuration")
            print("   - Data extraction process") 
            print("   - Dataset combination process")
        else:
            print("✅ Dataset appears to have different data for different thresholds")
            
        print(f"\nTo fix this issue:")
        print("1. Check simulation logs for deflection threshold parameter application")
        print("2. Verify data extraction scripts are reading from correct sources")
        print("3. Re-run simulations if configuration is incorrect")
        print("4. Re-run data extraction if processing is incorrect")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're in the correct directory with the combined dataset.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()
