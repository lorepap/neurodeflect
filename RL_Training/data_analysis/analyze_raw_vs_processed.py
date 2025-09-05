#!/usr/bin/env python3
"""
Raw vs Processed Data Collection Analysis

Analyze the difference between raw data collection counts and packets 
that make it into the final processed dataset to identify data loss.
"""

import pandas as pd
import os

def analyze_raw_vs_processed_data():
    """Analyze discrepancy between raw collection and final dataset"""
    
    print("=" * 80)
    print("RAW vs PROCESSED DATA COLLECTION ANALYSIS")
    print("=" * 80)
    
    thresholds = [15000, 25000, 37500, 50000]
    threshold_fractions = [0.30, 0.50, 0.75, 1.00]
    
    print("Analyzing data collection and processing pipeline...")
    print("Thresholds: 15000 (0.3), 25000 (0.5), 37500 (0.75), 50000 (1.0) bytes\n")
    
    results = []
    
    for i, threshold in enumerate(thresholds):
        print(f"=== THRESHOLD {threshold} BYTES ({threshold_fractions[i]:.2f} buffer) ===")
        
        # 1. Check raw PacketAction data
        raw_packet_action_file = f"extracted_results/PACKET_ACTION/sim_*_{threshold}_deflection_threshold_{threshold}.csv"
        
        try:
            # Find the actual file
            import glob
            raw_files = glob.glob(raw_packet_action_file)
            if raw_files:
                raw_file = raw_files[0]
                print(f"Raw PacketAction file: {os.path.basename(raw_file)}")
                
                # Count raw observations
                with open(raw_file, 'r') as f:
                    raw_lines = f.readlines()
                    raw_total = len(raw_lines) - 1  # Exclude header
                    raw_deflections = sum(1 for line in raw_lines[1:] if line.strip().split(',')[1] == '1')
                    raw_forwards = raw_total - raw_deflections
                
                print(f"  Raw total observations: {raw_total}")
                print(f"  Raw deflections: {raw_deflections}")
                print(f"  Raw forwards: {raw_forwards}")
                print(f"  Raw deflection rate: {raw_deflections/raw_total:.4f} ({raw_deflections/raw_total*100:.2f}%)")
                
            else:
                print(f"  ❌ Raw PacketAction file not found")
                raw_total = raw_deflections = raw_forwards = 0
                
        except Exception as e:
            print(f"  ❌ Error reading raw data: {e}")
            raw_total = raw_deflections = raw_forwards = 0
        
        # 2. Check other raw signal files to understand total data collection
        signal_types = ['SWITCH_ID', 'FLOW_ID', 'QUEUE_LEN', 'QUEUE_CAPACITY', 'TTL']
        raw_signal_counts = {}
        
        print(f"\n  Raw signal collection counts:")
        for signal in signal_types:
            try:
                signal_files = glob.glob(f"extracted_results/{signal}/sim_*_{threshold}_deflection_threshold_{threshold}.csv")
                if signal_files:
                    with open(signal_files[0], 'r') as f:
                        signal_count = len(f.readlines()) - 1  # Exclude header
                    raw_signal_counts[signal] = signal_count
                    print(f"    {signal}: {signal_count} observations")
                else:
                    raw_signal_counts[signal] = 0
                    print(f"    {signal}: FILE NOT FOUND")
            except Exception as e:
                raw_signal_counts[signal] = 0
                print(f"    {signal}: ERROR - {e}")
        
        # 3. Check processed merged_final.csv
        merged_file = f"results_1G_thr_{threshold}/merged_final.csv"
        
        try:
            if os.path.exists(merged_file):
                print(f"\n  Processed file: {merged_file}")
                df = pd.read_csv(merged_file)
                
                processed_total = len(df)
                processed_deflections = (df['action'] == 1).sum()
                processed_forwards = (df['action'] == 0).sum()
                
                print(f"  Processed total observations: {processed_total}")
                print(f"  Processed deflections: {processed_deflections}")
                print(f"  Processed forwards: {processed_forwards}")
                print(f"  Processed deflection rate: {processed_deflections/processed_total:.4f} ({processed_deflections/processed_total*100:.2f}%)")
                
            else:
                print(f"  ❌ Processed file not found: {merged_file}")
                processed_total = processed_deflections = processed_forwards = 0
                
        except Exception as e:
            print(f"  ❌ Error reading processed data: {e}")
            processed_total = processed_deflections = processed_forwards = 0
        
        # 4. Calculate data loss/discrepancy
        print(f"\n  📊 DATA PROCESSING ANALYSIS:")
        
        if raw_total > 0 and processed_total > 0:
            # Overall data retention
            retention_rate = processed_total / raw_total
            data_loss_rate = 1 - retention_rate
            
            print(f"    Data retention: {processed_total}/{raw_total} = {retention_rate:.4f} ({retention_rate*100:.2f}%)")
            print(f"    Data loss: {data_loss_rate:.4f} ({data_loss_rate*100:.2f}%)")
            
            # Deflection accuracy
            if raw_deflections > 0:
                deflection_accuracy = processed_deflections / raw_deflections
                print(f"    Deflection accuracy: {processed_deflections}/{raw_deflections} = {deflection_accuracy:.4f} ({deflection_accuracy*100:.2f}%)")
            else:
                deflection_accuracy = 1.0 if processed_deflections == 0 else 0.0
                print(f"    Deflection accuracy: N/A (no raw deflections)")
            
            # Check if deflection rates are preserved
            raw_deflection_rate = raw_deflections / raw_total if raw_total > 0 else 0
            processed_deflection_rate = processed_deflections / processed_total if processed_total > 0 else 0
            rate_preservation = abs(raw_deflection_rate - processed_deflection_rate) < 0.001
            
            print(f"    Deflection rate preservation: {'✓ YES' if rate_preservation else '✗ NO'}")
            print(f"      Raw rate: {raw_deflection_rate:.4f}, Processed rate: {processed_deflection_rate:.4f}")
            
        else:
            retention_rate = deflection_accuracy = 0.0
            print(f"    ❌ Cannot calculate - missing data")
        
        # 5. Check signal consistency
        print(f"\n  🔍 SIGNAL CONSISTENCY CHECK:")
        if raw_signal_counts:
            max_signal_count = max(raw_signal_counts.values())
            min_signal_count = min([v for v in raw_signal_counts.values() if v > 0] or [0])
            
            print(f"    Max signal observations: {max_signal_count}")
            print(f"    Min signal observations: {min_signal_count}")
            
            if max_signal_count > 0:
                consistency = min_signal_count / max_signal_count
                print(f"    Signal consistency: {consistency:.4f} ({consistency*100:.2f}%)")
                
                if consistency < 0.95:
                    print(f"    ⚠️  Inconsistent signal collection detected")
                else:
                    print(f"    ✓ Good signal consistency")
        
        # Store results
        results.append({
            'threshold': threshold,
            'threshold_fraction': threshold_fractions[i],
            'raw_total': raw_total,
            'raw_deflections': raw_deflections,
            'processed_total': processed_total,
            'processed_deflections': processed_deflections,
            'retention_rate': retention_rate,
            'deflection_accuracy': deflection_accuracy,
            'max_signal_count': max(raw_signal_counts.values()) if raw_signal_counts else 0
        })
        
        print("\n" + "-" * 60 + "\n")
    
    # Summary analysis
    print("=" * 80)
    print("SUMMARY: RAW vs PROCESSED DATA ANALYSIS")
    print("=" * 80)
    
    if results:
        summary_df = pd.DataFrame(results)
        
        print("Threshold  Fraction  Raw_Total  Processed_Total  Retention%  Deflection_Accuracy%")
        print("-" * 80)
        for _, row in summary_df.iterrows():
            retention_pct = row['retention_rate'] * 100
            deflection_pct = row['deflection_accuracy'] * 100
            print(f"{int(row['threshold']):8d}  {row['threshold_fraction']:8.2f}  {int(row['raw_total']):9d}  {int(row['processed_total']):15d}  {retention_pct:10.2f}  {deflection_pct:18.2f}")
        
        # Overall statistics
        total_raw = summary_df['raw_total'].sum()
        total_processed = summary_df['processed_total'].sum()
        overall_retention = total_processed / total_raw if total_raw > 0 else 0
        
        print(f"\nOVERALL STATISTICS:")
        print(f"  Total raw observations: {total_raw}")
        print(f"  Total processed observations: {total_processed}")
        print(f"  Overall retention rate: {overall_retention:.4f} ({overall_retention*100:.2f}%)")
        print(f"  Overall data loss: {1-overall_retention:.4f} ({(1-overall_retention)*100:.2f}%)")
        
        # Identify patterns
        print(f"\nDATA LOSS PATTERNS:")
        avg_retention = summary_df['retention_rate'].mean()
        std_retention = summary_df['retention_rate'].std()
        
        print(f"  Average retention rate: {avg_retention:.4f} ± {std_retention:.4f}")
        
        consistent_thresholds = summary_df[summary_df['retention_rate'] > 0.95]
        problematic_thresholds = summary_df[summary_df['retention_rate'] < 0.95]
        
        if len(consistent_thresholds) > 0:
            print(f"  ✓ Consistent thresholds: {list(consistent_thresholds['threshold'])}")
        
        if len(problematic_thresholds) > 0:
            print(f"  ⚠️  Problematic thresholds: {list(problematic_thresholds['threshold'])}")
            print(f"     These show significant data loss during processing")
    
    return results

if __name__ == "__main__":
    try:
        results = analyze_raw_vs_processed_data()
        print("\n✓ Analysis completed successfully!")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
