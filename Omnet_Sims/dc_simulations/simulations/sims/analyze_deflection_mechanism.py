#!/usr/bin/env python3
"""
Analyze deflection behavior by examining simulation logs for each threshold.
This script checks if deflection is actually working by looking for:
1. Queue occupancy exceeding deflection thresholds
2. Deflection actions being taken
3. "No bouncing" vs actual bouncing events
"""

import re
import os
from collections import defaultdict

def extract_occupancy_values(log_file):
    """Extract all queue occupancy values from a log file."""
    occupancy_values = []
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Look for: "Queue data occupancy is X B"
                match = re.search(r'Queue data occupancy is (\d+) B', line)
                if match:
                    occupancy_values.append(int(match.group(1)))
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
    return occupancy_values

def count_bouncing_events(log_file):
    """Count bouncing vs no-bouncing events."""
    no_bouncing_count = 0
    bouncing_count = 0
    deflection_tag_calls = 0
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "No bouncing" in line:
                    no_bouncing_count += 1
                elif "bouncing" in line.lower() and "No bouncing" not in line:
                    bouncing_count += 1
                elif "get_packet_deflection_tag called" in line:
                    deflection_tag_calls += 1
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
    
    return {
        'no_bouncing': no_bouncing_count,
        'bouncing': bouncing_count,
        'deflection_tag_calls': deflection_tag_calls
    }

def analyze_threshold_simulation(threshold_bytes, threshold_dir):
    """Analyze a specific threshold simulation."""
    print(f"\n{'='*60}")
    print(f"ANALYZING THRESHOLD: {threshold_bytes} bytes")
    print(f"Directory: {threshold_dir}")
    print(f"{'='*60}")
    
    # Find the .out file
    out_files = [f for f in os.listdir(threshold_dir) if f.endswith('.out')]
    if not out_files:
        print(f"❌ No .out files found in {threshold_dir}")
        return None
    
    log_file = os.path.join(threshold_dir, out_files[0])
    print(f"Analyzing log file: {os.path.basename(log_file)}")
    
    # Extract occupancy values
    occupancy_values = extract_occupancy_values(log_file)
    
    if not occupancy_values:
        print("❌ No occupancy values found")
        return None
    
    # Count bouncing events
    bouncing_stats = count_bouncing_events(log_file)
    
    # Calculate statistics
    max_occupancy = max(occupancy_values)
    min_occupancy = min(occupancy_values)
    avg_occupancy = sum(occupancy_values) / len(occupancy_values)
    
    # Count how many times occupancy exceeded threshold
    threshold_exceeded_count = sum(1 for occ in occupancy_values if occ > threshold_bytes)
    threshold_exceeded_pct = (threshold_exceeded_count / len(occupancy_values)) * 100
    
    # Print results
    print(f"\n📊 OCCUPANCY ANALYSIS:")
    print(f"   - Total occupancy samples: {len(occupancy_values):,}")
    print(f"   - Min occupancy: {min_occupancy:,} B")
    print(f"   - Max occupancy: {max_occupancy:,} B")
    print(f"   - Avg occupancy: {avg_occupancy:.1f} B")
    print(f"   - Deflection threshold: {threshold_bytes:,} B")
    print(f"   - Times exceeded threshold: {threshold_exceeded_count:,} ({threshold_exceeded_pct:.2f}%)")
    
    print(f"\n🎯 DEFLECTION BEHAVIOR:")
    print(f"   - Deflection tag calls: {bouncing_stats['deflection_tag_calls']:,}")
    print(f"   - 'No bouncing' events: {bouncing_stats['no_bouncing']:,}")
    print(f"   - Actual bouncing events: {bouncing_stats['bouncing']:,}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    if max_occupancy > threshold_bytes:
        print(f"   ✅ Queue occupancy DID exceed deflection threshold")
        print(f"     (Max: {max_occupancy:,}B > Threshold: {threshold_bytes:,}B)")
        
        if bouncing_stats['bouncing'] > 0:
            print(f"   ✅ Deflection actions WERE taken ({bouncing_stats['bouncing']} times)")
        else:
            print(f"   ❌ NO deflection actions were taken despite threshold being exceeded!")
            print(f"     This indicates a BUG in the deflection logic!")
    else:
        print(f"   ⚠️  Queue occupancy NEVER exceeded deflection threshold")
        print(f"     (Max: {max_occupancy:,}B ≤ Threshold: {threshold_bytes:,}B)")
        print(f"     This explains why no deflection occurred.")
    
    return {
        'threshold_bytes': threshold_bytes,
        'max_occupancy': max_occupancy,
        'threshold_exceeded_count': threshold_exceeded_count,
        'bouncing_events': bouncing_stats['bouncing'],
        'no_bouncing_events': bouncing_stats['no_bouncing'],
        'deflection_working': bouncing_stats['bouncing'] > 0 if max_occupancy > threshold_bytes else None
    }

def main():
    """Main analysis function."""
    print("🔍 DEFLECTION MECHANISM ANALYSIS")
    print("="*60)
    print("This script analyzes whether deflection is working correctly")
    print("by examining queue occupancy vs deflection thresholds.")
    
    # Define thresholds and their corresponding directories
    thresholds = [
        (15000, 'results/threshold_15000'),
        (25000, 'results/threshold_25000'),
        (35000, 'results/threshold_35000'),
        (45000, 'results/threshold_45000'),
        (50000, 'results/threshold_50000')
    ]
    
    results = []
    
    for threshold_bytes, threshold_dir in thresholds:
        if os.path.exists(threshold_dir):
            result = analyze_threshold_simulation(threshold_bytes, threshold_dir)
            if result:
                results.append(result)
        else:
            print(f"\n❌ Directory not found: {threshold_dir}")
    
    # Summary analysis
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print("="*60)
    
    if not results:
        print("❌ No valid results to analyze")
        return
    
    print(f"\n📋 THRESHOLD COMPARISON:")
    print(f"{'Threshold':<10} {'Max Occupancy':<15} {'Exceeded?':<10} {'Deflections':<12} {'Working?'}")
    print("-" * 70)
    
    deflection_working_count = 0
    should_have_deflected_count = 0
    
    for result in results:
        threshold = result['threshold_bytes']
        max_occ = result['max_occupancy']
        exceeded = "Yes" if max_occ > threshold else "No"
        deflections = result['bouncing_events']
        
        if max_occ > threshold:
            should_have_deflected_count += 1
            working = "Yes" if deflections > 0 else "❌ NO"
            if deflections > 0:
                deflection_working_count += 1
        else:
            working = "N/A"
        
        print(f"{threshold:<10} {max_occ:<15,} {exceeded:<10} {deflections:<12} {working}")
    
    print(f"\n🎯 FINAL VERDICT:")
    if should_have_deflected_count == 0:
        print("⚠️  INCONCLUSIVE: No simulations had occupancy exceeding deflection thresholds")
        print("   This could mean:")
        print("   1. Simulation time is too short to build up queues")
        print("   2. Traffic load is too low")
        print("   3. Network topology prevents congestion")
    elif deflection_working_count == should_have_deflected_count:
        print("✅ DEFLECTION IS WORKING: All thresholds that should deflect are deflecting")
    elif deflection_working_count == 0:
        print("❌ DEFLECTION IS BROKEN: No deflections occurred despite thresholds being exceeded")
        print("   This indicates a critical bug in the deflection implementation!")
    else:
        print("⚠️  DEFLECTION IS PARTIALLY WORKING:")
        print(f"   {deflection_working_count}/{should_have_deflected_count} thresholds are working correctly")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if deflection_working_count < should_have_deflected_count:
        print("1. Check the deflection threshold comparison logic in the simulation code")
        print("2. Verify that deflection decisions are being logged properly")
        print("3. Ensure the deflection mechanism is enabled in the configuration")
        print("4. Look for bugs in the queue occupancy calculation")
    else:
        print("1. Increase simulation time or traffic load to test deflection more thoroughly")
        print("2. Verify that different thresholds produce different deflection rates")

if __name__ == "__main__":
    main()
