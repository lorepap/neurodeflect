#!/bin/bash

# Script for running deflection threshold variation experiments
# This script runs the DCTCP_SD_THRESHOLD_VARIATION configuration
# which tests different deflection threshold values
#
# Usage: ./run_deflection_threshold_experiments.sh <simulation_time>
# Example: ./run_deflection_threshold_experiments.sh 10s
#
# FIXED: Updated to use correct byte-based thresholds reflecting 30%, 50%, 75%, 100%
# of a 50KB buffer: (15000, 25000, 37500, 50000) instead of older values.

# Check if simulation time parameter is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <simulation_time>"
    echo "Example: $0 10s (for 10 seconds)"
    echo "Example: $0 1000ms (for 1000 milliseconds)"
    echo "Example: $0 0.1s (for 0.1 seconds)"
    exit 1
fi

SIMULATION_TIME=$1

echo "================================================="
echo "Running Deflection Threshold Variation Experiments"
echo "Simulation Time: $SIMULATION_TIME"
echo "================================================="

do_extract_with_threshold () {
    local threshold=$1
    local config_name=$2
    echo "Extracting results for threshold ${threshold}..."
    

    python3 ./extractor_shell_creator.py threshold_${threshold}
    pushd ./results
    pwd
    bash ../extractor.sh
    popd
    sleep 5
}

# Clean up previous results
# rm -rf results
# rm -rf logs/deflection_threshold_*

# Create the directory structure to save extracted results
bash dir_creator.sh

echo "\n\n-------------------------------------------"
echo "Running DCTCP_SD with Threshold Variation"
echo "This will test thresholds: 15000, 25000, 37500, 50000 bytes (30%, 50%, 75%, 100%)"
echo "Simulation time: $SIMULATION_TIME"
echo "-------------------------------------------"

# Run the threshold variation configuration
# This will automatically create separate runs for each threshold value

opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_THRESHOLD_VARIATION -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME

# Process results for each threshold
thresholds=("15000" "25000" "37500" "50000")

for threshold in "${thresholds[@]}"; do
    echo "\n\n-------------------------------------------"
    echo "Processing results for threshold ${threshold}"
    echo "-------------------------------------------"
    
    # Check if results directory exists for this threshold
    if [ -d "results/threshold_${threshold}" ]; then
        # Extract and organize results
        do_extract_with_threshold ${threshold} "dctcp_sd_threshold"
        
        # Create logs directory for this threshold
        mkdir -p logs/deflection_threshold_${threshold}
        
        # Copy output files to logs
        if [ -d "results/threshold_${threshold}" ]; then
            cp results/threshold_${threshold}/*.out logs/deflection_threshold_${threshold}/ 2>/dev/null || echo "No .out files found for threshold ${threshold}"
            cp results/threshold_${threshold}/*.sca logs/deflection_threshold_${threshold}/ 2>/dev/null || echo "No .sca files found for threshold ${threshold}"
            cp results/threshold_${threshold}/*.vec logs/deflection_threshold_${threshold}/ 2>/dev/null || echo "No .vec files found for threshold ${threshold}"
        fi
        
        echo "Results for threshold ${threshold} saved to logs/deflection_threshold_${threshold}/"
    else
        echo "No results found for threshold ${threshold}"
    fi
done

echo "\n\n================================================="
echo "Deflection Threshold Experiments Completed"
echo "================================================="
echo "Results are organized by threshold in:"
for threshold in "${thresholds[@]}"; do
    if [ -d "logs/deflection_threshold_${threshold}" ]; then
        echo "  - logs/deflection_threshold_${threshold}/"
    fi
done

# Add this section to help debug the structure
echo "\n\nDirectory structure for extracted results:"
for threshold in "${thresholds[@]}"; do
    if [ -d "results/threshold_${threshold}" ]; then
        echo "Threshold ${threshold} directories:"
        ls -la results/threshold_${threshold}/
    fi
done

# Save extracted results in threshold-specific directories
cats=(QUEUE_LEN QUEUES_TOT_LEN QUEUE_CAPACITY QUEUES_TOT_CAPACITY PACKET_ACTION RCV_TS_SEQ_NUM SND_TS_SEQ_NUM OOO_SEG SWITCH_SEQ_NUM TTL ACTION_SEQ_NUM SWITCH_ID SWITCH_ID_ACTION INTERFACE_ID RETRANSMITTED FLOW_ID PACKET_SIZE REQUESTER_ID FLOW_STARTED FLOW_ENDED)
extracted_dir="extracted_results"

for threshold in "${thresholds[@]}"; do
    echo "Processing threshold ${threshold} for CSV organization..."
    
    if [ "$extracted_dir" = "extracted_results" ]; then
        for c in "${cats[@]}"; do
            mkdir -p "results_1G_thr_${threshold}/${c}"
            # copy only matching CSVs for this threshold - prefer manually extracted files
            copied_files=0
            
            # Fixed syntax error: missing 'if' and proper condition structure
            if compgen -G "${extracted_dir}/${c}/*threshold_${threshold}*.csv" > /dev/null 2>&1; then
                cp ${extracted_dir}/${c}/*threshold_${threshold}*.csv "results_1G_thr_${threshold}/${c}/"
                copied_files=1
            fi
            
            if [ $copied_files -eq 0 ]; then
                echo "⚠ No files found for category $c and threshold $threshold"
            fi
        done
    else
        # Already per-threshold structured; copy the whole directory
        cp -r "$extracted_dir" "results_1G_thr_${threshold}"
    fi

    # Sanity: ensure we have CSVs
    csv_count=$(find "results_1G_thr_${threshold}" -name "*.csv" -type f 2>/dev/null | wc -l)
    if [ "$csv_count" -eq 0 ]; then
        echo "✗ No CSV files found after preparation for threshold ${threshold}"
        # Changed 'return 1' to 'continue' since we're in a loop, not a function
        continue
    fi
    echo "✓ Staged $csv_count CSV files into results_1G_thr_${threshold}/"
done

echo "\n\nAll done! Extracted results are in results_1G_thr_<threshold>/ directories."
