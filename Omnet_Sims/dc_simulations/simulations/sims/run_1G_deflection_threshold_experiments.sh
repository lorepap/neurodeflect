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

opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_THRESHOLD_VARIATION -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME

# Process results for each threshold (include trailing 'B' everywhere)
thresholds=("15000B" "25000B" "37500B" "50000B")

for threshold in "${thresholds[@]}"; do
    echo "\n\n-------------------------------------------"
    echo "Processing results for threshold ${threshold}"
    echo "-------------------------------------------"
    
    # results_dir must match the threshold value exactly (threshold includes trailing 'B')
    results_dir="results/threshold_${threshold}"
    if [ ! -d "$results_dir" ]; then
        # fallback: any matching glob
        results_dir=$(ls -d results/threshold_${threshold}* 2>/dev/null | head -n1 || true)
    fi

    if [ -n "$results_dir" ] && [ -d "$results_dir" ]; then
        # Extract and organize results (original behavior)
        do_extract_with_threshold ${threshold} "dctcp_sd_threshold"

        # Create logs directory for this threshold
        mkdir -p logs/deflection_threshold_${threshold}

        # Copy output files to logs (use detected results_dir)
        cp ${results_dir}/*.out logs/deflection_threshold_${threshold}/ 2>/dev/null || echo "No .out files found for threshold ${threshold} in ${results_dir}"
        cp ${results_dir}/*.sca logs/deflection_threshold_${threshold}/ 2>/dev/null || echo "No .sca files found for threshold ${threshold} in ${results_dir}"
        cp ${results_dir}/*.vec logs/deflection_threshold_${threshold}/ 2>/dev/null || echo "No .vec files found for threshold ${threshold} in ${results_dir}"

        echo "Results for threshold ${threshold} saved to logs/deflection_threshold_${threshold}/ (from ${results_dir})"
    else
    echo "No results found for threshold ${threshold} (expected results/threshold_${threshold})"
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

# Add this section to help debug the structure (prefer trailing-B folders)
echo "\n\nDirectory structure for extracted results:"
for threshold in "${thresholds[@]}"; do
    dir_to_list="results/threshold_${threshold}"
    if [ -d "$dir_to_list" ]; then
        echo "Threshold ${threshold} directories (using ${dir_to_list}):"
        ls -la "$dir_to_list"
    fi
done

# Save extracted results in threshold-specific directories
cats=(QUEUE_LEN QUEUES_TOT_LEN QUEUE_CAPACITY QUEUES_TOT_CAPACITY PACKET_ACTION RCV_TS_SEQ_NUM SND_TS_SEQ_NUM OOO_SEG SWITCH_SEQ_NUM TTL ACTION_SEQ_NUM SWITCH_ID SWITCH_ID_ACTION INTERFACE_ID RETRANSMITTED FLOW_ID PACKET_SIZE REQUESTER_ID FLOW_STARTED FLOW_ENDED)
extracted_dir="extracted_results"

for threshold in "${thresholds[@]}"; do
    echo "Processing threshold ${threshold} for CSV organization..."
    
    # Stage extracted CSVs under a single common root where each category folder
    # contains CSVs for all thresholds (no per-threshold subfolders).
    root_out_dir="results_1G_deflection_thresholds"
    mkdir -p "${root_out_dir}"

    if [ "$extracted_dir" = "extracted_results" ]; then
        for c in "${cats[@]}"; do
            mkdir -p "${root_out_dir}/${c}"
            copied_files=0
            if compgen -G "${extracted_dir}/${c}/*threshold_${threshold}*.csv" > /dev/null 2>&1; then
                cp ${extracted_dir}/${c}/*threshold_${threshold}*.csv "${root_out_dir}/${c}/"
                copied_files=1
            fi
            if [ $copied_files -eq 0 ]; then
                echo "⚠ No files found for category $c and threshold $threshold"
            fi
        done
    else
        # If extracted_dir already organized (e.g., per-category), copy matching files
        for c in "${cats[@]}"; do
            mkdir -p "${root_out_dir}/${c}"
            # Copy any files that include the threshold tag in their filename
            if compgen -G "${extracted_dir}/${c}/*threshold_${threshold}*.csv" > /dev/null 2>&1; then
                cp ${extracted_dir}/${c}/*threshold_${threshold}*.csv "${root_out_dir}/${c}/" || true
            fi
        done
    fi

    # Sanity: ensure we have CSVs for this threshold
    csv_count=$(find "${root_out_dir}" -name "*threshold_${threshold}*.csv" -type f 2>/dev/null | wc -l)
    if [ "$csv_count" -eq 0 ]; then
        echo "✗ No CSV files found after preparation for threshold ${threshold}"
        continue
    fi
    echo "✓ Staged $csv_count CSV files for threshold ${threshold} into ${root_out_dir}/ (category subfolders)"
done

echo "\n\nAll done! Extracted results are in results_1G_thr_<threshold>/ directories."
