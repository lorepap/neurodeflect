#!/bin/bash

# Script for running deflection threshold variation experiments
# This script runs the DCTCP_SD_THRESHOLD_VARIATION configuration
# which tests different deflection threshold values
#
# Usage: ./run_deflection_threshold_experiments.sh <simulation_time>
# Example: ./run_deflection_threshold_experiments.sh 10s
#
# FIXED: Updated to use correct byte-based thresholds (15000, 25000, 35000, 45000)
# instead of percentage values (0.3, 0.5, 0.7, 0.9) due to type mismatch bug

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
    
    # Enter the threshold results directory first so the extractor generator
    # scans the correct files (it expects to be run from the directory
    # containing the .vec/.sca files).
    pushd ./results/threshold_${threshold}/
    # Run generator (relative path back to the generator in the sims dir)
    python3 ../../extractor_shell_creator.py ${config_name}_threshold_${threshold}
    # Run the generated extractor script (script is written three levels up
    # by the generator and is designed to be executed from inside the
    # threshold directory so scavetool can open the vector files by name).
    bash ../../../extractor.sh
    popd
    sleep 5
}

# Clean up previous results
rm -rf results
rm -rf logs/deflection_threshold_*

# Create the directory structure to save extracted results
bash dir_creator.sh

echo "\n\n-------------------------------------------"
echo "Running DCTCP_SD with Threshold Variation"
echo "This will test thresholds: 15000, 25000, 35000, 45000 50000 bytes (30%, 50%, 70%, 90%, 100%)"
echo "Simulation time: $SIMULATION_TIME"
echo "-------------------------------------------"

# Run the threshold variation configuration
# This will automatically create separate runs for each threshold value
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_THRESHOLD_VARIATION -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME

# Process results for each threshold
thresholds=("15000" "25000" "35000" "45000" "50000")

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
