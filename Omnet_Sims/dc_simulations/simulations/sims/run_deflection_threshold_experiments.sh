#!/bin/bash

# Script for running deflection threshold variation experiments
# This script runs the DCTCP_SD_THRESHOLD_VARIATION configuration
# which tests different deflection threshold values

echo "================================================="
echo "Running Deflection Threshold Variation Experiments"
echo "================================================="

do_extract_with_threshold () {
    local threshold=$1
    local config_name=$2
    echo "Extracting results for threshold ${threshold}..."
    
    # Create threshold-specific extraction script
    python3 ./extractor_shell_creator.py ${config_name}_threshold_${threshold}
    
    # Extract results into threshold-specific directory
    pushd ./results/threshold_${threshold}/
    bash ../extractor.sh
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
echo "This will test thresholds: 0.3, 0.5, 0.7, 0.9"
echo "-------------------------------------------"

# Run the threshold variation configuration
# This will automatically create separate runs for each threshold value
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_THRESHOLD_VARIATION -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini

# Process results for each threshold
thresholds=("0.3" "0.5" "0.7" "0.9")

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

echo "\nTo analyze the results, you can use the create_dataset.py script"
echo "on each threshold directory separately or combine them as needed."
