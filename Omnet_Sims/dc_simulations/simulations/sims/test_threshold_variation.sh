#!/bin/bash

# Test script for deflection threshold variation
# Runs a short simulation to verify the configuration works

echo "===================================================================="
echo "Testing Deflection Threshold Variation Configuration"
echo "===================================================================="

cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims

# Create necessary directories
mkdir -p results
for threshold in 0.3 0.5 0.7 0.9; do
    mkdir -p results/threshold_${threshold}
done
mkdir -p logs/threshold_test

# Make sure we have the right files
if [ ! -d "distributions" ]; then
    echo "Error: 'distributions' directory not found."
    echo "Please run 'bash extract_dist_files_LS_1Gbps.sh' first."
    exit 1
fi

# Run a short simulation with the threshold variation config
echo "Running DCTCP_SD_THRESHOLD_VARIATION for 0.001s simulation time..."
opp_runall -j1 ../../src/dc_simulations -m -u Cmdenv \
    -c DCTCP_SD_THRESHOLD_VARIATION \
    -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases \
    --image-path=../../../inet/images \
    -l ../../../inet/src/INET \
    --sim-time-limit=0.001s \
    --cmdenv-express-mode=false \
    --cmdenv-status-frequency=1s \
    --cmdenv-runs-to-execute=0 \
    omnetpp_1G.ini

# Check if the simulation ran successfully
if [ $? -eq 0 ]; then
    echo "Success! The threshold variation configuration ran without errors."
    
    # List generated results
    echo "Result files:"
    find results/threshold_* -type f 2>/dev/null | head -8
    
    # Copy results to logs
    cp -r results/threshold_* logs/threshold_test/ 2>/dev/null || echo "No threshold result files generated"
else
    echo "Error: The threshold variation configuration failed to run."
fi

echo "===================================================================="
