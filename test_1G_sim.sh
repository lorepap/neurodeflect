#!/bin/bash

# Simplified script to test if 1G simulation works
# This runs for a very short time just to verify functionality

echo "===================================================================="
echo "Testing Basic 1G Simulation Functionality"
echo "===================================================================="

cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims

# Create results directory
mkdir -p results

echo "Running DCTCP_ECMP for a short time..."
opp_runall -j1 ../../src/dc_simulations -m -u Cmdenv \
    -c DCTCP_ECMP \
    -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases \
    --image-path=../../../inet/images \
    -l ../../../inet/src/INET \
    --cmdenv-runs-to-execute=0 \
    --cmdenv-express-mode=false \
    --sim-time-limit=0.001s \
    omnetpp_1G.ini

# Check if simulation produced any output
if [ -d "results" ] && [ "$(ls -A results)" ]; then
    echo "✓ Simulation produced output files"
    ls -la results/
else
    echo "✗ No output files produced"
fi

echo "===================================================================="
echo "Test Completed"
echo "===================================================================="
