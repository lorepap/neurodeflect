#!/bin/bash

# Generate threshold dataset with short simulation
# Creates a combined CSV with deflection threshold as a feature

echo "===================================================================="
echo "Generating Threshold Dataset - Short Simulation"
echo "===================================================================="

cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims

# Clean previous results
rm -rf results
rm -rf logs
rm -rf figs
rm -rf extracted_results
rm -rf results_1G

# Create directories
bash dir_creator.sh
sudo chmod -R +777 ./

echo "Running DCTCP_SD_THRESHOLD_VARIATION for 10ms simulation..."
echo "This will generate datasets for thresholds: 0.3, 0.5, 0.7, 0.9"

# Run simulation with short time limit for all thresholds
if opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv \
    -c DCTCP_SD_THRESHOLD_VARIATION \
    -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases \
    --image-path=../../../inet/images \
    -l ../../../inet/src/INET \
    omnetpp_1G.ini \
    --sim-time-limit=0.01s; then
    
    echo "✓ Simulation completed successfully"
    echo "✓ Generated data for all thresholds"
    
    # Check results structure
    echo ""
    echo "Results structure:"
    find results -name "*.out" | head -10
    
else
    echo "✗ Simulation failed"
    exit 1
fi

echo ""
echo "===================================================================="
echo "Processing Data for All Thresholds"
echo "===================================================================="

# Extract data for each threshold
echo "Extracting data for all threshold configurations..."
python3 ./extractor_shell_creator.py dctcp_sd_threshold_variation
pushd ./results/
bash extractor.sh
popd

# Move extracted results
echo "Moving extracted results to results_1G"
rm -rf results_1G
mv extracted_results results_1G

# Create individual datasets and combine them
echo "Creating combined threshold dataset..."
python3 create_dataset.py

echo "Filtering collisions..."
python3 filter_collisions.py

echo "Filtering overlapping timestamps..."
python3 filter_overlapping_timestamps.py

echo ""
echo "===================================================================="
echo "Dataset Generation Complete"
echo "===================================================================="

# Check final dataset
if [ -f "dataset_final.csv" ]; then
    echo "✓ Final dataset created: dataset_final.csv"
    echo "Dataset info:"
    echo "Lines: $(wc -l < dataset_final.csv)"
    echo "Columns: $(head -1 dataset_final.csv | tr ',' '\n' | wc -l)"
    echo "Sample columns: $(head -1 dataset_final.csv)"
    echo ""
    echo "Sample data (first 5 rows):"
    head -5 dataset_final.csv
else
    echo "✗ Final dataset not found"
    echo "Available files:"
    ls -la *.csv 2>/dev/null || echo "No CSV files found"
fi

echo ""
echo "===================================================================="
echo "Complete!"
echo "===================================================================="
