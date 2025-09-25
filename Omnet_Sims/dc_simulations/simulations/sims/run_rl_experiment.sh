#!/bin/bash

# Script to run RL policy experiment and extract results
# This script runs the DCTCP_SD_RL_POLICY configuration for a short simulation time (0.01s)
# then processes the results using the same approach as threshold experiments

echo "================================================="
echo "RL Policy Experiment"
echo "================================================="
echo ""

# Check if simulation time parameter is provided (with default)
if [ $# -eq 0 ]; then
    SIMULATION_TIME="0.01s"
    echo "Using default simulation time: $SIMULATION_TIME"
else
    SIMULATION_TIME=$1
    echo "Using provided simulation time: $SIMULATION_TIME"
fi

echo "================================================="
echo "Step 1: Ensuring Correct RL Normalization Parameters"
echo "================================================="

# Update RL configuration with correct normalization parameters
echo "Checking and updating RL normalization parameters..."
python3 update_rl_config.py

if [ $? -ne 0 ]; then
    echo "✗ Failed to update RL configuration"
    echo "  Please check that normalization parameters are available"
    echo "  Either run RL training first or ensure training_normalization_params.json exists"
    exit 1
fi

echo ""
echo "================================================="
echo "Step 2: Running RL Policy Experiment"
echo "Configuration: DCTCP_SD_RL_POLICY"
echo "Simulation Time: $SIMULATION_TIME"
echo "================================================="

# Function to extract data from RL results (similar to do_extract)
do_extract_rl() {
    local config_name=$1
    echo "Extracting data for configuration: $config_name"
    
    # First run the simulation
    echo ""
    echo "-------------------------------------------"
    echo "Running DCTCP_SD with RL Policy"
    echo "This will use the trained RL model for deflection decisions"
    echo "Simulation time: $SIMULATION_TIME"
    echo "-------------------------------------------"
    echo "Starting OMNeT++ simulation..."
    
    ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_RL_POLICY -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME -s --cmdenv-redirect-output=true -r 0
    
    if [ $? -eq 0 ]; then
        echo "✓ OMNeT++ simulation completed successfully"
        
        # Check for generated files
        vec_files=$(find results/ -name "*rl_policy*.vec" 2>/dev/null | wc -l)
        sca_files=$(find results/ -name "*rl_policy*.sca" 2>/dev/null | wc -l)
        log_files=$(find results/ -name "*rl_policy*.out" 2>/dev/null | wc -l)
        
        echo "  Generated files:"
        echo "    Vector files: $vec_files"
        echo "    Scalar files: $sca_files"
        echo "    Output files: $log_files"
        echo ""
    else
        echo "✗ OMNeT++ simulation failed!"
        return 1
    fi
    
    echo "-------------------------------------------"
    echo "Processing RL policy results"
    echo "-------------------------------------------"
    echo "Step 1: Extracting data from RL simulation results..."
    
    # Generate the extractor script first
    echo "Generating data extraction script for RL policy results..."
    python3 extractor_rl_policy_creator.py
    
    if [ ! -f "../../../extractor_rl_policy.sh" ]; then
        echo "ERROR: Failed to generate extractor script!"
        return 1
    fi
    
    echo "Running data extraction for RL policy experiments..."
    chmod +x ../../../extractor_rl_policy.sh
    ../../../extractor_rl_policy.sh
    
    if [ $? -eq 0 ]; then
        echo "✓ Data extraction completed for $config_name"
        sleep 2
    else
        echo "✗ Data extraction failed for $config_name"
        return 1
    fi
}

# Function to process RL data into dataset (adapted from run_1G_thr_dataset.sh)
process_rl_dataset() {
    local config_name=$1
    
    echo ""
    echo "============================================="
    echo "Processing RL dataset: $config_name"
    echo "============================================="
    
    # Check if we have extracted data
    if [ ! -d "extracted_results" ]; then
        echo "✗ No extracted_results directory found"
        echo "  Please run data extraction first"
        return 1
    fi
    
    # Verify that we have actual CSV files in the extracted directories
    local csv_count=$(find "extracted_results" -name "*.csv" -type f 2>/dev/null | wc -l)
    if [ $csv_count -eq 0 ]; then
        echo "✗ No CSV files found in extracted_results"
        echo "  The extraction may have failed or produced no data"
        return 1
    fi
    echo "✓ Found $csv_count CSV files in extracted data"
    
    # Prepare results directory for dataset creation
    echo "Preparing data for dataset creation..."
    rm -rf "results_${config_name}"
    mkdir -p "results_${config_name}"
    
    # Known categories from the threshold script
    local cats=(QUEUE_LEN QUEUES_TOT_LEN QUEUE_CAPACITY QUEUES_TOT_CAPACITY PACKET_ACTION RCV_TS_SEQ_NUM SND_TS_SEQ_NUM OOO_SEG SWITCH_SEQ_NUM TTL ACTION_SEQ_NUM SWITCH_ID SWITCH_ID_ACTION INTERFACE_ID RETRANSMITTED FLOW_ID PACKET_SIZE FLOW_STARTED FLOW_ENDED)

    # Copy extracted data to results directory
    for c in "${cats[@]}"; do
        if [ -d "extracted_results/${c}" ]; then
            mkdir -p "results_${config_name}/${c}"
            cp extracted_results/${c}/*.csv "results_${config_name}/${c}/" 2>/dev/null || true
        fi
    done

    # Create symbolic link for compatibility with existing scripts
    rm -f results_1G
    ln -sf results_${config_name} results_1G
    echo "✓ Created symbolic link: results_1G -> results_${config_name}"
    
    # Sanity check: ensure we have CSVs (check the actual directory, not symlink)
    local csv_count=$(find "results_${config_name}" -name "*.csv" -type f 2>/dev/null | wc -l)
    if [ "$csv_count" -eq 0 ]; then
        echo "✗ No CSV files found after preparation for RL config ${config_name}"
        rm -f results_1G
        return 1
    fi
    echo "✓ Staged $csv_count CSV files into results_1G/"
    
    # Run the dataset creation pipeline (same as threshold script)
    echo ""
    echo "RL Policy Analysis Ready"
    echo "-----------------------------------------"
    echo "ℹ  Dataset creation is not needed for RL analysis."
    echo "ℹ  The extracted CSV files are ready for analysis."
    echo "ℹ  Run the RL analysis script to analyze deflection patterns:"
    echo "   cd /home/ubuntu/practical_deflection/RL_Training/data_analysis"
    echo "   python analyze_rl_policy.py"
    echo ""
    echo "✓ RL experiment data extraction completed successfully!"
    echo "  Results available in: results_${config_name}/"
    echo "  Key data files:"
    echo "    - PACKET_ACTION: Deflection decisions (0=forward, 1=deflect)"
    echo "    - QUEUE_LEN: Queue occupancy data"
    echo ""
    
    # Save a copy of the processed results for debugging/archival
    mkdir -p processed_rl/${config_name}
    cp -r results_${config_name}/* processed_rl/${config_name}/ 2>/dev/null
    echo "✓ Archived processed results to processed_rl/${config_name}/"
    
    echo ""
    echo "✓ RL config $config_name processing completed successfully!"
    return 0
}

# Clean up previous results for RL policy experiment
echo "Cleaning up previous RL policy results..."
rm -rf results/rl_policy
rm -rf logs/rl_policy_*
rm -f rl_policy_dataset_*.csv
echo "Creating result directories..."
mkdir -p results/rl_policy
mkdir -p logs

# First, run the simulation and extract data
do_extract_rl rl_policy

# Then process the dataset
if [ $? -eq 0 ]; then
    echo "Proceeding to process RL dataset..."
    process_rl_dataset rl_policy
else
    echo "✗ Data extraction failed, skipping dataset processing"
    exit 1
fi

echo ""
echo "================================================="
echo "RL Policy Experiment Completed!"
echo "================================================="
echo ""
echo "Next steps:"
echo "1. Run the RL analysis script:"
echo "   cd /home/ubuntu/practical_deflection/RL_Training/data_analysis"
echo "   python analyze_rl_policy.py"
echo ""
echo "2. Results are available in:"
echo "   - extracted_results/: Raw extracted CSV data"
echo "   - results_rl_policy/: Organized data by category"
echo "   - processed_rl/rl_policy/: Archived results"
echo ""
