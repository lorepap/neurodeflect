#!/bin/bash

# Script to run threshold experiments and create datasets for each threshold
# This script follows the same pattern as run_1G_dataset.sh but for multiple thresholds

# Default thresholds if not provided as argument
THRESHOLDS=${1:-"0.3,0.5,0.7,0.9"}

# Convert comma-separated thresholds to array
IFS=',' read -ra THRESHOLD_ARRAY <<< "$THRESHOLDS"

echo "Running threshold experiments for thresholds: ${THRESHOLDS}"

# Function to extract data for a given configuration
do_extract () {
    local config_name=$1
    echo "Extracting data for configuration: $config_name"
    python3 ./extractor_shell_creator.py $config_name
    pushd ./results/
    bash extractor.sh
    popd
    sleep 5
}

# Function to process data for a single threshold
process_threshold() {
    local threshold=$1
    local config_name="dctcp_sd_thr_${threshold}"
    local threshold_dir="results_backup/threshold_${threshold}"
    
    echo ""
    echo "============================================="
    echo "Processing threshold: $threshold"
    echo "============================================="
    
    # Check if threshold data exists
    if [ ! -d "$threshold_dir" ]; then
        echo "✗ Threshold directory $threshold_dir not found"
        return 1
    fi
    
    # Clean up previous results
    rm -rf results
    rm -rf logs
    rm -rf figs
    rm -rf extracted_results
    rm -rf results_1G
    
    # Create the directory to save extracted_results
    bash dir_creator.sh
    
    sudo chmod -R +777 ./
    
    echo ""
    echo "-------------------------------------------"
    echo "Processing existing data for threshold $threshold"
    echo "-------------------------------------------"
    
    # Copy threshold data to results directory (removing threshold suffix for compatibility)
    mkdir -p results
    for file in "$threshold_dir"/*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            # Remove threshold suffix from filename to match expected naming
            new_filename=$(echo "$filename" | sed "s/_threshold_${threshold}//g")
            cp "$file" "results/$new_filename"
        fi
    done
    
    # Extract data using the same pattern as run_1G_dataset.sh
    do_extract $config_name
    
    # Create logs directory for this threshold
    mkdir -p logs/${config_name}
    cp results/*.out logs/${config_name}/
    
    # Move the extracted results to threshold-specific directory
    echo "Moving the extracted results to results_1G_thr_${threshold}"
    rm -rf results_1G_thr_${threshold}
    mv extracted_results results_1G_thr_${threshold}
    
    # Create dataset for this threshold using the same logic as the original script
    echo "Creating dataset for threshold $threshold"
    # Temporarily link results_1G to the threshold-specific directory so the existing scripts work
    ln -sf results_1G_thr_${threshold} results_1G
    
    python3 create_dataset.py
    echo "Dataset created for threshold $threshold"
    
    echo "Filtering collisions for threshold $threshold"
    python3 filter_collisions.py
    echo "Collisions filtered for threshold $threshold"
    
    echo "Filtering overlapping timestamps for threshold $threshold"
    python3 filter_overlapping_timestamps.py
    echo "Overlapping timestamps filtered for threshold $threshold"
    
    # Copy final dataset to a threshold-specific name
    if [ -f "results_1G/final_dataset.csv" ]; then
        cp results_1G/final_dataset.csv threshold_dataset_${threshold}.csv
        echo "✓ Created threshold_dataset_${threshold}.csv"
    else
        echo "✗ final_dataset.csv not found for threshold $threshold"
        return 1
    fi
    
    # Remove the symbolic link
    rm -f results_1G
    
    echo "✓ Threshold $threshold processing completed!"
    return 0
}

# Process each threshold
successful_thresholds=()
for threshold in "${THRESHOLD_ARRAY[@]}"; do
    # Remove any whitespace
    threshold=$(echo "$threshold" | xargs)
    if process_threshold "$threshold"; then
        successful_thresholds+=("$threshold")
    else
        echo "✗ Failed to process threshold $threshold"
    fi
done

if [ ${#successful_thresholds[@]} -eq 0 ]; then
    echo "✗ No thresholds were successfully processed"
    exit 1
fi

echo ""
echo "✓ Successfully processed thresholds: ${successful_thresholds[*]}"

echo ""
echo "============================================="
echo "Combining all threshold datasets"
echo "============================================="

# Create a Python script to combine all threshold datasets
python3 combine_threshold_datasets.py "${THRESHOLDS}"

echo ""
echo "============================================="
echo "All threshold processing completed!"
echo "Final combined dataset: combined_threshold_dataset.csv"
echo "============================================="
