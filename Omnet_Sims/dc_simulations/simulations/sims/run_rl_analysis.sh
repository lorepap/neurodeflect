#!/bin/bash

# Script to analyze RL policy results using the same approach as threshold experiments
# This processes the RL experiment results into datasets for analysis

echo "================================================="
echo "RL Policy Results Analysis"
echo "================================================="
echo ""

# Function to extract data from RL results (similar to do_extract)
do_extract_rl() {
    local config_name=$1
    echo "Extracting data for configuration: $config_name"
    
    # Use extractor to process the results
    python3 ./extractor_shell_creator.py $config_name
    
    if [ -f "./results/extractor.sh" ]; then
        pushd ./results/
        bash extractor.sh
        popd
        echo "✓ Data extraction completed for $config_name"
        sleep 2
    else
        echo "✗ Extractor script not created for $config_name"
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
    rm -rf "results_rl_${config_name}"
    mkdir -p "results_rl_${config_name}"
    
    # Known categories from the threshold script
    local cats=(QUEUE_LEN QUEUES_TOT_LEN QUEUE_CAPACITY QUEUES_TOT_CAPACITY PACKET_ACTION RCV_TS_SEQ_NUM SND_TS_SEQ_NUM OOO_SEG SWITCH_SEQ_NUM TTL ACTION_SEQ_NUM SWITCH_ID SWITCH_ID_ACTION INTERFACE_ID RETRANSMITTED FLOW_ID PACKET_SIZE)

    # Copy extracted data to results directory
    for c in "${cats[@]}"; do
        if [ -d "extracted_results/${c}" ]; then
            mkdir -p "results_rl_${config_name}/${c}"
            cp extracted_results/${c}/*.csv "results_rl_${config_name}/${c}/" 2>/dev/null || true
        fi
    done

    # Create symbolic link for compatibility with existing scripts
    rm -f results_1G
    ln -sf results_rl_${config_name} results_1G
    echo "✓ Created symbolic link: results_1G -> results_rl_${config_name}"
    
    # Sanity check: ensure we have CSVs
    local csv_count=$(find "results_1G" -name "*.csv" -type f 2>/dev/null | wc -l)
    if [ "$csv_count" -eq 0 ]; then
        echo "✗ No CSV files found after preparation for RL config ${config_name}"
        rm -f results_1G
        return 1
    fi
    echo "✓ Staged $csv_count CSV files into results_1G/"
    
    # Run the dataset creation pipeline (same as threshold script)
    echo ""
    echo "Running dataset creation pipeline..."
    echo "-----------------------------------------"
    
    # Step 1: Create initial dataset
    if [ -f "create_dataset.py" ]; then
        echo "Creating dataset..."
        
        # Debug: Check file counts and sizes before running create_dataset.py
        echo "Debug info for RL config $config_name:"
        echo "Files in results_1G directories:"
        for dir in QUEUE_CAPACITY QUEUES_TOT_CAPACITY QUEUE_LEN QUEUES_TOT_LEN SWITCH_SEQ_NUM TTL PACKET_SIZE ACTION_SEQ_NUM PACKET_ACTION; do
            if [ -d "results_1G/$dir" ]; then
                file_count=$(ls results_1G/$dir/*.csv 2>/dev/null | wc -l)
                if [ $file_count -gt 0 ]; then
                    first_file=$(ls results_1G/$dir/*.csv 2>/dev/null | head -1)
                    row_count=$(wc -l < "$first_file" 2>/dev/null || echo "0")
                    echo "  $dir: $file_count files, $row_count rows"
                else
                    echo "  $dir: No CSV files"
                fi
            else
                echo "  $dir: Directory missing"
            fi
        done
        
        # Clean any existing merged files to ensure fresh creation
        rm -f results_1G/merged_final.csv results_1G/dataset.csv results_1G/final_dataset.csv
        rm -f results_1G/removed_seq_nums.csv results_1G/filtered_*.csv
        
        # Also clean any temporary files
        rm -f merged_final.csv dataset.csv final_dataset.csv
        
        if python3 create_dataset.py 2>&1 | tee /tmp/create_dataset_rl_${config_name}.log; then
            if [ -f "results_1G/merged_final.csv" ]; then
                # Verify the merged file has the correct columns including packet_size
                header=$(head -1 results_1G/merged_final.csv)
                row_count=$(wc -l < results_1G/merged_final.csv)
                echo "✓ Dataset creation completed"
                echo "  Created merged_final.csv with $row_count rows"
                echo "  Header: $header"
                
                # Check if packet_size column is present
                if echo "$header" | grep -q "packet_size"; then
                    echo "  ✓ packet_size column found in dataset"
                else
                    echo "  ⚠ WARNING: packet_size column missing from dataset!"
                fi
            else
                echo "✗ Dataset creation failed - no output file created"
                cat /tmp/create_dataset_rl_${config_name}.log
                rm -f results_1G
                return 1
            fi
        else
            echo "✗ Dataset creation failed (exit code $?)"
            echo "Error details:"
            cat /tmp/create_dataset_rl_${config_name}.log
            rm -f results_1G
            return 1
        fi
    else
        echo "✗ create_dataset.py not found"
        rm -f results_1G
        return 1
    fi
    
    # Step 2: Filter collisions
    echo ""
    echo "Running collision filtering on merged dataset..."
    if [ -f "filter_collisions.py" ] && [ -f "results_1G/merged_final.csv" ]; then
        if python3 filter_collisions.py --input-file results_1G/merged_final.csv --output-file results_1G/collision_filtered.csv 2>&1 | tee /tmp/filter_collisions_rl_${config_name}.log; then
            if [ -f "results_1G/collision_filtered.csv" ]; then
                echo "✓ Collision filtering completed successfully"
            else
                echo "⚠ Collision filtering completed but no output file created"
                cp results_1G/merged_final.csv results_1G/collision_filtered.csv
            fi
        else
            echo "⚠ Collision filtering failed, using unfiltered data"
            cp results_1G/merged_final.csv results_1G/collision_filtered.csv
        fi
    else
        echo "⚠ Collision filtering skipped (missing filter_collisions.py or merged_final.csv)"
        cp results_1G/merged_final.csv results_1G/collision_filtered.csv
    fi
    
    # Step 3: Add out-of-order features
    echo ""
    echo "Adding out-of-order features to dataset..."
    if [ -f "filter_overlapping_timestamps.py" ] && [ -f "results_1G/collision_filtered.csv" ]; then
        if python3 filter_overlapping_timestamps.py --input-file results_1G/collision_filtered.csv --output-file results_1G/ooo_enhanced.csv 2>&1 | tee /tmp/filter_overlap_rl_${config_name}.log; then
            if [ -f "results_1G/ooo_enhanced.csv" ]; then
                echo "✓ Out-of-order feature computation completed successfully"
                # Use the OOO-enhanced dataset as the base for flow enrichment
                cp results_1G/ooo_enhanced.csv results_1G/final_dataset.csv
            else
                echo "⚠ OOO feature computation completed but no output file created"
                cp results_1G/collision_filtered.csv results_1G/final_dataset.csv
            fi
        else
            echo "⚠ OOO feature computation failed, using collision-filtered data"
            cp results_1G/collision_filtered.csv results_1G/final_dataset.csv
        fi
    else
        echo "⚠ OOO feature computation skipped (missing filter_overlapping_timestamps.py or collision_filtered.csv)"
        cp results_1G/collision_filtered.csv results_1G/final_dataset.csv
    fi
    
    # Step 4: Enrich packet-level data with flow information
    echo ""
    echo "Enriching packet-level data with flow information..."
    
    # Ensure we have a final_dataset.csv (should be OOO-enhanced now)
    if [ ! -f "results_1G/final_dataset.csv" ]; then
        echo "✗ No final_dataset.csv found"
        rm -f results_1G
        return 1
    fi
    
    if [ -f "enrich_packets_with_flows.py" ]; then
        # Use the OOO-enhanced dataset as input
        input_file="results_1G/final_dataset.csv"
        
        if python3 enrich_packets_with_flows.py "$input_file" results_1G results_1G/packet_level_with_flows.csv --log-file /tmp/enrich_packets_rl_${config_name}.log 2>&1; then
            if [ -s results_1G/packet_level_with_flows.csv ]; then
                echo "✓ Packet-level flow enrichment completed"
                # Use the enriched packet-level dataset as final
                mv results_1G/packet_level_with_flows.csv results_1G/final_dataset.csv
            else
                echo "⚠ Packet-level flow enrichment produced empty output, keeping OOO-enhanced dataset"
            fi
        else
            echo "⚠ Packet-level flow enrichment failed (exit code $?). See /tmp/enrich_packets_rl_${config_name}.log"
            echo "✓ Keeping OOO-enhanced dataset"
        fi
    else
        echo "⚠ enrich_packets_with_flows.py not found, using OOO-enhanced dataset without flow enrichment"
    fi
    
    # Step 5: Save the final dataset with RL-specific name
    echo ""
    echo "Saving final RL dataset..."
    
    if [ -f "results_1G/final_dataset.csv" ]; then
        cp results_1G/final_dataset.csv rl_policy_dataset_${config_name}.csv
        local lines=$(wc -l < rl_policy_dataset_${config_name}.csv)
        echo "✓ Created rl_policy_dataset_${config_name}.csv ($lines lines)"
    elif [ -f "results_1G/dataset.csv" ]; then
        # Fallback if final_dataset.csv doesn't exist but dataset.csv does
        cp results_1G/dataset.csv rl_policy_dataset_${config_name}.csv
        local lines=$(wc -l < rl_policy_dataset_${config_name}.csv)
        echo "✓ Created rl_policy_dataset_${config_name}.csv from dataset.csv ($lines lines)"
    else
        echo "✗ No dataset.csv or final_dataset.csv found for RL config $config_name"
        rm -f results_1G
        return 1
    fi
    
    # Remove the symbolic link
    rm -f results_1G
    
    # Save a copy of the processed results for debugging/archival
    mkdir -p processed_rl/${config_name}
    cp -r results_rl_${config_name}/* processed_rl/${config_name}/ 2>/dev/null
    echo "✓ Archived processed results to processed_rl/${config_name}/"
    
    echo ""
    echo "✓ RL config $config_name processing completed successfully!"
    return 0
}

# Main execution
echo "Starting RL policy analysis..."

# Configuration name
CONFIG_NAME="rl_policy"

# Clean previous artifacts
rm -f rl_policy_dataset_*.csv
rm -f results_1G
rm -rf processed_rl

# Step 1: Extract data from RL results
echo "Step 1: Extracting data from RL simulation results..."
if do_extract_rl "$CONFIG_NAME"; then
    echo "✓ Data extraction successful"
else
    echo "✗ Data extraction failed"
    exit 1
fi

# Step 2: Process into dataset
echo ""
echo "Step 2: Processing extracted data into analysis dataset..."
if process_rl_dataset "$CONFIG_NAME"; then
    echo "✓ Dataset processing successful"
else
    echo "✗ Dataset processing failed"
    exit 1
fi

# Summary report
echo ""
echo "================================================="
echo "RL Policy Analysis Summary"
echo "================================================="

if [ -f "rl_policy_dataset_${CONFIG_NAME}.csv" ]; then
    size=$(du -h "rl_policy_dataset_${CONFIG_NAME}.csv" | cut -f1)
    lines=$(wc -l < "rl_policy_dataset_${CONFIG_NAME}.csv")
    echo "✓ Generated dataset: rl_policy_dataset_${CONFIG_NAME}.csv"
    echo "  Size: ${size}"
    echo "  Lines: ${lines}"
    echo ""
    
    # Show dataset structure
    echo "Dataset structure:"
    head -1 "rl_policy_dataset_${CONFIG_NAME}.csv"
    echo ""
    
    # Show some basic statistics
    echo "Basic statistics:"
    echo "  Total rows: $(tail -n +2 rl_policy_dataset_${CONFIG_NAME}.csv | wc -l)"
    
    # Check for RL-specific features
    header=$(head -1 "rl_policy_dataset_${CONFIG_NAME}.csv")
    if echo "$header" | grep -q "packet_action"; then
        echo "  ✓ Packet actions recorded (for RL decision analysis)"
    fi
    if echo "$header" | grep -q "queue_len"; then
        echo "  ✓ Queue length data available (RL state feature)"
    fi
    if echo "$header" | grep -q "ttl"; then
        echo "  ✓ TTL data available (RL state feature)"  
    fi
    if echo "$header" | grep -q "packet_size"; then
        echo "  ✓ Packet size data available (RL state feature)"
    fi
    
else
    echo "✗ No dataset generated"
    exit 1
fi

echo ""
echo "Archived results:"
echo "  Processed data: processed_rl/${CONFIG_NAME}/"
echo ""
echo "Log files saved in /tmp/:"
echo "  - /tmp/create_dataset_rl_${CONFIG_NAME}.log"
echo "  - /tmp/filter_collisions_rl_${CONFIG_NAME}.log"
echo "  - /tmp/filter_overlap_rl_${CONFIG_NAME}.log"
echo "  - /tmp/enrich_packets_rl_${CONFIG_NAME}.log"

echo ""
echo "================================================="
echo "RL Policy Analysis completed!"
echo "================================================="
echo ""
echo "Next steps:"
echo "1. Analyze RL decision patterns in the dataset"
echo "2. Compare with baseline deflection strategies"
echo "3. Evaluate flow completion times and network performance"
echo "4. Create visualizations of RL policy behavior"
echo ""
echo "The dataset is ready for analysis with Python/R scripts!"
echo "================================================="
