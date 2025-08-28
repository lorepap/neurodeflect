#!/bin/bash

# Script to process already-extracted threshold data and create datasets
# This version SKIPS extraction since it's already done by run_deflection_threshold_experiments.sh
# It directly uses the extracted results from the previous stage

# Default thresholds if not provided as argument
THRESHOLDS=${1:-"15000 25000 35000 45000 50000"}

# Convert space-separated thresholds to array
IFS=' ' read -ra THRESHOLD_ARRAY <<< "$THRESHOLDS"

echo "================================================="
echo "Processing Threshold Datasets"
echo "Thresholds: ${THRESHOLDS}"
echo "================================================="
echo ""
echo "Note: This script assumes extraction was already done by"
echo "run_deflection_threshold_experiments.sh"
echo ""

# Function to process data for a single threshold
process_threshold() {
    local threshold=$1
    
    echo ""
    echo "============================================="
    echo "Processing threshold: $threshold"
    echo "============================================="
    
    # The extracted results should be in one of these locations after run_deflection_threshold_experiments.sh
    local extracted_dir=""
    
    # Check where the extracted results are located
    # Option 1: Check if extracted results are already in the expected format
    if [ -d "results/threshold_${threshold}" ]; then
        echo "Found directory: results/threshold_${threshold}/"
        echo "Contents:"
        ls -la "results/threshold_${threshold}/"
        
        # The extraction created directories like QUEUE_LEN, PACKET_ACTION, etc.
        # inside results/threshold_${threshold}/
        if [ -d "results/threshold_${threshold}/QUEUE_LEN" ] || [ -d "results/threshold_${threshold}/PACKET_ACTION" ]; then
            extracted_dir="results/threshold_${threshold}"
            echo "✓ Found extracted data in results/threshold_${threshold}/"
        else
            echo "⚠ Directory exists but no extracted subdirectories found"
            echo "  Looking for raw files to check extraction status..."
            
            # Check if there are any .vec files that could be extracted
            if ls "results/threshold_${threshold}"/*.vec >/dev/null 2>&1; then
                echo "  Found .vec files - need to run extraction first"
                echo "  Attempting to extract now..."
                
                # Try to run extractor on these files
                python3 ./extractor_shell_creator.py "dctcp_sd_threshold_${threshold}"
                if [ -f "results/extractor.sh" ]; then
                    pushd "results/threshold_${threshold}/" >/dev/null
                    if [ -f "../extractor.sh" ]; then
                        bash "../extractor.sh"
                        popd >/dev/null
                        echo "✓ Extraction completed"
                        
                        # Check if extraction created necessary directories
                        if [ -d "results/threshold_${threshold}/QUEUE_LEN" ] || [ -d "results/threshold_${threshold}/PACKET_ACTION" ]; then
                            extracted_dir="results/threshold_${threshold}"
                            echo "✓ Successfully extracted data"
                        fi
                    else
                        popd >/dev/null
                        echo "✗ extractor.sh not found"
                    fi
                else
                    echo "✗ Failed to create extractor.sh"
                fi
            fi
        fi
    fi
    
    # Option 2: Check if we need to look in extracted_results from a previous partial run
    if [ -z "$extracted_dir" ] && [ -d "extracted_results_threshold_${threshold}" ]; then
        extracted_dir="extracted_results_threshold_${threshold}"
        echo "✓ Found extracted data in extracted_results_threshold_${threshold}/"
    fi
    
    # Option 3: The extraction might have put results directly in extracted_results
    if [ -z "$extracted_dir" ] && [ -d "extracted_results" ]; then
        # Check if this is the current threshold's data
        if [ -f "extracted_results/QUEUE_LEN/"*"threshold_${threshold}"* ] 2>/dev/null; then
            extracted_dir="extracted_results"
            echo "✓ Found extracted data in extracted_results/ (for threshold ${threshold})"
        else
            echo "Checking extracted_results directory contents:"
            find "extracted_results" -type f -name "*.csv" | head -n 5
        fi
    fi
    
    # If no extracted data found, we need to run extraction
    if [ -z "$extracted_dir" ]; then
        echo "✗ No extracted data found for threshold ${threshold}"
        echo "  You may need to run the extraction step first using run_deflection_threshold_experiments.sh"
        echo "  Or manually extract using the extractor scripts"
        return 1
    fi
    
    # Verify that we have actual CSV files in the extracted directories
    local csv_count=$(find "$extracted_dir" -name "*.csv" -type f 2>/dev/null | wc -l)
    if [ $csv_count -eq 0 ]; then
        echo "✗ No CSV files found in $extracted_dir"
        echo "  The extraction may have failed or produced no data"
        return 1
    fi
    echo "✓ Found $csv_count CSV files in extracted data"
    
    # Move/copy the extracted results to the expected location for dataset creation
    echo "Preparing data for dataset creation..."
    rm -rf results_1G_thr_${threshold}
    
    # Copy the extracted data to the working directory
    if [ "$extracted_dir" != "results_1G_thr_${threshold}" ]; then
        cp -r "$extracted_dir" results_1G_thr_${threshold}
        echo "✓ Copied extracted data to results_1G_thr_${threshold}/"
    fi
    
    # Create symbolic link so existing Python scripts work with expected directory name
    rm -f results_1G
    ln -sf results_1G_thr_${threshold} results_1G
    echo "✓ Created symbolic link: results_1G -> results_1G_thr_${threshold}"
    
    # Run the dataset creation pipeline
    echo ""
    echo "Running dataset creation pipeline..."
    echo "-----------------------------------------"
    
    # Step 1: Create initial dataset
    if [ -f "create_dataset.py" ]; then
        echo "Creating dataset..."
        if python3 create_dataset.py 2>&1 | tee /tmp/create_dataset_${threshold}.log | grep -E "(Created|Error|Warning|CSV|rows|dataset)" ; then
            echo "✓ Dataset creation completed"
        else
            echo "⚠ Dataset creation may have encountered issues (check /tmp/create_dataset_${threshold}.log)"
        fi
    else
        echo "✗ create_dataset.py not found"
        rm -f results_1G
        return 1
    fi
    
    # Step 2: Filter collisions (optional)
    if [ -f "filter_collisions.py" ]; then
        echo ""
        echo "Filtering collisions..."
        if python3 filter_collisions.py 2>&1 | tee /tmp/filter_collisions_${threshold}.log | grep -E "(Filtered|Removed|Error|Warning|rows)" ; then
            echo "✓ Collision filtering completed"
        else
            echo "⚠ Collision filtering may have encountered issues"
        fi
    else
        echo "⚠ filter_collisions.py not found, skipping collision filtering"
    fi
    
    # Step 3: Filter overlapping timestamps (optional)
    if [ -f "filter_overlapping_timestamps.py" ]; then
        echo ""
        echo "Filtering overlapping timestamps..."
        if python3 filter_overlapping_timestamps.py 2>&1 | tee /tmp/filter_overlap_${threshold}.log | grep -E "(Filtered|Removed|Error|Warning|rows)" ; then
            echo "✓ Overlapping timestamp filtering completed"
        else
            echo "⚠ Overlapping timestamp filtering may have encountered issues"
        fi
    else
        echo "⚠ filter_overlapping_timestamps.py not found, skipping overlap filtering"
    fi
    
    # Step 4: Save the final dataset with threshold-specific name
    echo ""
    echo "Saving final dataset..."
    
    if [ -f "results_1G/final_dataset.csv" ]; then
        cp results_1G/final_dataset.csv threshold_dataset_${threshold}.csv
        local lines=$(wc -l < threshold_dataset_${threshold}.csv)
        echo "✓ Created threshold_dataset_${threshold}.csv ($lines lines)"
    elif [ -f "results_1G/dataset.csv" ]; then
        # Fallback if final_dataset.csv doesn't exist but dataset.csv does
        cp results_1G/dataset.csv threshold_dataset_${threshold}.csv
        local lines=$(wc -l < threshold_dataset_${threshold}.csv)
        echo "✓ Created threshold_dataset_${threshold}.csv from dataset.csv ($lines lines)"
    else
        echo "✗ No dataset.csv or final_dataset.csv found for threshold $threshold"
        rm -f results_1G
        return 1
    fi
    
    # Remove the symbolic link
    rm -f results_1G
    
    # Save a copy of the processed results for debugging/archival
    mkdir -p processed_thresholds/${threshold}
    cp -r results_1G_thr_${threshold}/* processed_thresholds/${threshold}/ 2>/dev/null
    echo "✓ Archived processed results to processed_thresholds/${threshold}/"
    
    echo ""
    echo "✓ Threshold $threshold processing completed successfully!"
    return 0
}

# Main execution loop
successful_thresholds=()
failed_thresholds=()

echo "Starting processing of ${#THRESHOLD_ARRAY[@]} thresholds..."

for threshold in "${THRESHOLD_ARRAY[@]}"; do
    # Remove any whitespace
    threshold=$(echo "$threshold" | xargs)
    
    if process_threshold "$threshold"; then
        successful_thresholds+=("$threshold")
    else
        failed_thresholds+=("$threshold")
        echo "✗ Failed to process threshold $threshold"
    fi
done

# Summary report
echo ""
echo "================================================="
echo "Processing Summary"
echo "================================================="

if [ ${#successful_thresholds[@]} -gt 0 ]; then
    echo "✓ Successfully processed: ${successful_thresholds[*]}"
    echo ""
    echo "Generated datasets:"
    for t in "${successful_thresholds[@]}"; do
        if [ -f "threshold_dataset_${t}.csv" ]; then
            local size=$(du -h "threshold_dataset_${t}.csv" | cut -f1)
            local lines=$(wc -l < "threshold_dataset_${t}.csv")
            echo "  - threshold_dataset_${t}.csv (${size}, ${lines} lines)"
        fi
    done
fi

if [ ${#failed_thresholds[@]} -gt 0 ]; then
    echo ""
    echo "✗ Failed: ${failed_thresholds[*]}"
fi

if [ ${#successful_thresholds[@]} -eq 0 ]; then
    echo "✗ No thresholds were successfully processed"
    exit 1
fi

# Combine datasets if multiple were successful
if [ ${#successful_thresholds[@]} -gt 1 ]; then
    echo ""
    echo "================================================="
    echo "Combining threshold datasets"
    echo "================================================="
    
    if [ -f "combine_threshold_datasets.py" ]; then
        echo "Running dataset combination..."
        python3 combine_threshold_datasets.py "${successful_thresholds[*]}"
        
        if [ -f "combined_threshold_dataset.csv" ]; then
            local size=$(du -h "combined_threshold_dataset.csv" | cut -f1)
            local lines=$(wc -l < "combined_threshold_dataset.csv")
            echo "✓ Created combined_threshold_dataset.csv (${size}, ${lines} lines)"
        else
            echo "⚠ combine_threshold_datasets.py did not create output file"
        fi
    else
        echo "⚠ combine_threshold_datasets.py not found"
        echo "Individual threshold datasets are available as threshold_dataset_*.csv"
    fi
fi

echo ""
echo "================================================="
echo "All processing completed!"
echo "================================================="
echo ""
echo "Output files:"
echo "  Individual datasets: threshold_dataset_*.csv"
if [ -f "combined_threshold_dataset.csv" ]; then
    echo "  Combined dataset: combined_threshold_dataset.csv"
fi
echo "  Archived results: processed_thresholds/"
echo ""
echo "Log files saved in /tmp/:"
for t in "${successful_thresholds[@]}"; do
    echo "  - /tmp/create_dataset_${t}.log"
    [ -f "/tmp/filter_collisions_${t}.log" ] && echo "  - /tmp/filter_collisions_${t}.log"
    [ -f "/tmp/filter_overlap_${t}.log" ] && echo "  - /tmp/filter_overlap_${t}.log"
done
echo "================================================="