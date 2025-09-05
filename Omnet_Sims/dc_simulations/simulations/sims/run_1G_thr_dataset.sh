#!/bin/bash

# Script to process already-extracted threshold data and create datasets
# This version SKIPS extraction since it's already done by run_deflection_threshold_experiments.sh
# It directly uses the extracted results from the previous stage

# Default thresholds if not provided as argument
THRESHOLDS=${1:-"15000 25000 37500 50000"}

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

# Function to process data for a single threshold (no extraction here)
process_threshold() {
    local threshold=$1
    
    echo ""
    echo "============================================="
    echo "Processing threshold: $threshold"
    echo "============================================="
    
    # Locate already-extracted CSVs (produced by run_deflection_threshold_experiments.sh)
    # Preferred layout: per-threshold extracted subdirs under results/threshold_<thr>/
    local extracted_dir=""
    if [ -d "results/threshold_${threshold}/QUEUE_LEN" ] || [ -d "results/threshold_${threshold}/PACKET_ACTION" ]; then
        extracted_dir="results/threshold_${threshold}"
        echo "✓ Using extracted data in results/threshold_${threshold}/"
    elif [ -d "extracted_results" ]; then
        echo "✓ Using shared extracted_results/; will filter files for threshold ${threshold}"
        extracted_dir="extracted_results"
    fi

    if [ -z "$extracted_dir" ]; then
        echo "✗ No extracted data found for threshold ${threshold}"
        echo "  Please run ./run_deflection_threshold_experiments.sh <time> first"
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
    rm -rf "results_1G_thr_${threshold}"

    # Populate results_1G_thr_<thr> with ONLY this threshold's CSVs
    mkdir -p "results_1G_thr_${threshold}"
    # Known categories
    local cats=(QUEUE_LEN QUEUES_TOT_LEN QUEUE_CAPACITY QUEUES_TOT_CAPACITY PACKET_ACTION RCV_TS_SEQ_NUM SND_TS_SEQ_NUM OOO_SEG SWITCH_SEQ_NUM TTL ACTION_SEQ_NUM SWITCH_ID SWITCH_ID_ACTION INTERFACE_ID RETRANSMITTED FLOW_ID PACKET_SIZE)

    if [ "$extracted_dir" = "extracted_results" ]; then
        for c in "${cats[@]}"; do
            mkdir -p "results_1G_thr_${threshold}/${c}"
            # copy only matching CSVs for this threshold - prefer manually extracted files
            copied_files=0
            
            # Pattern 1: *_${threshold}_${threshold}.csv (for our manually extracted files) - PREFER THESE
            if compgen -G "${extracted_dir}/${c}/*_${threshold}_${threshold}.csv" > /dev/null 2>&1; then
                cp ${extracted_dir}/${c}/*_${threshold}_${threshold}.csv "results_1G_thr_${threshold}/${c}/"
                copied_files=1
            # Pattern 2: *threshold_${threshold}*.csv (fallback)
            elif compgen -G "${extracted_dir}/${c}/*threshold_${threshold}*.csv" > /dev/null 2>&1; then
                cp ${extracted_dir}/${c}/*threshold_${threshold}*.csv "results_1G_thr_${threshold}/${c}/"
                copied_files=1
            # Pattern 3: *_${threshold}.csv (general fallback)
            elif compgen -G "${extracted_dir}/${c}/*_${threshold}.csv" > /dev/null 2>&1; then
                cp ${extracted_dir}/${c}/*_${threshold}.csv "results_1G_thr_${threshold}/${c}/"
                copied_files=1
            fi
            
            if [ $copied_files -eq 0 ]; then
                echo "⚠ No files found for category $c and threshold $threshold"
            fi
        done
    else
        # Already per-threshold structured; copy the whole directory
        cp -r "$extracted_dir" "results_1G_thr_${threshold}"
    fi

    # Sanity: ensure we have CSVs
    local csv_count=$(find "results_1G_thr_${threshold}" -name "*.csv" -type f 2>/dev/null | wc -l)
    if [ "$csv_count" -eq 0 ]; then
        echo "✗ No CSV files found after preparation for threshold ${threshold}"
        return 1
    fi
    echo "✓ Staged $csv_count CSV files into results_1G_thr_${threshold}/"
    
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
        
        # Debug: Check file counts and sizes before running create_dataset.py
        echo "Debug info for threshold $threshold:"
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
        
        if python3 create_dataset.py 2>&1 | tee /tmp/create_dataset_${threshold}.log; then
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
                cat /tmp/create_dataset_${threshold}.log
                rm -f results_1G
                return 1
            fi
        else
            echo "✗ Dataset creation failed (exit code $?)"
            echo "Error details:"
            cat /tmp/create_dataset_${threshold}.log
            rm -f results_1G
            return 1
        fi
    else
        echo "✗ create_dataset.py not found"
        rm -f results_1G
        return 1
    fi
    
    # Step 2: Filter collisions (RE-ENABLED - now works with merged CSV format)
    echo ""
    echo "Running collision filtering on merged dataset..."
    if [ -f "filter_collisions.py" ] && [ -f "results_1G/merged_final.csv" ]; then
        if python3 filter_collisions.py --input-file results_1G/merged_final.csv --output-file results_1G/collision_filtered.csv 2>&1 | tee /tmp/filter_collisions_${threshold}.log; then
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
    
    # Step 3: Add out-of-order features (RE-ENABLED - now works with merged CSV format)
    echo ""
    echo "Adding out-of-order features to dataset..."
    if [ -f "filter_overlapping_timestamps.py" ] && [ -f "results_1G/collision_filtered.csv" ]; then
        if python3 filter_overlapping_timestamps.py --input-file results_1G/collision_filtered.csv --output-file results_1G/ooo_enhanced.csv 2>&1 | tee /tmp/filter_overlap_${threshold}.log; then
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
    
    # Step 3.5: Enrich packet-level data with flow information (FlowID, FCT, SWITCH_ID)
    # This creates a packet-level dataset where each row is a packet with attached flow info
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
        
        if python3 enrich_packets_with_flows.py "$input_file" results_1G results_1G/packet_level_with_flows.csv --log-file /tmp/enrich_packets_${threshold}.log 2>&1; then
            if [ -s results_1G/packet_level_with_flows.csv ]; then
                echo "✓ Packet-level flow enrichment completed"
                # Use the enriched packet-level dataset as final
                mv results_1G/packet_level_with_flows.csv results_1G/final_dataset.csv
            else
                echo "⚠ Packet-level flow enrichment produced empty output, keeping OOO-enhanced dataset"
            fi
        else
            echo "⚠ Packet-level flow enrichment failed (exit code $?). See /tmp/enrich_packets_${threshold}.log"
            echo "✓ Keeping OOO-enhanced dataset"
        fi
    else
        echo "⚠ enrich_packets_with_flows.py not found, using OOO-enhanced dataset without flow enrichment"
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

# Clean previous artifacts from earlier runs
rm -f threshold_dataset_*.csv combined_threshold_dataset.csv
rm -f results_1G
rm -rf processed_thresholds

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
            size=$(du -h "threshold_dataset_${t}.csv" | cut -f1)
            lines=$(wc -l < "threshold_dataset_${t}.csv")
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
            size=$(du -h "combined_threshold_dataset.csv" | cut -f1)
            lines=$(wc -l < "combined_threshold_dataset.csv")
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