# Save extracted results in threshold-specific directories
threshold=$1
extracted_dir="extracted_results"
cats=(QUEUE_LEN QUEUES_TOT_LEN QUEUE_CAPACITY QUEUES_TOT_CAPACITY PACKET_ACTION RCV_TS_SEQ_NUM SND_TS_SEQ_NUM OOO_SEG SWITCH_SEQ_NUM TTL ACTION_SEQ_NUM SWITCH_ID SWITCH_ID_ACTION INTERFACE_ID RETRANSMITTED FLOW_ID PACKET_SIZE REQUESTER_ID FLOW_STARTED FLOW_ENDED)
echo "Testing copy process for threshold ${threshold}..."
echo "Source directory: ${extracted_dir}"
echo "Checking if source exists: $(ls -la ${extracted_dir} 2>/dev/null | wc -l) items"

if [ "$extracted_dir" = "extracted_results" ]; then
    for c in "${cats[@]}"; do
        echo "Processing category: $c"
        mkdir -p "results_1G_thr_${threshold}/${c}"
        
        # Check if source directory exists
        if [ -d "${extracted_dir}/${c}" ]; then
            echo "  Source dir exists: ${extracted_dir}/${c}"
            # List what files match the pattern
            echo "  Files matching pattern:"
            ls -la ${extracted_dir}/${c}/*threshold_${threshold}*.csv 2>/dev/null || echo "    No matching files"
            
            copied_files=0
            if compgen -G "${extracted_dir}/${c}/*threshold_${threshold}*.csv" > /dev/null 2>&1; then
                cp ${extracted_dir}/${c}/*threshold_${threshold}*.csv "results_1G_thr_${threshold}/${c}/"
                copied_files=1
                echo "  ✓ Copied files for $c"
            fi
            
            if [ $copied_files -eq 0 ]; then
                echo "  ⚠ No files found for category $c and threshold $threshold"
            fi
        else
            echo "  ✗ Source directory doesn't exist: ${extracted_dir}/${c}"
        fi
    done
else
    cp -r "$extracted_dir" "results_1G_thr_${threshold}"
fi

# Final check
csv_count=$(find "results_1G_thr_${threshold}" -name "*.csv" -type f 2>/dev/null | wc -l)
echo "Final result: $csv_count CSV files in results_1G_thr_${threshold}/"