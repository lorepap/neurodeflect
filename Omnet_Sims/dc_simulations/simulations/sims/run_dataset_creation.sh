#!/bin/bash

#!/bin/bash

# Usage: run_dataset_creation.sh <extracted_results_folder> [output_name]
# Example: bash run_dataset_creation.sh results_1G_uniform_random

if [ $# -lt 1 ]; then
    echo "Usage: $0 <extracted_results_folder> [output_name]"
    exit 1
fi

SOURCE_DIR=$1
OUTPUT_NAME=${2:-$(basename "$SOURCE_DIR")}

# Directory where this script lives (used to find helper scripts)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "================================================="
echo "Creating dataset from source: $SOURCE_DIR"
echo "Output base name: $OUTPUT_NAME"
echo "================================================="

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: source directory '$SOURCE_DIR' does not exist"
    exit 2
fi

# Prepare staging directory results_1G
rm -f results_1G
rm -rf results_1G_staging
mkdir -p results_1G_staging

# Known categories expected by the pipeline
cats=(QUEUE_LEN QUEUES_TOT_LEN QUEUE_CAPACITY QUEUES_TOT_CAPACITY PACKET_ACTION RCV_TS_SEQ_NUM SND_TS_SEQ_NUM OOO_SEG SWITCH_SEQ_NUM TTL ACTION_SEQ_NUM SWITCH_ID SWITCH_ID_ACTION INTERFACE_ID RETRANSMITTED FLOW_ID PACKET_SIZE REQUESTER_ID FLOW_STARTED FLOW_ENDED)

if [ -d "$SOURCE_DIR/QUEUE_LEN" ] || [ -d "$SOURCE_DIR/PACKET_ACTION" ]; then
    # If source already has category subdirectories, mirror them into staging
    for c in "${cats[@]}"; do
        if [ -d "$SOURCE_DIR/$c" ]; then
            mkdir -p "results_1G_staging/$c"
            cp -r "$SOURCE_DIR/$c/"* "results_1G_staging/$c/" 2>/dev/null || true
        fi
    done
else
    # Try to distribute CSVs into categories by filename, otherwise keep them in top-level of staging
    for c in "${cats[@]}"; do
        mkdir -p "results_1G_staging/$c"
        if compgen -G "$SOURCE_DIR/*${c}*.csv" > /dev/null 2>&1; then
            cp $SOURCE_DIR/*${c}*.csv "results_1G_staging/$c/" 2>/dev/null || true
        fi
    done
    # Also copy any CSVs into a generic place if they weren't categorized
    if compgen -G "$SOURCE_DIR/*.csv" > /dev/null 2>&1; then
        mkdir -p "results_1G_staging/UNCATEGORIZED"
        cp $SOURCE_DIR/*.csv "results_1G_staging/UNCATEGORIZED/" 2>/dev/null || true
    fi
fi

# Sanity check for CSVs
csv_count=$(find results_1G_staging -name "*.csv" -type f 2>/dev/null | wc -l)
if [ "$csv_count" -eq 0 ]; then
    echo "✗ No CSV files found in staged data (from $SOURCE_DIR)"
    exit 3
fi
echo "✓ Staged $csv_count CSV files into results_1G_staging/"

# Create symlink expected by downstream scripts
ln -sfn results_1G_staging results_1G

# Calculate deflection statistics per original PACKET_ACTION CSV (each CSV is one run)
echo "Calculating per-CSV deflection statistics (one line per PACKET_ACTION CSV)..."

# Prefer original files under SOURCE_DIR/PACKET_ACTION if present
pa_dir="$SOURCE_DIR/PACKET_ACTION"
if [ -d "$pa_dir" ] && compgen -G "$pa_dir/*.csv" > /dev/null 2>&1; then
    for f in "$pa_dir"/*.csv; do
        [ -f "$f" ] || continue
        # Skip empty files
        if [ ! -s "$f" ]; then
            continue
        fi

        # Detect whether the first row contains a numeric action (no header) or a header
        first_second=$(awk -F, 'NR==1{print $2}' "$f" 2>/dev/null || echo "")
        if [[ "$first_second" =~ ^[01]$ ]]; then
            # Two-column no-header format: timestamp,action
            total=$(wc -l < "$f" 2>/dev/null || echo 0)
            deflections=$(awk -F, '{ if(($2+0)==1) c++ } END{print c+0}' "$f" 2>/dev/null || echo 0)
        else
            # Has header: skip header row when counting
            total=$(awk 'END{print NR-1}' "$f" 2>/dev/null || echo 0)
            deflections=$(awk -F, 'NR>1{ if(($2+0)==1) c++ } END{print c+0}' "$f" 2>/dev/null || echo 0)
        fi

        pct=0.0
        if [ "$total" -gt 0 ]; then
            pct=$(awk -v d="$deflections" -v t="$total" 'BEGIN{printf("%.1f", (t>0? d/t*100: 0))}')
        fi

        echo "Deflection statistics for $(basename "$f"): $deflections deflections out of $total packets (${pct}% deflection rate)"
    done
else
    # Fallback: inspect staged CSVs (older behavior)
    find results_1G_staging -type f -name "*.csv" | sort | while read -r f; do
        # Skip empty files
        if [ ! -s "$f" ]; then
            continue
        fi

        # Try to detect 'action' column index (case-insensitive)
        idx=$(awk -F, 'NR==1{for(i=1;i<=NF;i++){if(tolower($i)=="action"){print i; exit}}}' "$f" 2>/dev/null || true)

        if [ -z "$idx" ]; then
            # No header with action column found -> skip detailed deflection count
            total=$(awk 'END{print NR}' "$f" 2>/dev/null || echo 0)
            echo "Deflection statistics for $(basename "$f"): header has no 'action' column; total rows=$total"
            continue
        fi

        # Count deflections where action==1 (skip header)
        deflections=$(awk -F, -v col="$idx" 'NR>1{ if(($col+0)==1) c++ } END{print c+0}' "$f" 2>/dev/null || echo 0)
        total=$(awk 'END{print NR-1}' "$f" 2>/dev/null || echo 0)
        if [ "$total" -le 0 ]; then
            total=$(awk 'END{print NR}' "$f" 2>/dev/null || echo 0)
        fi

        pct=0.0
        if [ "$total" -gt 0 ]; then
            pct=$(awk -v d="$deflections" -v t="$total" 'BEGIN{printf("%.1f", (t>0? d/t*100: 0))}')
        fi

        echo "Deflection statistics for $(basename "$f"): $deflections deflections out of $total packets (${pct}% deflection rate)"
    done
fi



# === NEW LOGIC: If multiple experiment CSVs present, run full pipeline per experiment ===
echo "\n=== Detecting experiments in staged data... ==="
exp_set=()
# For each category, examine filenames and extract experiment basenames (filename without the trailing _<CATEGORY>.csv)
for c in "${cats[@]}"; do
    dir="results_1G_staging/$c"
    if [ -d "$dir" ]; then
        for f in "$dir"/*.csv; do
            [ -f "$f" ] || continue
            base=$(basename "$f" .csv)
            # if base ends with _<category>, strip it to get experiment id
            suffix="_${c}"
            if [[ "$base" == *"$suffix" ]]; then
                exp=${base%$suffix}
            else
                # fallback: use full basename (will match similarly-named files across categories)
                exp=$base
            fi
            exp_set+=("$exp")
        done
    fi
done

# deduplicate experiments
IFS=$'\n' read -r -d '' -a experiments < <(printf "%s\n" "${exp_set[@]}" | awk '!seen[$0]++' && printf '\0')

if [ ${#experiments[@]} -eq 0 ]; then
    echo "✗ No experiments detected in staged data (unexpected)"
    rm -f results_1G
    exit 8
fi

echo "✓ Detected ${#experiments[@]} experiments: ${experiments[*]}"

final_datasets=()

for exp in "${experiments[@]}"; do
    echo "\n--- Running pipeline for experiment: $exp ---"
    tmp_stage="results_1G_exp_$exp"
    rm -rf "$tmp_stage"
    mkdir -p "$tmp_stage"
    # create category dirs and copy only files matching this experiment
    for c in "${cats[@]}"; do
        mkdir -p "$tmp_stage/$c"
        # try patterns: <exp>_<category>.csv or *<exp>* in filename
        if compgen -G "results_1G_staging/$c/${exp}_$c.csv" > /dev/null 2>&1; then
            cp results_1G_staging/$c/${exp}_$c.csv "$tmp_stage/$c/" 2>/dev/null || true
        elif compgen -G "results_1G_staging/$c/*${exp}*.csv" > /dev/null 2>&1; then
            cp results_1G_staging/$c/*${exp}*.csv "$tmp_stage/$c/" 2>/dev/null || true
        fi
    done

    # Verify we copied something
    cnt=$(find "$tmp_stage" -name "*.csv" | wc -l)
    if [ "$cnt" -eq 0 ]; then
        echo "⚠ No CSVs found for experiment $exp, skipping"
        rm -rf "$tmp_stage"
        continue
    fi

    # Point results_1G to the tmp stage and run create_dataset.py (default invocation)
    ln -sfn "$tmp_stage" results_1G

    echo "Running create_dataset.py for $exp (logs -> /tmp/create_dataset_${exp}.log)"
    if python3 "$SCRIPT_DIR/create_dataset.py" 2>&1 | tee "/tmp/create_dataset_${exp}.log"; then
        echo "✓ create_dataset.py completed for $exp"
    else
        echo "✗ create_dataset.py failed for $exp (see /tmp/create_dataset_${exp}.log)"
        rm -f results_1G
        rm -rf "$tmp_stage"
        continue
    fi

    # Post-processing and enrichment (operate on results_1G as usual)
    if [ -f "$SCRIPT_DIR/filter_collisions.py" ] && [ -f "results_1G/merged_final.csv" ]; then
        python3 "$SCRIPT_DIR/filter_collisions.py" --input-file results_1G/merged_final.csv --output-file results_1G/collision_filtered.csv 2>&1 | tee "/tmp/filter_collisions_${exp}.log" || true
    else
        cp results_1G/merged_final.csv results_1G/collision_filtered.csv 2>/dev/null || true
    fi

    if [ -f "$SCRIPT_DIR/filter_overlapping_timestamps.py" ] && [ -f "results_1G/collision_filtered.csv" ]; then
        python3 "$SCRIPT_DIR/filter_overlapping_timestamps.py" --input-file results_1G/collision_filtered.csv --output-file results_1G/ooo_enhanced.csv 2>&1 | tee "/tmp/filter_overlap_${exp}.log" || true
        if [ -f "results_1G/ooo_enhanced.csv" ]; then
            cp results_1G/ooo_enhanced.csv results_1G/final_dataset.csv
        else
            cp results_1G/collision_filtered.csv results_1G/final_dataset.csv
        fi
    else
        cp results_1G/collision_filtered.csv results_1G/final_dataset.csv 2>/dev/null || true
    fi

    if [ -f "$SCRIPT_DIR/enrich_packets_with_flows.py" ] && [ -f "results_1G/final_dataset.csv" ]; then
        python3 "$SCRIPT_DIR/enrich_packets_with_flows.py" "results_1G/final_dataset.csv" results_1G results_1G/packet_level_with_flows.csv --log-file "/tmp/enrich_packets_${exp}.log" 2>&1 || true
        if [ -s results_1G/packet_level_with_flows.csv ]; then
            mv results_1G/packet_level_with_flows.csv results_1G/final_dataset.csv
        fi
    fi

    # Save per-experiment final dataset
    if [ -f "results_1G/final_dataset.csv" ]; then
        out_name="${OUTPUT_NAME}_${exp}_dataset.csv"
        cp results_1G/final_dataset.csv "$out_name"
        echo "✓ Saved per-experiment final dataset: $out_name"
        final_datasets+=("$out_name")
    else
        echo "✗ No final_dataset.csv produced for experiment $exp"
    fi

    # cleanup tmp stage for this experiment
    rm -f results_1G
    rm -rf "$tmp_stage"
done

# Merge per-experiment datasets into single CSV
echo "\n=== Merging per-experiment final datasets ==="
merged_out="${OUTPUT_NAME}_all_final_datasets.csv"
if [ ${#final_datasets[@]} -gt 0 ]; then
    head -n 1 "${final_datasets[0]}" > "$merged_out"
    for f in "${final_datasets[@]}"; do
        tail -n +2 "$f" >> "$merged_out"
    done
    echo "✓ Merged ${#final_datasets[@]} datasets into $merged_out"
    mkdir -p results_data_collection
    cp -f "$merged_out" results_data_collection/ 2>/dev/null || true
    echo "✓ Archived merged dataset to results_data_collection/$merged_out"
else
    echo "⚠ No per-experiment final datasets were produced"
fi

# Archive staged results
mkdir -p processed_${OUTPUT_NAME}
cp -r results_1G_staging/* processed_${OUTPUT_NAME}/ 2>/dev/null || true
echo "✓ Archived staged results to processed_${OUTPUT_NAME}/"

echo "All done. Per-experiment datasets: ${final_datasets[*]}"
exit 0

# Example: to create datasets for probabilistic runs produced by run_1G_probabilistic.sh
# bash run_dataset_creation.sh results_1G_probabilistic probabilistic
