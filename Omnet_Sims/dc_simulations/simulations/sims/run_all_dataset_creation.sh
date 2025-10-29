#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

POLICIES=(
    dibs
    ecmp
    probabilistic
    probabilistic_tb
    random
    random_tb
    sd
    threshold
    threshold_tb
    uniform_random
    vertigo
)

RUN_SCRIPT="$SCRIPT_DIR/run_dataset_creation.sh"
if [[ ! -x "$RUN_SCRIPT" ]]; then
    echo "Error: dataset creation runner not found or not executable at $RUN_SCRIPT" >&2
    exit 1
fi

extra_args=("$@")

for policy in "${POLICIES[@]}"; do
    results_dir="$SCRIPT_DIR/results_1G_${policy}"
    output_dir="$SCRIPT_DIR/tmp/data/data_1G_${policy}"

    if [[ ! -d "$results_dir" ]]; then
        echo "[run_all_dataset_creation] Skipping ${policy}: missing results dir $results_dir" >&2
        continue
    fi

    mkdir -p "$output_dir"
    echo "[run_all_dataset_creation] Processing ${policy}"
    "$RUN_SCRIPT" --results-dir "$results_dir" --output-dir "$output_dir" "${extra_args[@]}"
done
