#!/bin/bash

# Orchestration script to run all data-collection policies and create datasets.
# Usage: ./run_all_data_collection.sh <simulation_time>
# Example: ./run_all_data_collection.sh 10s

if [ $# -eq 0 ]; then
    echo "Usage: $0 <simulation_time>"
    exit 1
fi
SIM_TIME=$1

set -euo pipefail

SCRIPTDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPTDIR"

policies=("run_1G_ECMP.sh" "run_1G_uniform_random.sh" "run_deflection_threshold_experiments.sh")
results_dirs=("results_1G_dctcp_ecmp" "results_1G_uniform_random" "results_1G_deflection_thresholds")

# Run each policy script
for i in "${!policies[@]}"; do
    pol=${policies[$i]}
    echo "\n============================"
    echo "Running policy: $pol (sim time: $SIM_TIME)"
    echo "============================\n"

    if [ ! -x "$pol" ]; then
        if [ -f "$pol" ]; then
            chmod +x "$pol"
        else
            echo "Policy script $pol not found in $SCRIPTDIR"
            exit 2
        fi
    fi

    # Run policy script; allow it to fail without stopping the whole orchestration
    if ./$pol "$SIM_TIME"; then
        echo "Policy $pol finished successfully"
    else
        echo "Warning: policy $pol failed (continuing with next)"
    fi

    # After running policy, run dataset creation for the expected results dir
    outdir=${results_dirs[$i]}
    echo "Creating dataset for $outdir"
    if ./run_dataset_creation.sh "$outdir"; then
        echo "Dataset created for $outdir"
    else
        echo "Warning: dataset creation for $outdir failed"
    fi
done

# Final message
echo "\nAll policies processed. Datasets are saved in the working directory and archived under results_data_collection/."
exit 0
