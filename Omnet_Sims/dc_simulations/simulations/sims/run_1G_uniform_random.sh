#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <simulation_time>"
    echo "Example: $0 10s (for 10 seconds)"
    echo "Example: $0 1000ms (for 1000 milliseconds)"
    echo "Example: $0 0.1s (for 0.1 seconds)"
    exit 1
fi

SIMULATION_TIME=$1

do_extract () {
    python3 ./extractor_shell_creator.py $1
    pushd ./results/
    bash ../extractor.sh
    popd
    sleep 5
}

rm -rf results

# create the directory to save extracted_results
bash dir_creator.sh

# ensure results and per-config results directories exist so omnet can open output files
mkdir -p results
mkdir -p results/uniform_random

# UNIFORM_RANDOM RUN ONLY
echo "\n\n-------------------------------------------"
echo "Running UNIFORM_RANDOM"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c UNIFORM_RANDOM -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME

do_extract uniform_random
mkdir -p logs/uniform_random_1G
cp results/uniform_random/*.out logs/uniform_random_1G/ || true

# move the extracted results
echo "Moving the extracted results to results_1G_uniform_random"
rm -rf results_1G_uniform_random
mkdir -p results_1G_uniform_random
cp -r extracted_results/* results_1G_uniform_random/
rm -rf extracted_results

# python3 sample_qct.py
