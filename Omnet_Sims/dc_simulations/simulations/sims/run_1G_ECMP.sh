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
mkdir -p results
mkdir -p results/dctcp_ecmp

# create the directory to save extracted_results
bash dir_creator.sh

# DCTCP_ECMP RUN ONLY
echo "\n\n-------------------------------------------"
echo "Running DCTCP_ECMP"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_ECMP -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME
do_extract dctcp_ecmp
mkdir -p logs/dctcp_ecmp_1G
cp results/dctcp_ecmp/*.out logs/dctcp_ecmp_1G/

# move the extracted results
echo "Moving the extracted results to results_1G_dctcp_ecmp"
rm -rf results_1G_dctcp_ecmp
mkdir -p results_1G_dctcp_ecmp
cp -r extracted_results/* results_1G_dctcp_ecmp/
rm -rf extracted_results

# python3 sample_qct.py
