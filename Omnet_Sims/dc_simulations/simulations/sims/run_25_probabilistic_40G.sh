#!/bin/bash

# This script requires one positional argument: the simulation time limit
# (for example: 0.001s). Theta/beta sweeps are driven from the INI (see
# omnetpp_1G_collection.ini where relayUnit.deflect_prob_theta and
# relayUnit.deflect_prob_beta are defined as sweep variables). The script
# runs opp_runall with the PROBABILISTIC config so opp_runall expands theta/beta.

# helper to extract results and move them into results_1G_probabilistic
do_extract () {
    python3 ./extractor_shell_creator.py probabilistic
    pushd ./results/
    bash ../extractor.sh
    popd
    sleep 2
}

# validate args
if [ -z "$1" ]; then
    echo "Usage: $0 <SIMULATION_TIME>    e.g. $0 0.001s"
    exit 1
fi
SIMULATION_TIME="$1"

rm -rf results

# create the directory to save extracted_results
# call the local dir_creator.sh (script may be invoked from another cwd)
bash ./dir_creator.sh

# ensure results and per-config results directories exist so omnet can open output files
mkdir -p results
mkdir -p results/probabilistic

# Run the simulation using opp_runall and the PROBABILISTIC config (theta/beta
# sweep variables are defined in omnetpp_1G_collection.ini; no overrides here)
echo "\n\n-------------------------------------------"
echo "Running PROBABILISTIC (driven from ini sweep variables)"
mkdir -p logs/probabilistic_1G
OPP_LOG=logs/probabilistic_1G/opp_runall.log
echo "Running opp_runall; logging to $OPP_LOG"
# Resolve absolute paths relative to this script so running from elsewhere works reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INET_SRC_ABS="$(readlink -f "$SCRIPT_DIR/../../../inet/src")"
DC_SIM_EXE="$(readlink -f "$SCRIPT_DIR/../../src/dc_simulations")"
export LD_LIBRARY_PATH="${INET_SRC_ABS}":${LD_LIBRARY_PATH:-}
# Use NED paths that prioritize the project's NED folders and the INET package
NED_PATHS="${SCRIPT_DIR}/..:${SCRIPT_DIR}/../../src:${INET_SRC_ABS}/inet"
opp_runall -j50 "${DC_SIM_EXE}" -m -u Cmdenv -c PROBABILISTIC -n "$NED_PATHS" --image-path="${SCRIPT_DIR}/../../../inet/images" -l "${SCRIPT_DIR}/../../../inet/src/INET" "${SCRIPT_DIR}/omnetpp_25_bg_dqps_collection.ini" --sim-time-limit=$SIMULATION_TIME >"$OPP_LOG" 2>&1 || {
    echo "opp_runall failed. See $OPP_LOG" >&2
    tail -n 200 "$OPP_LOG" >&2 || true
    exit 2
}

echo "opp_runall finished successfully; log saved to $OPP_LOG"

# extract and move
do_extract probabilistic
cp results/probabilistic/*.out logs/probabilistic_1G/ || true

# move the extracted results
echo "Moving the extracted results to results_1G_probabilistic"
rm -rf results_1G_probabilistic
mkdir -p results_1G_probabilistic
cp -r extracted_results/* results_1G_probabilistic/
rm -rf extracted_results

# no cleanup required (no temporary INI created)

echo "Done."
