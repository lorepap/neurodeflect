#!/bin/bash

# This script mirrors run_25_probabilistic_40G.sh but targets the 75% load
# collection configuration (omnetpp_75_bg_dqps_collection.ini).

set -euo pipefail

# helper to extract results and move them into results_1G_probabilistic
# (same helper used by the 25% script)
do_extract () {
    python3 ./extractor_shell_creator.py probabilistic
    pushd ./results/ >/dev/null
    bash ../extractor.sh
    popd >/dev/null
    sleep 2
}

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <SIMULATION_TIME>    e.g. $0 0.001s"
    exit 1
fi
SIMULATION_TIME="$1"

rm -rf results

# ensure base directories exist
bash ./dir_creator.sh
mkdir -p results/probabilistic
mkdir -p logs/probabilistic_1G

OPP_LOG=logs/probabilistic_1G/opp_runall.log

echo "\n\n-------------------------------------------"
echo "Running PROBABILISTIC (75% load collection config)"
echo "Running opp_runall; logging to $OPP_LOG"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INET_SRC_ABS="$(readlink -f "$SCRIPT_DIR/../../../inet/src")"
DC_SIM_EXE="$(readlink -f "$SCRIPT_DIR/../../src/dc_simulations")"
export LD_LIBRARY_PATH="${INET_SRC_ABS}":${LD_LIBRARY_PATH:-}
NED_PATHS="${SCRIPT_DIR}/..:${SCRIPT_DIR}/../../src:${INET_SRC_ABS}/inet"

touch "$OPP_LOG"
opp_runall -j50 "$DC_SIM_EXE" -m -u Cmdenv -c PROBABILISTIC \
    -n "$NED_PATHS" --image-path="${SCRIPT_DIR}/../../../inet/images" \
    -l "${SCRIPT_DIR}/../../../inet/src/INET" \
    "${SCRIPT_DIR}/omnetpp_75_bg_dqps_collection.ini" \
    --sim-time-limit="$SIMULATION_TIME" >"$OPP_LOG" 2>&1 || {
        echo "opp_runall failed. See $OPP_LOG" >&2
        tail -n 200 "$OPP_LOG" >&2 || true
        exit 2
    }

echo "opp_runall finished successfully; log saved to $OPP_LOG"

do_extract probabilistic
cp results/probabilistic/*.out logs/probabilistic_1G/ || true

echo "Moving the extracted results to results_1G_probabilistic"
rm -rf results_1G_probabilistic
mkdir -p results_1G_probabilistic
cp -r extracted_results/* results_1G_probabilistic/
rm -rf extracted_results

echo "Done."
