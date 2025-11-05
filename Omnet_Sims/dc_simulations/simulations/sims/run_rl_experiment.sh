#!/bin/bash

# Wrapper around the RL policy configuration so that it mirrors the 1G baseline scripts.
# Usage: ./run_rl_experiment.sh <sim_time> [--fast]
# Example: ./run_rl_experiment.sh 0.01s

set -euo pipefail

usage() {
    echo "Usage: $0 <simulation_time> [--fast]" >&2
    echo "Example: $0 0.01s" >&2
    echo "         $0 10s --fast" >&2
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

SIMULATION_TIME="$1"
shift

DEBUG_OPTION=""
while [ $# -gt 0 ]; do
    case "$1" in
        --fast|fast)
            DEBUG_OPTION="--cmdenv-express-mode=true"
            ;;
        *)
            echo "Unexpected argument: $1" >&2
            usage
            exit 2
            ;;
    esac
    shift
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

INI_FILE="$SCRIPT_DIR/omnetpp_1G.ini"
if [ ! -f "$INI_FILE" ]; then
    echo "✗ Cannot find $INI_FILE in $(pwd)" >&2
    exit 1
fi

# Locate rl_model_path from the INI and ensure the TorchScript file exists
RL_MODEL_PATH=$(python3 - <<PY
from pathlib import Path
import re
ini = Path("$INI_FILE")
pattern = re.compile(r"\*\*\.agg\[\*\]\.rl_model_path\s*=\s*\"(.*)\"")
path = ""
for line in ini.read_text().splitlines():
    m = pattern.match(line.strip())
    if m:
        path = m.group(1)
        break
print(Path(path).expanduser())
PY
)

if [[ "${RL_MODEL_PATH}" != /* ]]; then
    RL_MODEL_PATH="$RL_MODEL_PATH"
    RL_MODEL_PATH=$(python3 - <<PY
from pathlib import Path
print(Path("$SCRIPT_DIR").joinpath(Path("${RL_MODEL_PATH}")).resolve())
PY
)
fi

if [ -z "$RL_MODEL_PATH" ]; then
    echo "✗ Unable to locate rl_model_path in $INI_FILE" >&2
    exit 1
fi

if [ ! -f "$RL_MODEL_PATH" ]; then
    echo "✗ RL model checkpoint not found at: $RL_MODEL_PATH" >&2
    echo "  Please export the TorchScript actor and update $INI_FILE" >&2
    exit 1
fi

echo "Using RL TorchScript checkpoint: $RL_MODEL_PATH"

do_extract() {
    if [ -z "$1" ]; then
        echo "do_extract requires one argument: the name of the subdir in results/" >&2
        exit 3
    fi
    echo "Extracting results from results/$1/"
    bash ./dir_creator.sh --keep-logs
    python3 ./extractor_shell_creator.py "$1"
    pushd ./results/ >/dev/null
    bash ../extractor.sh
    popd >/dev/null
    sleep 2
}

prepare_logs_dir() {
    local subdir="$1"
    mkdir -p logs
    if [ -n "$subdir" ]; then
        rm -rf "logs/${subdir}"
        mkdir -p "logs/${subdir}"
    fi
}

echo "Creating base directories"
bash ./dir_creator.sh --keep-logs
mkdir -p results

LOG_SUBDIR="rl_policy"
prepare_logs_dir "$LOG_SUBDIR"

echo "Running OMNeT++ configuration: DCTCP_SD_RL_POLICY"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_RL_POLICY \
    -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases \
    --image-path=../../../inet/images -l ../../../inet/src/INET \
    "$INI_FILE" --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >"logs/${LOG_SUBDIR}/opp_runall.log" 2>&1

echo "Simulation completed; log stored in logs/${LOG_SUBDIR}/opp_runall.log"

do_extract rl_policy

if ls results/rl_policy/*.out >/dev/null 2>&1; then
    cp results/rl_policy/*.out "logs/${LOG_SUBDIR}/" || true
fi

echo "Staging extracted results"
rm -rf results_rl_policy
mkdir -p results_rl_policy
cp -r extracted_results/* results_rl_policy/
rm -rf extracted_results

echo "RL experiment completed successfully."
echo "Results available in results_rl_policy/ and logs/${LOG_SUBDIR}/"
