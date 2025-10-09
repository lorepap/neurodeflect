#!/bin/bash

# This script requires one positional argument: the simulation time limit
# (for example: 0.001s). Theta/beta sweeps are driven from the INI (see
# omnetpp_1G_collection.ini where relayUnit.deflect_prob_theta and
# relayUnit.deflect_prob_beta are defined as sweep variables). The script
# runs opp_runall with the PROBABILISTIC config so opp_runall expands theta/beta.


# helper to extract results and move them into results_1G_probabilistic
do_extract () {
    if [ -z "$1" ]; then
        echo "do_extract requires one argument: the name of the subdir in results/ to extract from"
        exit 3
    fi
    echo "Extracting results from results/$1/"
    python3 ./extractor_shell_creator.py "$1"
    pushd ./results/
    bash ../extractor.sh
    popd
    sleep 2
}

prepare_logs_dir () {
    local subdir="$1"
    mkdir -p logs
    if [ -n "$subdir" ]; then
        rm -rf "logs/${subdir}"
        mkdir -p "logs/${subdir}"
    fi
}

# validate args
if [ -z "$1" ]; then
    # update usage
    echo "Usage: $0 <simulation_time> [config]"
    echo "Example: $0 10s (for 10 seconds)"
    echo "Example: $0 1000ms (for 1000 milliseconds)"
    echo "Example: $0 0.1s (for 0.1 seconds)"
    echo "Optional second argument: config to run (e.g., ECMP, PROBABILISTIC, RANDOM, THRESHOLD). If not given, all three will be run."
    exit 1
fi

SIMULATION_TIME="$1"
shift

DEBUG_OPTION=""
TYPE_RAW=""

while [ $# -gt 0 ]; do
    case "$1" in
        debug|--debug)
            DEBUG_OPTION="--cmdenv-express-mode=true"
            ;;
        *)
            if [ -n "$TYPE_RAW" ]; then
                echo "Error: Unexpected argument '$1'" >&2
                exit 2
            fi
            TYPE_RAW="$1"
            ;;
    esac
    shift
done


# create the directory to save extracted_results
# call the local dir_creator.sh (script may be invoked from another cwd)
bash ./dir_creator.sh --keep-logs

# ensure results and per-config results directories exist so omnet can open output files
mkdir -p results

# if $2 is given, let's check if it's a valid config in the omnetpp_1G_collection.ini
if [ -n "$TYPE_RAW" ]; then
    TYPE_UPPER=$(echo "$TYPE_RAW" | tr '[:lower:]' '[:upper:]')
    if ! grep -q "^\[Config ${TYPE_UPPER}\]" omnetpp_1G_collection.ini; then
        echo "Error: Config [${TYPE_UPPER}] not found in omnetpp_1G_collection.ini" >&2
        exit 4
    fi
    echo "Config [${TYPE_UPPER}] found in omnetpp_1G_collection.ini; proceeding"
    TYPE_LOWER=$(echo "$TYPE_RAW" | tr '[:upper:]' '[:lower:]')
else
    TYPE_UPPER=""
    TYPE_LOWER=""
    # otherwise create the a directory for each of the configs we know about (ecmp, random, probabilistic, threshold.)
    for cfg in ecmp random probabilistic threshold; do
        mkdir -p "results/$cfg"
    done
fi

# create the per-config results directory
if [ -n "$TYPE_LOWER" ]; then
    mkdir -p "results/${TYPE_LOWER}"
fi

# if type is given run only the type one and exit the program
if [ -n "$TYPE_UPPER" ]; then
    echo "Running only config [${TYPE_UPPER}]"
    case "$TYPE_UPPER" in
        PROBABILISTIC)
            echo "Running PROBABILISTIC"
            prepare_logs_dir "probabilistic_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c PROBABILISTIC -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > opp_runall.log 2>&1
            echo "opp_runall finished successfully; log saved to opp_runall.log"
            # extract and move
            do_extract probabilistic
            cp results/probabilistic/*.out logs/probabilistic_1G/ || true
            # move the extracted results
            echo "Moving the extracted results to results_1G_probabilistic"
            rm -rf results_1G_probabilistic
            mkdir -p results_1G_probabilistic
            cp -r extracted_results/* results_1G_probabilistic/
            rm -rf extracted_results
            ;;
        RANDOM)
            echo "Running RANDOM"
            prepare_logs_dir "random_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c RANDOM -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > opp_runall.log 2>&1
            echo "opp_runall finished successfully; log saved to opp_runall.log"
            # extract and move
            do_extract random
            cp results/random/*.out logs/random_1G/ || true
            # move the extracted results
            echo "Moving the extracted results to results_1G_random"
            rm -rf results_1G_random
            mkdir -p results_1G_random
            cp -r extracted_results/* results_1G_random/
            rm -rf extracted_results
            ;;
        THRESHOLD)
            echo "Running THRESHOLD"
            prepare_logs_dir "threshold_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c THRESHOLD -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > opp_runall.log 2>&1
            echo "opp_runall finished successfully; log saved to opp_runall.log"
            # extract and move  
            do_extract threshold
            cp results/threshold/*.out logs/threshold_1G/ || true
            # move the extracted results    
            echo "Moving the extracted results to results_1G_threshold"
            rm -rf results_1G_threshold
            mkdir -p results_1G_threshold
            cp -r extracted_results/* results_1G_threshold/
            rm -rf extracted_results
            ;;
        ECMP)
            echo "Running ECMP"
            prepare_logs_dir "ecmp_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c ECMP -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > opp_runall.log 2>&1
            echo "opp_runall finished successfully; log saved to opp_runall.log"
            # extract and move
            do_extract ecmp
            cp results/ecmp/*.out logs/ecmp_1G/ || true
            # move the extracted results    
            echo "Moving the extracted results to results_1G_ecmp"
            rm -rf results_1G_ecmp
            mkdir -p results_1G_ecmp
            cp -r extracted_results/* results_1G_ecmp/
            rm -rf extracted_results
            ;;
        *)
            echo "Error: Unsupported config [${TYPE_UPPER}]" >&2
            exit 5
            ;;
    esac
    echo "Done."
    exit 0
fi

echo "\n\n-------------------------------------------"
echo "Running 1G experiments with simulation time limit: $SIMULATION_TIME"

# ecmp run
echo "\n\n-------------------------------------------"
echo "Running ECMP"
prepare_logs_dir "ecmp_1G"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c ECMP -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > logs/ecmp_1G/opp_runall.log 2>&1
echo "opp_runall finished successfully; log saved to opp_runall.log"
# extract and move
do_extract ecmp
cp results/ecmp/*.out logs/ecmp_1G/ || true
# move the extracted results
echo "Moving the extracted results to results_1G_ecmp"
rm -rf results_1G_ecmp
mkdir -p results_1G_ecmp
cp -r extracted_results/* results_1G_ecmp/
rm -rf extracted_results

# PROBABILISTIC RUNS
echo "\n\n-------------------------------------------"
echo "Running PROBABILISTIC"
prepare_logs_dir "probabilistic_1G"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c PROBABILISTIC -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > logs/probabilistic_1G/opp_runall.log 2>&1
echo "opp_runall finished successfully; log saved to logs/probabilistic_1G/opp_runall.log"

# extract and move
do_extract probabilistic
cp results/probabilistic/*.out logs/probabilistic_1G/ || true

# move the extracted results
echo "Moving the extracted results to results_1G_probabilistic"
rm -rf results_1G_probabilistic
mkdir -p results_1G_probabilistic
cp -r extracted_results/* results_1G_probabilistic/
rm -rf extracted_results

# UNIFORM RANDOM
echo "\n\n-------------------------------------------"
echo "Running RANDOM"
prepare_logs_dir "random_1G"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c RANDOM -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > logs/random_1G/opp_runall.log 2>&1

echo "opp_runall finished successfully; log saved to $OPP_LOG"
# extract and move
do_extract random
cp results/random/*.out logs/random_1G/ || true
# move the extracted results
echo "Moving the extracted results to results_1G_random"
rm -rf results_1G_random
mkdir -p results_1G_random
cp -r extracted_results/* results_1G_random/
rm -rf extracted_results

# FIXED THRESHOLD
echo "\n\n-------------------------------------------"
echo "Running THRESHOLD"
prepare_logs_dir "threshold_1G"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c THRESHOLD -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > logs/threshold_1G/opp_runall.log 2>&1

echo "opp_runall finished successfully; log saved to $OPP_LOG"
# extract and move
do_extract threshold
cp results/threshold/*.out logs/threshold_1G/ || true
# move the extracted results    
echo "Moving the extracted results to results_1G_threshold"
rm -rf results_1G_threshold
mkdir -p results_1G_threshold
cp -r extracted_results/* results_1G_threshold/
rm -rf extracted_results

echo "Done."
