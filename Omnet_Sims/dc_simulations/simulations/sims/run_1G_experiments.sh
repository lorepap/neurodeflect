#!/bin/bash

# This script requires one positional argument: the simulation time limit
# (for example: 0.001s). Theta/beta sweeps are driven from the INI (see
# omnetpp_1G_collection.ini where relayUnit.deflect_prob_theta and
# relayUnit.deflect_prob_beta are defined as sweep variables). The script
# runs opp_runall with the PROBABILISTIC config so opp_runall expands theta/beta.

# TODO: add input flags to avoid input errors

# helper to extract results and move them into results_1G_probabilistic
do_extract () {
    if [ -z "$1" ]; then
        echo "do_extract requires one argument: the name of the subdir in results/ to extract from"
        exit 3
    fi
    echo "Extracting results from results/$1/"
    bash ./dir_creator.sh --keep-logs
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
    echo "Optional second argument: config to run (e.g., ECMP, PROBABILISTIC, PROBABILISTIC_TB, RANDOM, RANDOM_TB, THRESHOLD, THRESHOLD_TB). If not given, all configs will be run."
    exit 1
fi

SIMULATION_TIME="$1"
shift

DEBUG_OPTION=""
TYPE_RAW=""

while [ $# -gt 0 ]; do
    case "$1" in
        fast|--fast)
            echo "Debug mode disabled: enabling express mode for faster simulations"
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

# if $2 is given, let's check if it's a valid config in the omnetpp_1G_collection.ini or omnetpp_1G.ini
if [ -n "$TYPE_RAW" ]; then
    TYPE_UPPER=$(echo "$TYPE_RAW" | tr '[:lower:]' '[:upper:]')
    # case probabilistic_tb, random_tb, threshold_tb. Check only if the _TB_S, _TB_M, _TB_L config exists without modifying type_upper
    if [[ "$TYPE_UPPER" == *_TB ]]; then
        if ! grep -q "^\[Config ${TYPE_UPPER}_S\]" omnetpp_1G_collection.ini && ! grep -q "^\[Config ${TYPE_UPPER}_S\]" omnetpp_1G.ini; then
            echo "Error: Config [${TYPE_UPPER}_S] not found in omnetpp_1G_collection.ini or omnetpp_1G.ini" >&2
            exit 4
        fi
        echo "Config [${TYPE_UPPER}_S] found in omnetpp_1G_collection.ini or omnetpp_1G.ini; proceeding"
    # for other types check directly

    else
        if ! grep -q "^\[Config ${TYPE_UPPER}\]" omnetpp_1G_collection.ini && ! grep -q "^\[Config ${TYPE_UPPER}\]" omnetpp_1G.ini; then
        echo "Error: Config [${TYPE_UPPER}] not found in omnetpp_1G_collection.ini or omnetpp_1G.ini" >&2
        exit 4
        fi
    echo "Config [${TYPE_UPPER}] found in omnetpp_1G_collection.ini or omnetpp_1G.ini; proceeding"
    TYPE_LOWER=$(echo "$TYPE_RAW" | tr '[:upper:]' '[:lower:]')
    fi
else
    TYPE_UPPER=""
    TYPE_LOWER=""
    # otherwise create the directories for each supported config
    for cfg in ecmp random probabilistic threshold dctcp_dibs dctcp_sd dctcp_vertigo probabilistic_tb random_tb threshold_tb; do
        mkdir -p "results/$cfg"
    done
fi




# if type is given run only the type one and exit the program
if [ -n "$TYPE_UPPER" ]; then
    echo "Running only config [${TYPE_UPPER}]"
    case "$TYPE_UPPER" in
        PROBABILISTIC)
            echo "Running PROBABILISTIC"
            prepare_logs_dir "probabilistic_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c PROBABILISTIC -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/probabilistic_1G/opp_runall.log 2>&1
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
        PROBABILISTIC_TB)
            # run TB_S, TB_M and TB_L sequentially
            for tb_cfg in PROBABILISTIC_TB_S PROBABILISTIC_TB_M PROBABILISTIC_TB_L; do
                echo "Running PROBABILISTIC_TB config: $tb_cfg"
                # lower tb_cfg
                # tb_cfg_lower=$(echo "$tb_cfg" | tr '[:upper:]' '[:lower:]') # probabilistic_tb_s
                prepare_logs_dir "${tb_cfg_lower}_1G"
                opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c ${tb_cfg} -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/probabilistic_tb_1G/opp_runall.log 2>&1
                echo "opp_runall finished successfully for ${tb_cfg_lower} config: $tb_cfg; log saved to opp_runall.log"
            done
            # extract and move
            do_extract probabilistic_tb
            cp results/probabilistic_tb/*.out logs/probabilistic_tb_1G/ || true
            # move the extracted results
            echo "Moving the extracted results to results_1G_probabilistic_tb_${tb_cfg}"
            rm -rf results_1G_probabilistic_tb
            mkdir -p results_1G_probabilistic_tb
            cp -r extracted_results/* results_1G_probabilistic_tb/
            rm -rf extracted_results
            rm -rf results/probabilistic_tb
            ;;
        RANDOM)
            echo "Running RANDOM"
            prepare_logs_dir "random_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c RANDOM -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/random_1G/opp_runall.log 2>&1
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
        RANDOM_TB)
            # run TB_S, TB_M and TB_L sequentially
            for tb_cfg in RANDOM_TB_S RANDOM_TB_M RANDOM_TB_L; do
                echo "Running RANDOM_TB config: $tb_cfg"
                prepare_logs_dir "random_tb_1G"
                opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c ${tb_cfg} -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/random_tb_1G/opp_runall.log 2>&1
                echo "opp_runall finished successfully for RANDOM_TB config: $tb_cfg; log saved to opp_runall.log"
            done
            # extract and move
            do_extract random_tb
            cp results/random_tb/*.out logs/random_tb_1G/ || true
            # move the extracted results
            echo "Moving the extracted results to results_1G_random_tb_${tb_cfg}"
            rm -rf results_1G_random_tb
            mkdir -p results_1G_random_tb
            cp -r extracted_results/* results_1G_random_tb/
            rm -rf extracted_results
            ;;
        THRESHOLD)
            echo "Running THRESHOLD"
            prepare_logs_dir "threshold_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c THRESHOLD -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/threshold_1G/opp_runall.log 2>&1
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
        THRESHOLD_TB)
            # run TB_S, TB_M and TB_L sequentially
            for tb_cfg in THRESHOLD_TB_S THRESHOLD_TB_M THRESHOLD_TB_L; do
                echo "Running THRESHOLD_TB config: $tb_cfg"
                tb_cfg_lower=$(echo "$tb_cfg" | tr '[:upper:]' '[:lower:]')
                prepare_logs_dir "${tb_cfg_lower}_1G"
                opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c ${tb_cfg} -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/threshold_tb_1G/opp_runall.log 2>&1
                echo "opp_runall finished successfully for THRESHOLD_TB config: $tb_cfg; log saved to opp_runall.log"
            done
             # extract and move
            do_extract threshold_tb
            cp results/threshold_tb/*.out logs/threshold_tb_1G/ || true
            # move the extracted results
            echo "Moving the extracted results to results_1G_threshold_tb"
            rm -rf results_1G_threshold_tb
            mkdir -p results_1G_threshold_tb
            cp -r extracted_results/* results_1G_threshold_tb/
            rm -rf extracted_results
            ;;
        ECMP)
            echo "Running ECMP"
            prepare_logs_dir "ecmp_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c ECMP -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/ecmp_1G/opp_runall.log 2>&1
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
        DCTCP_DIBS)
            echo "Running DCTCP_DIBS"
            prepare_logs_dir "dctcp_dibs_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_DIBS -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME >logs/dctcp_dibs_1G/opp_runall.log 2>&1
            do_extract dctcp_dibs
            cp results/*.out logs/dctcp_dibs_1G/
            echo "Moving the extracted results to results_1G_dibs"
            rm -rf results_1G_dibs
            mkdir -p results_1G_dibs
            cp -r extracted_results/* results_1G_dibs/
            rm -rf extracted_results
            ;;
        DCTCP_SD)
            echo "Running DCTCP_SD"
            prepare_logs_dir "dctcp_sd_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME >logs/dctcp_sd_1G/opp_runall.log 2>&1
            do_extract dctcp_sd
            cp results/*.out logs/dctcp_sd_1G/
            echo "Moving the extracted results to results_1G_sd"
            rm -rf results_1G_sd
            mkdir -p results_1G_sd
            cp -r extracted_results/* results_1G_sd/
            rm -rf extracted_results
            ;;
        DCTCP_VERTIGO)
            echo "Running DCTCP_VERTIGO"
            prepare_logs_dir "dctcp_vertigo_1G"
            opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_VERTIGO -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME  >logs/dctcp_vertigo_1G/opp_runall.log 2>&1
            do_extract dctcp_vertigo
            cp results/*.out logs/dctcp_vertigo_1G/
            echo "Moving the extracted results to results_1G_vertigo"
            rm -rf results_1G_vertigo
            mkdir -p results_1G_vertigo
            cp -r extracted_results/* results_1G_vertigo/
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

# PROBABILISTIC WITH TB
for tb_cfg in PROBABILISTIC_TB_S PROBABILISTIC_TB_M PROBABILISTIC_TB_L; do
    echo "Running PROBABILISTIC_TB config: $tb_cfg"
    prepare_logs_dir "probabilistic_tb_${tb_cfg}_1G"
    opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c PROBABILISTIC_${tb_cfg} -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/probabilistic_tb_${tb_cfg}_1G/opp_runall.log 2>&1
    echo "opp_runall finished successfully for PROBABILISTIC_TB config: $tb_cfg; log saved to opp_runall.log"
    # extract and move  
    do_extract probabilistic_tb
    cp results/probabilistic_tb/*.out logs/probabilistic_tb_${tb_cfg}_1G/ || true
    # move the extracted results    
    echo "Moving the extracted results to results_1G_probabilistic_tb_${tb_cfg}"
    rm -rf results_1G_probabilistic_tb_${tb_cfg}
    mkdir -p results_1G_probabilistic_tb_${tb_cfg}
    cp -r extracted_results/* results_1G_probabilistic_tb_${tb_cfg}/
    rm -rf extracted_results
done

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

# RANDOM WITH TB
for tb_cfg in RANDOM_TB_S RANDOM_TB_M RANDOM_TB_L; do
    echo "Running RANDOM_TB config: $tb_cfg"
    prepare_logs_dir "random_tb_${tb_cfg}_1G"
    opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c RANDOM_${tb_cfg} -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/random_tb_${tb_cfg}_1G/opp_runall.log 2>&1
    echo "opp_runall finished successfully for RANDOM_TB config: $tb_cfg; log saved to opp_runall.log"
    # extract and move  
    do_extract random_tb
    cp results/random_tb/*.out logs/random_tb_${tb_cfg}_1G/ || true
    # move the extracted results    
    echo "Moving the extracted results to results_1G_random_tb_${tb_cfg}"
    rm -rf results_1G_random_tb_${tb_cfg}
    mkdir -p results_1G_random_tb_${tb_cfg}
    cp -r extracted_results/* results_1G_random_tb_${tb_cfg}/
    rm -rf extracted_results
done

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

# THRESHOLD WITH TB (S, M, L)
 for tb_cfg in THRESHOLD_TB_S THRESHOLD_TB_M THRESHOLD_TB_L; do
    echo "Running THRESHOLD_TB config: $tb_cfg"
    prepare_logs_dir "threshold_tb_${tb_cfg}_1G"
    opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c THRESHOLD_${tb_cfg} -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G_collection.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION >logs/threshold_tb_${tb_cfg}_1G/opp_runall.log 2>&1
    echo "opp_runall finished successfully for THRESHOLD_TB config: $tb_cfg; log saved to opp_runall.log"
    # extract and move  
    do_extract threshold_tb
    cp results/threshold_tb/*.out logs/threshold_tb_${tb_cfg}_1G/ || true
    # move the extracted results    
    echo "Moving the extracted results to results_1G_threshold_tb_${tb_cfg}"
    rm -rf results_1G_threshold_tb_${tb_cfg}
    mkdir -p results_1G_threshold_tb_${tb_cfg}
    cp -r extracted_results/* results_1G_threshold_tb_${tb_cfg}/
    rm -rf extracted_results
done

# BASELINES

# DIBS
echo "\n\n-------------------------------------------"
echo "Running DCTCP_DIBS"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_DIBS -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > logs/dctcp_dibs_1G/opp_runall.log 2>&1
echo "opp_runall finished successfully; log saved to logs/dctcp_dibs_1G/opp_runall.log"
# extract and move
do_extract dctcp_dibs
mkdir logs/dctcp_dibs_1G
cp results/*.out logs/dctcp_dibs_1G/

# move the extracted results
echo "Moving the extracted results to results_1G_dibs"
rm -rf results_1G_dibs
mkdir -p results_1G_dibs
cp -r extracted_results/* results_1G_dibs/
rm -rf extracted_results

# SIMPLE DEFLECTION
echo "\n\n-------------------------------------------"
echo "Running DCTCP_SD"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > logs/dctcp_sd_1G/opp_runall.log 2>&1
echo "opp_runall finished successfully; log saved to logs/dctcp_sd_1G/opp_runall.log"   
do_extract dctcp_sd
mkdir logs/dctcp_sd_1G
cp results/*.out logs/dctcp_sd_1G/

# move the extracted results
echo "Moving the extracted results to results_1G_sd"
rm -rf results_1G_sd
mkdir -p results_1G_sd
cp -r extracted_results/* results_1G_sd/
rm -rf extracted_results

# VERTIGO
echo "\n\n-------------------------------------------"
echo "Running DCTCP_VERTIGO"
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_VERTIGO -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME $DEBUG_OPTION > logs/dctcp_vertigo_1G/opp_runall.log 2>&1
echo "opp_runall finished successfully; log saved to logs/dctcp_vertigo_1G/opp_runall.log"
# extract and move
do_extract dctcp_vertigo
mkdir logs/dctcp_vertigo_1G
cp results/*.out logs/dctcp_vertigo_1G/

# move the extracted results
echo "Moving the extracted results to results_1G_vertigo"
rm -rf results_1G_vertigo
mkdir -p results_1G_vertigo
cp -r extracted_results/* results_1G_vertigo/
rm -rf extracted_results

echo "Done."
