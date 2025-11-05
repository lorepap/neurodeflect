#!/bin/bash
#
# Orchestrates the RL simulation, dataset extraction, and model training end-to-end.
# Usage: ./run_full_pipeline.sh <simulation_time> <link_bandwidth>

set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: run_full_pipeline.sh <simulation_time> <link_bandwidth>

Example:
  ./run_full_pipeline.sh 1s 1G
EOF
}

if [ $# -ne 2 ]; then
    usage
    exit 1
fi

SIMULATION_TIME="$1"
LINK_BANDWIDTH="$2"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../../" && pwd)

PIPELINE_LOG_ROOT="$SCRIPT_DIR/logs/pipelines"
mkdir -p "$PIPELINE_LOG_ROOT"

RUNS_DIR="$REPO_ROOT/runs"
mkdir -p "$RUNS_DIR"
PLOTS_DIR_ROOT="$SCRIPT_DIR/plots"
mkdir -p "$PLOTS_DIR_ROOT"

timestamp_utc() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

sanitize_id_component() {
    echo "$1" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-z0-9_.-]/-/g'
}

RUN_TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
SIM_COMPONENT=$(sanitize_id_component "$SIMULATION_TIME")
BW_COMPONENT=$(sanitize_id_component "$LINK_BANDWIDTH")
RUN_ID="${RUN_TIMESTAMP}_algo-cql_sim-${SIM_COMPONENT}_bw-${BW_COMPONENT}"
LOG_FILE="$PIPELINE_LOG_ROOT/${RUN_ID}.log"
TRAIN_OUT_DIR="$RUNS_DIR/$RUN_ID"
PLOTS_OUT_DIR="$PLOTS_DIR_ROOT/$RUN_ID"

log() {
    local level="$1"
    shift
    local message="$*"
    local now
    now=$(timestamp_utc)
    printf "%s [%s] %s\n" "$now" "$level" "$message" | tee -a "$LOG_FILE"
}

handle_exit() {
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log INFO "Pipeline completed successfully (run_id=${RUN_ID})."
    else
        log ERROR "Pipeline failed with exit code $exit_code (run_id=${RUN_ID}). See $LOG_FILE for details."
    fi
}
trap handle_exit EXIT

run_stage() {
    local stage_name="$1"
    shift
    log INFO "Starting stage: ${stage_name}"
    local stage_start
    stage_start=$(date +%s)
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        local stage_end
        stage_end=$(date +%s)
        local duration=$((stage_end - stage_start))
        log INFO "Completed stage: ${stage_name} (duration=${duration}s)"
    else
        local err=$?
        log ERROR "Stage failed: ${stage_name} (exit_code=${err})"
        exit $err
    fi
}

log INFO "===== RL Experiment Pipeline ====="
log INFO "run_id=${RUN_ID}"
log INFO "simulation_time=${SIMULATION_TIME}"
log INFO "link_bandwidth=${LINK_BANDWIDTH}"
log INFO "logs=${LOG_FILE}"
log INFO "train_out_dir=${TRAIN_OUT_DIR}"
log INFO "plots_out_dir=${PLOTS_OUT_DIR}"

run_stage "RL simulation" \
    bash "$SCRIPT_DIR/run_rl_experiment.sh" "$SIMULATION_TIME" --fast

run_stage "Dataset extraction" \
    bash "$SCRIPT_DIR/run_all_dataset_creation.sh"

run_stage "Offline RL training" \
    python3 "$REPO_ROOT/RL_Training/train.py" \
        --algo cql \
        --out-dir "$TRAIN_OUT_DIR" \
        --steps 50000 \
        --batch-size 2048

run_stage "RL vs baselines plotting" \
    python3 "$SCRIPT_DIR/plots/plot_rl_vs_baselines.py" \
        --rl-dir "$SCRIPT_DIR/results_rl_policy" \
        --out-dir "$PLOTS_OUT_DIR"

log INFO "Artifacts:"
log INFO " - Simulation logs: $SCRIPT_DIR/logs/"
log INFO " - Dataset directory: $SCRIPT_DIR/tmp/data/"
log INFO " - Training outputs: $TRAIN_OUT_DIR"
log INFO " - Plots: $PLOTS_OUT_DIR"
