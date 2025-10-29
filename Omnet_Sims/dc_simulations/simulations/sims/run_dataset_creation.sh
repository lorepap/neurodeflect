#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

RESULTS_DIR=""
OUTPUT_DIR=""
RUNS=()
DRY_RUN=false

usage() {
    cat <<USAGE
Usage: $(basename "$0") --results-dir DIR --output-dir DIR [options]

Options:
  --runs r1 r2 ...    Optional list of run IDs to include.
  --dry-run           Only list detected runs/modules without writing files.
  -h, --help          Show this help and exit.
USAGE
}

# Parse arguments
after_opts=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --runs)
            shift
            while [[ $# -gt 0 ]] && [[ $1 != --* ]]; do
                RUNS+=("$1")
                shift
            done
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            after_opts=true
            shift
            ;;
        *)
            if $after_opts; then
                RUNS+=("$1")
                shift
            else
                echo "Error: Unknown option $1" >&2
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$RESULTS_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --results-dir and --output-dir are required" >&2
    usage
    exit 1
fi

PYTHON_SCRIPT="$SCRIPT_DIR/new_dataset_builder.py"
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: dataset builder script not found at $PYTHON_SCRIPT" >&2
    exit 1
fi

CMD=(python "$PYTHON_SCRIPT" --results-dir "$RESULTS_DIR" --output-dir "$OUTPUT_DIR")
if [[ ${#RUNS[@]} -gt 0 ]]; then
    CMD+=(--runs "${RUNS[@]}")
fi
if $DRY_RUN; then
    CMD+=(--dry-run)
fi

echo "[run_dataset_creation_new] Invoking: ${CMD[*]}"
"${CMD[@]}"
