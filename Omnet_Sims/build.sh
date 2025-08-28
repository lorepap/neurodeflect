#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory to use absolute paths
BASE_DIR=$(cd "$(dirname "$0")" && pwd)

# make inet
coreNum=$(grep -c '^processor' /proc/cpuinfo || echo 1)
echo "$coreNum hyper-threads used for building."

echo -e "-------------------------- make clean -C $BASE_DIR/inet --------------------------"
make clean -C "$BASE_DIR/inet"

echo -e "\n\n-------------------------- make -C $BASE_DIR/inet makefiles --------------------------"
make -C "$BASE_DIR/inet" makefiles

echo -e "\n\n-------------------------- make -j $coreNum -C $BASE_DIR/inet MODE=release all --------------------------"
make -j "$coreNum" -C "$BASE_DIR/inet" MODE=release all

# make simulations/src
echo -e "\n\n-------------------------- make clean -C $BASE_DIR/dc_simulations --------------------------"
make clean -C "$BASE_DIR/dc_simulations"

echo -e "\n\n-------------------------- make -C $BASE_DIR/dc_simulations MODE=release all --------------------------"
make -j "$coreNum" -C "$BASE_DIR/dc_simulations" MODE=release all