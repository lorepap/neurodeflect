# Updated Deflection Threshold Experiment Script

## Script Location
`/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/run_deflection_threshold_experiments.sh`

## Key Updates Made

### 1. Added Simulation Time Parameter
The script now requires a simulation time parameter as input.

**Usage:**
```bash
./run_deflection_threshold_experiments.sh <simulation_time>
```

**Examples:**
```bash
# Run for 10 seconds
./run_deflection_threshold_experiments.sh 10s

# Run for 1000 milliseconds 
./run_deflection_threshold_experiments.sh 1000ms

# Run for 0.1 seconds (100ms)
./run_deflection_threshold_experiments.sh 0.1s
```

### 2. Fixed Threshold Values
Updated from broken percentage values to correct byte-based thresholds:

- **Old (broken):** 0.3, 0.5, 0.7, 0.9 (cast to 0 due to type mismatch)
- **New (fixed):** 15000, 25000, 35000, 45000 bytes (30%, 50%, 70%, 90% of 50KB buffer)

### 3. Updated OMNeT++ Command
The `opp_runall` command now includes:
```bash
--sim-time-limit=$SIMULATION_TIME
```

### 4. Corrected Result Organization
- Updated threshold array values in processing loop
- Log directories now use correct threshold values: `logs/deflection_threshold_15000/`, etc.
- Output messaging reflects byte-based thresholds

## Command Breakdown

**Original command:**
```bash
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_THRESHOLD_VARIATION -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini
```

**Updated command:**
```bash
opp_runall -j50 ../../src/dc_simulations -m -u Cmdenv -c DCTCP_SD_THRESHOLD_VARIATION -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases --image-path=../../../inet/images -l ../../../inet/src/INET omnetpp_1G.ini --sim-time-limit=$SIMULATION_TIME
```

**What changed:**
- Added: `--sim-time-limit=$SIMULATION_TIME` parameter

## How to Run

1. **Navigate to the simulation directory:**
   ```bash
   cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims
   ```

2. **Run the script with desired simulation time:**
   ```bash
   ./run_deflection_threshold_experiments.sh 10s
   ```

3. **Results will be organized by threshold:**
   ```
   logs/deflection_threshold_15000/   # 30% threshold (15KB)
   logs/deflection_threshold_25000/   # 50% threshold (25KB)  
   logs/deflection_threshold_35000/   # 70% threshold (35KB)
   logs/deflection_threshold_45000/   # 90% threshold (45KB)
   ```

## Benefits

1. **Configurable simulation time** - Can adjust based on experimental needs
2. **Fixed threshold bug** - Now tests actual threshold variations
3. **Better organization** - Clear byte-based threshold naming
4. **Proper parameter validation** - Script shows usage if run incorrectly

## Next Steps

After running with the fixed script:
1. Verify deflection occurs at expected buffer utilization levels (30%, 50%, 70%, 90%)
2. Compare results across different thresholds
3. Generate new RL training dataset with proper threshold variation
4. Proceed with RL training using meaningful threshold differences
