# Deflection Threshold Variation Experiments

This guide explains how to run network simulations with different deflection threshold values and create datasets for reinforcement learning training.

## Overview

The deflection threshold feature allows you to configure how aggressively switches deflect packets when queues are full. This implementation supports threshold values between 0.3 and 0.9, where:
- Lower values (0.3) = more aggressive deflection
- Higher values (0.9) = less aggressive deflection

## Prerequisites

1. **Build the simulation environment:**
   ```bash
   cd /home/ubuntu/practical_deflection/Omnet_Sims
   bash build.sh
   ```

2. **Extract 1Gbps simulation data (if not already done):**
   ```bash
   cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims
   bash extract_dist_files_LS_1Gbps.sh
   ```

## Running Threshold Experiments

### Quick Test (Short Duration)
To test if the threshold feature is working:
```bash
cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims
bash test_threshold_variation.sh
```

This runs a 1.2-second simulation with threshold 0.5 to verify the implementation.

### Full Threshold Variation Experiments
To run experiments with multiple threshold values:
```bash
cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims
bash run_deflection_threshold_experiments.sh
```

This script runs simulations with thresholds: 0.3, 0.5, 0.7, and 0.9.

### Custom Threshold Values
To run with specific threshold values, modify the configuration in `omnetpp_1G.ini`:

```ini
[Config DCTCP_SD_THRESHOLD_VARIATION]
extends = DCTCP_SD
**.deflectionThreshold = ${deflectionThreshold=0.3,0.5,0.7,0.9}
```

Then run:
```bash
../../../out/gcc-release/src/dc_simulations -c DCTCP_SD_THRESHOLD_VARIATION omnetpp_1G.ini
```

## Creating Datasets for RL Training

### Automatic Dataset Creation
Use the complete pipeline script to extract data and create ML-ready datasets:

```bash
cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims
python3 threshold_pipeline.py --thresholds 0.3,0.5,0.7,0.9 --results_dir results --output_dir dataset_output
```

### Pipeline Options
- `--thresholds`: Comma-separated threshold values to process
- `--results_dir`: Directory containing simulation results (default: "results")
- `--output_dir`: Where to store extracted data and datasets (default: "dataset_output")
- `--skip_extraction`: Skip data extraction if already done
- `--combine_only`: Only combine existing datasets

### Manual Steps (if needed)

1. **Extract specific threshold data:**
   ```bash
   python3 threshold_pipeline.py --threshold 0.5 --results_dir results --output_dir dataset_output
   ```

2. **Combine all threshold datasets:**
   ```bash
   python3 threshold_pipeline.py --combine_all_thresholds --results_dir results --output_dir dataset_output
   ```

## Dataset Structure

The generated dataset contains the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `timestamp` | Simulation time | float64 |
| `capacity` | Queue capacity (bytes) | int64 |
| `total_capacity` | Total capacity across all queues | int64 |
| `occupancy` | Current queue length (packets) | int64 |
| `total_occupancy` | Total occupancy across all queues | int64 |
| `seq_num` | Packet sequence number | int64 |
| `ttl` | Time-to-live value | int64 |
| `action` | Packet forwarding action (0=forward, 1=deflect, 2=drop) | int64 |
| `deflection_threshold` | Threshold value used in experiment | float64 |

### Output Files
- `combined_threshold_dataset.csv` - Main dataset with all thresholds
- `threshold_dataset_X.X.csv` - Individual datasets per threshold
- `extract_threshold_X.X.sh` - Extraction scripts (for reproducibility)

## Using the Dataset for RL Training

The dataset is designed for offline reinforcement learning where:

- **State Space**: `[capacity, total_capacity, occupancy, total_occupancy, seq_num, ttl]`
- **Action Space**: `action` (discrete: 0, 1, 2)
- **Context**: `deflection_threshold` provides experimental condition
- **Temporal**: `timestamp` enables time-series analysis

### Example Loading in Python:
```python
import pandas as pd

# Load the combined dataset
df = pd.read_csv('dataset_output/combined_threshold_dataset.csv')

# Split by threshold for analysis
threshold_groups = df.groupby('deflection_threshold')

# Prepare for RL training
features = ['capacity', 'total_capacity', 'occupancy', 'total_occupancy', 'seq_num', 'ttl']
X = df[features].values
y = df['action'].values
thresholds = df['deflection_threshold'].values
```

## Configuration Details

### Threshold Implementation
The deflection threshold is implemented in `V2PacketBuffer.cc`:
- `getCurrentThreshold()`: Returns current threshold value
- `setDeflectionThreshold()`: Sets threshold value
- `is_queue_full_DT()`: Checks if queue exceeds threshold

### Simulation Parameters
Key parameters in the threshold experiments:
- **Simulation Time**: 1.2s (short test) or longer for full experiments
- **Network Topology**: 4 spines, 8 aggregation switches, 40 servers
- **Queue Type**: V2PIFO with deflection capabilities
- **Traffic**: Mix of bursty and background flows

## Troubleshooting

### Common Issues

1. **"No .vec files found"**: Ensure simulations completed successfully
   ```bash
   ls -la results/threshold_*/
   ```

2. **"Empty dataset"**: Check if vector names match in extraction filters
   ```bash
   scavetool q -v results/threshold_0.3/*.vec
   ```

3. **Build errors**: Ensure INET library is properly built
   ```bash
   cd /home/ubuntu/practical_deflection/Omnet_Sims
   bash build.sh
   ```

### Verification
To verify the threshold feature is working:
```bash
# Check if threshold parameter appears in simulation output
grep -r "deflectionThreshold" results/threshold_*/
```

## Next Steps

1. **Extend experiments**: Add more threshold values or longer simulation times
2. **Analyze results**: Compare performance metrics across thresholds
3. **Train RL models**: Use the dataset for offline reinforcement learning
4. **Validate models**: Test trained models in live simulations

For questions or issues, refer to the simulation logs in the `logs/` directory or check the OMNeT++ documentation.
