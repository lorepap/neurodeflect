# Signal Extraction Tests

This directory contains tests to validate the signal extraction pipeline for the deflection threshold experiments.

## Test Files

### `test_signal_extraction.py`
Comprehensive test that validates:
- Signal availability in simulation output files
- Correct extraction of critical signals using scavetool
- Proper signal naming conventions (PascalCase vs camelCase)
- Extractor script generation functionality

#### Critical Signals Tested:
1. **PacketAction:vector** - Deflection decisions (0=forward, 1=deflect)
2. **QueueLen:vector** - Queue occupancy over time
3. **QueueCapacity:vector** - Queue capacity over time
4. **QueuesTotLen:vector** - Total queue length across all queues
5. **QueuesTotCapacity:vector** - Total queue capacity across all queues
6. **FlowID:vector** - Flow identification for packet tracking
7. **RequesterID:vector** - Requester identification for flow tracking
8. **switchId:vector** - Switch identification for path tracking

## Running the Tests

```bash
cd /home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/tests
python test_signal_extraction.py
```

## Test Requirements

- Simulation data must be available in `../results/threshold_*/` directories
- `scavetool` must be available in the system PATH
- pandas library must be installed for CSV validation

## Expected Output

The test will:
1. Find available simulation files
2. Query signals in the simulation
3. Test extraction of each critical signal
4. Validate extractor script generation
5. Provide a comprehensive pass/fail summary

A successful test run ensures that the dataset creation pipeline will work correctly with the current signal extraction configuration.
