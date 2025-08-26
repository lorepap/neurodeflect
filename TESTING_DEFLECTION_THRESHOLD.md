# Testing the Deflection Threshold Feature

## Prerequisites (from README.md):
1. Install OMNeT++ 5.6.2:
   ```bash
   wget https://github.com/omnetpp/omnetpp/releases/download/omnetpp-5.6.2/omnetpp-5.6.2-src-linux.tgz
   tar xvfz omnetpp-5.6.2-src-linux.tgz
   cd omnetpp-5.6.2/
   . setenv
   echo "export PATH=$HOME/omnetpp-5.6.2/bin:\$PATH" >> ~/.bashrc
   source ~/.bashrc
   ./configure WITH_QTENV=no WITH_OSG=no WITH_OSGEARTH=no
   make
   ```

2. Build the project:
   ```bash
   cd practical_deflection/Omnet_Sims/
   bash build.sh
   ```

3. Download distribution files:
   ```bash
   cd dc_simulations/simulations/sims
   git submodule init
   git submodule update
   bash extract_dist_files_LS_1Gbps.sh
   ```

## Testing Our Deflection Threshold Feature:

### Option 1: Quick Test (Single Threshold)
```bash
cd practical_deflection/Omnet_Sims/dc_simulations/simulations/sims
./test_threshold.sh
```

### Option 2: Full Threshold Variation Experiment
```bash
cd practical_deflection/Omnet_Sims/dc_simulations/simulations/sims
./run_deflection_threshold_experiments.sh
```

### Option 3: Manual Single Configuration
```bash
opp_runall -j1 ../../src/dc_simulations -m -u Cmdenv \
    -c DCTCP_SD_THRESHOLD_VARIATION \
    -n ..:../../src:../../../inet/src:../../../inet/examples:../../../inet/tutorials:../../../inet/showcases \
    --image-path=../../../inet/images \
    -l ../../../inet/src/INET \
    omnetpp_1G.ini
```

## Expected Results:
- Output files with threshold values in filenames (e.g., `*_threshold_0.5_*`)
- Data showing different deflection behavior for different threshold values
- CSV dataset with threshold-specific analysis via `create_threshold_dataset.py`

## What Our Implementation Does:
1. **V2PacketBuffer** now uses configurable thresholds instead of fixed dt_alpha
2. **Multiple threshold testing** in one experiment run (0.3, 0.5, 0.7, 0.9)
3. **Enhanced output naming** includes threshold values for easy identification
4. **Backward compatibility** with existing configurations (default threshold = dt_alpha)
5. **Automated data collection** scripts that understand threshold variations
