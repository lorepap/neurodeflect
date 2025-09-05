# Data Analysis Scripts

This directory contains various analysis scripts for the practical deflection routing project.

## Analysis Scripts

### FCT and Timing Analysis
- **`analyze_fct_vs_delays.py`** - Analyzes the relationship between Flow Completion Time (FCT) and individual packet one-way delays
- **`analyze_0.01s_experiment.py`** - Analysis of 0.01 second experiment data

### Deflection Analysis
- **`analyze_deflection_trend.py`** - Analyzes deflection trends across different scenarios
- **`analyze_correct_deflection_rates.py`** - Validates deflection rate calculations
- **`comprehensive_deflection_analysis.py`** - Comprehensive deflection behavior analysis
- **`detailed_deflection_analysis.py`** - Detailed deflection pattern analysis
- **`final_deflection_analysis.py`** - Final deflection analysis results

### Data Quality and Validation
- **`analyze_data_loss.py`** - Analyzes data loss patterns and missing data
- **`analyze_raw_vs_processed.py`** - Compares raw simulation data with processed datasets
- **`final_data_discrepancy_analysis.py`** - Identifies and analyzes data discrepancies

### Switch and Network Analysis
- **`correct_switch_analysis.py`** - Switch behavior and routing analysis

## Analysis Results

### Flow Analysis Results
- **`flow_analysis/`** - Directory containing:
  - Flow-specific analysis results (CSV files)
  - Flow timing visualizations (PNG files)
  - FCT vs packet delay analysis outputs

## Usage Examples

### Analyze FCT vs Packet Delays
```bash
python analyze_fct_vs_delays.py --dataset ../../../Omnet_Sims/dc_simulations/simulations/sims/threshold_dataset_15000.csv --threshold 15000
```

### Run Deflection Analysis
```bash
python comprehensive_deflection_analysis.py --input-file dataset.csv
```

### Compare Raw vs Processed Data
```bash
python analyze_raw_vs_processed.py
```

## Dependencies

Most scripts require:
- pandas
- numpy
- matplotlib
- seaborn (for some visualizations)

Install with:
```bash
pip install pandas numpy matplotlib seaborn
```

## Notes

- Scripts assume dataset files are located in the simulations directory
- Many scripts can be run with default parameters for quick analysis
- Output files and visualizations are typically saved in the current directory or subdirectories
- Some scripts may require adjustment of file paths when run from this new location
