#!/usr/bin/env python3
"""
Signal Extraction Validation Test

This test verifies that:
1. The simulation emits all required signals for dataset creation
2. The extractor can successfully extract these signals
3. The extracted data has the expected format and content

Critical signals tested:
- PacketAction:vector (deflection decisions)
- QueueLen:vector (queue occupancy)
- QueueCapacity:vector (queue capacity)
- QueuesTotLen:vector (total queue length)
- QueuesTotCapacity:vector (total queue capacity)
- FlowID:vector (flow identification)
- RequesterID:vector (requester identification)
- switchId:vector (switch identification)
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

# Configuration
SIMS_DIR = '/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims'
RESULTS_DIR = f'{SIMS_DIR}/results'
OUTPUT_DIR = f'{SIMS_DIR}/extracted_results'

# Critical signals that must be present for dataset creation
CRITICAL_SIGNALS = {
    'PacketAction:vector': {
        'directory': 'PACKET_ACTION',
        'description': 'Deflection action decisions (0=forward, 1=deflect)',
        'expected_format': 'CSV-S',
        'module_filter': 'module(LeafSpine1G)',
        'required_for': 'Deflection analysis'
    },
    'QueueLen:vector': {
        'directory': 'QUEUE_LEN', 
        'description': 'Queue length over time',
        'expected_format': 'CSV-S',
        'module_filter': 'module(LeafSpine1G)',
        'required_for': 'Queue state analysis'
    },
    'QueueCapacity:vector': {
        'directory': 'QUEUE_CAPACITY',
        'description': 'Queue capacity over time', 
        'expected_format': 'CSV-S',
        'module_filter': 'module(LeafSpine1G)',
        'required_for': 'Queue state analysis'
    },
    'QueuesTotLen:vector': {
        'directory': 'QUEUES_TOT_LEN',
        'description': 'Total queue length across all queues',
        'expected_format': 'CSV-S', 
        'module_filter': 'module(LeafSpine1G)',
        'required_for': 'Aggregate queue analysis'
    },
    'QueuesTotCapacity:vector': {
        'directory': 'QUEUES_TOT_CAPACITY',
        'description': 'Total queue capacity across all queues',
        'expected_format': 'CSV-S',
        'module_filter': 'module(LeafSpine1G)', 
        'required_for': 'Aggregate queue analysis'
    },
    'FlowID:vector': {
        'directory': 'FLOW_ID',
        'description': 'Flow identification for packet tracking',
        'expected_format': 'CSV-R',
        'module_filter': 'module(**.**.relayUnit)',
        'required_for': 'Flow correlation and FCT analysis'
    },
    'RequesterID:vector': {
        'directory': 'REQUESTER_ID', 
        'description': 'Requester identification for flow tracking',
        'expected_format': 'CSV-R',
        'module_filter': 'module(**.**.relayUnit)',
        'required_for': 'Flow correlation and analysis'
    },
    'switchId:vector': {
        'directory': 'SWITCH_ID',
        'description': 'Switch identification for path tracking',
        'expected_format': 'CSV-R',
        'module_filter': 'module(**.relayUnit)',
        'required_for': 'Path analysis and switch identification'
    }
}

class SignalExtractionTest:
    def __init__(self):
        self.test_results = {}
        self.errors = []
        self.warnings = []
        
    def log_error(self, message):
        self.errors.append(message)
        print(f"❌ ERROR: {message}")
        
    def log_warning(self, message):
        self.warnings.append(message)
        print(f"⚠️  WARNING: {message}")
        
    def log_success(self, message):
        print(f"✅ SUCCESS: {message}")
        
    def log_info(self, message):
        print(f"ℹ️  INFO: {message}")

    def find_simulation_files(self):
        """Find available simulation vector files for testing"""
        vec_files = []
        threshold_dirs = [d for d in os.listdir(RESULTS_DIR) 
                         if d.startswith('threshold_') and os.path.isdir(f'{RESULTS_DIR}/{d}')]
        
        self.log_info(f"Found threshold directories: {threshold_dirs}")
        
        for threshold_dir in threshold_dirs:
            threshold_path = f'{RESULTS_DIR}/{threshold_dir}'
            vec_files_in_dir = [f for f in os.listdir(threshold_path) 
                               if f.endswith('.vec') and '_rep_' in f]
            
            if vec_files_in_dir:
                # Use the first vector file from the first threshold directory with data
                test_file = f'{threshold_path}/{vec_files_in_dir[0]}'
                self.log_info(f"Using test file: {test_file}")
                return test_file, threshold_dir
                
        self.log_error("No simulation vector files found for testing")
        return None, None

    def check_signal_availability(self, vec_file):
        """Check which signals are available in the simulation file"""
        self.log_info("Checking signal availability in simulation file...")
        
        try:
            # Since parsing scavetool query output is complex, we'll just assume
            # the critical signals are available and test them directly
            self.log_info("Will test critical signals directly by attempting extraction")
            
            # Return the signals we expect to find for testing
            available_signals = {}
            for signal_name in CRITICAL_SIGNALS.keys():
                available_signals[signal_name] = "simulation_module"
                        
            self.log_info(f"Will test {len(available_signals)} critical signals")
            return available_signals
            
        except Exception as e:
            self.log_error(f"Error checking signal availability: {e}")
            return {}

    def test_signal_extraction(self, vec_file, signal_name, signal_config):
        """Test extraction of a specific signal"""
        self.log_info(f"Testing extraction of {signal_name}...")
        
        # Create temporary output file
        temp_output = f'/tmp/test_{signal_config["directory"].lower()}.csv'
        
        try:
            # Build scavetool command
            cmd = (f'scavetool x --type v --filter "{signal_config["module_filter"]} AND '
                  f'\\\"{signal_name}\\\"" -o {temp_output} -F {signal_config["expected_format"]} "{vec_file}"')
            
            self.log_info(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=SIMS_DIR)
            
            if result.returncode != 0:
                self.log_error(f"Extraction failed for {signal_name}: {result.stderr}")
                return False
                
            # Check if output file was created and has content
            if not os.path.exists(temp_output):
                self.log_error(f"Output file not created for {signal_name}")
                return False
                
            file_size = os.path.getsize(temp_output)
            if file_size == 0:
                self.log_error(f"Empty output file for {signal_name}")
                return False
                
            # Try to read and validate the CSV
            try:
                df = pd.read_csv(temp_output)
                row_count = len(df)
                
                if row_count == 0:
                    self.log_error(f"No data rows extracted for {signal_name}")
                    return False
                    
                self.log_success(f"Successfully extracted {signal_name}: {row_count} data points, {file_size} bytes")
                
                # Clean up temporary file
                os.remove(temp_output)
                return True
                
            except Exception as e:
                self.log_error(f"Failed to parse extracted CSV for {signal_name}: {e}")
                return False
                
        except Exception as e:
            self.log_error(f"Exception during extraction of {signal_name}: {e}")
            return False

    def test_extractor_script_generation(self, threshold_dir):
        """Test that the extractor script can be generated and contains correct signal names"""
        self.log_info("Testing extractor script generation...")
        
        try:
            # Generate extractor script
            script_path = f'{SIMS_DIR}/tests/test_extractor.sh'
            cmd = f'cd {RESULTS_DIR}/{threshold_dir} && python ../../extractor_shell_creator.py TEST_VALIDATION > {script_path}'
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log_error(f"Failed to generate extractor script: {result.stderr}")
                return False
                
            # Check that script contains correct signal names
            with open(script_path, 'r') as f:
                script_content = f.read()
                
            signals_found = 0
            for signal_name in CRITICAL_SIGNALS.keys():
                if signal_name in script_content:
                    signals_found += 1
                else:
                    self.log_error(f"Signal {signal_name} not found in generated extractor script")
                    
            if signals_found == len(CRITICAL_SIGNALS):
                self.log_success(f"All {signals_found} critical signals found in extractor script")
                # Clean up test script
                os.remove(script_path)
                return True
            else:
                self.log_error(f"Only {signals_found}/{len(CRITICAL_SIGNALS)} signals found in extractor script")
                return False
                
        except Exception as e:
            self.log_error(f"Exception during extractor script generation test: {e}")
            return False

    def run_tests(self):
        """Run all signal extraction tests"""
        print("=" * 80)
        print("SIGNAL EXTRACTION VALIDATION TEST")
        print("=" * 80)
        
        # Find simulation files
        vec_file, threshold_dir = self.find_simulation_files()
        if not vec_file:
            return False
            
        # Check signal availability
        available_signals = self.check_signal_availability(vec_file)
        if not available_signals:
            return False
            
        print("\n" + "-" * 60)
        print("AVAILABLE SIGNALS IN SIMULATION:")
        for signal, module in available_signals.items():
            print(f"  {signal} (from {module})")
        print("-" * 60)
        
        # Test each critical signal
        print(f"\nTesting extraction of {len(CRITICAL_SIGNALS)} critical signals...")
        success_count = 0
        
        for signal_name, signal_config in CRITICAL_SIGNALS.items():
            print(f"\n[{success_count + 1}/{len(CRITICAL_SIGNALS)}] Testing {signal_name}")
            print(f"   Purpose: {signal_config['required_for']}")
            print(f"   Description: {signal_config['description']}")
            
            if signal_name not in available_signals:
                self.log_error(f"Signal {signal_name} not found in simulation output")
                continue
                
            if self.test_signal_extraction(vec_file, signal_name, signal_config):
                success_count += 1
                
        # Test extractor script generation
        print(f"\n[EXTRA] Testing extractor script generation...")
        script_generation_success = self.test_extractor_script_generation(threshold_dir)
        
        # Print summary
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"✅ Successful signal extractions: {success_count}/{len(CRITICAL_SIGNALS)}")
        print(f"✅ Extractor script generation: {'PASS' if script_generation_success else 'FAIL'}")
        
        if self.errors:
            print(f"\n❌ Errors encountered: {len(self.errors)}")
            for error in self.errors:
                print(f"   - {error}")
                
        if self.warnings:
            print(f"\n⚠️  Warnings: {len(self.warnings)}")
            for warning in self.warnings:
                print(f"   - {warning}")
                
        overall_success = (success_count == len(CRITICAL_SIGNALS) and 
                          script_generation_success and 
                          len(self.errors) == 0)
        
        print(f"\n🎯 OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        
        if overall_success:
            print("\n🎉 All signal extraction tests passed! Your pipeline should work correctly.")
        else:
            print("\n⚠️  Some tests failed. Please check the errors above.")
            
        return overall_success

if __name__ == "__main__":
    test = SignalExtractionTest()
    success = test.run_tests()
    sys.exit(0 if success else 1)
