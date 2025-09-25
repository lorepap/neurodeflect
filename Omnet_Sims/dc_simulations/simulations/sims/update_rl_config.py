#!/usr/bin/env python3
"""
Script to update OMNeT++ configuration with correct RL normalization parameters.

This script:
1. Loads normalization parameters from the training process
2. Updates the omnetpp_1G.ini file with the correct values
3. Ensures consistency between training and inference
"""

import json
import re
import os
from pathlib import Path

def load_normalization_params(source='auto'):
    """
    Load normalization parameters from different sources.
    
    Args:
        source: 'auto', 'training', or 'manual'
    """
    # Try to load from RL training models directory first
    training_norm_file = Path("/home/ubuntu/practical_deflection/RL_Training/models/normalization_params.json")
    
    # Try to load from simulation directory (manual calculation)
    manual_norm_file = Path("training_normalization_params.json")
    
    if source == 'auto':
        if training_norm_file.exists():
            print(f"✓ Loading normalization parameters from training: {training_norm_file}")
            with open(training_norm_file, 'r') as f:
                return json.load(f)
        elif manual_norm_file.exists():
            print(f"✓ Loading normalization parameters from manual calculation: {manual_norm_file}")
            with open(manual_norm_file, 'r') as f:
                return json.load(f)
        else:
            print("✗ No normalization parameters found. Please run training first or calculate manually.")
            return None
    elif source == 'training':
        if training_norm_file.exists():
            with open(training_norm_file, 'r') as f:
                return json.load(f)
        else:
            print(f"✗ Training normalization file not found: {training_norm_file}")
            return None
    elif source == 'manual':
        if manual_norm_file.exists():
            with open(manual_norm_file, 'r') as f:
                return json.load(f)
        else:
            print(f"✗ Manual normalization file not found: {manual_norm_file}")
            return None

def update_omnet_config(norm_params, config_file="omnetpp_1G.ini", backup=True):
    """
    Update the OMNeT++ configuration file with correct normalization parameters.
    
    Args:
        norm_params: Dictionary with normalization parameters
        config_file: Path to the OMNeT++ configuration file
        backup: Whether to create a backup of the original file
    """
    config_path = Path(config_file)
    
    if not config_path.exists():
        print(f"✗ Configuration file not found: {config_path}")
        return False
    
    # Create backup if requested
    if backup:
        backup_path = config_path.with_suffix('.ini.backup')
        import shutil
        shutil.copy2(config_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
    
    # Read the configuration file
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Format the normalization parameters
    if 'mean' in norm_params and 'std' in norm_params:
        means = norm_params['mean']
        stds = norm_params['std']
        
        mean_str = f"{means[0]:.6f},{means[1]:.6f},{means[2]:.6f}"
        std_str = f"{stds[0]:.6f},{stds[1]:.6f},{stds[2]:.6f}"
    else:
        print("✗ Invalid normalization parameters format")
        return False
    
    # Update the RL_POLICY configuration section
    # Pattern to match the rl_state_mean and rl_state_std lines
    mean_pattern = r'(\*\*\.(?:agg|spine)\[\*\]\.rl_state_mean\s*=\s*")[^"]*(")'
    std_pattern = r'(\*\*\.(?:agg|spine)\[\*\]\.rl_state_std\s*=\s*")[^"]*(")'
    
    # Replace mean values
    mean_count = len(re.findall(mean_pattern, content))
    new_content = re.sub(mean_pattern, rf'\g<1>{mean_str}\g<2>', content)
    
    # Replace std values  
    std_count = len(re.findall(std_pattern, new_content))
    new_content = re.sub(std_pattern, rf'\g<1>{std_str}\g<2>', new_content)
    
    print(f"  Found {mean_count} mean parameters and {std_count} std parameters")
    
    # Check if any replacements were made
    if new_content == content:
        # Check if current values are already correct
        current_mean_matches = re.findall(r'rl_state_mean\s*=\s*"([^"]*)"', content)
        current_std_matches = re.findall(r'rl_state_std\s*=\s*"([^"]*)"', content)
        
        if (current_mean_matches and current_mean_matches[0] == mean_str and
            current_std_matches and current_std_matches[0] == std_str):
            print("✓ Configuration already has the correct normalization parameters!")
            print(f"  Current mean: {current_mean_matches[0]}")
            print(f"  Current std: {current_std_matches[0]}")
            return True
        else:
            print("⚠ Warning: No RL normalization parameters found in configuration file")
            print("  Make sure the DCTCP_SD_RL_POLICY section exists and has rl_state_mean/rl_state_std parameters")
            return False
    
    # Write the updated configuration
    with open(config_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Updated OMNeT++ configuration: {config_path}")
    print(f"  rl_state_mean = \"{mean_str}\"")
    print(f"  rl_state_std = \"{std_str}\"")
    
    return True

def verify_config_update(config_file="omnetpp_1G.ini"):
    """Verify that the configuration was updated correctly."""
    config_path = Path(config_file)
    
    with open(config_path, 'r') as f:
        content = f.read()
    
    # Find RL policy section and extract parameters
    rl_section_start = content.find('[Config DCTCP_SD_RL_POLICY]')
    if rl_section_start == -1:
        print("✗ DCTCP_SD_RL_POLICY section not found")
        return False
    
    # Extract the next section or end of file
    next_section = content.find('\n[Config ', rl_section_start + 1)
    if next_section == -1:
        rl_section = content[rl_section_start:]
    else:
        rl_section = content[rl_section_start:next_section]
    
    # Find mean and std parameters
    mean_match = re.search(r'rl_state_mean\s*=\s*"([^"]*)"', rl_section)
    std_match = re.search(r'rl_state_std\s*=\s*"([^"]*)"', rl_section)
    
    if mean_match and std_match:
        print("✓ Configuration verification:")
        print(f"  Mean: {mean_match.group(1)}")
        print(f"  Std: {std_match.group(1)}")
        return True
    else:
        print("✗ Could not find RL normalization parameters in configuration")
        return False

def main():
    """Main function."""
    print("RL Configuration Updater")
    print("=" * 50)
    
    # Load normalization parameters
    norm_params = load_normalization_params('auto')
    if norm_params is None:
        print("\n❌ Failed to load normalization parameters")
        print("Please either:")
        print("1. Run RL training which will save normalization_params.json")
        print("2. Run the manual calculation script to generate training_normalization_params.json")
        return 1
    
    print(f"✓ Loaded normalization parameters:")
    if 'state_features' in norm_params:
        print(f"  Features: {norm_params['state_features']}")
    if 'mean' in norm_params:
        print(f"  Mean: {norm_params['mean']}")
    if 'std' in norm_params:
        print(f"  Std: {norm_params['std']}")
    print()
    
    # Update OMNeT++ configuration
    success = update_omnet_config(norm_params)
    if not success:
        return 1
    
    print()
    
    # Verify the update
    verify_config_update()
    
    print()
    print("✅ RL configuration update completed!")
    print("You can now run the RL experiment with: ./run_rl_experiment.sh")
    
    return 0

if __name__ == "__main__":
    exit(main())
