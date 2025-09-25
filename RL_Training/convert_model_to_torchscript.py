#!/usr/bin/env python3
"""
Convert trained IQL model to TorchScrip    # Create policy network with same architecture as training
    # State space: 3 dimensions (queue_utilization, total_occupancy, ttl_priority)
    # Action space: 2 actions (0=forward, 1=deflect)
    state_dim = 3
    action_dim = 2mat for C++ inference.

This script loads the saved IQL model state dictionaries and converts
only the policy network to TorchScript format, which is what we need
for inference in the C++ simulation.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

class PolicyNetwork(nn.Module):
    """Policy network matching the IQL agent implementation."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=[256]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, action_dim),
            nn.Softmax(dim=-1)
        ])
            
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """Forward pass returning action probabilities."""
        return self.network(state)

def convert_model_to_torchscript():
    """Convert the trained model to TorchScript format."""
    
    # Paths
    model_path = Path("/home/ubuntu/practical_deflection/RL_Training/models/final_model.pth")
    torchscript_path = Path("/home/ubuntu/practical_deflection/RL_Training/models/final_model_torchscript.pt")
    
    print(f"Loading model from: {model_path}")
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    
    # Load the saved model checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("Model checkpoint loaded successfully")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Create policy network with same architecture as training
    # State space: 6 dimensions (queue_util, total_util, ttl_priority, ooo_indicator, packet_delay, fct_contribution)
    # Action space: 2 actions (0=forward, 1=deflect)
    state_dim = 3
    action_dim = 2
    hidden_dims = [256]  # Single hidden layer with 256 units as per IQL agent
    
    print(f"Creating policy network: {state_dim} -> {hidden_dims} -> {action_dim}")
    
    # Create policy network
    policy = PolicyNetwork(state_dim, action_dim, hidden_dims)
    
    # Load the policy state dict
    try:
        policy.load_state_dict(checkpoint['policy'])
        print("Policy state dict loaded successfully")
    except KeyError as e:
        print(f"Error: 'policy' key not found in checkpoint. Available keys: {list(checkpoint.keys())}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading policy state dict: {e}")
        sys.exit(1)
    
    # Set to evaluation mode
    policy.eval()
    
    # Create example input for tracing (batch_size=1, state_dim=3)
    example_input = torch.randn(1, state_dim)
    print(f"Example input shape: {example_input.shape}")
    
    # Test the policy with example input
    with torch.no_grad():
        test_output = policy(example_input)
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output (action probabilities): {test_output}")
        
        # Get action (0 or 1) - action probabilities are already softmax'd
        action = torch.argmax(test_output, dim=1)
        print(f"Selected action: {action.item()}")
        print(f"Action 0 probability: {test_output[0, 0].item():.4f}")
        print(f"Action 1 probability: {test_output[0, 1].item():.4f}")
    
    # Convert to TorchScript using tracing
    try:
        print("Converting to TorchScript...")
        traced_policy = torch.jit.trace(policy, example_input)
        print("TorchScript conversion successful")
        
        # Test the traced model
        with torch.no_grad():
            traced_output = traced_policy(example_input)
            print(f"Traced model output: {traced_output}")
            
            # Verify outputs match
            if torch.allclose(test_output, traced_output, atol=1e-6):
                print("✓ Traced model output matches original model")
            else:
                print("✗ Warning: Traced model output differs from original")
                print(f"Max difference: {torch.max(torch.abs(test_output - traced_output))}")
    
    except Exception as e:
        print(f"Error during TorchScript conversion: {e}")
        sys.exit(1)
    
    # Save the TorchScript model
    try:
        torchscript_path.parent.mkdir(parents=True, exist_ok=True)
        traced_policy.save(str(torchscript_path))
        print(f"TorchScript model saved to: {torchscript_path}")
    except Exception as e:
        print(f"Error saving TorchScript model: {e}")
        sys.exit(1)
    
    # Verify the saved model can be loaded
    try:
        print("Verifying saved TorchScript model...")
        loaded_model = torch.jit.load(str(torchscript_path))
        
        with torch.no_grad():
            loaded_output = loaded_model(example_input)
            print(f"Loaded model output: {loaded_output}")
            
            if torch.allclose(test_output, loaded_output, atol=1e-6):
                print("✓ Loaded TorchScript model works correctly")
            else:
                print("✗ Warning: Loaded model output differs")
                
    except Exception as e:
        print(f"Error verifying saved model: {e}")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("SUCCESS: Model conversion completed!")
    print(f"Original model: {model_path}")
    print(f"TorchScript model: {torchscript_path}")
    print(f"File size: {torchscript_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("="*60)
    
    return str(torchscript_path)

if __name__ == "__main__":
    torchscript_path = convert_model_to_torchscript()
    print(f"\nUse this path in your C++ code: {torchscript_path}")
