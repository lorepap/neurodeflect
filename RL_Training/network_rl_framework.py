"""
Offline Reinforcement Learning for Datacenter Network Deflection Optimization

This module provides a complete framework for training RL agents to optimize
packet deflection decisions in datacenter networks using offline datasets
collected from network simulations.

The framework supports:
- PPO (Proximal Policy Optimization) for policy learning
- Custom reward engineering for network performance optimization
- Feature preprocessing for network state representation
- Model evaluation and analysis tools
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import pickle
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NetworkState:
    """Represents the network state at a given timestamp."""
    timestamp: float
    capacity: int
    total_capacity: int
    occupancy: int
    total_occupancy: int
    seq_num: int
    ttl: int
    deflection_threshold: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor representation."""
        return torch.tensor([
            self.capacity / 1e6,  # Normalize capacity
            self.total_capacity / 1e8,  # Normalize total capacity
            self.occupancy / 1000,  # Normalize occupancy
            self.total_occupancy / 10000,  # Normalize total occupancy
            self.seq_num / 1e10,  # Normalize sequence number
            self.ttl / 250.0,  # Normalize TTL (max 250)
            self.deflection_threshold  # Already normalized
        ], dtype=torch.float32)


@dataclass
class Experience:
    """Represents a single experience tuple for RL training."""
    state: NetworkState
    action: int
    reward: float
    next_state: Optional[NetworkState] = None
    done: bool = False


class NetworkEnvironmentDataset(Dataset):
    """PyTorch Dataset for offline RL training from network simulation data."""
    
    def __init__(self, 
                 dataset_path: str,
                 reward_function: Optional[callable] = None,
                 sequence_length: int = 1,
                 normalize: bool = True):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to the CSV dataset file
            reward_function: Function to compute rewards from network metrics
            sequence_length: Length of state sequences (for temporal modeling)
            normalize: Whether to normalize features
        """
        self.dataset_path = Path(dataset_path)
        self.reward_function = reward_function or self._default_reward_function
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Load and preprocess data
        self.data = self._load_data()
        self.experiences = self._create_experiences()
        
        # Normalization
        if self.normalize:
            self.scaler = StandardScaler()
            self._fit_scaler()
        
        logger.info(f"Loaded {len(self.experiences)} experiences from {dataset_path}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        
        # Validate required columns
        required_columns = [
            'timestamp', 'capacity', 'total_capacity', 'occupancy',
            'total_occupancy', 'seq_num', 'ttl', 'action', 'deflection_threshold'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by timestamp for temporal consistency
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Loaded dataset with {len(df)} records and {len(df.columns)} features")
        return df
    
    def _default_reward_function(self, state: NetworkState, action: int, next_state: Optional[NetworkState] = None) -> float:
        """
        Default reward function based on network performance metrics.
        
        Reward components:
        - Queue utilization penalty (avoid congestion)
        - TTL preservation reward (avoid packet drops)
        - Deflection cost (deflection has overhead)
        
        Args:
            state: Current network state
            action: Action taken (0=forward, 1=deflect, 2=drop)
            next_state: Next network state (if available)
        
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Base reward for packet processing
        if action == 0:  # Forward
            reward += 1.0
        elif action == 1:  # Deflect
            reward += 0.5  # Deflection has cost but avoids drops
        else:  # Drop (action == 2)
            reward -= 2.0  # Heavy penalty for drops
        
        # Queue utilization penalty
        utilization = state.occupancy / max(state.capacity, 1)
        if utilization > 0.8:
            reward -= 1.0 * utilization  # Penalty for high utilization
        
        # TTL preservation reward
        ttl_ratio = state.ttl / 250.0
        reward += 0.5 * ttl_ratio  # Reward for preserving TTL
        
        # Capacity efficiency reward
        if state.total_occupancy < state.total_capacity * 0.7:
            reward += 0.3  # Reward for maintaining spare capacity
        
        return reward
    
    def _create_experiences(self) -> List[Experience]:
        """Create experience tuples from the dataset."""
        experiences = []
        
        # Group by deflection threshold to maintain experiment consistency
        for threshold in self.data['deflection_threshold'].unique():
            threshold_data = self.data[self.data['deflection_threshold'] == threshold]
            
            for i in range(len(threshold_data) - 1):
                current_row = threshold_data.iloc[i]
                next_row = threshold_data.iloc[i + 1]
                
                # Create current state
                state = NetworkState(
                    timestamp=current_row['timestamp'],
                    capacity=current_row['capacity'],
                    total_capacity=current_row['total_capacity'],
                    occupancy=current_row['occupancy'],
                    total_occupancy=current_row['total_occupancy'],
                    seq_num=current_row['seq_num'],
                    ttl=current_row['ttl'],
                    deflection_threshold=current_row['deflection_threshold']
                )
                
                # Create next state
                next_state = NetworkState(
                    timestamp=next_row['timestamp'],
                    capacity=next_row['capacity'],
                    total_capacity=next_row['total_capacity'],
                    occupancy=next_row['occupancy'],
                    total_occupancy=next_row['total_occupancy'],
                    seq_num=next_row['seq_num'],
                    ttl=next_row['ttl'],
                    deflection_threshold=next_row['deflection_threshold']
                )
                
                action = int(current_row['action'])
                reward = self.reward_function(state, action, next_state)
                
                # Determine if episode is done (significant time gap or threshold change)
                time_gap = next_row['timestamp'] - current_row['timestamp']
                done = time_gap > 0.01 or i == len(threshold_data) - 2  # 10ms gap or end of sequence
                
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done
                )
                
                experiences.append(experience)
        
        return experiences
    
    def _fit_scaler(self):
        """Fit the feature scaler on the state data."""
        state_features = []
        for exp in self.experiences:
            state_features.append(exp.state.to_tensor().numpy())
        
        state_features = np.array(state_features)
        self.scaler.fit(state_features)
    
    def __len__(self) -> int:
        return len(self.experiences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single experience as tensors."""
        experience = self.experiences[idx]
        
        state_tensor = experience.state.to_tensor()
        if self.normalize and hasattr(self, 'scaler'):
            state_tensor = torch.tensor(
                self.scaler.transform(state_tensor.unsqueeze(0).numpy())[0],
                dtype=torch.float32
            )
        
        next_state_tensor = None
        if experience.next_state:
            next_state_tensor = experience.next_state.to_tensor()
            if self.normalize and hasattr(self, 'scaler'):
                next_state_tensor = torch.tensor(
                    self.scaler.transform(next_state_tensor.unsqueeze(0).numpy())[0],
                    dtype=torch.float32
                )
        
        return {
            'state': state_tensor,
            'action': torch.tensor(experience.action, dtype=torch.long),
            'reward': torch.tensor(experience.reward, dtype=torch.float32),
            'next_state': next_state_tensor if next_state_tensor is not None else torch.zeros_like(state_tensor),
            'done': torch.tensor(experience.done, dtype=torch.bool)
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics for analysis."""
        rewards = [exp.reward for exp in self.experiences]
        actions = [exp.action for exp in self.experiences]
        
        stats = {
            'total_experiences': len(self.experiences),
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
            'action_distribution': np.bincount(actions),
            'unique_thresholds': len(self.data['deflection_threshold'].unique()),
            'threshold_values': sorted(self.data['deflection_threshold'].unique())
        }
        
        return stats


class PolicyNetwork(nn.Module):
    """Neural network for policy approximation in PPO."""
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [128, 64],
                 activation: nn.Module = nn.ReLU):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)
    
    def get_action_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities using softmax."""
        logits = self.forward(state)
        return torch.softmax(logits, dim=-1)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic policy
        
        Returns:
            Tuple of (action, log_probability)
        """
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
            log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)))
        else:
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """Neural network for value function approximation in PPO."""
    
    def __init__(self, 
                 state_dim: int, 
                 hidden_dims: List[int] = [128, 64],
                 activation: nn.Module = nn.ReLU):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimension of state space
            hidden_dims: List of hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                activation(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer (single value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state).squeeze(-1)


if __name__ == "__main__":
    # Example usage and testing
    print("Network RL Framework - Core Components")
    print("=" * 50)
    
    # Test dataset loading (assuming dataset exists)
    try:
        dataset_path = "../Omnet_Sims/dc_simulations/simulations/sims/dataset_output/combined_threshold_dataset.csv"
        
        if Path(dataset_path).exists():
            dataset = NetworkEnvironmentDataset(dataset_path)
            stats = dataset.get_statistics()
            
            print(f"Dataset loaded successfully!")
            print(f"Total experiences: {stats['total_experiences']}")
            print(f"Reward statistics: mean={stats['reward_mean']:.3f}, std={stats['reward_std']:.3f}")
            print(f"Action distribution: {stats['action_distribution']}")
            print(f"Threshold values: {stats['threshold_values']}")
            
            # Test data loading
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            batch = next(iter(dataloader))
            print(f"Batch shape - State: {batch['state'].shape}, Action: {batch['action'].shape}")
            
            # Test networks
            state_dim = 7  # Based on NetworkState features
            action_dim = 3  # 0=forward, 1=deflect, 2=drop
            
            policy_net = PolicyNetwork(state_dim, action_dim)
            value_net = ValueNetwork(state_dim)
            
            # Test forward pass
            test_state = batch['state'][:5]  # Test with 5 samples
            action_probs = policy_net.get_action_probabilities(test_state)
            values = value_net(test_state)
            
            print(f"Policy output shape: {action_probs.shape}")
            print(f"Value output shape: {values.shape}")
            print("Framework components initialized successfully!")
            
        else:
            print(f"Dataset not found at {dataset_path}")
            print("Please run the threshold pipeline first to generate the dataset.")
            
    except Exception as e:
        print(f"Error testing framework: {e}")
        print("This is normal if the dataset hasn't been created yet.")
