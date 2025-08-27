"""
Datacenter Network Deflection Environment

This environment simulates the packet deflection decision-making process
for optimizing datacenter network performance using the collected threshold dataset.
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import logging


class DatacenterDeflectionEnv(gym.Env):
    """
    OpenAI Gym environment for learning optimal packet deflection policies
    in datacenter networks using offline RL from collected threshold data.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, 
                 dataset_path: str,
                 reward_config: Optional[Dict] = None,
                 max_steps: int = 1000,
                 normalize_states: bool = True):
        """
        Initialize the datacenter deflection environment.
        
        Args:
            dataset_path: Path to the threshold dataset CSV file
            reward_config: Configuration for reward engineering
            max_steps: Maximum steps per episode
            normalize_states: Whether to normalize observations
        """
        super().__init__()
        
        # Load dataset
        self.dataset = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(self.dataset)} records")
        
        # Handle different threshold column names
        if 'threshold' in self.dataset.columns:
            threshold_col = 'threshold'
        elif 'deflection_threshold' in self.dataset.columns:
            threshold_col = 'deflection_threshold'
        else:
            raise ValueError("Dataset must contain either 'threshold' or 'deflection_threshold' column")
        
        # Extract features based on available columns
        feature_columns = []
        
        # Map available columns to our standard features
        column_mapping = {
            'occupancy': 'queue_length',
            'capacity': 'arrival_rate', 
            'total_capacity': 'service_rate',
            'total_occupancy': 'utilization',
            threshold_col: 'threshold',
            'timestamp': 'load'  # Use timestamp as a proxy for load
        }
        
        # Build feature matrix from available columns
        feature_data = []
        self.feature_names = []
        
        for original_col, standard_name in column_mapping.items():
            if original_col in self.dataset.columns:
                feature_data.append(self.dataset[original_col].values)
                self.feature_names.append(standard_name)
        
        # If we don't have enough features, add derived ones
        if len(feature_data) < 6:
            # Add utilization ratio if we have occupancy and capacity
            if 'occupancy' in self.dataset.columns and 'capacity' in self.dataset.columns:
                util_ratio = self.dataset['occupancy'] / (self.dataset['capacity'] + 1e-8)
                feature_data.append(util_ratio.values)
                self.feature_names.append('utilization_ratio')
        
        # Combine features
        self.features = np.column_stack(feature_data)
        
        # Extract actions (map to our action space)
        if 'action' in self.dataset.columns:
            # Assume actions are already in correct format (0, 1, 2)
            self.actions = self.dataset['action'].values
        else:
            # If no action column, create random actions for testing
            self.actions = np.random.randint(0, 3, len(self.dataset))
            print("Warning: No action column found, using random actions")
        
        print(f"Features extracted: {self.feature_names}")
        print(f"Feature matrix shape: {self.features.shape}")
        print(f"Action range: {self.actions.min()} to {self.actions.max()}")
        
        # Normalize features if requested
        if normalize_states:
            self.feature_mean = np.mean(self.features, axis=0)
            self.feature_std = np.std(self.features, axis=0) + 1e-8
            self.features = (self.features - self.feature_mean) / self.feature_std
            print("Features normalized")
        
        # Set current index
        self.current_index = 0
        self.max_steps = max_steps
        self.step_count = 0
    
    def _default_reward_config(self) -> Dict:
        """Default reward engineering configuration."""
        return {
            'queue_penalty_weight': -0.1,      # Penalty for high queue occupancy
            'deflection_cost': -0.01,          # Small cost for deflecting
            'drop_penalty': -1.0,              # Large penalty for dropping
            'throughput_reward': 1.0,          # Reward for maintaining throughput
            'latency_penalty_weight': -0.5,    # Penalty for increased latency
            'baseline_reward': 0.1             # Baseline reward for forwarding
        }
    
    def _preprocess_dataset(self):
        """Preprocess the dataset for RL training."""
        # Extract state features (exclude timestamp, action, threshold)
        feature_columns = ['capacity', 'total_capacity', 'occupancy', 
                          'total_occupancy', 'seq_num', 'ttl']
        
        self.state_features = self.dataset[feature_columns]
        self.actions = self.dataset['action'].values
        self.thresholds = self.dataset['deflection_threshold'].values
        self.timestamps = self.dataset['timestamp'].values
        
        # Compute normalization parameters
        if self.normalization:
            self.feature_means = self.state_features.mean()
            self.feature_stds = self.state_features.std()
            
            # Normalize features
            self.normalized_features = (self.state_features - self.feature_means) / (self.feature_stds + 1e-8)
        
        # Compute additional metrics for reward engineering
        self._compute_network_metrics()
        
        self.logger.info("Dataset preprocessing completed")
    
    def _compute_network_metrics(self):
        """Compute network performance metrics for reward engineering."""
        # Queue utilization ratio
        self.queue_utilization = self.dataset['occupancy'] / (self.dataset['capacity'] + 1e-8)
        
        # Total network utilization
        self.total_utilization = self.dataset['total_occupancy'] / (self.dataset['total_capacity'] + 1e-8)
        
        # TTL as a proxy for latency/hops
        self.normalized_ttl = (250 - self.dataset['ttl']) / 250.0  # Higher = more hops
        
        # Sequence number gaps as proxy for out-of-order delivery
        self.seq_gaps = np.abs(np.diff(self.dataset['seq_num'], prepend=self.dataset['seq_num'].iloc[0]))
    
    def _sample_episode_data(self):
        """Sample a random episode from the dataset."""
        # Ensure we have enough data for an episode
        max_start = max(0, len(self.dataset) - self.episode_length)
        
        if max_start == 0:
            # Dataset smaller than episode length, use entire dataset
            start_idx = 0
            end_idx = len(self.dataset)
        else:
            # Random start position
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + self.episode_length
        
        # Extract episode data
        episode_indices = slice(start_idx, end_idx)
        
        if self.normalization:
            episode_states = self.normalized_features.iloc[episode_indices].values
        else:
            episode_states = self.state_features.iloc[episode_indices].values
            
        episode_actions = self.actions[episode_indices]
        episode_thresholds = self.thresholds[episode_indices]
        episode_timestamps = self.timestamps[episode_indices]
        
        return {
            'states': episode_states,
            'actions': episode_actions,
            'thresholds': episode_thresholds,
            'timestamps': episode_timestamps,
            'indices': (start_idx, end_idx)
        }
    
    def _compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Compute reward based on the current state, action, and next state.
        This implements the reward engineering for network performance optimization.
        """
        reward = 0.0
        
        # Extract current queue utilization (state[2] = occupancy, state[0] = capacity)
        current_util = state[2] / (state[0] + 1e-8)
        next_util = next_state[2] / (next_state[0] + 1e-8)
        
        # 1. Queue occupancy penalty - penalize high utilization
        queue_penalty = current_util * self.reward_config['queue_penalty_weight']
        reward += queue_penalty
        
        # 2. Action-specific rewards/penalties
        if action == 0:  # Forward normally
            reward += self.reward_config['baseline_reward']
            
        elif action == 1:  # Deflect packet
            # Small cost for deflection but potential benefit if it reduces congestion
            reward += self.reward_config['deflection_cost']
            
            # Bonus if deflection helps reduce future queue utilization
            if next_util < current_util:
                reward += 0.1 * (current_util - next_util)
                
        elif action == 2:  # Drop packet
            # Large penalty for dropping packets
            reward += self.reward_config['drop_penalty']
        
        # 3. Network efficiency reward
        # Reward maintaining low overall network utilization
        total_util = state[3] / (state[1] + 1e-8)  # total_occupancy / total_capacity
        efficiency_reward = (1.0 - total_util) * self.reward_config['throughput_reward'] * 0.1
        reward += efficiency_reward
        
        # 4. TTL-based latency penalty
        # Higher TTL indicates more network hops/latency
        ttl_penalty = (250 - state[5]) / 250.0 * self.reward_config['latency_penalty_weight'] * 0.1
        reward += ttl_penalty
        
        return reward
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        
        # Sample new episode data
        self.current_episode_data = self._sample_episode_data()
        self.current_step = 0
        
        # Reset history
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
        # Get initial observation
        initial_state = self.current_episode_data['states'][0].astype(np.float32)
        
        info = {
            'episode_length': len(self.current_episode_data['states']),
            'threshold': self.current_episode_data['thresholds'][0],
            'timestamp': self.current_episode_data['timestamps'][0]
        }
        
        return initial_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.current_episode_data is None:
            raise ValueError("Environment not reset. Call reset() before step().")
        
        # Get current and next states
        current_state = self.current_episode_data['states'][self.current_step]
        
        # Check if episode is done
        self.current_step += 1
        done = self.current_step >= len(self.current_episode_data['states'])
        truncated = False
        
        if not done:
            next_state = self.current_episode_data['states'][self.current_step]
        else:
            # Use current state as next state for final step
            next_state = current_state
        
        # Compute reward
        reward = self._compute_reward(current_state, action, next_state)
        
        # Store history
        self.state_history.append(current_state.copy())
        self.action_history.append(action)
        self.reward_history.append(reward)
        
        # Prepare info
        info = {
            'step': self.current_step - 1,
            'original_action': self.current_episode_data['actions'][self.current_step - 1],
            'threshold': self.current_episode_data['thresholds'][self.current_step - 1],
            'timestamp': self.current_episode_data['timestamps'][self.current_step - 1],
            'reward_components': {
                'queue_utilization': current_state[2] / (current_state[0] + 1e-8),
                'total_utilization': current_state[3] / (current_state[1] + 1e-8),
                'ttl': current_state[5]
            }
        }
        
        # Return next observation
        if not done:
            observation = next_state.astype(np.float32)
        else:
            observation = current_state.astype(np.float32)
            
            # Add episode summary to info
            info['episode_summary'] = {
                'total_reward': sum(self.reward_history),
                'average_reward': np.mean(self.reward_history),
                'actions_taken': self.action_history.copy(),
                'final_utilization': current_state[2] / (current_state[0] + 1e-8)
            }
        
        return observation, reward, done, truncated, info
    
    def render(self, mode='human'):
        """Render the environment state."""
        if self.current_episode_data is None:
            return
        
        if self.current_step > 0:
            current_state = self.current_episode_data['states'][self.current_step - 1]
            
            print(f"Step: {self.current_step - 1}")
            print(f"Queue Utilization: {current_state[2] / (current_state[0] + 1e-8):.3f}")
            print(f"Total Utilization: {current_state[3] / (current_state[1] + 1e-8):.3f}")
            print(f"TTL: {current_state[5]}")
            if self.action_history:
                print(f"Last Action: {self.action_history[-1]}")
                print(f"Last Reward: {self.reward_history[-1]:.3f}")
            print("-" * 40)
    
    def get_dataset_info(self) -> Dict:
        """Get information about the loaded dataset."""
        return {
            'total_samples': len(self.dataset),
            'unique_thresholds': sorted(self.dataset['deflection_threshold'].unique()),
            'time_range': (self.dataset['timestamp'].min(), self.dataset['timestamp'].max()),
            'action_distribution': self.dataset['action'].value_counts().to_dict(),
            'feature_statistics': self.state_features.describe().to_dict()
        }


# Helper function to create environment with common configurations
def create_deflection_env(dataset_path: str, config: Optional[Dict] = None) -> DatacenterDeflectionEnv:
    """
    Factory function to create a datacenter deflection environment with common configurations.
    
    Args:
        dataset_path: Path to the threshold dataset
        config: Optional configuration dictionary
        
    Returns:
        Configured DatacenterDeflectionEnv instance
    """
    default_config = {
        'episode_length': 1000,
        'normalization': True,
        'reward_config': {
            'queue_penalty_weight': -0.1,
            'deflection_cost': -0.01,
            'drop_penalty': -1.0,
            'throughput_reward': 1.0,
            'latency_penalty_weight': -0.5,
            'baseline_reward': 0.1
        }
    }
    
    if config:
        default_config.update(config)
    
    return DatacenterDeflectionEnv(
        dataset_path=dataset_path,
        **default_config
    )
