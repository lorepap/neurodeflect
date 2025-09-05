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
        
        # Define optimal real-time features for autonomous deflection decisions
        # NO threshold or switch type - model must learn optimal deflection autonomously
        
        # Extract base features
        feature_data = []
        self.feature_names = []
        
        # 1. Queue utilization (fundamental congestion indicator)
        queue_utilization = self.dataset['occupancy'] / (self.dataset['capacity'] + 1e-8)
        feature_data.append(queue_utilization.values)
        self.feature_names.append('queue_utilization')
        
        # 2. Total network utilization (global congestion context)
        total_utilization = self.dataset['total_occupancy'] / (self.dataset['total_capacity'] + 1e-8)
        feature_data.append(total_utilization.values)
        self.feature_names.append('total_utilization')
        
        # 3. TTL priority (packet urgency - higher value = more hops traveled)
        ttl_priority = (250 - self.dataset['ttl']) / 250.0
        feature_data.append(ttl_priority.values)
        self.feature_names.append('ttl_priority')
        
        # 4. Out-of-order indicator (deflection cost - OOO packets hurt performance)
        ooo_indicator = self.dataset['ooo'].values.astype(float)
        feature_data.append(ooo_indicator)
        self.feature_names.append('ooo_indicator')
        
        # 5. Packet delay (one-way delay: end_time - start_time)
        packet_delay = (self.dataset['end_time'] - self.dataset['start_time']).values
        feature_data.append(packet_delay)
        self.feature_names.append('packet_delay')
        
        # 6. FCT contribution (how this packet's delay affects flow completion time)
        # Compute how much this packet's delay contributes to overall FCT (vectorized)
        packet_delays = (self.dataset['end_time'] - self.dataset['start_time']).values
        flow_fcts = self.dataset['FCT'].values
        
        # FCT contribution: packet delay as fraction of total flow completion time
        # Avoid division by zero
        fct_contribution = np.divide(packet_delays, flow_fcts, 
                                   out=np.zeros_like(packet_delays), 
                                   where=flow_fcts!=0)
        
        feature_data.append(fct_contribution)
        self.feature_names.append('fct_contribution')
        
        # Combine features into state matrix
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
        
        # Define observation and action spaces
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_features,), 
            dtype=np.float32
        )
        
        # Action space: 0=forward, 1=deflect, 2=drop (but our data only has 0,1)
        unique_actions = np.unique(self.actions)
        n_actions = max(len(unique_actions), 3)  # At least 3 actions
        self.action_space = spaces.Discrete(n_actions)
        
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
        self.episode_length = max_steps  # Add missing episode_length attribute
        
        # Reward configuration
        if reward_config is None:
            self.reward_config = self._default_reward_config()
        else:
            self.reward_config = reward_config
        
        # Additional attributes needed for environment
        self.normalize_states = normalize_states
        self.current_episode_data = None
    
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
        max_start = max(0, len(self.features) - self.episode_length)
        
        if max_start == 0:
            # Dataset smaller than episode length, use entire dataset
            start_idx = 0
            end_idx = len(self.features)
        else:
            # Random start position
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + self.episode_length
        
        # Extract episode data
        episode_states = self.features[start_idx:end_idx]
        episode_actions = self.actions[start_idx:end_idx]
        
        return {
            'states': episode_states,
            'actions': episode_actions,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'current_step': 0
        }
    
    def _compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Packet-level reward for offline RL training.
        Focuses on packet characteristics only, avoiding misleading network state comparisons.
        """
        reward = 0.0
        
        # Extract current state features
        queue_util = state[0]         # Queue utilization
        total_util = state[1]         # Total network utilization  
        ttl_priority = state[2]       # TTL priority
        ooo_indicator = state[3]      # Out-of-order indicator
        packet_delay = state[4]       # Packet one-way delay
        fct_contribution = state[5]   # FCT contribution factor
        
        if action == 0:  # FORWARD
            reward = 0.1  # Normal operation baseline
            
        elif action == 1:  # DEFLECT  
            reward = -0.1  # Deflection is inherently costly (latency)
            
            # REMOVED: Network state improvement bonus - misleading in offline setting
            # In offline RL, next_state is from pre-recorded data, not caused by our action
        
        # PACKET-LEVEL PERFORMANCE PENALTIES (valid in offline setting):
        # These penalties are based on packet characteristics, not network state transitions
        
        # 1. OUT-OF-ORDER PENALTY: Always penalize OOO packets
        if ooo_indicator > 0:  # This packet is out-of-order
            ooo_penalty = -0.2 * ooo_indicator
            reward += ooo_penalty
        
        # 2. PACKET DELAY PENALTY: Penalize high delay packets
        # Normalized delay penalty (assuming delays are normalized)
        if packet_delay > 0:  # High delay is bad for performance
            delay_penalty = -0.1 * abs(packet_delay)  # Penalty proportional to delay
            reward += delay_penalty
        
        # 3. FCT CONTRIBUTION PENALTY: Penalize actions that hurt flow completion
        # Higher FCT contribution means this packet is more critical to flow completion
        if fct_contribution > 0:
            # If deflecting a critical packet (high FCT impact), bigger penalty
            if action == 1:  # Deflection
                fct_penalty = -0.15 * fct_contribution  # Critical packets shouldn't be deflected
                reward += fct_penalty
            else:  # Forwarding critical packets is good
                fct_bonus = 0.05 * fct_contribution
                reward += fct_bonus
        
        return np.clip(reward, -1.0, 1.0)
    
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
            'start_idx': self.current_episode_data['start_idx'],
            'end_idx': self.current_episode_data['end_idx']
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
        original_action = self.current_episode_data['actions'][self.current_step - 1] if self.current_step - 1 < len(self.current_episode_data['actions']) else action
        
        info = {
            'step': self.current_step - 1,
            'original_action': original_action,
            'reward_components': {
                'queue_utilization': current_state[0] if len(current_state) > 0 else 0.0,
                'action_taken': action,
                'step_reward': reward
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
                'average_reward': np.mean(self.reward_history) if self.reward_history else 0.0,
                'episode_length': len(self.reward_history)
            }
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
