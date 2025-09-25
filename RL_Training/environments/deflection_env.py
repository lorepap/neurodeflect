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
        # Using the agreed 3-feature state space: queue_utilization, total_occupancy, ttl_priority
        
        # Extract base features
        feature_data = []
        self.feature_names = []
        
        # 1. Queue utilization (fundamental congestion indicator)
        queue_utilization = self.dataset['occupancy'] / (self.dataset['capacity'] + 1e-8)
        feature_data.append(queue_utilization.values)
        self.feature_names.append('queue_utilization')
        
        # 2. Total occupancy (global congestion context - raw value)
        total_occupancy = self.dataset['total_occupancy']
        feature_data.append(total_occupancy.values)
        self.feature_names.append('total_occupancy')
        
        # 3. TTL priority (packet urgency - higher value = more hops traveled)
        ttl_priority = (250 - self.dataset['ttl']) / 250.0
        feature_data.append(ttl_priority.values)
        self.feature_names.append('ttl_priority')
        
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
            'current_step': 0,
            'dataset_indices': list(range(start_idx, end_idx))  # Track dataset row indices
        }
    
    def _compute_reward(self, state: np.ndarray, action: int, next_state: np.ndarray) -> float:
        """
        Performance-based reward function using actual network outcomes.
        State features: [queue_utilization, total_occupancy, ttl_priority]
        
        Key principle: Reward based on OUTCOMES (delays, FCT, OOO), 
        not state features (to avoid double-counting).
        """
        # Get the current dataset row index
        if (hasattr(self, 'current_episode_data') and 
            self.current_episode_data is not None and 
            'dataset_indices' in self.current_episode_data):
            
            step_idx = self.current_step - 1 if self.current_step > 0 else 0
            if step_idx < len(self.current_episode_data['dataset_indices']):
                current_idx = self.current_episode_data['dataset_indices'][step_idx]
            else:
                current_idx = self.current_episode_data['dataset_indices'][-1]
        else:
            # Fallback
            current_idx = 0
        
        # Extract performance metrics from dataset row
        row = self.dataset.iloc[current_idx]
        
        # Performance metrics (outcomes)
        fct = row['FCT']                           # Flow completion time
        ooo_flag = row['ooo']                      # Out-of-order flag (0/1)
        packet_delay = row['end_time'] - row['start_time']  # One-way packet delay
        
        # Flow contribution: How much this packet's delay contributes to the overall FCT
        # This measures whether deflecting this packet helps or hurts the flow
        flow_contribution = self._compute_flow_contribution(row, current_idx)
        
        # Base action rewards
        if action == 0:  # FORWARD
            reward = 0.1  # Normal operation baseline
        elif action == 1:  # DEFLECT  
            reward = -0.05  # Small deflection cost
        else:
            reward = 0.0  # Unknown action
        
        # OUTCOME-BASED PENALTIES (what we actually care about):
        
        # 1. FCT PENALTY: Penalize actions that lead to high FCT
        # FCT is the ultimate performance metric
        if fct > 0:
            # Normalize FCT (typical values range from 1e-5 to 0.002 seconds)
            normalized_fct = min(fct / 0.01, 1.0)  # Cap at 1.0
            
            if action == 1:  # Deflection
                # Deflection might increase FCT, so penalty if FCT is high
                fct_penalty = -0.3 * normalized_fct
                reward += fct_penalty
            else:  # Forward
                # Forwarding should ideally minimize FCT
                # Small penalty if FCT is still high despite forwarding
                fct_penalty = -0.1 * normalized_fct
                reward += fct_penalty
        
        # 2. OUT-OF-ORDER PENALTY: Strong penalty for OOO packets
        # OOO packets hurt application performance significantly
        if ooo_flag > 0:
            if action == 1:  # Deflection caused/contributed to OOO
                reward += -0.4  # Strong penalty
            else:  # Forward but still OOO (might be from previous deflections)
                reward += -0.2  # Moderate penalty
        
        # 3. PACKET DELAY PENALTY: Penalize high one-way delays
        # High delays hurt user experience
        if packet_delay > 0:
            # Normalize delay (typical values range from 1e-5 to 0.1 seconds)
            normalized_delay = min(packet_delay / 0.1, 1.0)  # Cap at 1.0
            
            if action == 1:  # Deflection adds delay
                delay_penalty = -0.2 * normalized_delay
                reward += delay_penalty
            # No penalty for forwarding - delay might be due to congestion
        
        # 4. DEFLECTION SUCCESS BONUS: Reward deflection if it actually helps
        # Only reward deflection if it's in a high-FCT, high-delay context
        # where deflection might be beneficial for load balancing
        if action == 1 and fct > 0.001 and packet_delay > 0.01:
            # Deflection in high-congestion scenario might help overall
            reward += 0.1  # Small bonus for potentially helpful deflection
        
        # 5. FLOW CONTRIBUTION: Reward/penalize based on how packet delay affects FCT
        # Positive contribution = packet helps flow, negative = packet hurts flow
        if flow_contribution != 0:
            if action == 1:  # Deflection
                # If deflection reduces negative flow contribution, reward it
                # If deflection increases positive flow contribution, penalize it
                contribution_reward = -0.15 * flow_contribution
                reward += contribution_reward
            else:  # Forward
                # If forwarding maintains positive flow contribution, small reward
                # If forwarding maintains negative contribution, small penalty
                contribution_reward = 0.05 * max(0, flow_contribution)
                reward += contribution_reward
        
        return np.clip(reward, -1.0, 1.0)
    
    def _compute_flow_contribution(self, packet_row, current_idx: int) -> float:
        """
        Compute how much this packet's delay contributes to the flow's FCT.
        
        Flow contribution = (packet_delay - avg_flow_packet_delay) / FCT
        
        Positive values mean packet delay is above average for the flow (hurts FCT)
        Negative values mean packet delay is below average for the flow (helps FCT)
        
        Args:
            packet_row: Current packet's data row
            current_idx: Index in the dataset
            
        Returns:
            Normalized flow contribution score [-1, 1]
        """
        try:
            flow_id = packet_row['FlowID']
            packet_delay = packet_row['end_time'] - packet_row['start_time']
            fct = packet_row['FCT']
            
            if fct <= 0:
                return 0.0
            
            # Get all packets for this flow
            flow_packets = self.dataset[self.dataset['FlowID'] == flow_id]
            
            if len(flow_packets) <= 1:
                return 0.0  # Single packet flow, no contribution to compute
            
            # Compute average packet delay for this flow
            flow_packet_delays = (flow_packets['end_time'] - flow_packets['start_time'])
            avg_flow_delay = flow_packet_delays.mean()
            
            # Flow contribution: how much this packet deviates from flow average
            delay_deviation = packet_delay - avg_flow_delay
            
            # Normalize by FCT to get relative contribution
            contribution = delay_deviation / fct
            
            # Clip to reasonable range [-1, 1]
            return np.clip(contribution, -1.0, 1.0)
            
        except (KeyError, ZeroDivisionError, ValueError):
            # If any required columns are missing or computation fails
            return 0.0
    
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
