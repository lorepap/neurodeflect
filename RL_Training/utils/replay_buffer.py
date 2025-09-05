"""
Replay Buffer for Offline RL

This buffer loads a pre-collected dataset and provides sampling for offline training.
No new data is collected during training - only the fixed dataset is used.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple, Optional
import logging


class OfflineReplayBuffer:
    """
    Replay buffer for offline RL that loads from a pre-collected dataset.
    
    Maps dataset columns to (state, action, reward, next_state, done) tuples
    and provides batch sampling for training.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 state_features: list,
                 action_column: str = 'action',
                 reward_column: Optional[str] = None,
                 normalize_states: bool = True,
                 sequence_column: str = 'FlowID',
                 time_column: str = 'timestamp'):
        """
        Initialize replay buffer from dataset.
        
        Args:
            dataset_path: Path to CSV dataset
            state_features: List of columns to use as state features
            action_column: Column containing actions
            reward_column: Column containing rewards (if None, computed from environment logic)
            normalize_states: Whether to normalize state features
            sequence_column: Column defining sequences (for determining done flags)
            time_column: Column for ordering within sequences
        """
        self.dataset_path = dataset_path
        self.state_features = state_features
        self.action_column = action_column
        self.reward_column = reward_column
        self.normalize_states = normalize_states
        self.sequence_column = sequence_column
        self.time_column = time_column
        
        self.logger = logging.getLogger(__name__)
        
        # Load and process dataset
        self._load_dataset()
        self._process_data()
        
        self.logger.info(f"Offline replay buffer initialized with {len(self.states)} transitions")
    
    def _load_dataset(self):
        """Load dataset from CSV file."""
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        self.dataset = pd.read_csv(self.dataset_path)
        self.logger.info(f"Loaded {len(self.dataset)} records")
        
        # Sort by sequence and time to ensure proper ordering
        self.dataset = self.dataset.sort_values([self.sequence_column, self.time_column])
    
    def _compute_rewards(self, row: pd.Series) -> float:
        """
        Compute reward for a transition (mirrors the environment reward function).
        
        Args:
            row: Dataset row containing packet information
            
        Returns:
            Computed reward value
        """
        # Extract features for reward computation
        queue_util = (row['occupancy'] / (row['capacity'] + 1e-8))
        total_util = (row['total_occupancy'] / (row['total_capacity'] + 1e-8))
        ttl_priority = (250 - row['ttl']) / 250.0
        ooo_indicator = row['ooo']
        
        # Compute packet delay (time since flow start)
        packet_delay = row['timestamp'] - row['flow_start_time']
        # Normalize by flow completion time (avoid division by zero)
        fct = row['FCT'] if row['FCT'] > 0 else 1e-6
        packet_delay = packet_delay / fct
        
        # Compute FCT contribution
        fct_contribution = packet_delay  # Higher delay = higher FCT impact
        
        # Base action reward
        action = row[self.action_column]
        if action == 0:  # Forward
            reward = 0.1
        elif action == 1:  # Deflect
            reward = -0.1
        else:
            reward = 0.0
        
        # Packet-level penalties
        if ooo_indicator > 0:
            reward += -0.2 * ooo_indicator
        
        if packet_delay > 0:
            reward += -0.1 * abs(packet_delay)
        
        if fct_contribution > 0:
            if action == 1:  # Deflecting critical packet
                reward += -0.15 * fct_contribution
            else:  # Forwarding critical packet
                reward += 0.05 * fct_contribution
        
        return np.clip(reward, -1.0, 1.0)
    
    def _process_data(self):
        """Process dataset into (s, a, r, s', done) format."""
        self.logger.info("Processing dataset into replay buffer format")
        
        # Extract state features
        state_data = self.dataset[self.state_features].values.astype(np.float32)
        
        # Normalize states if requested
        if self.normalize_states:
            self.state_mean = np.mean(state_data, axis=0)
            self.state_std = np.std(state_data, axis=0) + 1e-8
            state_data = (state_data - self.state_mean) / self.state_std
            self.logger.info("States normalized")
        
        # Extract actions
        actions = self.dataset[self.action_column].values.astype(np.int64)
        
        # Compute rewards
        if self.reward_column and self.reward_column in self.dataset.columns:
            rewards = self.dataset[self.reward_column].values.astype(np.float32)
        else:
            self.logger.info("Computing rewards from packet characteristics")
            rewards = np.array([self._compute_rewards(row) for _, row in self.dataset.iterrows()])
        
        # Create next states and done flags
        states = []
        next_states = []
        valid_actions = []
        valid_rewards = []
        dones = []
        
        # Group by sequence to handle episode boundaries
        for sequence_id, group in self.dataset.groupby(self.sequence_column):
            group_indices = group.index.tolist()
            
            for i, idx in enumerate(group_indices):
                # Current state
                states.append(state_data[self.dataset.index.get_loc(idx)])
                valid_actions.append(actions[self.dataset.index.get_loc(idx)])
                valid_rewards.append(rewards[self.dataset.index.get_loc(idx)])
                
                # Next state and done flag
                if i < len(group_indices) - 1:
                    # Not last in sequence - use next state
                    next_idx = group_indices[i + 1]
                    next_states.append(state_data[self.dataset.index.get_loc(next_idx)])
                    dones.append(False)
                else:
                    # Last in sequence - use current state and mark as done
                    next_states.append(state_data[self.dataset.index.get_loc(idx)])
                    dones.append(True)
        
        # Convert to numpy arrays
        self.states = np.array(states, dtype=np.float32)
        self.actions = np.array(valid_actions, dtype=np.int64)
        self.rewards = np.array(valid_rewards, dtype=np.float32)
        self.next_states = np.array(next_states, dtype=np.float32)
        self.dones = np.array(dones, dtype=bool)
        
        self.size = len(self.states)
        
        self.logger.info(f"Processed {self.size} transitions")
        self.logger.info(f"State shape: {self.states.shape}")
        self.logger.info(f"Action distribution: {np.bincount(self.actions)}")
        self.logger.info(f"Reward range: [{self.rewards.min():.3f}, {self.rewards.max():.3f}]")
        self.logger.info(f"Done episodes: {self.dones.sum()}")
    
    def sample_batch(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to
            
        Returns:
            Dictionary containing batch tensors
        """
        # Random sampling
        indices = np.random.choice(self.size, batch_size, replace=True)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.LongTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }
        
        return batch
    
    def get_dataset_stats(self) -> Dict[str, float]:
        """Get statistics about the dataset."""
        return {
            'size': self.size,
            'action_0_pct': (self.actions == 0).mean() * 100,
            'action_1_pct': (self.actions == 1).mean() * 100,
            'mean_reward': self.rewards.mean(),
            'std_reward': self.rewards.std(),
            'episodes': self.dones.sum(),
            'avg_episode_length': self.size / max(self.dones.sum(), 1)
        }
    
    def create_train_val_split(self, val_ratio: float = 0.2) -> Tuple['OfflineReplayBuffer', 'OfflineReplayBuffer']:
        """
        Create train/validation split for offline policy evaluation.
        
        Args:
            val_ratio: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_buffer, val_buffer)
        """
        # Split by episodes to avoid data leakage
        episode_ends = np.where(self.dones)[0]
        n_episodes = len(episode_ends)
        n_val_episodes = int(n_episodes * val_ratio)
        
        # Randomly select validation episodes
        val_episode_indices = np.random.choice(n_episodes, n_val_episodes, replace=False)
        val_episode_ends = episode_ends[val_episode_indices]
        
        # Create episode boundaries
        episode_starts = np.concatenate([[0], episode_ends[:-1] + 1])
        
        # Collect validation indices
        val_indices = []
        for ep_idx in val_episode_indices:
            start = episode_starts[ep_idx]
            end = episode_ends[ep_idx] + 1
            val_indices.extend(range(start, end))
        
        val_indices = np.array(val_indices)
        train_indices = np.setdiff1d(np.arange(self.size), val_indices)
        
        # Create new buffers (simplified - just store indices)
        # For full implementation, you'd create new OfflineReplayBuffer objects
        # with subset data
        
        self.logger.info(f"Created train/val split: {len(train_indices)} train, {len(val_indices)} val")
        
        return train_indices, val_indices
