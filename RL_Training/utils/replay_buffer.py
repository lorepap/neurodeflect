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
        Compute reward based on actual performance outcomes.
        Uses FCT, packet delay, and OOO flag instead of network state features.
        """
        # Performance metrics (outcomes)
        fct = row['FCT']                                    # Flow completion time
        ooo_flag = row['ooo']                              # Out-of-order flag (0/1)
        packet_delay = row['end_time'] - row['start_time'] # One-way packet delay
        action = row[self.action_column]
        
        # Flow contribution: How much this packet's delay contributes to the overall FCT
        flow_contribution = self._compute_flow_contribution(row)
        
        # Base action rewards
        if action == 0:  # FORWARD
            reward = 0.1  # Normal operation baseline
        elif action == 1:  # DEFLECT  
            reward = -0.05  # Small deflection cost
        else:
            reward = 0.0  # Unknown action
        
        # OUTCOME-BASED PENALTIES:
        
        # 1. FCT PENALTY: Ultimate performance metric
        if fct > 0:
            normalized_fct = min(fct / 0.01, 1.0)  # Normalize to [0,1]
            if action == 1:  # Deflection
                fct_penalty = -0.3 * normalized_fct
                reward += fct_penalty
            else:  # Forward
                fct_penalty = -0.1 * normalized_fct
                reward += fct_penalty
        
        # 2. OUT-OF-ORDER PENALTY: Strong penalty for OOO
        if ooo_flag > 0:
            if action == 1:  # Deflection caused/contributed to OOO
                reward += -0.4  # Strong penalty
            else:  # Forward but still OOO
                reward += -0.2  # Moderate penalty
        
        # 3. PACKET DELAY PENALTY: Penalize high delays
        if packet_delay > 0:
            normalized_delay = min(packet_delay / 0.1, 1.0)  # Normalize to [0,1]
            if action == 1:  # Deflection adds delay
                delay_penalty = -0.2 * normalized_delay
                reward += delay_penalty
        
        # 4. DEFLECTION SUCCESS BONUS: Reward helpful deflection
        if action == 1 and fct > 0.001 and packet_delay > 0.01:
            reward += 0.1  # Bonus for potentially helpful deflection
        
        # 5. FLOW CONTRIBUTION: Reward/penalize based on how packet delay affects FCT
        # Positive contribution = packet hurts flow, negative = packet helps flow
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
    
    def _compute_flow_contribution(self, packet_row) -> float:
        """
        Compute how much this packet's delay contributes to the flow's FCT.
        
        Flow contribution = (packet_delay - avg_flow_packet_delay) / FCT
        
        Positive values mean packet delay is above average for the flow (hurts FCT)
        Negative values mean packet delay is below average for the flow (helps FCT)
        
        Args:
            packet_row: Current packet's data row
            
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
    
    def _process_data(self):
        """Process dataset into (s, a, r, s', done) format."""
        self.logger.info("Processing dataset into replay buffer format")
        
        # Compute derived state features as agreed
        if set(self.state_features) == {'queue_utilization', 'total_occupancy', 'ttl_priority'}:
            self.logger.info("Computing derived state features: queue_utilization, total_occupancy, ttl_priority")
            
            # 1. Queue utilization = occupancy / capacity
            queue_utilization = self.dataset['occupancy'] / (self.dataset['capacity'] + 1e-8)
            
            # 2. Total occupancy (raw value, not ratio)
            total_occupancy = self.dataset['total_occupancy']
            
            # 3. TTL priority = (250 - ttl) / 250 (higher = more hops)
            ttl_priority = (250 - self.dataset['ttl']) / 250.0
            
            # Stack features in the specified order
            state_data = np.column_stack([
                queue_utilization.values,
                total_occupancy.values, 
                ttl_priority.values
            ]).astype(np.float32)
            
        else:
            # Fallback to raw column extraction
            self.logger.info(f"Using raw columns as state features: {self.state_features}")
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
    
    def sample_batch(self, batch_size: int, device: str = "cpu", 
                     stratified: bool = True, deflect_ratio: float = 0.15) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer with optional stratified sampling.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to
            stratified: Whether to use stratified sampling to ensure deflection representation
            deflect_ratio: Target ratio of deflection samples in each batch (0.1-0.2 recommended)
            
        Returns:
            Dictionary containing batch tensors
        """
        if stratified and deflect_ratio > 0:
            # Stratified sampling to ensure deflection representation
            deflect_samples = int(batch_size * deflect_ratio)
            forward_samples = batch_size - deflect_samples
            
            # Get indices for each action type
            deflect_indices = np.where(self.actions == 1)[0]
            forward_indices = np.where(self.actions == 0)[0]
            
            # Sample from each group
            if len(deflect_indices) > 0:
                deflect_batch_indices = np.random.choice(
                    deflect_indices, deflect_samples, replace=True
                )
            else:
                # Fallback if no deflect samples exist
                deflect_batch_indices = np.array([], dtype=int)
                forward_samples = batch_size
            
            forward_batch_indices = np.random.choice(
                forward_indices, forward_samples, replace=True
            )
            
            # Combine and shuffle
            indices = np.concatenate([deflect_batch_indices, forward_batch_indices])
            np.random.shuffle(indices)
            
        else:
            # Standard random sampling
            indices = np.random.choice(self.size, batch_size, replace=True)
        
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(device),
            'actions': torch.LongTensor(self.actions[indices]).to(device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(device),
            'dones': torch.FloatTensor(self.dones[indices]).to(device)
        }
        
        return batch
    
    def sample_batch_advantage_weighted(self, batch_size: int, device: str = "cpu", 
                                       deflect_weight: float = 10.0) -> Dict[str, torch.Tensor]:
        """
        Sample a batch using advantage-weighted sampling to oversample rare deflect transitions.
        
        This method computes sampling weights based on action rarity and potentially 
        advantage estimates to focus on the most informative transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to
            deflect_weight: Weight multiplier for deflection samples (higher = more oversampling)
            
        Returns:
            Dictionary containing batch tensors
        """
        # Compute sampling weights based on action rarity
        weights = np.ones(self.size)
        
        # Heavily weight deflection samples (rare action)
        deflect_mask = (self.actions == 1)
        weights[deflect_mask] *= deflect_weight
        
        # Additional weighting based on reward magnitude (higher rewards = more important)
        # This helps focus on transitions with higher learning value
        abs_rewards = np.abs(self.rewards)
        reward_percentile_75 = np.percentile(abs_rewards, 75)
        high_reward_mask = (abs_rewards >= reward_percentile_75)
        weights[high_reward_mask] *= 2.0
        
        # Normalize weights to probabilities
        weights = weights / weights.sum()
        
        # Sample according to weights
        indices = np.random.choice(self.size, batch_size, replace=True, p=weights)
        
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
