"""
Proximal Policy Optimization (PPO) Agent for Datacenter Network Deflection

This module implements a PPO agent specifically designed for learning
optimal packet deflection policies in datacenter networks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from collections import deque


class PolicyNetwork(nn.Module):
    """
    Neural network for the PPO policy.
    Takes network state as input and outputs action probabilities.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions
        """
        super(PolicyNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)  # Light dropout for regularization
            ])
            input_dim = hidden_dim
        
        # Output layer for action probabilities
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        logits = self.network(state)
        return F.softmax(logits, dim=-1)
    
    def get_action_and_log_prob(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action and its log probability for the given state.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action, log_probability)
        """
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Neural network for the PPO value function.
    Estimates the value of a given state.
    """
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 64]):
        """
        Initialize the value network.
        
        Args:
            state_dim: Dimension of the state space
            hidden_dims: List of hidden layer dimensions
        """
        super(ValueNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        # Output layer for value estimation
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)


class PPOAgent:
    """
    Proximal Policy Optimization agent for datacenter network deflection optimization.
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 lr_policy: float = 3e-4,
                 lr_value: float = 3e-4,
                 gamma: float = 0.99,
                 lambda_gae: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            lr_policy: Learning rate for policy network
            lr_value: Learning rate for value network
            gamma: Discount factor
            lambda_gae: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            entropy_coeff: Entropy regularization coefficient
            value_loss_coeff: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on (cpu/cuda)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': deque(maxlen=100),
            'value_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'total_loss': deque(maxlen=100)
        }
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Get action for the given state.
        
        Args:
            state: Current state
            deterministic: If True, select action deterministically
            
        Returns:
            Tuple of (action, log_probability)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
                log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1))).squeeze()
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for the given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.value_net(state_tensor)
        
        return value.item()
    
    def compute_gae_advantages(self, 
                              rewards: List[float], 
                              values: List[float], 
                              dones: List[bool]) -> Tuple[List[float], List[float]]:
        """
        Compute Generalized Advantage Estimation (GAE) advantages and returns.
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            
        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = 0
        
        # Process in reverse order
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[i]
                next_value = 0  # Terminal state
            else:
                next_non_terminal = 1.0 - dones[i]
                next_value = values[i + 1]
            
            delta = rewards[i] + self.gamma * next_value * next_non_terminal - values[i]
            gae = delta + self.gamma * self.lambda_gae * next_non_terminal * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def update(self, 
               states: List[np.ndarray],
               actions: List[int],
               rewards: List[float],
               log_probs: List[float],
               values: List[float],
               dones: List[bool],
               epochs: int = 10,
               batch_size: int = 64) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.
        
        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            log_probs: List of log probabilities
            values: List of value estimates
            dones: List of done flags
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dictionary of training statistics
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        values_tensor = torch.FloatTensor(values).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae_advantages(rewards, values, dones)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Training loop
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for epoch in range(epochs):
            # Create random indices for batch sampling
            indices = torch.randperm(len(states)).to(self.device)
            
            for start_idx in range(0, len(states), batch_size):
                end_idx = min(start_idx + batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass through networks
                action_probs = self.policy_net(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                current_values = self.value_net(batch_states).squeeze()
                
                # Compute PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(current_values, batch_returns)
                
                # Total loss
                total_loss = (policy_loss + 
                             self.value_loss_coeff * value_loss - 
                             self.entropy_coeff * entropy)
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss_for_backward = policy_loss - self.entropy_coeff * entropy
                policy_loss_for_backward.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        # Calculate averages
        num_updates = epochs * (len(states) // batch_size + (1 if len(states) % batch_size > 0 else 0))
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        avg_total_loss = avg_policy_loss + self.value_loss_coeff * avg_value_loss
        
        # Update statistics
        self.training_stats['policy_loss'].append(avg_policy_loss)
        self.training_stats['value_loss'].append(avg_value_loss)
        self.training_stats['entropy'].append(avg_entropy)
        self.training_stats['total_loss'].append(avg_total_loss)
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'total_loss': avg_total_loss
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'training_stats': dict(self.training_stats)
        }, filepath)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            for key, values in checkpoint['training_stats'].items():
                self.training_stats[key] = deque(values, maxlen=100)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get recent training statistics."""
        return {
            key: np.mean(values) if values else 0.0 
            for key, values in self.training_stats.items()
        }
