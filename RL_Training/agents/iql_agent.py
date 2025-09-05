"""
Implicit Q-Learning (IQL) Agent for Offline RL

IQL is a stable offline RL algorithm that learns:
1. Value function V(s) via expectile regression
2. Q-function Q(s,a) via standard TD learning  
3. Policy π(a|s) via advantage-weighted regression

This avoids the distribution shift issues of on-policy methods like PPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List
import logging


class MLP(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256], 
                 output_activation: str = None):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if output_activation == 'softmax':
            layers.append(nn.Softmax(dim=-1))
            
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class IQLAgent:
    """
    Implicit Q-Learning agent for offline RL.
    
    IQL avoids distribution shift by:
    1. Learning value function via expectile regression (no max over actions)
    2. Training policy via advantage-weighted regression (stays close to data)
    3. Using separate Q-networks for stability
    """
    
    def __init__(self, state_dim: int, action_dim: int, lr_q: float = 3e-4,
                 lr_v: float = 3e-4, lr_policy: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, expectile: float = 0.8, temperature: float = 3.0,
                 hidden_dim: int = 256, device: str = 'cpu'):
        """
        Initialize IQL agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            lr_q: Learning rate for Q-networks
            lr_v: Learning rate for value network
            lr_policy: Learning rate for policy network
            gamma: Discount factor
            tau: Target network update rate
            expectile: Expectile parameter for value function (0.8 = 80th percentile)
            temperature: Temperature for advantage weighting (beta)
            hidden_dim: Hidden layer dimension
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.beta = temperature  # Store temperature as beta for advantage weighting
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize networks
        self.q1 = MLP(state_dim, action_dim, [hidden_dim]).to(device)
        self.q2 = MLP(state_dim, action_dim, [hidden_dim]).to(device)
        self.v = MLP(state_dim, 1, [hidden_dim]).to(device)
        self.policy = MLP(state_dim, action_dim, [hidden_dim], output_activation='softmax').to(device)
        
        # Initialize target networks
        self.q1_target = MLP(state_dim, action_dim, [hidden_dim]).to(device)
        self.q2_target = MLP(state_dim, action_dim, [hidden_dim]).to(device)
        self.v_target = MLP(state_dim, 1, [hidden_dim]).to(device)
        
        # Copy main networks to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.v_target.load_state_dict(self.v.state_dict())
        
        # Initialize optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr_q)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr_q)
        self.v_optimizer = optim.Adam(self.v.parameters(), lr=lr_v)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_policy)
        
        self.logger.info(f"IQL agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Get action from policy.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_prob)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.policy(state_tensor)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.log_softmax(logits, dim=-1)[0, action]
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).squeeze()
                log_prob = torch.log(probs[0, action])
        
        return action.item(), log_prob.item()
    
    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.v(state_tensor)
        
        return value.item()
    
    def expectile_loss(self, diff: torch.Tensor, expectile: float) -> torch.Tensor:
        """Compute expectile loss for value function."""
        weight = torch.where(diff > 0, expectile, 1 - expectile)
        return weight * (diff ** 2)
    
    def update_v(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update value function via expectile regression."""
        try:
            with torch.no_grad():
                q1_values = self.q1_target(states)
                q2_values = self.q2_target(states)
                q_values = torch.min(q1_values, q2_values)
                target_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()
            
            current_values = self.v(states).squeeze()
            value_loss = self.expectile_loss(target_values - current_values, self.expectile).mean()
            
            # Check for NaN
            if torch.isnan(value_loss):
                self.logger.warning("NaN detected in value loss")
                return 0.0
            
            self.v_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v.parameters(), 1.0)  # Gradient clipping
            self.v_optimizer.step()
            
            return value_loss.item()
            
        except Exception as e:
            self.logger.error(f"Error in update_v: {e}")
            return 0.0
    
    def update_q(self, states: torch.Tensor, actions: torch.Tensor, 
                 rewards: torch.Tensor, next_states: torch.Tensor, 
                 dones: torch.Tensor) -> tuple:
        """Update Q-functions via temporal difference."""
        try:
            with torch.no_grad():
                next_values = self.v_target(next_states).squeeze()
                q_targets = rewards + self.gamma * next_values * (1 - dones)
            
            current_q1 = self.q1(states).gather(1, actions.unsqueeze(1)).squeeze()
            current_q2 = self.q2(states).gather(1, actions.unsqueeze(1)).squeeze()
            
            q1_loss = F.mse_loss(current_q1, q_targets)
            q2_loss = F.mse_loss(current_q2, q_targets)
            
            # Check for NaN
            if torch.isnan(q1_loss) or torch.isnan(q2_loss):
                self.logger.warning("NaN detected in Q loss")
                return 0.0, 0.0
            
            # Update Q1
            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
            self.q1_optimizer.step()
            
            # Update Q2
            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
            self.q2_optimizer.step()
            
            return q1_loss.item(), q2_loss.item()
            
        except Exception as e:
            self.logger.error(f"Error in update_q: {e}")
            return 0.0, 0.0
    
    def update_policy(self, states: torch.Tensor, actions: torch.Tensor) -> float:
        """Update policy via advantage-weighted regression."""
        try:
            with torch.no_grad():
                v_values = self.v(states).squeeze()
                q1_values = self.q1(states)
                q2_values = self.q2(states) 
                q_values = torch.min(q1_values, q2_values)
                
                advantages = q_values.gather(1, actions.unsqueeze(1)).squeeze() - v_values
                weights = torch.exp(advantages / self.beta).clamp(max=100.0)  # Clamp weights
            
            action_probs = self.policy(states)
            log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze() + 1e-8)
            
            policy_loss = -(weights * log_probs).mean()
            
            # Check for NaN
            if torch.isnan(policy_loss):
                self.logger.warning("NaN detected in policy loss")
                return 0.0
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.policy_optimizer.step()
            
            return policy_loss.item()
            
        except Exception as e:
            self.logger.error(f"Error in update_policy: {e}")
            return 0.0
    
    def update_targets(self):
        """Soft update of target networks."""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for param, target_param in zip(self.v.parameters(), self.v_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update all networks with a batch of data.
        
        Args:
            batch: Dictionary containing states, actions, rewards, next_states, dones
            
        Returns:
            Dictionary of loss values
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Update value function
        v_loss = self.update_v(states, actions)
        
        # Update Q-functions
        q1_loss, q2_loss = self.update_q(states, actions, rewards, next_states, dones)
        
        # Update policy
        policy_loss = self.update_policy(states, actions)
        
        # Update target networks
        self.update_targets()
        
        return {
            'v_loss': v_loss,
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'policy_loss': policy_loss,
            'total_loss': v_loss + q1_loss + q2_loss + policy_loss
        }
    
    def save_model(self, path: str):
        """Save model parameters."""
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'v': self.v.state_dict(),
            'policy': self.policy.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.v.load_state_dict(checkpoint['v'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.logger.info(f"Model loaded from {path}")
