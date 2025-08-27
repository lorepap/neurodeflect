"""
PPO (Proximal Policy Optimization) Implementation for Network Deflection Learning

This module implements PPO for offline reinforcement learning on network 
deflection optimization. The algorithm learns to make deflection decisions
that optimize datacenter network performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from network_rl_framework import (
    NetworkEnvironmentDataset, 
    PolicyNetwork, 
    ValueNetwork,
    NetworkState
)

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # Value loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    
    # Training parameters
    batch_size: int = 64
    mini_batch_size: int = 16
    epochs_per_update: int = 4
    max_grad_norm: float = 0.5
    
    # Network architecture
    hidden_dims: List[int] = None
    state_dim: int = 7
    action_dim: int = 3
    
    # Training schedule
    total_timesteps: int = 100000
    eval_freq: int = 1000
    save_freq: int = 5000
    
    # Data parameters
    train_split: float = 0.8
    normalize_advantages: bool = True
    use_gae: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64]


class PPOAgent:
    """PPO Agent for network deflection optimization."""
    
    def __init__(self, config: PPOConfig, device: str = 'cpu'):
        """
        Initialize PPO agent.
        
        Args:
            config: PPO configuration
            device: Device to run on ('cpu' or 'cuda')
        """
        self.config = config
        self.device = torch.device(device)
        
        # Initialize networks
        self.policy_net = PolicyNetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims
        ).to(self.device)
        
        self.value_net = ValueNetwork(
            state_dim=config.state_dim,
            hidden_dims=config.hidden_dims
        ).to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), 
            lr=config.learning_rate
        )
        
        # Training metrics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'explained_variance': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
        
        logger.info(f"PPO Agent initialized on device: {self.device}")
    
    def compute_gae(self, 
                   rewards: torch.Tensor, 
                   values: torch.Tensor, 
                   dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Reward tensor [batch_size]
            values: Value predictions [batch_size]
            dones: Done flags [batch_size]
        
        Returns:
            Tuple of (advantages, returns)
        """
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for step in reversed(range(batch_size - 1)):
            delta = rewards[step] + self.config.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]
        
        # Handle last step
        advantages[-1] = rewards[-1] - values[-1]
        returns[-1] = rewards[-1]
        
        return advantages, returns
    
    def compute_policy_loss(self, 
                           states: torch.Tensor,
                           actions: torch.Tensor,
                           old_log_probs: torch.Tensor,
                           advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO policy loss.
        
        Args:
            states: State tensor
            actions: Action tensor
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
        
        Returns:
            Tuple of (loss, info_dict)
        """
        # Get current policy outputs
        logits = self.policy_net(states)
        action_probs = torch.softmax(logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        # Compute current log probabilities
        current_log_probs = action_dist.log_prob(actions)
        
        # Compute ratio
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Compute surrogate losses
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
        
        # Policy loss
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus
        entropy = action_dist.entropy().mean()
        entropy_loss = -self.config.entropy_coef * entropy
        
        # Total policy loss
        total_policy_loss = policy_loss + entropy_loss
        
        # Compute statistics
        with torch.no_grad():
            kl_divergence = (old_log_probs - current_log_probs).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()
        
        info = {
            'policy_loss': policy_loss.item(),
            'entropy': entropy.item(),
            'kl_divergence': kl_divergence.item(),
            'clip_fraction': clip_fraction.item()
        }
        
        return total_policy_loss, info
    
    def compute_value_loss(self, 
                          states: torch.Tensor,
                          returns: torch.Tensor) -> torch.Tensor:
        """
        Compute value function loss.
        
        Args:
            states: State tensor
            returns: Target returns
        
        Returns:
            Value loss
        """
        predicted_values = self.value_net(states)
        value_loss = nn.MSELoss()(predicted_values, returns)
        return value_loss
    
    def update_policy(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """
        Update policy and value networks using PPO.
        
        Args:
            batch: Batch of experience data
        
        Returns:
            Training statistics
        """
        states = batch['state'].to(self.device)
        actions = batch['action'].to(self.device)
        rewards = batch['reward'].to(self.device)
        dones = batch['done'].to(self.device)
        
        # Compute values and advantages
        with torch.no_grad():
            values = self.value_net(states)
            
            if self.config.use_gae:
                advantages, returns = self.compute_gae(rewards, values, dones)
            else:
                # Simple advantage computation
                returns = rewards  # Simplified for offline learning
                advantages = returns - values
            
            # Normalize advantages
            if self.config.normalize_advantages and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Get old log probabilities
            logits = self.policy_net(states)
            action_probs = torch.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            old_log_probs = action_dist.log_prob(actions)
        
        # Perform multiple epochs of updates
        total_stats = {key: [] for key in ['policy_loss', 'value_loss', 'entropy', 'kl_divergence', 'clip_fraction']}
        
        batch_size = states.size(0)
        indices = torch.randperm(batch_size)
        
        for epoch in range(self.config.epochs_per_update):
            # Mini-batch updates
            for start_idx in range(0, batch_size, self.config.mini_batch_size):
                end_idx = min(start_idx + self.config.mini_batch_size, batch_size)
                mini_batch_indices = indices[start_idx:end_idx]
                
                # Extract mini-batch
                mini_states = states[mini_batch_indices]
                mini_actions = actions[mini_batch_indices]
                mini_advantages = advantages[mini_batch_indices]
                mini_returns = returns[mini_batch_indices]
                mini_old_log_probs = old_log_probs[mini_batch_indices]
                
                # Compute losses
                policy_loss, policy_info = self.compute_policy_loss(
                    mini_states, mini_actions, mini_old_log_probs, mini_advantages
                )
                value_loss = self.compute_value_loss(mini_states, mini_returns)
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()
                
                # Collect statistics
                total_stats['policy_loss'].append(policy_info['policy_loss'])
                total_stats['value_loss'].append(value_loss.item())
                total_stats['entropy'].append(policy_info['entropy'])
                total_stats['kl_divergence'].append(policy_info['kl_divergence'])
                total_stats['clip_fraction'].append(policy_info['clip_fraction'])
        
        # Average statistics
        avg_stats = {key: np.mean(values) for key, values in total_stats.items()}
        
        # Compute explained variance
        with torch.no_grad():
            predicted_values = self.value_net(states)
            var_y = torch.var(returns)
            explained_var = 1 - torch.var(returns - predicted_values) / (var_y + 1e-8)
            avg_stats['explained_variance'] = explained_var.item()
        
        return avg_stats
    
    def evaluate(self, eval_loader: DataLoader) -> Dict:
        """
        Evaluate the current policy.
        
        Args:
            eval_loader: DataLoader for evaluation data
        
        Returns:
            Evaluation metrics
        """
        self.policy_net.eval()
        self.value_net.eval()
        
        total_reward = 0
        total_samples = 0
        action_counts = np.zeros(self.config.action_dim)
        
        with torch.no_grad():
            for batch in eval_loader:
                states = batch['state'].to(self.device)
                rewards = batch['reward'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Get policy actions
                _, policy_actions = self.policy_net.get_action(states, deterministic=True)
                
                # Accumulate metrics
                total_reward += rewards.sum().item()
                total_samples += states.size(0)
                
                # Count actions
                for action in actions.cpu().numpy():
                    action_counts[action] += 1
        
        self.policy_net.train()
        self.value_net.train()
        
        eval_metrics = {
            'avg_reward': total_reward / total_samples,
            'total_reward': total_reward,
            'action_distribution': action_counts / total_samples,
            'total_samples': total_samples
        }
        
        return eval_metrics
    
    def save_checkpoint(self, filepath: str, epoch: int, stats: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'policy_state_dict': self.policy_net.state_dict(),
            'value_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'config': asdict(self.config),
            'training_stats': stats
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['training_stats']


class NetworkRLTrainer:
    """High-level trainer for network deflection RL."""
    
    def __init__(self, 
                 dataset_path: str,
                 config: PPOConfig,
                 experiment_name: str = "network_deflection_ppo",
                 device: str = 'cpu'):
        """
        Initialize the trainer.
        
        Args:
            dataset_path: Path to the dataset CSV file
            config: PPO configuration
            experiment_name: Name for the experiment
            device: Device to run on
        """
        self.dataset_path = dataset_path
        self.config = config
        self.experiment_name = experiment_name
        self.device = device
        
        # Create experiment directory
        self.experiment_dir = Path(f"experiments/{experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "tensorboard"))
        
        # Save configuration
        with open(self.experiment_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2)
        
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def setup_data(self):
        """Setup training and evaluation datasets."""
        # Load full dataset
        full_dataset = NetworkEnvironmentDataset(self.dataset_path)
        
        # Split into train/eval
        train_size = int(len(full_dataset) * self.config.train_split)
        eval_size = len(full_dataset) - train_size
        
        self.train_dataset, self.eval_dataset = random_split(
            full_dataset, [train_size, eval_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Training samples: {train_size}, Evaluation samples: {eval_size}")
    
    def train(self):
        """Main training loop."""
        # Setup data and agent
        self.setup_data()
        agent = PPOAgent(self.config, self.device)
        
        # Training loop
        global_step = 0
        epoch = 0
        
        logger.info("Starting training...")
        start_time = time.time()
        
        while global_step < self.config.total_timesteps:
            epoch_stats = []
            
            # Training epoch
            for batch_idx, batch in enumerate(self.train_loader):
                # Update policy
                stats = agent.update_policy(batch)
                epoch_stats.append(stats)
                
                global_step += batch['state'].size(0)
                
                # Log statistics
                if global_step % 100 == 0:
                    avg_stats = {key: np.mean([s[key] for s in epoch_stats[-10:]]) 
                               for key in stats.keys()}
                    
                    for key, value in avg_stats.items():
                        self.writer.add_scalar(f"train/{key}", value, global_step)
                    
                    logger.info(f"Step {global_step}: Policy Loss: {avg_stats['policy_loss']:.4f}, "
                              f"Value Loss: {avg_stats['value_loss']:.4f}, "
                              f"Entropy: {avg_stats['entropy']:.4f}")
                
                # Evaluation
                if global_step % self.config.eval_freq == 0:
                    eval_metrics = agent.evaluate(self.eval_loader)
                    
                    for key, value in eval_metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f"eval/{key}", value, global_step)
                    
                    logger.info(f"Evaluation - Avg Reward: {eval_metrics['avg_reward']:.4f}")
                
                # Save checkpoint
                if global_step % self.config.save_freq == 0:
                    checkpoint_path = self.experiment_dir / f"checkpoint_{global_step}.pt"
                    agent.save_checkpoint(str(checkpoint_path), epoch, epoch_stats)
                
                if global_step >= self.config.total_timesteps:
                    break
            
            epoch += 1
        
        # Final save
        final_checkpoint = self.experiment_dir / "final_model.pt"
        agent.save_checkpoint(str(final_checkpoint), epoch, epoch_stats)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Close tensorboard writer
        self.writer.close()
        
        return agent


if __name__ == "__main__":
    # Example training script
    print("PPO Network Deflection Trainer")
    print("=" * 40)
    
    # Configuration
    config = PPOConfig(
        learning_rate=3e-4,
        batch_size=64,
        total_timesteps=50000,
        eval_freq=1000
    )
    
    # Check if dataset exists
    dataset_path = "../Omnet_Sims/dc_simulations/simulations/sims/dataset_output/combined_threshold_dataset.csv"
    
    if Path(dataset_path).exists():
        print(f"Found dataset at {dataset_path}")
        
        # Initialize trainer
        trainer = NetworkRLTrainer(
            dataset_path=dataset_path,
            config=config,
            experiment_name="test_network_deflection",
            device='cpu'
        )
        
        print("Starting training...")
        agent = trainer.train()
        print("Training completed!")
        
    else:
        print(f"Dataset not found at {dataset_path}")
        print("Please run the threshold pipeline first to generate the dataset.")
        print("\nTo test the framework:")
        print("1. Run: cd ../Omnet_Sims/dc_simulations/simulations/sims")
        print("2. Run: python3 threshold_pipeline.py --thresholds 0.3,0.5,0.7,0.9")
        print("3. Then run this training script")
