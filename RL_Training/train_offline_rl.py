"""
Offline RL Training Script for Datacenter Network Deflection Optimization

This script implements offline reinforcement learning training using the collected
threshold experiment data to learn optimal packet deflection policies.
"""

import os
import sys
import logging
import argparse
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from pathlib import Path

# Add the RL_Training directory to the path
sys.path.append('/home/ubuntu/practical_deflection/RL_Training')

from environments.deflection_env import DatacenterDeflectionEnv
from agents.ppo_agent import PPOAgent


class OfflineRLTrainer:
    """
    Offline RL trainer for learning deflection policies from collected data.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 log_dir: str = "/home/ubuntu/practical_deflection/RL_Training/logs",
                 model_dir: str = "/home/ubuntu/practical_deflection/RL_Training/models",
                 device: str = "cpu"):
        """
        Initialize the offline RL trainer.
        
        Args:
            dataset_path: Path to the threshold dataset CSV file
            log_dir: Directory for saving logs
            model_dir: Directory for saving models
            device: Device to run training on
        """
        self.dataset_path = dataset_path
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.device = device
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize environment and agent
        self.env = None
        self.agent = None
        self.dataset = None
        
        # Training statistics
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropies': [],
            'learning_rates': []
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"training_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Offline RL training initialized. Logs saved to {log_file}")
    
    def load_dataset(self):
        """Load and preprocess the threshold dataset."""
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        try:
            self.dataset = pd.read_csv(self.dataset_path)
            self.logger.info(f"Dataset loaded successfully with {len(self.dataset)} records")
            
            # Log dataset statistics
            self.logger.info("Dataset statistics:")
            
            # Check for threshold column (handle different names)
            threshold_col = None
            if 'threshold' in self.dataset.columns:
                threshold_col = 'threshold'
            elif 'deflection_threshold' in self.dataset.columns:
                threshold_col = 'deflection_threshold'
            
            if threshold_col:
                self.logger.info(f"  - Unique threshold values: {self.dataset[threshold_col].unique()}")
            else:
                self.logger.info("  - No threshold column found")
            
            self.logger.info(f"  - Available columns: {list(self.dataset.columns)}")
            self.logger.info(f"  - Data shape: {self.dataset.shape}")
            
            return self.dataset
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def initialize_environment(self, normalize_states: bool = True):
        """Initialize the datacenter deflection environment."""
        if self.dataset is None:
            self.load_dataset()
        
        self.logger.info("Initializing datacenter deflection environment")
        
        self.env = DatacenterDeflectionEnv(
            dataset_path=self.dataset_path,
            normalize_states=normalize_states
        )
        
        self.logger.info(f"Environment initialized:")
        self.logger.info(f"  - State space: {self.env.observation_space}")
        self.logger.info(f"  - Action space: {self.env.action_space}")
        self.logger.info(f"  - Dataset size: {len(self.env.features)}")
    
    def initialize_agent(self, 
                        lr_policy: float = 3e-4,
                        lr_value: float = 3e-4,
                        **kwargs):
        """Initialize the PPO agent."""
        if self.env is None:
            raise ValueError("Environment must be initialized before agent")
        
        self.logger.info("Initializing PPO agent")
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_policy=lr_policy,
            lr_value=lr_value,
            device=self.device,
            **kwargs
        )
        
        self.logger.info(f"PPO agent initialized:")
        self.logger.info(f"  - State dimension: {state_dim}")
        self.logger.info(f"  - Action dimension: {action_dim}")
        self.logger.info(f"  - Device: {self.device}")
    
    def collect_episode_data(self, max_steps: int = 100) -> Tuple[List, List, List, List, List, List]:
        """
        Collect data from a single episode in the environment.
        
        Args:
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (states, actions, rewards, log_probs, values, dones)
        """
        states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []
        
        state, info = self.env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Get action and value from agent
            action, log_prob = self.agent.get_action(state)
            value = self.agent.get_value(state)
            
            # Take step in environment
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # Store data
            states.append(state.copy())
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            dones.append(done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        self.logger.debug(f"Episode completed: {len(states)} steps, total reward: {total_reward:.3f}")
        
        return states, actions, rewards, log_probs, values, dones
    
    def train_epoch(self, 
                   num_episodes: int = 10,
                   max_steps_per_episode: int = 100,
                   update_frequency: int = 5) -> Dict[str, float]:
        """
        Train the agent for one epoch.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            update_frequency: How often to update the agent
            
        Returns:
            Dictionary of training statistics
        """
        epoch_rewards = []
        epoch_lengths = []
        
        # Collect data from multiple episodes
        all_states, all_actions, all_rewards = [], [], []
        all_log_probs, all_values, all_dones = [], [], []
        
        for episode in range(num_episodes):
            states, actions, rewards, log_probs, values, dones = self.collect_episode_data(max_steps_per_episode)
            
            # Store episode data
            all_states.extend(states)
            all_actions.extend(actions)
            all_rewards.extend(rewards)
            all_log_probs.extend(log_probs)
            all_values.extend(values)
            all_dones.extend(dones)
            
            # Track episode statistics
            episode_reward = sum(rewards)
            episode_length = len(rewards)
            
            epoch_rewards.append(episode_reward)
            epoch_lengths.append(episode_length)
            
            self.logger.debug(f"Episode {episode + 1}/{num_episodes}: "
                            f"reward={episode_reward:.3f}, length={episode_length}")
        
        # Update agent with collected data
        if len(all_states) > 0:
            training_stats = self.agent.update(
                states=all_states,
                actions=all_actions,
                rewards=all_rewards,
                log_probs=all_log_probs,
                values=all_values,
                dones=all_dones
            )
        else:
            training_stats = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0, 'total_loss': 0}
        
        # Compile epoch statistics
        epoch_stats = {
            'mean_reward': np.mean(epoch_rewards),
            'std_reward': np.std(epoch_rewards),
            'mean_length': np.mean(epoch_lengths),
            'num_episodes': num_episodes,
            **training_stats
        }
        
        return epoch_stats
    
    def train(self,
              num_epochs: int = 100,
              episodes_per_epoch: int = 10,
              max_steps_per_episode: int = 100,
              save_frequency: int = 10,
              evaluation_frequency: int = 5) -> Dict[str, List]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of training epochs
            episodes_per_epoch: Episodes per epoch
            max_steps_per_episode: Maximum steps per episode
            save_frequency: How often to save models
            evaluation_frequency: How often to evaluate
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting offline RL training:")
        self.logger.info(f"  - Epochs: {num_epochs}")
        self.logger.info(f"  - Episodes per epoch: {episodes_per_epoch}")
        self.logger.info(f"  - Max steps per episode: {max_steps_per_episode}")
        
        best_reward = float('-inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train for one epoch
            epoch_stats = self.train_epoch(
                num_episodes=episodes_per_epoch,
                max_steps_per_episode=max_steps_per_episode
            )
            
            # Update training history
            self.training_history['episode_rewards'].append(epoch_stats['mean_reward'])
            self.training_history['episode_lengths'].append(epoch_stats['mean_length'])
            self.training_history['policy_losses'].append(epoch_stats['policy_loss'])
            self.training_history['value_losses'].append(epoch_stats['value_loss'])
            self.training_history['entropies'].append(epoch_stats['entropy'])
            
            # Log progress
            self.logger.info(f"  Mean reward: {epoch_stats['mean_reward']:.3f} ± {epoch_stats['std_reward']:.3f}")
            self.logger.info(f"  Mean length: {epoch_stats['mean_length']:.1f}")
            self.logger.info(f"  Policy loss: {epoch_stats['policy_loss']:.4f}")
            self.logger.info(f"  Value loss: {epoch_stats['value_loss']:.4f}")
            self.logger.info(f"  Entropy: {epoch_stats['entropy']:.4f}")
            
            # Save model if improved
            if epoch_stats['mean_reward'] > best_reward:
                best_reward = epoch_stats['mean_reward']
                best_model_path = self.model_dir / "best_model.pth"
                self.agent.save_model(str(best_model_path))
                self.logger.info(f"New best model saved with reward: {best_reward:.3f}")
            
            # Periodic saves
            if (epoch + 1) % save_frequency == 0:
                checkpoint_path = self.model_dir / f"checkpoint_epoch_{epoch + 1}.pth"
                self.agent.save_model(str(checkpoint_path))
                self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Evaluation
            if (epoch + 1) % evaluation_frequency == 0:
                eval_stats = self.evaluate(num_episodes=5)
                self.logger.info(f"Evaluation - Mean reward: {eval_stats['mean_reward']:.3f}")
        
        # Save final model
        final_model_path = self.model_dir / "final_model.pth"
        self.agent.save_model(str(final_model_path))
        
        # Save training history
        self.save_training_history()
        
        self.logger.info("Training completed!")
        return self.training_history
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
            
        Returns:
            Evaluation statistics
        """
        self.logger.info(f"Evaluating agent for {num_episodes} episodes")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _ = self.agent.get_action(state, deterministic=deterministic)
                state, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if done or truncated or steps >= 100:  # Max steps for evaluation
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }
        
        return eval_stats
    
    def save_training_history(self):
        """Save training history to file."""
        history_file = self.log_dir / "training_history.json"
        
        # Convert numpy types to Python types for JSON serialization
        serializable_history = {}
        for key, values in self.training_history.items():
            serializable_history[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self.logger.info(f"Training history saved to {history_file}")
    
    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True)
        
        # Policy loss
        axes[0, 1].plot(self.training_history['policy_losses'])
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Value loss
        axes[1, 0].plot(self.training_history['value_losses'])
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Entropy
        axes[1, 1].plot(self.training_history['entropies'])
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Entropy')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training progress plot saved to {save_path}")
        
        return fig


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Offline RL Training for Datacenter Deflection')
    
    parser.add_argument('--dataset', type=str, 
                       default='/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv',
                       help='Path to the threshold dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--episodes-per-epoch', type=int, default=10,
                       help='Episodes per epoch')
    parser.add_argument('--max-steps', type=int, default=100,
                       help='Maximum steps per episode')
    parser.add_argument('--lr-policy', type=float, default=3e-4,
                       help='Learning rate for policy network')
    parser.add_argument('--lr-value', type=float, default=3e-4,
                       help='Learning rate for value network')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    parser.add_argument('--log-dir', type=str, 
                       default='/home/ubuntu/practical_deflection/RL_Training/logs',
                       help='Directory for logs')
    parser.add_argument('--model-dir', type=str,
                       default='/home/ubuntu/practical_deflection/RL_Training/models',
                       help='Directory for models')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = OfflineRLTrainer(
        dataset_path=args.dataset,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        device=args.device
    )
    
    # Load dataset and initialize environment
    trainer.load_dataset()
    trainer.initialize_environment()
    
    # Initialize agent
    trainer.initialize_agent(
        lr_policy=args.lr_policy,
        lr_value=args.lr_value
    )
    
    # Train the agent
    training_history = trainer.train(
        num_epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        max_steps_per_episode=args.max_steps
    )
    
    # Plot training progress
    plot_path = Path(args.log_dir) / "training_progress.png"
    trainer.plot_training_progress(save_path=plot_path)
    
    # Final evaluation
    final_eval = trainer.evaluate(num_episodes=20)
    print(f"\nFinal Evaluation Results:")
    print(f"Mean Reward: {final_eval['mean_reward']:.3f} ± {final_eval['std_reward']:.3f}")
    print(f"Min/Max Reward: {final_eval['min_reward']:.3f} / {final_eval['max_reward']:.3f}")
    print(f"Mean Episode Length: {final_eval['mean_length']:.1f}")


if __name__ == "__main__":
    main()
