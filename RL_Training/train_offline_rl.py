"""
True Offline RL Training Script for Datacenter Network Deflection Optimization

This script implements TRUE offline reinforcement learning using IQL (Implicit Q-Learning).
Key differences from the previous version:
1. Uses only pre-collected dataset for training (no environment interaction)
2. Implements IQL algorithm suitable for offline RL
3. Uses replay buffer sampling instead of online rollouts
4. Includes offline policy evaluation for model selection

The environment is only used for final evaluation after training is complete.
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
from agents.iql_agent import IQLAgent
from utils.replay_buffer import OfflineReplayBuffer


class OfflineRLTrainer:
    """
    True offline RL trainer using IQL algorithm.
    
    This trainer:
    1. Loads a pre-collected dataset into a replay buffer
    2. Trains IQL agent using only dataset samples (no environment interaction)
    3. Uses offline policy evaluation for model selection
    4. Only touches the environment for final validation
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
        
        # Initialize components
        self.replay_buffer = None
        self.agent = None
        self.env = None  # Only for final evaluation
        
        # Training statistics
        self.training_history = {
            'v_losses': [],
            'q_losses': [],
            'policy_losses': [],
            'total_losses': [],
            'offline_eval_scores': []
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
    
    def load_replay_buffer(self):
        """Load dataset into replay buffer for offline training."""
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        # Define state features (same as environment)
        state_features = ['capacity', 'total_capacity', 'occupancy', 'total_occupancy', 'ttl', 'ooo']
        
        self.replay_buffer = OfflineReplayBuffer(
            dataset_path=self.dataset_path,
            state_features=state_features,
            action_column='action',
            normalize_states=True,
            sequence_column='FlowID',
            time_column='timestamp'
        )
        
        # Log dataset statistics
        stats = self.replay_buffer.get_dataset_stats()
        self.logger.info("Replay buffer statistics:")
        for key, value in stats.items():
            self.logger.info(f"  - {key}: {value}")
        
        return self.replay_buffer
    
    def initialize_agent(self, 
                        lr_q: float = 3e-4,
                        lr_v: float = 3e-4,
                        lr_policy: float = 3e-4,
                        expectile: float = 0.8,
                        temperature: float = 3.0,
                        **kwargs):
        """Initialize the IQL agent for offline RL."""
        if self.replay_buffer is None:
            raise ValueError("Replay buffer must be loaded before agent")
        
        self.logger.info("Initializing IQL agent for offline RL")
        
        # Get dimensions from replay buffer
        state_dim = self.replay_buffer.states.shape[1]
        action_dim = len(np.unique(self.replay_buffer.actions))
        
        self.agent = IQLAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr_q=lr_q,
            lr_v=lr_v,
            lr_policy=lr_policy,
            expectile=expectile,
            temperature=temperature,
            device=self.device,
            **kwargs
        )
        
        self.logger.info(f"IQL agent initialized:")
        self.logger.info(f"  - State dimension: {state_dim}")
        self.logger.info(f"  - Action dimension: {action_dim}")
        self.logger.info(f"  - Device: {self.device}")
        self.logger.info(f"  - Expectile: {expectile}")
        self.logger.info(f"  - Temperature: {temperature}")
    
    def train_step(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Perform one training step using batch from replay buffer.
        
        Args:
            batch_size: Size of batch to sample from replay buffer
            
        Returns:
            Dictionary of training statistics
        """
        try:
            # Sample batch from replay buffer (no environment interaction!)
            batch = self.replay_buffer.sample_batch(batch_size, self.device)
            
            # Update agent using offline data
            losses = self.agent.update(batch)
            
            return losses
            
        except Exception as e:
            self.logger.error(f"Error in train_step: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def offline_policy_evaluation(self, num_samples: int = 1000) -> float:
        """
        Perform offline policy evaluation using Fitted Q Evaluation (FQE).
        
        This estimates the policy's performance without environment interaction
        by evaluating Q-values on dataset states.
        
        Args:
            num_samples: Number of samples to evaluate
            
        Returns:
            Estimated policy value
        """
        # Sample states from replay buffer
        indices = np.random.choice(self.replay_buffer.size, num_samples, replace=True)
        states = self.replay_buffer.states[indices]
        
        total_value = 0.0
        
        with torch.no_grad():
            for state in states:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Get policy action
                action, _ = self.agent.get_action(state, deterministic=True)
                
                # Get Q-value for policy action
                q_values = self.agent.q1(state_tensor)
                q_value = q_values[0, action].item()
                
                total_value += q_value
        
        return total_value / num_samples
    
    def train(self,
              num_steps: int = 100000,
              batch_size: int = 256,
              eval_frequency: int = 5000,
              save_frequency: int = 10000,
              log_frequency: int = 100) -> Dict[str, List]:
        """
        Main offline RL training loop.
        
        Args:
            num_steps: Number of training steps (batches)
            batch_size: Batch size for sampling from replay buffer
            eval_frequency: How often to perform offline evaluation
            save_frequency: How often to save model checkpoints
            log_frequency: How often to log training progress
            
        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting TRUE offline RL training:")
        self.logger.info(f"  - Training steps: {num_steps}")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Log frequency: every {log_frequency} steps")
        self.logger.info(f"  - NO environment interaction during training")
        
        best_eval_score = float('-inf')
        
        self.logger.info("Beginning training loop...")
        
        for step in range(num_steps):
            try:
                # Training step using only replay buffer data
                if step == 0:
                    self.logger.info("Performing first training step...")
                
                losses = self.train_step(batch_size)
                
                if step == 0:
                    self.logger.info(f"First step completed! Losses: {losses}")
                
                # Update training history
                self.training_history['v_losses'].append(losses['v_loss'])
                self.training_history['q_losses'].append(losses['q1_loss'] + losses['q2_loss'])
                self.training_history['policy_losses'].append(losses['policy_loss'])
                self.training_history['total_losses'].append(losses['total_loss'])
                
                # Regular progress logging
                if (step + 1) % log_frequency == 0:
                    self.logger.info(f"Step {step + 1}/{num_steps} - "
                                   f"V: {losses['v_loss']:.4f}, "
                                   f"Q: {losses['q1_loss'] + losses['q2_loss']:.4f}, "
                                   f"Policy: {losses['policy_loss']:.4f}")
                
                # Periodic offline evaluation
                if (step + 1) % eval_frequency == 0:
                    self.logger.info(f"Performing offline evaluation at step {step + 1}")
                    eval_score = self.offline_policy_evaluation()
                    self.training_history['offline_eval_scores'].append(eval_score)
                    
                    self.logger.info(f"=== EVALUATION STEP {step + 1}/{num_steps} ===")
                    self.logger.info(f"  V loss: {losses['v_loss']:.4f}")
                    self.logger.info(f"  Q loss: {losses['q1_loss'] + losses['q2_loss']:.4f}")
                    self.logger.info(f"  Policy loss: {losses['policy_loss']:.4f}")
                    self.logger.info(f"  Offline eval score: {eval_score:.3f}")
                    
                    # Save best model based on offline evaluation
                    if eval_score > best_eval_score:
                        best_eval_score = eval_score
                        best_model_path = self.model_dir / "best_model.pth"
                        self.agent.save_model(str(best_model_path))
                        self.logger.info(f"New best model saved with eval score: {best_eval_score:.3f}")
                
                # Periodic saves
                if (step + 1) % save_frequency == 0:
                    checkpoint_path = self.model_dir / f"checkpoint_step_{step + 1}.pth"
                    self.agent.save_model(str(checkpoint_path))
                    self.logger.info(f"Checkpoint saved: {checkpoint_path}")
                    
            except Exception as e:
                self.logger.error(f"Error at training step {step + 1}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
        
        # Save final model
        final_model_path = self.model_dir / "final_model.pth"
        self.agent.save_model(str(final_model_path))
        
        # Save training history
        self.save_training_history()
        
        self.logger.info("Offline RL training completed!")
        return self.training_history
    
    def initialize_environment_for_eval(self):
        """Initialize environment ONLY for final evaluation after training."""
        self.logger.info("Initializing environment for post-training evaluation only")
        
        self.env = DatacenterDeflectionEnv(
            dataset_path=self.dataset_path,
            normalize_states=True
        )
        
        self.logger.info(f"Environment initialized for evaluation:")
        self.logger.info(f"  - State space: {self.env.observation_space}")
        self.logger.info(f"  - Action space: {self.env.action_space}")
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the trained agent in the environment.
        
        This is the ONLY place where environment interaction happens.
        Used only for final validation after offline training is complete.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions
            
        Returns:
            Evaluation statistics
        """
        if self.env is None:
            self.initialize_environment_for_eval()
        
        self.logger.info(f"Evaluating trained agent for {num_episodes} episodes")
        
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
        """Plot offline RL training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Value loss
        axes[0, 0].plot(self.training_history['v_losses'])
        axes[0, 0].set_title('Value Function Loss')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Q loss
        axes[0, 1].plot(self.training_history['q_losses'])
        axes[0, 1].set_title('Q-Function Loss')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # Policy loss
        axes[1, 0].plot(self.training_history['policy_losses'])
        axes[1, 0].set_title('Policy Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        
        # Offline evaluation scores
        if self.training_history['offline_eval_scores']:
            axes[1, 1].plot(self.training_history['offline_eval_scores'])
            axes[1, 1].set_title('Offline Policy Evaluation')
            axes[1, 1].set_xlabel('Evaluation Point')
            axes[1, 1].set_ylabel('Estimated Value')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training progress plot saved to {save_path}")
        
        return fig


def main():
    """Main training function for true offline RL."""
    parser = argparse.ArgumentParser(description='True Offline RL Training for Datacenter Deflection')
    
    parser.add_argument('--dataset', type=str, 
                       default='/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv',
                       help='Path to the threshold dataset')
    parser.add_argument('--steps', type=int, default=50000,
                       help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size for training')
    parser.add_argument('--lr-q', type=float, default=3e-4,
                       help='Learning rate for Q-networks')
    parser.add_argument('--lr-v', type=float, default=3e-4,
                       help='Learning rate for value network')
    parser.add_argument('--lr-policy', type=float, default=3e-4,
                       help='Learning rate for policy network')
    parser.add_argument('--expectile', type=float, default=0.8,
                       help='Expectile for value function')
    parser.add_argument('--temperature', type=float, default=3.0,
                       help='Temperature for advantage weighting')
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
    
    # Load dataset into replay buffer (NO environment initialization)
    trainer.load_replay_buffer()
    
    # Initialize IQL agent
    trainer.initialize_agent(
        lr_q=args.lr_q,
        lr_v=args.lr_v,
        lr_policy=args.lr_policy,
        expectile=args.expectile,
        temperature=args.temperature
    )
    
    # Train the agent using ONLY the dataset
    training_history = trainer.train(
        num_steps=args.steps,
        batch_size=args.batch_size
    )
    
    # Plot training progress
    plot_path = Path(args.log_dir) / "offline_training_progress.png"
    trainer.plot_training_progress(save_path=plot_path)
    
    # Final evaluation in environment (ONLY post-training validation)
    print(f"\n🔍 FINAL ENVIRONMENT EVALUATION (post-training validation only):")
    print("=" * 70)
    final_eval = trainer.evaluate(num_episodes=20)
    print(f"Mean Reward: {final_eval['mean_reward']:.3f} ± {final_eval['std_reward']:.3f}")
    print(f"Min/Max Reward: {final_eval['min_reward']:.3f} / {final_eval['max_reward']:.3f}")
    print(f"Mean Episode Length: {final_eval['mean_length']:.1f}")
    print("\noffline RL training completed!")

if __name__ == "__main__":
    main()
