#!/usr/bin/env python3
"""
Network Deflection RL Training Script

This script orchestrates the complete training pipeline for learning
optimal deflection policies in datacenter networks.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from RL_Training.network_rl_framework import NetworkEnvironmentDataset
from RL_Training.ppo_trainer import PPOAgent, PPOConfig, NetworkRLTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def check_dataset(dataset_path: str) -> bool:
    """Check if dataset exists and is valid."""
    if not Path(dataset_path).exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return False
    
    try:
        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)} features")
        logger.info(f"Columns: {list(df.columns)}")
        return True
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return False


def create_experiment_config(args) -> PPOConfig:
    """Create PPO configuration from arguments."""
    config = PPOConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        epochs_per_update=args.epochs_per_update,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        train_split=args.train_split,
        hidden_dims=[int(x) for x in args.hidden_dims.split(',')],
        normalize_advantages=args.normalize_advantages,
        use_gae=args.use_gae
    )
    return config


def analyze_dataset(dataset_path: str):
    """Analyze the dataset and print statistics."""
    df = pd.read_csv(dataset_path)
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    
    print("\nFeature statistics:")
    print(df.describe())
    
    if 'action' in df.columns:
        print("\nAction distribution:")
        action_counts = df['action'].value_counts().sort_index()
        for action, count in action_counts.items():
            percentage = count / len(df) * 100
            print(f"  Action {action}: {count} ({percentage:.1f}%)")
    
    if 'reward' in df.columns:
        print(f"\nReward statistics:")
        print(f"  Mean: {df['reward'].mean():.4f}")
        print(f"  Std: {df['reward'].std():.4f}")
        print(f"  Min: {df['reward'].min():.4f}")
        print(f"  Max: {df['reward'].max():.4f}")
    
    if 'threshold' in df.columns:
        print(f"\nThreshold distribution:")
        threshold_counts = df['threshold'].value_counts().sort_index()
        for threshold, count in threshold_counts.items():
            percentage = count / len(df) * 100
            print(f"  Threshold {threshold}: {count} ({percentage:.1f}%)")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for network deflection optimization')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, 
                       default='../Omnet_Sims/dc_simulations/simulations/sims/dataset_output/combined_threshold_dataset.csv',
                       help='Path to the dataset CSV file')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze the dataset without training')
    
    # Training arguments
    parser.add_argument('--experiment-name', type=str, default='network_deflection_rl',
                       help='Name for the experiment')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    # PPO hyperparameters
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--clip-epsilon', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--mini-batch-size', type=int, default=16,
                       help='Mini-batch size for PPO updates')
    parser.add_argument('--epochs-per-update', type=int, default=4,
                       help='Number of epochs per PPO update')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='Total training timesteps')
    parser.add_argument('--eval-freq', type=int, default=1000,
                       help='Evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=5000,
                       help='Model save frequency')
    
    # Data processing
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Training data split ratio')
    parser.add_argument('--normalize-advantages', action='store_true', default=True,
                       help='Normalize advantages')
    parser.add_argument('--use-gae', action='store_true', default=True,
                       help='Use Generalized Advantage Estimation')
    
    # Network architecture
    parser.add_argument('--hidden-dims', type=str, default='128,64',
                       help='Hidden layer dimensions (comma-separated)')
    
    # Resume training
    parser.add_argument('--resume', type=str,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Check dataset
    if not check_dataset(args.dataset):
        return 1
    
    # Analyze dataset
    analyze_dataset(args.dataset)
    
    if args.analyze_only:
        return 0
    
    # Create configuration
    config = create_experiment_config(args)
    
    logger.info("Training Configuration:")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Batch Size: {config.batch_size}")
    logger.info(f"  Total Timesteps: {config.total_timesteps}")
    logger.info(f"  Hidden Dimensions: {config.hidden_dims}")
    
    try:
        # Initialize trainer
        trainer = NetworkRLTrainer(
            dataset_path=args.dataset,
            config=config,
            experiment_name=args.experiment_name,
            device=device
        )
        
        # Train the agent
        logger.info("Starting training...")
        agent = trainer.train()
        
        logger.info("Training completed successfully!")
        
        # Final evaluation
        trainer.setup_data()
        final_eval = agent.evaluate(trainer.eval_loader)
        
        logger.info("Final Evaluation Results:")
        for key, value in final_eval.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, np.ndarray):
                logger.info(f"  {key}: {value}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
