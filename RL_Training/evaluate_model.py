"""
Model Evaluation and Testing Script for Trained Deflection Policies

This script provides comprehensive evaluation capabilities for trained PPO models,
including performance analysis, policy visualization, and comparison with baselines.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

# Add the RL_Training directory to the path
sys.path.append('/home/ubuntu/practical_deflection/RL_Training')

from environments.deflection_env import DatacenterDeflectionEnv
from agents.ppo_agent import PPOAgent


class ModelEvaluator:
    """
    Comprehensive evaluation framework for trained deflection policies.
    """
    
    def __init__(self,
                 model_path: str,
                 dataset_path: str,
                 device: str = "cpu"):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model
            dataset_path: Path to the dataset
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.device = device
        
        # Initialize environment and agent
        self.env = DatacenterDeflectionEnv(dataset_path=dataset_path)
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        
        # Load the trained model
        self.agent.load_model(model_path)
        
        print(f"Model loaded from: {model_path}")
        print(f"Environment initialized with dataset: {dataset_path}")
    
    def evaluate_performance(self, 
                           num_episodes: int = 100,
                           deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate the model's performance over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary of performance metrics
        """
        print(f"Evaluating model performance over {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        action_counts = {0: 0, 1: 0, 2: 0}  # forward, deflect, drop
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                action, _ = self.agent.get_action(state, deterministic=deterministic)
                action_counts[action] += 1
                
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if done or steps >= 100:  # Max steps
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            
            if (episode + 1) % 20 == 0:
                print(f"Episode {episode + 1}/{num_episodes} completed")
        
        # Calculate statistics
        total_actions = sum(action_counts.values())
        action_probabilities = {k: v / total_actions for k, v in action_counts.items()}
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'total_episodes': num_episodes,
            'action_distribution': action_probabilities,
            'raw_rewards': episode_rewards,
            'raw_lengths': episode_lengths
        }
        
        return results
    
    def compare_with_baselines(self, num_episodes: int = 50) -> Dict[str, Dict]:
        """
        Compare the trained model with baseline policies.
        
        Args:
            num_episodes: Number of episodes for comparison
            
        Returns:
            Dictionary with results for each policy
        """
        print(f"Comparing with baseline policies over {num_episodes} episodes...")
        
        policies = {
            'trained_model': self._evaluate_trained_model,
            'always_forward': self._evaluate_always_forward,
            'always_deflect': self._evaluate_always_deflect,
            'random_policy': self._evaluate_random_policy
        }
        
        results = {}
        
        for policy_name, policy_func in policies.items():
            print(f"Evaluating {policy_name}...")
            policy_results = policy_func(num_episodes)
            results[policy_name] = policy_results
        
        return results
    
    def _evaluate_trained_model(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate the trained model."""
        return self.evaluate_performance(num_episodes, deterministic=True)
    
    def _evaluate_always_forward(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate always forward baseline."""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Always choose forward action (0)
                state, reward, done, info = self.env.step(0)
                total_reward += reward
                steps += 1
                
                if done or steps >= 100:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'raw_rewards': episode_rewards,
            'raw_lengths': episode_lengths
        }
    
    def _evaluate_always_deflect(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate always deflect baseline."""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Always choose deflect action (1)
                state, reward, done, info = self.env.step(1)
                total_reward += reward
                steps += 1
                
                if done or steps >= 100:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'raw_rewards': episode_rewards,
            'raw_lengths': episode_lengths
        }
    
    def _evaluate_random_policy(self, num_episodes: int) -> Dict[str, float]:
        """Evaluate random policy baseline."""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # Random action
                action = np.random.choice(3)
                state, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                if done or steps >= 100:
                    break
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'raw_rewards': episode_rewards,
            'raw_lengths': episode_lengths
        }
    
    def analyze_action_patterns(self, num_episodes: int = 50) -> Dict[str, List]:
        """
        Analyze the action patterns of the trained model.
        
        Args:
            num_episodes: Number of episodes to analyze
            
        Returns:
            Dictionary with action pattern analysis
        """
        print(f"Analyzing action patterns over {num_episodes} episodes...")
        
        states_history = []
        actions_history = []
        rewards_history = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_rewards = []
            
            while True:
                action, _ = self.agent.get_action(state, deterministic=True)
                next_state, reward, done, info = self.env.step(action)
                
                episode_states.append(state.copy())
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = next_state
                
                if done or len(episode_states) >= 100:
                    break
            
            states_history.extend(episode_states)
            actions_history.extend(episode_actions)
            rewards_history.extend(episode_rewards)
        
        # Analyze patterns
        state_action_pairs = list(zip(states_history, actions_history))
        
        # Group by action for analysis
        action_groups = {0: [], 1: [], 2: []}
        for state, action in state_action_pairs:
            action_groups[action].append(state)
        
        # Calculate mean state values for each action
        action_state_means = {}
        for action, states in action_groups.items():
            if states:
                action_state_means[action] = np.mean(states, axis=0)
            else:
                action_state_means[action] = np.zeros(len(states_history[0]))
        
        return {
            'states_history': states_history,
            'actions_history': actions_history,
            'rewards_history': rewards_history,
            'action_state_means': action_state_means,
            'action_counts': {action: len(states) for action, states in action_groups.items()}
        }
    
    def plot_evaluation_results(self, 
                              results: Dict[str, Dict],
                              save_path: Optional[str] = None):
        """
        Plot comprehensive evaluation results.
        
        Args:
            results: Results from compare_with_baselines
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for plotting
        policies = list(results.keys())
        mean_rewards = [results[policy]['mean_reward'] for policy in policies]
        std_rewards = [results[policy]['std_reward'] for policy in policies]
        median_rewards = [results[policy]['median_reward'] for policy in policies]
        mean_lengths = [results[policy]['mean_episode_length'] for policy in policies]
        
        # Bar plot of mean rewards with error bars
        axes[0, 0].bar(policies, mean_rewards, yerr=std_rewards, capsize=5)
        axes[0, 0].set_title('Mean Episode Rewards by Policy')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Median rewards comparison
        axes[0, 1].bar(policies, median_rewards)
        axes[0, 1].set_title('Median Episode Rewards by Policy')
        axes[0, 1].set_ylabel('Median Reward')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode lengths comparison
        axes[1, 0].bar(policies, mean_lengths)
        axes[1, 0].set_title('Mean Episode Lengths by Policy')
        axes[1, 0].set_ylabel('Mean Episode Length')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distribution for trained model
        if 'trained_model' in results:
            trained_rewards = results['trained_model']['raw_rewards']
            axes[1, 1].hist(trained_rewards, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Reward Distribution (Trained Model)')
            axes[1, 1].set_xlabel('Episode Reward')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation plot saved to: {save_path}")
        
        return fig
    
    def plot_action_analysis(self, 
                           action_patterns: Dict[str, List],
                           save_path: Optional[str] = None):
        """
        Plot action pattern analysis.
        
        Args:
            action_patterns: Results from analyze_action_patterns
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Action distribution
        actions = action_patterns['actions_history']
        action_names = ['Forward', 'Deflect', 'Drop']
        action_counts = [actions.count(i) for i in range(3)]
        
        axes[0, 0].pie(action_counts, labels=action_names, autopct='%1.1f%%')
        axes[0, 0].set_title('Action Distribution')
        
        # Action sequence over time (first 100 steps)
        if len(actions) > 100:
            axes[0, 1].plot(actions[:100], 'o-', markersize=3)
            axes[0, 1].set_title('Action Sequence (First 100 Steps)')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Action')
            axes[0, 1].set_yticks([0, 1, 2])
            axes[0, 1].set_yticklabels(action_names)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Reward vs Action
        rewards = action_patterns['rewards_history']
        df = pd.DataFrame({'Action': actions, 'Reward': rewards})
        
        # Box plot of rewards by action
        action_rewards = [df[df['Action'] == i]['Reward'].values for i in range(3)]
        axes[1, 0].boxplot(action_rewards, labels=action_names)
        axes[1, 0].set_title('Reward Distribution by Action')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].grid(True, alpha=0.3)
        
        # State features when taking each action
        if 'action_state_means' in action_patterns:
            state_means = action_patterns['action_state_means']
            feature_names = ['Queue Length', 'Arrival Rate', 'Service Rate', 
                           'Util', 'Deflection Threshold', 'Load']
            
            x = np.arange(len(feature_names))
            width = 0.25
            
            for i, (action, mean_state) in enumerate(state_means.items()):
                if len(mean_state) >= len(feature_names):
                    axes[1, 1].bar(x + i * width, mean_state[:len(feature_names)], 
                                  width, label=action_names[action])
            
            axes[1, 1].set_title('Mean State Features by Action')
            axes[1, 1].set_xlabel('State Features')
            axes[1, 1].set_ylabel('Mean Value')
            axes[1, 1].set_xticks(x + width)
            axes[1, 1].set_xticklabels(feature_names, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Action analysis plot saved to: {save_path}")
        
        return fig
    
    def generate_evaluation_report(self, 
                                 output_dir: str,
                                 num_episodes: int = 100) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save the report
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Path to the generated report
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Generating comprehensive evaluation report...")
        
        # Evaluate performance
        performance = self.evaluate_performance(num_episodes)
        
        # Compare with baselines
        comparison = self.compare_with_baselines(num_episodes // 2)
        
        # Analyze action patterns
        action_patterns = self.analyze_action_patterns(num_episodes // 4)
        
        # Create plots
        comparison_plot_path = output_path / "policy_comparison.png"
        self.plot_evaluation_results(comparison, comparison_plot_path)
        
        action_plot_path = output_path / "action_analysis.png"
        self.plot_action_analysis(action_patterns, action_plot_path)
        
        # Generate report text
        report_path = output_path / "evaluation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Datacenter Deflection Policy Evaluation Report\n\n")
            f.write(f"**Model:** {self.model_path}\n")
            f.write(f"**Dataset:** {self.dataset_path}\n")
            f.write(f"**Evaluation Episodes:** {num_episodes}\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write(f"- **Mean Reward:** {performance['mean_reward']:.3f} ± {performance['std_reward']:.3f}\n")
            f.write(f"- **Median Reward:** {performance['median_reward']:.3f}\n")
            f.write(f"- **Min/Max Reward:** {performance['min_reward']:.3f} / {performance['max_reward']:.3f}\n")
            f.write(f"- **Mean Episode Length:** {performance['mean_episode_length']:.1f}\n\n")
            
            f.write("## Action Distribution\n\n")
            action_dist = performance['action_distribution']
            f.write(f"- **Forward:** {action_dist[0]:.1%}\n")
            f.write(f"- **Deflect:** {action_dist[1]:.1%}\n")
            f.write(f"- **Drop:** {action_dist[2]:.1%}\n\n")
            
            f.write("## Baseline Comparison\n\n")
            f.write("| Policy | Mean Reward | Std Reward | Median Reward |\n")
            f.write("|--------|-------------|------------|---------------|\n")
            
            for policy, results in comparison.items():
                f.write(f"| {policy.replace('_', ' ').title()} | "
                       f"{results['mean_reward']:.3f} | "
                       f"{results['std_reward']:.3f} | "
                       f"{results['median_reward']:.3f} |\n")
            
            f.write("\n## Action Pattern Analysis\n\n")
            action_counts = action_patterns['action_counts']
            total_actions = sum(action_counts.values())
            
            f.write("Action frequency during evaluation:\n")
            f.write(f"- **Forward:** {action_counts[0]} ({action_counts[0]/total_actions:.1%})\n")
            f.write(f"- **Deflect:** {action_counts[1]} ({action_counts[1]/total_actions:.1%})\n")
            f.write(f"- **Drop:** {action_counts[2]} ({action_counts[2]/total_actions:.1%})\n\n")
            
            f.write("## Visualizations\n\n")
            f.write(f"![Policy Comparison]({comparison_plot_path.name})\n\n")
            f.write(f"![Action Analysis]({action_plot_path.name})\n\n")
        
        # Save detailed results as JSON
        results_path = output_path / "detailed_results.json"
        detailed_results = {
            'performance': performance,
            'comparison': comparison,
            'action_patterns': {
                'action_counts': action_patterns['action_counts'],
                'action_state_means': {str(k): v.tolist() for k, v in action_patterns['action_state_means'].items()}
            }
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        detailed_results = convert_numpy(detailed_results)
        
        with open(results_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"Evaluation report generated: {report_path}")
        print(f"Detailed results saved: {results_path}")
        
        return str(report_path)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate Trained Deflection Policy')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--dataset', type=str,
                       default='/home/ubuntu/practical_deflection/threshold_combined_dataset.csv',
                       help='Path to the dataset')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes for evaluation')
    parser.add_argument('--output-dir', type=str,
                       default='/home/ubuntu/practical_deflection/RL_Training/evaluation',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        dataset_path=args.dataset,
        device=args.device
    )
    
    # Generate comprehensive evaluation report
    report_path = evaluator.generate_evaluation_report(
        output_dir=args.output_dir,
        num_episodes=args.episodes
    )
    
    print(f"\nEvaluation completed! Report available at: {report_path}")


if __name__ == "__main__":
    main()
