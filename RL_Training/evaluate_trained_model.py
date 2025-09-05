#!/usr/bin/env python3
"""
Evaluate the trained offline RL model and compare with threshold-based policies.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add RL_Training to path
sys.path.append('/home/ubuntu/practical_deflection/RL_Training')

from environments.deflection_env import DatacenterDeflectionEnv
from agents.ppo_agent import PPOAgent

def evaluate_trained_model():
    """Evaluate the trained model and compare with baseline policies."""
    
    # Load environment
    dataset_path = "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    env = DatacenterDeflectionEnv(dataset_path)
    
    # Load trained agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device='cpu')
    model_path = "/home/ubuntu/practical_deflection/RL_Training/models/best_model.pth"
    agent.load_model(model_path)
    
    print("Evaluating trained RL model...")
    
    # Test the trained model
    rl_rewards = []
    rl_actions = []
    
    for episode in range(10):
        state, info = env.reset()
        episode_reward = 0
        episode_actions = []
        
        for step in range(50):
            action, _ = agent.get_action(state, deterministic=True)
            state, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_actions.append(action)
            
            if done or truncated:
                break
        
        rl_rewards.append(episode_reward)
        rl_actions.extend(episode_actions)
    
    # Compare with baseline policies
    baseline_policies = {
        'Always Forward (0)': 0,
        'Always Deflect (1)': 1,
        'Random': 'random'
    }
    
    baseline_results = {}
    
    for policy_name, policy_action in baseline_policies.items():
        print(f"Evaluating {policy_name} policy...")
        
        policy_rewards = []
        policy_actions = []
        
        for episode in range(10):
            state, info = env.reset()
            episode_reward = 0
            episode_actions = []
            
            for step in range(50):
                if policy_action == 'random':
                    action = np.random.randint(0, 3)
                else:
                    action = policy_action
                
                state, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_actions.append(action)
                
                if done or truncated:
                    break
            
            policy_rewards.append(episode_reward)
            policy_actions.extend(episode_actions)
        
        baseline_results[policy_name] = {
            'rewards': policy_rewards,
            'actions': policy_actions,
            'mean_reward': np.mean(policy_rewards),
            'std_reward': np.std(policy_rewards)
        }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Trained RL Model:")
    print(f"  Mean Reward: {np.mean(rl_rewards):.3f} ± {np.std(rl_rewards):.3f}")
    print(f"  Action Distribution: {np.bincount(rl_actions, minlength=3)}")
    
    print("\nBaseline Policies:")
    for policy_name, results in baseline_results.items():
        print(f"  {policy_name}:")
        print(f"    Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"    Action Distribution: {np.bincount(results['actions'], minlength=3)}")
    
    # Calculate improvement
    best_baseline = max(baseline_results.values(), key=lambda x: x['mean_reward'])
    improvement = np.mean(rl_rewards) - best_baseline['mean_reward']
    improvement_pct = (improvement / abs(best_baseline['mean_reward'])) * 100
    
    print(f"\nRL Model Improvement over Best Baseline:")
    print(f"  Absolute: {improvement:.3f}")
    print(f"  Percentage: {improvement_pct:.1f}%")
    
    return {
        'rl_rewards': rl_rewards,
        'rl_actions': rl_actions,
        'baseline_results': baseline_results,
        'improvement': improvement,
        'improvement_pct': improvement_pct
    }

if __name__ == "__main__":
    results = evaluate_trained_model()
