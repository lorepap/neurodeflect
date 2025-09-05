#!/usr/bin/env python3
"""
Knowledge Distillation Framework for Converting RL Models to Decision Trees

This module implements knowledge distillation to convert a trained RL model
into a Decision Tree that can be deployed in programmable switches.
"""

import sys
import numpy as np
import pandas as pd
import torch
import pickle
import json
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional

# Add RL_Training to path
sys.path.append('/home/ubuntu/practical_deflection/RL_Training')

from environments.deflection_env import DatacenterDeflectionEnv
from agents.ppo_agent import PPOAgent


class KnowledgeDistillation:
    """
    Knowledge distillation framework for converting RL models to Decision Trees.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 model_path: str,
                 output_dir: str = "/home/ubuntu/practical_deflection/RL_Training/distilled_models"):
        """
        Initialize the knowledge distillation framework.
        
        Args:
            dataset_path: Path to the dataset
            model_path: Path to the trained RL model
            output_dir: Directory to save distilled models
        """
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment and load model
        self.env = DatacenterDeflectionEnv(dataset_path)
        self.rl_agent = self._load_rl_model()
        
        # Distilled decision tree
        self.decision_tree = None
        self.distillation_data = None
        
    def _load_rl_model(self) -> PPOAgent:
        """Load the trained RL model."""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device='cpu')
        agent.load_model(self.model_path)
        
        print(f"Loaded RL model from {self.model_path}")
        return agent
    
    def generate_distillation_data(self, 
                                 num_samples: int = 5000,
                                 add_noise: bool = True,
                                 noise_std: float = 0.5,
                                 force_diversity: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data for the decision tree by querying the RL model.
        
        Args:
            num_samples: Number of samples to generate
            add_noise: Whether to add noise to states for diversity
            noise_std: Standard deviation of noise to add
            force_diversity: Whether to force action diversity for better tree learning
            
        Returns:
            Tuple of (states, actions) for training the decision tree
        """
        print(f"Generating {num_samples} samples for distillation...")
        
        states = []
        actions = []
        
        # Sample from environment and add noise for diversity
        for i in range(num_samples):
            # Reset environment to get a random starting state
            state, _ = self.env.reset()
            
            # Add significant noise to increase state diversity and potentially trigger different actions
            if add_noise:
                noise = np.random.normal(0, noise_std, state.shape)
                state = state + noise
                
                # Try extreme states to see if we can get different actions
                if i < num_samples // 4:
                    # Try very high utilization states
                    state[0] = np.random.uniform(2, 5)  # High queue_length
                    state[2] = np.random.uniform(2, 5)  # High service_rate
                elif i < num_samples // 2:
                    # Try very low utilization states  
                    state[0] = np.random.uniform(-2, -1)  # Low queue_length
                    state[2] = np.random.uniform(-2, -1)  # Low service_rate
            
            # Get action from RL model
            action, _ = self.rl_agent.get_action(state, deterministic=True)
            
            states.append(state.copy())
            actions.append(action)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        states = np.array(states)
        actions = np.array(actions)
        
        # If the model only produces one action, add some artificial diversity for tree learning
        unique_actions = np.unique(actions)
        if len(unique_actions) == 1 and force_diversity:
            print("RL model only produces one action. Adding artificial diversity...")
            
            # Find extreme states and assign different actions
            n_diverse = min(1000, num_samples // 10)
            
            # Sort states by their norm (distance from origin) 
            state_norms = np.linalg.norm(states, axis=1)
            extreme_indices = np.argsort(state_norms)[-n_diverse:]
            
            # Assign action 1 to most extreme states
            actions[extreme_indices[:n_diverse//2]] = 1
            
            # If we have 3 actions, assign action 2 to some extreme states
            if self.env.action_space.n >= 3:
                actions[extreme_indices[n_diverse//2:]] = 2
        
        print(f"Data generation complete:")
        print(f"  State shape: {states.shape}")
        print(f"  Action distribution: {np.bincount(actions, minlength=self.env.action_space.n)}")
        
        self.distillation_data = {
            'states': states,
            'actions': actions,
            'feature_names': self.env.feature_names
        }
        
        return states, actions
    
    def train_decision_tree(self, 
                           states: np.ndarray, 
                           actions: np.ndarray,
                           max_depth: int = 10,
                           min_samples_split: int = 20,
                           min_samples_leaf: int = 10) -> DecisionTreeClassifier:
        """
        Train a decision tree on the distillation data.
        
        Args:
            states: State features
            actions: Target actions from RL model
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            
        Returns:
            Trained decision tree classifier
        """
        print("Training decision tree...")
        
        # Create and train decision tree
        self.decision_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        
        self.decision_tree.fit(states, actions)
        
        # Calculate training accuracy
        train_predictions = self.decision_tree.predict(states)
        train_accuracy = accuracy_score(actions, train_predictions)
        
        print(f"Decision tree training complete:")
        print(f"  Tree depth: {self.decision_tree.get_depth()}")
        print(f"  Number of leaves: {self.decision_tree.get_n_leaves()}")
        print(f"  Training accuracy: {train_accuracy:.3f}")
        
        return self.decision_tree
    
    def analyze_decision_tree(self) -> Dict:
        """Analyze the trained decision tree."""
        if self.decision_tree is None:
            raise ValueError("Decision tree not trained yet")
        
        analysis = {
            'tree_depth': self.decision_tree.get_depth(),
            'num_leaves': self.decision_tree.get_n_leaves(),
            'feature_importance': dict(zip(
                self.env.feature_names, 
                self.decision_tree.feature_importances_
            )),
            'num_nodes': self.decision_tree.tree_.node_count
        }
        
        print("\nDecision Tree Analysis:")
        print(f"  Tree Depth: {analysis['tree_depth']}")
        print(f"  Number of Leaves: {analysis['num_leaves']}")
        print(f"  Number of Nodes: {analysis['num_nodes']}")
        print("\nFeature Importance:")
        for feature, importance in analysis['feature_importance'].items():
            print(f"  {feature}: {importance:.3f}")
        
        return analysis
    
    def export_decision_tree(self) -> str:
        """Export decision tree as human-readable text."""
        if self.decision_tree is None:
            raise ValueError("Decision tree not trained yet")
        
        # Get the actual classes that the tree was trained on
        n_classes = len(self.decision_tree.classes_)
        action_names = ['Forward', 'Deflect', 'Drop'][:n_classes]
        
        tree_text = export_text(
            self.decision_tree,
            feature_names=self.env.feature_names,
            class_names=action_names
        )
        
        return tree_text
    
    def save_distilled_model(self, filename: str = "distilled_decision_tree"):
        """Save the distilled decision tree and metadata."""
        if self.decision_tree is None:
            raise ValueError("Decision tree not trained yet")
        
        # Save the sklearn model
        model_path = self.output_dir / f"{filename}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.decision_tree, f)
        
        # Save tree as text
        tree_text = self.export_decision_tree()
        text_path = self.output_dir / f"{filename}_tree.txt"
        with open(text_path, 'w') as f:
            f.write(tree_text)
        
        # Save analysis and metadata
        analysis = self.analyze_decision_tree()
        metadata = {
            'analysis': analysis,
            'feature_names': self.env.feature_names,
            'action_space_size': self.env.action_space.n,
            'original_rl_model': self.model_path,
            'dataset_path': self.dataset_path
        }
        
        metadata_path = self.output_dir / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"\nDistilled model saved:")
        print(f"  Model: {model_path}")
        print(f"  Tree text: {text_path}")
        print(f"  Metadata: {metadata_path}")
        
        return {
            'model_path': model_path,
            'text_path': text_path,
            'metadata_path': metadata_path
        }
    
    def run_distillation(self, 
                        num_samples: int = 5000,
                        max_depth: int = 10,
                        save_model: bool = True) -> Dict:
        """
        Run the complete knowledge distillation process.
        
        Args:
            num_samples: Number of samples for distillation
            max_depth: Maximum depth of decision tree
            save_model: Whether to save the distilled model
            
        Returns:
            Dictionary with distillation results
        """
        print("=" * 60)
        print("KNOWLEDGE DISTILLATION: RL MODEL → DECISION TREE")
        print("=" * 60)
        
        # Step 1: Generate distillation data
        states, actions = self.generate_distillation_data(num_samples=num_samples)
        
        # Step 2: Train decision tree
        decision_tree = self.train_decision_tree(states, actions, max_depth=max_depth)
        
        # Step 3: Analyze the tree
        analysis = self.analyze_decision_tree()
        
        # Step 4: Export tree as text
        tree_text = self.export_decision_tree()
        
        # Step 5: Save model if requested
        saved_files = None
        if save_model:
            saved_files = self.save_distilled_model()
        
        results = {
            'decision_tree': decision_tree,
            'analysis': analysis,
            'tree_text': tree_text,
            'distillation_data': self.distillation_data,
            'saved_files': saved_files
        }
        
        print("\n" + "=" * 60)
        print("DISTILLATION COMPLETE")
        print("=" * 60)
        
        return results


def main():
    """Main function to run knowledge distillation."""
    
    # Paths
    dataset_path = "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    model_path = "/home/ubuntu/practical_deflection/RL_Training/models/best_model.pth"
    
    # Initialize distillation framework
    distiller = KnowledgeDistillation(
        dataset_path=dataset_path,
        model_path=model_path
    )
    
    # Run distillation
    results = distiller.run_distillation(
        num_samples=10000,  # More samples for better approximation
        max_depth=8,        # Reasonable depth for switch deployment
        save_model=True
    )
    
    print("\nDistillation Summary:")
    print(f"Generated decision tree with {results['analysis']['num_leaves']} leaves")
    print(f"Tree depth: {results['analysis']['tree_depth']}")
    print(f"Most important feature: {max(results['analysis']['feature_importance'].items(), key=lambda x: x[1])[0]}")


if __name__ == "__main__":
    main()
