#!/usr/bin/env python3
"""
Knowledge Distillation: RL Model to Decision Tree

This module implements knowledge distillation to convert a trained RL model
into a Decision Tree that can be deployed in programmable switches.
"""

import sys
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import logging

# Add RL_Training to path
sys.path.append('/home/ubuntu/practical_deflection/RL_Training')

from environments.deflection_env import DatacenterDeflectionEnv
from agents.ppo_agent import PPOAgent


class RLToDecisionTreeDistiller:
    """
    Distills knowledge from a trained RL model to a Decision Tree
    suitable for deployment in programmable switches.
    """
    
    def __init__(self, 
                 rl_model_path: str,
                 dataset_path: str,
                 output_dir: str = "/home/ubuntu/practical_deflection/RL_Training/distilled_models"):
        """
        Initialize the distillation framework.
        
        Args:
            rl_model_path: Path to the trained RL model
            dataset_path: Path to the dataset for generating synthetic data
            output_dir: Directory to save distilled models
        """
        self.rl_model_path = rl_model_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment and RL agent
        self.env = None
        self.rl_agent = None
        self.decision_tree = None
        
        # Distillation data
        self.distillation_states = None
        self.distillation_actions = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_rl_model(self):
        """Load the trained RL model."""
        self.logger.info(f"Loading RL model from {self.rl_model_path}")
        
        # Initialize environment to get state/action dimensions
        self.env = DatacenterDeflectionEnv(self.dataset_path)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        # Load RL agent
        self.rl_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device='cpu')
        self.rl_agent.load_model(self.rl_model_path)
        
        self.logger.info("RL model loaded successfully")
        
    def generate_distillation_data(self, 
                                 num_samples: int = 10000,
                                 exploration_noise: float = 0.1,
                                 force_diversity: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data by querying the RL model on diverse states.
        
        Args:
            num_samples: Number of state-action pairs to generate
            exploration_noise: Noise level for state exploration
            force_diversity: Whether to force action diversity if model is too deterministic
            
        Returns:
            Tuple of (states, actions) for training the decision tree
        """
        self.logger.info(f"Generating {num_samples} distillation samples")
        
        if self.rl_agent is None:
            self.load_rl_model()
        
        states = []
        actions = []
        
        # Strategy 1: Sample from dataset and add noise
        dataset_states = self.env.features
        dataset_size = len(dataset_states)
        
        for i in range(num_samples):
            if i % 1000 == 0:
                self.logger.info(f"Generated {i}/{num_samples} samples")
            
            # Sample a base state from the dataset
            base_idx = np.random.randint(0, dataset_size)
            base_state = dataset_states[base_idx].copy()
            
            # Add exploration noise
            if exploration_noise > 0:
                noise = np.random.normal(0, exploration_noise, size=base_state.shape)
                noisy_state = base_state + noise
            else:
                noisy_state = base_state
            
            # Get action from RL model (use stochastic policy for diversity)
            action, _ = self.rl_agent.get_action(noisy_state, deterministic=False)
            
            states.append(noisy_state)
            actions.append(action)
        
        # Strategy 2: Add some completely random states for better coverage
        random_samples = min(2000, num_samples // 5)
        self.logger.info(f"Adding {random_samples} random exploration samples")
        
        for i in range(random_samples):
            # Generate random state within reasonable bounds
            # Assume normalized states are roughly in [-3, 3] range
            random_state = np.random.uniform(-3, 3, size=self.env.observation_space.shape[0])
            action, _ = self.rl_agent.get_action(random_state, deterministic=False)
            
            states.append(random_state)
            actions.append(action)
        
        # Strategy 3: Force diversity if the model is too deterministic
        if force_diversity:
            action_counts = np.bincount(actions, minlength=3)
            total_actions = len(actions)
            
            # If any action is underrepresented (<5%), add synthetic examples
            for action_idx in range(3):
                if action_counts[action_idx] < total_actions * 0.05:
                    needed_samples = int(total_actions * 0.1)  # Target 10% for each action
                    self.logger.info(f"Adding {needed_samples} synthetic samples for action {action_idx}")
                    
                    for _ in range(needed_samples):
                        # Create states that might favor this action
                        if action_idx == 1:  # Deflect
                            # High utilization scenarios
                            synthetic_state = np.random.normal([2, 1, 2, 1, -1, 1], 0.5)
                        elif action_idx == 2:  # Drop
                            # Very high congestion scenarios
                            synthetic_state = np.random.normal([3, 2, 3, 2, 1, 2], 0.5)
                        else:  # Forward
                            # Normal operation scenarios
                            synthetic_state = np.random.normal([0, 0, 0, 0, 0, 0], 0.5)
                        
                        states.append(synthetic_state)
                        actions.append(action_idx)
        
        self.distillation_states = np.array(states)
        self.distillation_actions = np.array(actions)
        
        self.logger.info(f"Generated distillation dataset:")
        self.logger.info(f"  States shape: {self.distillation_states.shape}")
        self.logger.info(f"  Actions shape: {self.distillation_actions.shape}")
        self.logger.info(f"  Action distribution: {np.bincount(self.distillation_actions, minlength=3)}")
        
        return self.distillation_states, self.distillation_actions
    
    def train_decision_tree(self, 
                          max_depth: Optional[int] = 10,
                          min_samples_split: int = 100,
                          min_samples_leaf: int = 50,
                          **kwargs) -> Dict:
        """
        Train a decision tree to mimic the RL model's behavior.
        
        Args:
            max_depth: Maximum depth of the decision tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            
        Returns:
            Training statistics
        """
        if self.distillation_states is None:
            self.generate_distillation_data()
        
        self.logger.info("Training decision tree")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            self.distillation_states, 
            self.distillation_actions,
            test_size=0.2,
            random_state=42,
            stratify=self.distillation_actions
        )
        
        # Train decision tree
        self.decision_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            **kwargs
        )
        
        self.decision_tree.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.decision_tree.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        stats = {
            'accuracy': accuracy,
            'tree_depth': self.decision_tree.get_depth(),
            'n_leaves': self.decision_tree.get_n_leaves(),
            'n_nodes': self.decision_tree.tree_.node_count,
            'classification_report': classification_report(y_val, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        self.logger.info(f"Decision tree training completed:")
        self.logger.info(f"  Accuracy: {accuracy:.4f}")
        self.logger.info(f"  Tree depth: {stats['tree_depth']}")
        self.logger.info(f"  Number of leaves: {stats['n_leaves']}")
        self.logger.info(f"  Number of nodes: {stats['n_nodes']}")
        
        return stats
    
    def optimize_tree_complexity(self, 
                                max_depth_range: List[int] = [3, 5, 7, 10, 15, 20],
                                min_samples_split_range: List[int] = [50, 100, 200]) -> Dict:
        """
        Find optimal tree complexity through hyperparameter search.
        
        Args:
            max_depth_range: Range of max_depth values to try
            min_samples_split_range: Range of min_samples_split values to try
            
        Returns:
            Best parameters and results
        """
        if self.distillation_states is None:
            self.generate_distillation_data()
        
        self.logger.info("Optimizing decision tree complexity")
        
        X_train, X_val, y_train, y_val = train_test_split(
            self.distillation_states, 
            self.distillation_actions,
            test_size=0.2,
            random_state=42,
            stratify=self.distillation_actions
        )
        
        best_score = 0
        best_params = {}
        results = []
        
        for max_depth in max_depth_range:
            for min_samples_split in min_samples_split_range:
                tree = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_split//2,
                    random_state=42
                )
                
                tree.fit(X_train, y_train)
                y_pred = tree.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                
                result = {
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'accuracy': accuracy,
                    'n_leaves': tree.get_n_leaves(),
                    'tree_depth': tree.get_depth()
                }
                results.append(result)
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_params = {
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_split//2
                    }
                
                self.logger.info(f"Depth {max_depth}, Split {min_samples_split}: "
                               f"Acc {accuracy:.4f}, Leaves {tree.get_n_leaves()}")
        
        # Train final model with best parameters
        self.decision_tree = DecisionTreeClassifier(random_state=42, **best_params)
        self.decision_tree.fit(X_train, y_train)
        
        return {
            'best_params': best_params,
            'best_accuracy': best_score,
            'all_results': results
        }
    
    def export_decision_tree(self, 
                           feature_names: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export the decision tree in various formats.
        
        Args:
            feature_names: Names of the features for better readability
            
        Returns:
            Dictionary with exported formats
        """
        if self.decision_tree is None:
            raise ValueError("Decision tree not trained yet")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(self.env.observation_space.shape[0])]
        
        action_names = ['forward', 'deflect', 'drop']
        
        # Export as text
        tree_text = export_text(
            self.decision_tree,
            feature_names=feature_names,
            class_names=action_names
        )
        
        # Save text representation
        text_path = self.output_dir / "decision_tree.txt"
        with open(text_path, 'w') as f:
            f.write(tree_text)
        
        # Save the model
        model_path = self.output_dir / "decision_tree.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.decision_tree, f)
        
        # Save feature names and action names
        metadata = {
            'feature_names': feature_names,
            'action_names': action_names,
            'tree_depth': int(self.decision_tree.get_depth()),
            'n_leaves': int(self.decision_tree.get_n_leaves()),
            'n_nodes': int(self.decision_tree.tree_.node_count)
        }
        
        metadata_path = self.output_dir / "tree_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create visualization
        self._visualize_tree(feature_names, action_names)
        
        self.logger.info(f"Decision tree exported:")
        self.logger.info(f"  Text: {text_path}")
        self.logger.info(f"  Model: {model_path}")
        self.logger.info(f"  Metadata: {metadata_path}")
        
        return {
            'text': tree_text,
            'model_path': str(model_path),
            'metadata_path': str(metadata_path)
        }
    
    def _visualize_tree(self, feature_names: List[str], action_names: List[str]):
        """Create a visualization of the decision tree."""
        plt.figure(figsize=(20, 12))
        plot_tree(
            self.decision_tree,
            feature_names=feature_names,
            class_names=action_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        
        viz_path = self.output_dir / "decision_tree_visualization.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Visualization: {viz_path}")
    
    def generate_switch_code(self, 
                           programming_language: str = "p4") -> str:
        """
        Generate code for deploying the decision tree in a programmable switch.
        
        Args:
            programming_language: Target language ("p4", "c", "pseudo")
            
        Returns:
            Generated code as string
        """
        if self.decision_tree is None:
            raise ValueError("Decision tree not trained yet")
        
        if programming_language.lower() == "p4":
            return self._generate_p4_code()
        elif programming_language.lower() == "c":
            return self._generate_c_code()
        else:
            return self._generate_pseudo_code()
    
    def _generate_p4_code(self) -> str:
        """Generate P4 code for the decision tree."""
        tree = self.decision_tree.tree_
        feature_names = [f'hdr.custom.feature_{i}' for i in range(tree.n_features)]
        
        code = """
// Auto-generated P4 code for RL-distilled decision tree
// Packet deflection policy

action forward_packet() {
    // Forward packet normally
    standard_metadata.egress_spec = standard_metadata.ingress_port;
}

action deflect_packet() {
    // Deflect packet to alternative path
    // Implementation depends on network topology
    standard_metadata.egress_spec = DEFLECTION_PORT;
}

action drop_packet() {
    mark_to_drop(standard_metadata);
}

control DeflectionDecisionTree(inout headers hdr,
                              inout metadata meta,
                              inout standard_metadata_t standard_metadata) {
    
    apply {
"""
        
        # Generate decision tree logic
        code += self._generate_p4_tree_logic(tree, 0, "        ")
        
        code += """
    }
}
"""
        
        # Save P4 code
        p4_path = self.output_dir / "deflection_decision_tree.p4"
        with open(p4_path, 'w') as f:
            f.write(code)
        
        self.logger.info(f"P4 code generated: {p4_path}")
        return code
    
    def _generate_p4_tree_logic(self, tree, node_id: int, indent: str) -> str:
        """Recursively generate P4 logic for tree nodes."""
        if tree.children_left[node_id] == tree.children_right[node_id]:
            # Leaf node
            predicted_class = np.argmax(tree.value[node_id][0])
            actions = ['forward_packet();', 'deflect_packet();', 'drop_packet();']
            return f"{indent}{actions[predicted_class]}\n"
        
        # Internal node
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        
        code = f"{indent}if (hdr.custom.feature_{feature_idx} <= {threshold:.6f}) {{\n"
        code += self._generate_p4_tree_logic(tree, tree.children_left[node_id], indent + "    ")
        code += f"{indent}}} else {{\n"
        code += self._generate_p4_tree_logic(tree, tree.children_right[node_id], indent + "    ")
        code += f"{indent}}}\n"
        
        return code
    
    def _generate_c_code(self) -> str:
        """Generate C code for the decision tree."""
        # Similar implementation for C
        # This would be useful for other programmable targets
        pass
    
    def _generate_pseudo_code(self) -> str:
        """Generate human-readable pseudo code."""
        tree_text = export_text(self.decision_tree)
        return f"# Pseudo-code representation:\n{tree_text}"


def main():
    """Main function to demonstrate the distillation process."""
    distiller = RLToDecisionTreeDistiller(
        rl_model_path="/home/ubuntu/practical_deflection/RL_Training/models/best_model.pth",
        dataset_path="/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    )
    
    # Load RL model
    distiller.load_rl_model()
    
    # Generate distillation data
    distiller.generate_distillation_data(num_samples=15000)
    
    # Optimize tree complexity
    optimization_results = distiller.optimize_tree_complexity()
    print(f"Best parameters: {optimization_results['best_params']}")
    print(f"Best accuracy: {optimization_results['best_accuracy']:.4f}")
    
    # Export the decision tree
    feature_names = ['queue_length', 'arrival_rate', 'service_rate', 'utilization', 'threshold', 'load']
    export_results = distiller.export_decision_tree(feature_names=feature_names)
    
    # Generate P4 code
    p4_code = distiller.generate_switch_code("p4")
    
    print("Knowledge distillation completed successfully!")
    print(f"Decision tree ready for deployment in programmable switches.")


if __name__ == "__main__":
    main()
