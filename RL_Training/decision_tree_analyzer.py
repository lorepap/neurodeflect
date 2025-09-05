#!/usr/bin/env python3
"""
Decision Tree Analysis and Deployment Tool

This module provides tools for analyzing and preparing decision trees
for deployment in programmable switches (without P4 code generation).
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.tree import DecisionTreeClassifier


class DecisionTreeAnalyzer:
    """
    Analyzer for decision trees to understand their structure and deployment requirements.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the analyzer with a trained decision tree.
        
        Args:
            model_path: Path to the pickled decision tree model
        """
        self.model_path = Path(model_path)
        self.decision_tree = self._load_model()
        self.metadata = self._load_metadata()
        
    def _load_model(self) -> DecisionTreeClassifier:
        """Load the decision tree model."""
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Loaded decision tree from {self.model_path}")
        return model
    
    def _load_metadata(self) -> Dict:
        """Load metadata if available."""
        metadata_path = self.model_path.parent / f"{self.model_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}
    
    def analyze_complexity(self) -> Dict:
        """Analyze the complexity of the decision tree for switch deployment."""
        tree = self.decision_tree.tree_
        
        # Calculate basic metrics first
        basic_analysis = {
            'total_nodes': tree.node_count,
            'total_leaves': self.decision_tree.get_n_leaves(),
            'max_depth': self.decision_tree.get_depth(),
            'avg_path_length': self._calculate_avg_path_length(),
            'memory_requirements': self._estimate_memory_requirements()
        }
        
        # Add feasibility assessment based on basic metrics
        basic_analysis['switch_feasibility'] = self._assess_switch_feasibility(basic_analysis)
        
        return basic_analysis
    
    def _calculate_avg_path_length(self) -> float:
        """Calculate average path length from root to leaves."""
        tree = self.decision_tree.tree_
        
        def get_path_lengths(node_id, depth=0):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                return [depth]
            else:
                # Internal node
                left_paths = get_path_lengths(tree.children_left[node_id], depth + 1)
                right_paths = get_path_lengths(tree.children_right[node_id], depth + 1)
                return left_paths + right_paths
        
        path_lengths = get_path_lengths(0)
        return np.mean(path_lengths)
    
    def _estimate_memory_requirements(self) -> Dict:
        """Estimate memory requirements for switch deployment."""
        tree = self.decision_tree.tree_
        
        # Rough estimates for switch table entries
        internal_nodes = tree.node_count - self.decision_tree.get_n_leaves()
        
        # Each internal node needs a table entry with feature, threshold, and next nodes
        # Each leaf needs an action entry
        memory_estimate = {
            'decision_table_entries': internal_nodes,
            'action_table_entries': self.decision_tree.get_n_leaves(),
            'total_table_entries': tree.node_count,
            'estimated_memory_kb': tree.node_count * 0.1  # Rough estimate
        }
        
        return memory_estimate
    
    def _assess_switch_feasibility(self, analysis: Dict) -> Dict:
        """Assess feasibility for switch deployment."""
        
        # Define thresholds for switch deployment
        feasibility = {
            'depth_feasible': analysis['max_depth'] <= 16,  # Typical switch pipeline depth
            'memory_feasible': analysis['memory_requirements']['estimated_memory_kb'] <= 1024,  # 1MB limit
            'complexity_feasible': analysis['total_nodes'] <= 1000,  # Reasonable complexity
            'overall_feasible': True
        }
        
        feasibility['overall_feasible'] = all([
            feasibility['depth_feasible'],
            feasibility['memory_feasible'], 
            feasibility['complexity_feasible']
        ])
        
        return feasibility
    
    def extract_decision_rules(self) -> List[Dict]:
        """Extract all decision rules from the tree."""
        tree = self.decision_tree.tree_
        feature_names = self.metadata.get('feature_names', [f'feature_{i}' for i in range(self.decision_tree.n_features_in_)])
        
        def extract_rules(node_id, path_conditions=[]):
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                action = np.argmax(tree.value[node_id][0])
                confidence = np.max(tree.value[node_id][0]) / np.sum(tree.value[node_id][0])
                
                return [{
                    'conditions': path_conditions.copy(),
                    'action': int(action),
                    'confidence': float(confidence),
                    'samples': int(tree.n_node_samples[node_id])
                }]
            else:
                # Internal node
                feature_idx = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                feature_name = feature_names[feature_idx]
                
                # Left child (feature <= threshold)
                left_condition = f"{feature_name} <= {threshold:.4f}"
                left_rules = extract_rules(
                    tree.children_left[node_id], 
                    path_conditions + [left_condition]
                )
                
                # Right child (feature > threshold)
                right_condition = f"{feature_name} > {threshold:.4f}"
                right_rules = extract_rules(
                    tree.children_right[node_id], 
                    path_conditions + [right_condition]
                )
                
                return left_rules + right_rules
        
        return extract_rules(0)
    
    def generate_deployment_summary(self) -> Dict:
        """Generate a comprehensive deployment summary."""
        complexity = self.analyze_complexity()
        rules = self.extract_decision_rules()
        
        summary = {
            'model_info': {
                'model_path': str(self.model_path),
                'feature_count': self.decision_tree.n_features_in_,
                'class_count': self.decision_tree.n_classes_
            },
            'complexity_analysis': complexity,
            'deployment_rules': {
                'total_rules': len(rules),
                'action_distribution': {},
                'confidence_stats': {
                    'mean_confidence': np.mean([r['confidence'] for r in rules]),
                    'min_confidence': np.min([r['confidence'] for r in rules]),
                    'max_confidence': np.max([r['confidence'] for r in rules])
                }
            },
            'switch_compatibility': {
                'recommended_for_deployment': complexity['switch_feasibility']['overall_feasible'],
                'deployment_notes': self._generate_deployment_notes(complexity)
            }
        }
        
        # Calculate action distribution
        action_counts = {}
        for rule in rules:
            action = rule['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        summary['deployment_rules']['action_distribution'] = action_counts
        
        return summary
    
    def _generate_deployment_notes(self, complexity: Dict) -> List[str]:
        """Generate deployment notes and recommendations."""
        notes = []
        
        if not complexity['switch_feasibility']['depth_feasible']:
            notes.append(f"Tree depth ({complexity['max_depth']}) may be too deep for switch pipeline")
        
        if not complexity['switch_feasibility']['memory_feasible']:
            notes.append("Memory requirements may exceed switch table capacity")
        
        if not complexity['switch_feasibility']['complexity_feasible']:
            notes.append("Tree complexity may be too high for efficient switch implementation")
        
        if complexity['switch_feasibility']['overall_feasible']:
            notes.append("Decision tree is suitable for switch deployment")
            notes.append(f"Expected average packet processing: {complexity['avg_path_length']:.1f} table lookups")
        
        return notes
    
    def print_analysis(self):
        """Print a comprehensive analysis of the decision tree."""
        summary = self.generate_deployment_summary()
        
        print("="*60)
        print("DECISION TREE DEPLOYMENT ANALYSIS")
        print("="*60)
        
        print(f"\nModel Information:")
        print(f"  Path: {summary['model_info']['model_path']}")
        print(f"  Features: {summary['model_info']['feature_count']}")
        print(f"  Classes: {summary['model_info']['class_count']}")
        
        print(f"\nComplexity Analysis:")
        complexity = summary['complexity_analysis']
        print(f"  Total Nodes: {complexity['total_nodes']}")
        print(f"  Leaves: {complexity['total_leaves']}")
        print(f"  Max Depth: {complexity['max_depth']}")
        print(f"  Avg Path Length: {complexity['avg_path_length']:.2f}")
        
        print(f"\nMemory Requirements:")
        mem = complexity['memory_requirements']
        print(f"  Decision Table Entries: {mem['decision_table_entries']}")
        print(f"  Action Table Entries: {mem['action_table_entries']}")
        print(f"  Estimated Memory: {mem['estimated_memory_kb']:.1f} KB")
        
        print(f"\nSwitch Feasibility:")
        feasibility = complexity['switch_feasibility']
        print(f"  Depth Feasible: {feasibility['depth_feasible']}")
        print(f"  Memory Feasible: {feasibility['memory_feasible']}")
        print(f"  Complexity Feasible: {feasibility['complexity_feasible']}")
        print(f"  Overall Feasible: {feasibility['overall_feasible']}")
        
        print(f"\nDeployment Rules:")
        rules = summary['deployment_rules']
        print(f"  Total Rules: {rules['total_rules']}")
        print(f"  Action Distribution: {rules['action_distribution']}")
        print(f"  Mean Confidence: {rules['confidence_stats']['mean_confidence']:.3f}")
        
        print(f"\nDeployment Notes:")
        for note in summary['switch_compatibility']['deployment_notes']:
            print(f"  - {note}")


def main():
    """Main function to analyze the distilled decision tree."""
    
    # Path to the distilled model
    model_path = "/home/ubuntu/practical_deflection/RL_Training/distilled_models/distilled_decision_tree.pkl"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please run knowledge_distillation.py first to generate the model.")
        return
    
    # Initialize analyzer
    analyzer = DecisionTreeAnalyzer(model_path)
    
    # Print comprehensive analysis
    analyzer.print_analysis()
    
    # Save deployment summary
    summary = analyzer.generate_deployment_summary()
    output_path = Path(model_path).parent / "deployment_analysis.json"
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nDetailed analysis saved to: {output_path}")


if __name__ == "__main__":
    main()
