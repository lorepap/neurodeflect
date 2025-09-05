"""
Real-Time Switch-Level Features for Line-Rate Deflection Decisions

This module defines features that can be efficiently extracted at line rate
in datacenter switches for real-time packet deflection decisions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class LineRateFeatureExtractor:
    """
    Feature extractor optimized for real-time switch inference.
    Features are categorized by computational cost and latency requirements.
    """
    
    def __init__(self):
        self.feature_categories = {
            'immediate': [],      # 0 latency - from packet headers
            'local_fast': [],     # <1μs - local switch state  
            'local_medium': [],   # <10μs - local computations
            'network_slow': []    # <1ms - network-aware features
        }
    
    def extract_immediate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract zero-latency features from packet headers.
        These are available immediately when packet arrives.
        """
        features = df.copy()
        
        # 1. TTL priority (lower TTL = higher priority)
        features['ttl_priority'] = (250 - features['ttl']) / 250.0
        
        # 2. Packet size category (for QoS)
        features['is_large_packet'] = (features['packet_size'] > 64).astype(int)
        
        # 3. Sequence number locality (for flow tracking)
        features['seq_num_hash'] = features['seq_num'] % 1000  # Simple hash for tracking
        
        self.feature_categories['immediate'] = ['ttl_priority', 'is_large_packet', 'seq_num_hash']
        return features
    
    def extract_local_fast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract fast local switch state features (<1μs latency).
        These require minimal computation on local state.
        """
        features = df.copy()
        
        # 1. Queue utilization ratio (most important feature)
        features['queue_utilization'] = features['occupancy'] / (features['capacity'] + 1e-8)
        
        # 2. Congestion binary indicator
        features['is_congested'] = (features['queue_utilization'] > 0.5).astype(int)
        
        # 3. Queue depth in packets (approximate)
        avg_packet_size = features['packet_size'].mean()
        features['queue_depth_packets'] = features['occupancy'] / avg_packet_size
        
        # 4. Binary overload indicator
        features['queue_overload'] = (features['queue_utilization'] > 0.8).astype(int)
        
        self.feature_categories['local_fast'] = [
            'queue_utilization', 'is_congested', 'queue_depth_packets', 'queue_overload'
        ]
        return features
    
    def extract_local_medium_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract medium-latency local features (<10μs latency).
        These require sliding window computations.
        """
        features = df.copy()
        
        # 1. Total switch utilization
        features['total_utilization'] = features['total_occupancy'] / (features['total_capacity'] + 1e-8)
        
        # 2. Relative queue pressure (this queue vs total switch)
        features['relative_queue_pressure'] = features['occupancy'] / (features['total_occupancy'] + 1e-8)
        
        # 3. Switch load category
        switch_load_cut = pd.cut(
            features['total_utilization'], 
            bins=[0, 0.2, 0.5, 0.8, 1.0], 
            labels=[0, 1, 2, 3]
        )
        features['switch_load_category'] = switch_load_cut.cat.codes.fillna(0).astype(int)
        
        # 4. Queue pressure level
        queue_pressure_cut = pd.cut(
            features['queue_utilization'],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=[0, 1, 2, 3]
        )
        features['queue_pressure_level'] = queue_pressure_cut.cat.codes.fillna(0).astype(int)
        
        self.feature_categories['local_medium'] = [
            'total_utilization', 'relative_queue_pressure', 
            'switch_load_category', 'queue_pressure_level'
        ]
        return features
    
    def extract_network_slow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract network-aware features (<1ms latency).
        These require communication or complex computation.
        """
        features = df.copy()
        
        # 1. Switch type context (spine vs agg have different behaviors)
        features['is_spine_switch'] = features['SWITCH_ID'].str.contains('spine').astype(int)
        features['is_agg_switch'] = features['SWITCH_ID'].str.contains('agg').astype(int)
        
        # 2. Deflection threshold utilization
        features['threshold_pressure'] = features['queue_utilization'] / (features['deflection_threshold'] + 1e-8)
        
        # 3. Policy compliance indicator
        features['should_deflect'] = (features['queue_utilization'] > features['deflection_threshold']).astype(int)
        
        # 4. Flow progress estimation (requires flow tracking)
        # This is expensive but useful for flow-aware decisions
        features['estimated_flow_progress'] = features.groupby('FlowID').cumcount() + 1
        
        self.feature_categories['network_slow'] = [
            'is_spine_switch', 'is_agg_switch', 'threshold_pressure', 
            'should_deflect', 'estimated_flow_progress'
        ]
        return features
    
    def get_recommended_feature_sets(self) -> Dict[str, List[str]]:
        """
        Get recommended feature combinations for different deployment scenarios.
        """
        return {
            'ultra_fast': [
                'queue_utilization',
                'ttl_priority', 
                'is_congested'
            ],
            'fast': [
                'queue_utilization',
                'total_utilization',
                'ttl_priority',
                'is_congested',
                'queue_pressure_level'
            ],
            'balanced': [
                'queue_utilization',
                'total_utilization', 
                'relative_queue_pressure',
                'ttl_priority',
                'is_congested',
                'switch_load_category',
                'is_spine_switch'
            ],
            'full_featured': [
                'queue_utilization',
                'total_utilization',
                'relative_queue_pressure', 
                'ttl_priority',
                'threshold_pressure',
                'switch_load_category',
                'queue_pressure_level',
                'is_spine_switch',
                'estimated_flow_progress'
            ]
        }
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all feature categories."""
        features = df.copy()
        features = self.extract_immediate_features(features)
        features = self.extract_local_fast_features(features)
        features = self.extract_local_medium_features(features)
        features = self.extract_network_slow_features(features)
        return features
    
    def analyze_feature_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze which features correlate best with deflection decisions."""
        features = self.extract_all_features(df)
        
        # Get deflection mask
        deflection_mask = features['action'] == 1
        
        analysis = {}
        all_features = []
        for category in self.feature_categories.values():
            all_features.extend(category)
        
        for feature in all_features:
            if feature in features.columns:
                deflect_mean = features[deflection_mask][feature].mean()
                forward_mean = features[~deflection_mask][feature].mean()
                
                # Calculate discriminative power
                if forward_mean != 0:
                    ratio = deflect_mean / forward_mean
                else:
                    ratio = float('inf') if deflect_mean > 0 else 1.0
                
                analysis[feature] = {
                    'deflect_mean': deflect_mean,
                    'forward_mean': forward_mean,
                    'discriminative_ratio': ratio,
                    'correlation': np.corrcoef(features[feature], features['action'])[0, 1]
                }
        
        return analysis


def test_line_rate_features():
    """Test the line rate feature extraction."""
    # Load dataset
    df = pd.read_csv('/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv')
    
    # Initialize extractor
    extractor = LineRateFeatureExtractor()
    
    # Extract features
    enhanced_df = extractor.extract_all_features(df)
    
    print("=== LINE-RATE FEATURE EXTRACTION RESULTS ===")
    print()
    
    # Show feature categories
    feature_sets = extractor.get_recommended_feature_sets()
    for set_name, features in feature_sets.items():
        print(f"{set_name.upper()} FEATURE SET ({len(features)} features):")
        for feature in features:
            if feature in enhanced_df.columns:
                values = enhanced_df[feature]
                print(f"  {feature}: range=[{values.min():.3f}, {values.max():.3f}], mean={values.mean():.3f}")
        print()
    
    # Analyze feature performance
    print("FEATURE DISCRIMINATIVE POWER (deflection vs forward):")
    analysis = extractor.analyze_feature_performance(df)
    
    # Sort by discriminative ratio
    sorted_features = sorted(analysis.items(), key=lambda x: abs(x[1]['discriminative_ratio'] - 1), reverse=True)
    
    for feature, stats in sorted_features[:10]:  # Top 10
        ratio = stats['discriminative_ratio']
        corr = stats['correlation']
        print(f"  {feature:<25} ratio={ratio:6.2f}, correlation={corr:6.3f}")
    
    return enhanced_df, analysis


if __name__ == "__main__":
    enhanced_df, analysis = test_line_rate_features()
