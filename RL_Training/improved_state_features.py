"""
Improved State Feature Engineering for Offline RL Training

This module defines an enhanced state representation for the datacenter deflection
optimization problem based on analysis of the threshold dataset.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


class ImprovedStateExtractor:
    """
    Enhanced state feature extractor for datacenter deflection RL.
    """
    
    def __init__(self, normalize_features: bool = True):
        """
        Initialize the state extractor.
        
        Args:
            normalize_features: Whether to apply feature normalization
        """
        self.normalize_features = normalize_features
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
    
    def extract_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Extract improved state features from the dataset.
        
        Args:
            df: DataFrame with threshold dataset
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []
        feature_names = []
        
        # 1. NETWORK STATE FEATURES (normalized utilizations)
        queue_utilization = df['occupancy'] / (df['capacity'] + 1e-8)
        features.append(queue_utilization.values)
        feature_names.append('queue_utilization')
        
        total_utilization = df['total_occupancy'] / (df['total_capacity'] + 1e-8) 
        features.append(total_utilization.values)
        feature_names.append('total_utilization')
        
        # 2. ABSOLUTE LOAD INDICATORS (log-scaled for better distribution)
        log_queue_load = np.log1p(df['occupancy'].values)
        features.append(log_queue_load)
        feature_names.append('log_queue_load')
        
        log_total_load = np.log1p(df['total_occupancy'].values)
        features.append(log_total_load)
        feature_names.append('log_total_load')
        
        # 3. SWITCH TYPE FEATURES (one-hot encoded)
        switch_features = self._encode_switch_type(df['SWITCH_ID'])
        features.extend(switch_features)
        feature_names.extend(['is_spine', 'is_agg', 'switch_tier'])
        
        # 4. FLOW CONTEXT FEATURES
        # Relative packet position within flow
        packet_progress = df['packet_position'] / (df['flow_packet_count'] + 1e-8)
        features.append(packet_progress.values)
        feature_names.append('packet_progress')
        
        # Flow size category (log-scaled)
        log_flow_size = np.log1p(df['flow_packet_count'].values)
        features.append(log_flow_size)
        feature_names.append('log_flow_size')
        
        # Flow completion time (performance indicator)
        features.append(df['FCT'].values)
        feature_names.append('flow_completion_time')
        
        # Historical deflection rate for this flow
        features.append(df['deflection_rate'].values)
        feature_names.append('flow_deflection_rate')
        
        # 5. POLICY/THRESHOLD FEATURES
        features.append(df['deflection_threshold'].values)
        feature_names.append('deflection_threshold')
        
        # Threshold utilization (how close we are to threshold)
        threshold_pressure = queue_utilization / (df['deflection_threshold'] + 1e-8)
        features.append(threshold_pressure.values)
        feature_names.append('threshold_pressure')
        
        # 6. TEMPORAL/LATENCY FEATURES
        # TTL utilization (how many hops packet has taken)
        ttl_utilization = (250 - df['ttl']) / 250.0
        features.append(ttl_utilization.values)
        feature_names.append('ttl_utilization')
        
        # Normalized timestamp within episode
        norm_timestamp = (df['timestamp'] - df['timestamp'].min()) / (df['timestamp'].max() - df['timestamp'].min() + 1e-8)
        features.append(norm_timestamp.values)
        feature_names.append('normalized_time')
        
        # 7. PERFORMANCE INDICATORS
        # Sequence number gaps (proxy for reordering)
        seq_gaps = np.abs(df['seq_num'].diff().fillna(0))
        log_seq_gaps = np.log1p(seq_gaps.values)
        features.append(log_seq_gaps)
        feature_names.append('log_seq_gaps')
        
        # Combine all features
        feature_matrix = np.column_stack(features)
        
        # Apply normalization if requested
        if self.normalize_features:
            feature_matrix = self._normalize_features(feature_matrix, feature_names)
        
        self.feature_names = feature_names
        return feature_matrix, feature_names
    
    def _encode_switch_type(self, switch_ids: pd.Series) -> List[np.ndarray]:
        """Encode switch type information."""
        # Extract switch type from switch ID
        is_spine = switch_ids.str.contains('spine').astype(float).values
        is_agg = switch_ids.str.contains('agg').astype(float).values
        
        # Switch tier (0=spine, 1=agg, with some numeric encoding)
        switch_tier = is_agg.astype(float)  # 0 for spine, 1 for agg
        
        return [is_spine, is_agg, switch_tier]
    
    def _normalize_features(self, feature_matrix: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Apply feature normalization."""
        if not self.is_fitted:
            # Fit scalers
            for i, name in enumerate(feature_names):
                scaler = StandardScaler()
                self.scalers[name] = scaler.fit(feature_matrix[:, i].reshape(-1, 1))
            self.is_fitted = True
        
        # Transform features
        normalized_features = []
        for i, name in enumerate(feature_names):
            normalized_col = self.scalers[name].transform(feature_matrix[:, i].reshape(-1, 1))
            normalized_features.append(normalized_col.flatten())
        
        return np.column_stack(normalized_features)
    
    def get_feature_importance_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze feature importance and correlations."""
        features, names = self.extract_features(df)
        
        analysis = {
            'feature_names': names,
            'feature_stats': {},
            'correlations': {},
            'action_correlations': {}
        }
        
        # Basic statistics
        for i, name in enumerate(names):
            analysis['feature_stats'][name] = {
                'mean': float(features[:, i].mean()),
                'std': float(features[:, i].std()),
                'min': float(features[:, i].min()),
                'max': float(features[:, i].max())
            }
        
        # Correlation with actions
        if 'action' in df.columns:
            for i, name in enumerate(names):
                corr = np.corrcoef(features[:, i], df['action'])[0, 1]
                analysis['action_correlations'][name] = float(corr) if not np.isnan(corr) else 0.0
        
        return analysis


def create_improved_state_dataset(dataset_path: str, 
                                output_path: str = None,
                                normalize: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create an improved state dataset for offline RL training.
    
    Args:
        dataset_path: Path to the threshold dataset CSV
        output_path: Optional path to save the processed dataset
        normalize: Whether to normalize features
        
    Returns:
        Tuple of (states, actions, feature_names)
    """
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Extract improved features
    extractor = ImprovedStateExtractor(normalize_features=normalize)
    states, feature_names = extractor.extract_features(df)
    actions = df['action'].values
    
    print(f"Created improved state representation:")
    print(f"  - Original dataset: {df.shape}")
    print(f"  - State features: {states.shape}")
    print(f"  - Feature names: {feature_names}")
    print(f"  - Actions: {len(np.unique(actions))} unique values")
    
    # Feature importance analysis
    analysis = extractor.get_feature_importance_analysis(df)
    print(f"\nFeature-Action Correlations:")
    for name, corr in analysis['action_correlations'].items():
        print(f"  {name}: {corr:.4f}")
    
    # Save if requested
    if output_path:
        processed_df = pd.DataFrame(states, columns=feature_names)
        processed_df['action'] = actions
        processed_df['original_threshold'] = df['deflection_threshold']
        processed_df.to_csv(output_path, index=False)
        print(f"\nSaved processed dataset to: {output_path}")
    
    return states, actions, feature_names


if __name__ == "__main__":
    # Test the improved state extraction
    dataset_path = "/home/ubuntu/practical_deflection/Omnet_Sims/dc_simulations/simulations/sims/combined_threshold_dataset.csv"
    output_path = "/home/ubuntu/practical_deflection/RL_Training/improved_state_dataset.csv"
    
    states, actions, feature_names = create_improved_state_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        normalize=True
    )
