#!/usr/bin/env python3
"""
Comparative Analysis: RL Policy vs Threshold-based Deflection

This script compares the performance of the RL-based deflection policy 
against traditional threshold-based deflection strategies.

Usage:
    python compare_rl_vs_baselines.py [--rl-dir rl_analysis] [--output-dir comparison_analysis]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RLBaselineComparator:
    def __init__(self, rl_dir="rl_analysis", output_dir="comparison_analysis"):
        self.rl_dir = Path(rl_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store analysis results
        self.results = {
            'rl_policy': {},
            'threshold_policies': {}
        }
        
    def load_rl_results(self):
        """Load RL policy analysis results"""
        print("Loading RL policy results...")
        
        # Load FCT data
        fct_file = self.rl_dir / 'fct_data.csv'
        if fct_file.exists():
            rl_fct_df = pd.read_csv(fct_file)
            self.results['rl_policy']['fct_data'] = rl_fct_df['fct'].values
            print(f"  Loaded {len(rl_fct_df)} RL FCT records")
        else:
            print(f"  Warning: {fct_file} not found")
            return False
            
        # Load summary stats from report
        report_file = self.rl_dir / 'rl_policy_analysis_report.txt'
        if report_file.exists():
            with open(report_file, 'r') as f:
                content = f.read()
                
            # Parse key metrics from report
            lines = content.split('\n')
            for line in lines:
                if 'Mean FCT:' in line:
                    self.results['rl_policy']['mean_fct'] = float(line.split(':')[1].strip().rstrip('s'))
                elif 'Median FCT:' in line:
                    self.results['rl_policy']['median_fct'] = float(line.split(':')[1].strip().rstrip('s'))
                elif '95th percentile FCT:' in line:
                    self.results['rl_policy']['p95_fct'] = float(line.split(':')[1].strip().rstrip('s'))
                elif '99th percentile FCT:' in line:
                    self.results['rl_policy']['p99_fct'] = float(line.split(':')[1].strip().rstrip('s'))
                elif 'Total flows completed:' in line:
                    self.results['rl_policy']['total_flows'] = int(line.split(':')[1].strip())
                elif 'Total requests sent:' in line:
                    self.results['rl_policy']['total_requests'] = int(line.split(':')[1].strip())
                    
        return True
        
    def load_threshold_baselines(self):
        """Load threshold-based baseline results"""
        print("Loading threshold baseline results...")
        
        # Find threshold dataset files
        threshold_files = list(Path('.').glob('threshold_dataset_*.csv'))
        
        if not threshold_files:
            print("  No threshold baseline datasets found")
            return False
            
        for threshold_file in threshold_files:
            # Extract threshold value from filename
            threshold_str = threshold_file.stem.replace('threshold_dataset_', '')
            try:
                threshold = int(threshold_str)
            except ValueError:
                print(f"  Warning: Could not parse threshold from {threshold_file}")
                continue
                
            print(f"  Loading threshold {threshold} data...")
            
            try:
                df = pd.read_csv(threshold_file)
                
                # Analyze this threshold's data
                threshold_stats = self.analyze_threshold_dataset(df, threshold)
                self.results['threshold_policies'][threshold] = threshold_stats
                
                print(f"    Loaded {len(df)} records for threshold {threshold}")
                
            except Exception as e:
                print(f"    Error loading {threshold_file}: {e}")
                continue
                
        return len(self.results['threshold_policies']) > 0
        
    def analyze_threshold_dataset(self, df, threshold):
        """Analyze a threshold-based dataset"""
        stats = {'threshold': threshold}
        
        # Check available columns
        if 'fct' in df.columns:
            fcts = df['fct'].dropna()
            if len(fcts) > 0:
                stats['fct_data'] = fcts.values
                stats['mean_fct'] = np.mean(fcts)
                stats['median_fct'] = np.median(fcts)
                stats['p95_fct'] = np.percentile(fcts, 95)
                stats['p99_fct'] = np.percentile(fcts, 99)
                stats['total_flows'] = len(fcts)
        
        # Check for other relevant metrics
        if 'deflection_action' in df.columns:
            deflections = df['deflection_action'].value_counts()
            if 1 in deflections:  # Assuming 1 = deflected
                stats['deflection_rate'] = deflections[1] / len(df)
            else:
                stats['deflection_rate'] = 0.0
        
        # Count total packets/requests
        stats['total_packets'] = len(df)
        
        return stats
        
    def create_fct_comparison(self):
        """Create FCT comparison plots"""
        print("\nCreating FCT comparison analysis...")
        
        plt.figure(figsize=(15, 10))
        
        # Prepare data for plotting
        all_fcts = []
        all_labels = []
        
        # Add RL policy data
        if 'fct_data' in self.results['rl_policy']:
            rl_fcts = self.results['rl_policy']['fct_data']
            all_fcts.append(rl_fcts)
            all_labels.append('RL Policy')
        
        # Add threshold policy data
        threshold_keys = sorted(self.results['threshold_policies'].keys())
        for threshold in threshold_keys:
            threshold_data = self.results['threshold_policies'][threshold]
            if 'fct_data' in threshold_data:
                all_fcts.append(threshold_data['fct_data'])
                all_labels.append(f'Threshold {threshold}')
        
        if not all_fcts:
            print("  No FCT data available for comparison")
            return
            
        # Plot 1: FCT distributions
        plt.subplot(2, 3, 1)
        for i, (fcts, label) in enumerate(zip(all_fcts, all_labels)):
            plt.hist(fcts, bins=20, alpha=0.6, label=label, density=True)
        plt.xlabel('Flow Completion Time (s)')
        plt.ylabel('Density')
        plt.title('FCT Distributions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: FCT CDFs
        plt.subplot(2, 3, 2)
        for i, (fcts, label) in enumerate(zip(all_fcts, all_labels)):
            sorted_fcts = np.sort(fcts)
            cdf = np.arange(1, len(sorted_fcts) + 1) / len(sorted_fcts)
            plt.plot(sorted_fcts, cdf, label=label, linewidth=2)
        plt.xlabel('Flow Completion Time (s)')
        plt.ylabel('Cumulative Probability')
        plt.title('FCT CDFs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Box plots
        plt.subplot(2, 3, 3)
        plt.boxplot(all_fcts, labels=all_labels)
        plt.ylabel('Flow Completion Time (s)')
        plt.title('FCT Box Plot Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Mean FCT comparison
        plt.subplot(2, 3, 4)
        means = [np.mean(fcts) for fcts in all_fcts]
        colors = ['red'] + ['blue'] * (len(means) - 1)  # RL in red, others in blue
        bars = plt.bar(all_labels, means, color=colors, alpha=0.7)
        plt.ylabel('Mean FCT (s)')
        plt.title('Mean FCT Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(means) * 0.01,
                    f'{mean:.4f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Percentile comparison
        plt.subplot(2, 3, 5)
        p95s = [np.percentile(fcts, 95) for fcts in all_fcts]
        p99s = [np.percentile(fcts, 99) for fcts in all_fcts]
        
        x = np.arange(len(all_labels))
        width = 0.35
        
        plt.bar(x - width/2, p95s, width, label='95th percentile', alpha=0.7)
        plt.bar(x + width/2, p99s, width, label='99th percentile', alpha=0.7)
        
        plt.ylabel('FCT (s)')
        plt.title('FCT Percentiles')
        plt.xticks(x, all_labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Flow count comparison
        plt.subplot(2, 3, 6)
        flow_counts = [len(fcts) for fcts in all_fcts]
        bars = plt.bar(all_labels, flow_counts, color=colors, alpha=0.7)
        plt.ylabel('Number of Flows')
        plt.title('Flow Count Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, flow_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(flow_counts) * 0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fct_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  FCT comparison saved to {self.output_dir}/fct_comparison.png")
        
    def create_performance_summary(self):
        """Create performance summary table"""
        print("\nCreating performance summary...")
        
        # Create comparison table
        comparison_data = []
        
        # Add RL policy
        if self.results['rl_policy']:
            rl_data = self.results['rl_policy']
            comparison_data.append({
                'Policy': 'RL Policy',
                'Configuration': 'IQL Agent',
                'Total Flows': rl_data.get('total_flows', 'N/A'),
                'Mean FCT (s)': f"{rl_data.get('mean_fct', 0):.6f}",
                'Median FCT (s)': f"{rl_data.get('median_fct', 0):.6f}",
                '95th Percentile (s)': f"{rl_data.get('p95_fct', 0):.6f}",
                '99th Percentile (s)': f"{rl_data.get('p99_fct', 0):.6f}",
                'Total Requests': rl_data.get('total_requests', 'N/A')
            })
        
        # Add threshold policies
        for threshold in sorted(self.results['threshold_policies'].keys()):
            threshold_data = self.results['threshold_policies'][threshold]
            comparison_data.append({
                'Policy': 'Threshold-based',
                'Configuration': f'Threshold {threshold}',
                'Total Flows': threshold_data.get('total_flows', 'N/A'),
                'Mean FCT (s)': f"{threshold_data.get('mean_fct', 0):.6f}",
                'Median FCT (s)': f"{threshold_data.get('median_fct', 0):.6f}",
                '95th Percentile (s)': f"{threshold_data.get('p95_fct', 0):.6f}",
                '99th Percentile (s)': f"{threshold_data.get('p99_fct', 0):.6f}",
                'Total Requests': threshold_data.get('total_packets', 'N/A')
            })
        
        # Create DataFrame and save
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(self.output_dir / 'performance_comparison.csv', index=False)
        
        # Print comparison table
        print("\nPerformance Comparison Summary:")
        print("=" * 100)
        print(comparison_df.to_string(index=False))
        
        print(f"\nPerformance comparison saved to {self.output_dir}/performance_comparison.csv")
        
    def create_comprehensive_report(self):
        """Create comprehensive comparison report"""
        print("\nCreating comprehensive comparison report...")
        
        report_path = self.output_dir / 'rl_vs_baselines_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("RL Policy vs Threshold-based Deflection Comparison\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Analysis Overview:\n")
            f.write("This report compares the performance of an RL-based deflection policy\n")
            f.write("against traditional threshold-based deflection strategies.\n\n")
            
            f.write("Simulation Configuration:\n")
            f.write("- Network: LeafSpine 1G (4 spines, 8 aggregates, 40 servers)\n")
            f.write("- Protocol: DCTCP\n")
            f.write("- Simulation Time: 0.001s (validation experiment)\n")
            f.write("- Workload: Mixed mice/elephant flows with incast patterns\n\n")
            
            # RL Policy Summary
            if self.results['rl_policy']:
                f.write("RL Policy Performance:\n")
                f.write("- Algorithm: Implicit Q-Learning (IQL)\n")
                f.write("- State Space: 6D (queue lengths, utilization, etc.)\n")
                f.write("- Action Space: 2 (deflect/forward)\n")
                
                rl_data = self.results['rl_policy']
                f.write(f"- Flows Completed: {rl_data.get('total_flows', 'N/A')}\n")
                f.write(f"- Mean FCT: {rl_data.get('mean_fct', 0):.6f}s\n")
                f.write(f"- Median FCT: {rl_data.get('median_fct', 0):.6f}s\n")
                f.write(f"- 95th Percentile FCT: {rl_data.get('p95_fct', 0):.6f}s\n")
                f.write(f"- 99th Percentile FCT: {rl_data.get('p99_fct', 0):.6f}s\n\n")
            
            # Threshold Baselines Summary
            if self.results['threshold_policies']:
                f.write("Threshold-based Baseline Performance:\n")
                for threshold in sorted(self.results['threshold_policies'].keys()):
                    threshold_data = self.results['threshold_policies'][threshold]
                    f.write(f"\nThreshold {threshold}:\n")
                    f.write(f"- Flows Completed: {threshold_data.get('total_flows', 'N/A')}\n")
                    f.write(f"- Mean FCT: {threshold_data.get('mean_fct', 0):.6f}s\n")
                    f.write(f"- Median FCT: {threshold_data.get('median_fct', 0):.6f}s\n")
                    f.write(f"- 95th Percentile FCT: {threshold_data.get('p95_fct', 0):.6f}s\n")
                    f.write(f"- 99th Percentile FCT: {threshold_data.get('p99_fct', 0):.6f}s\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("Key Findings:\n")
            
            # Performance analysis
            if self.results['rl_policy'] and self.results['threshold_policies']:
                rl_mean = self.results['rl_policy'].get('mean_fct', float('inf'))
                
                best_threshold = None
                best_threshold_mean = float('inf')
                
                for threshold, data in self.results['threshold_policies'].items():
                    threshold_mean = data.get('mean_fct', float('inf'))
                    if threshold_mean < best_threshold_mean:
                        best_threshold_mean = threshold_mean
                        best_threshold = threshold
                
                if best_threshold is not None:
                    improvement = ((best_threshold_mean - rl_mean) / best_threshold_mean) * 100
                    f.write(f"- RL Policy vs Best Threshold ({best_threshold}):\n")
                    f.write(f"  * Mean FCT improvement: {improvement:.2f}%\n")
                    
                    if improvement > 0:
                        f.write("  * RL policy shows BETTER performance\n")
                    elif improvement < -5:
                        f.write("  * RL policy shows WORSE performance\n")
                    else:
                        f.write("  * Performance is comparable\n")
            
            f.write("\nGenerated Analysis Files:\n")
            f.write("- fct_comparison.png: Comprehensive FCT comparison plots\n")
            f.write("- performance_comparison.csv: Detailed performance metrics\n")
            f.write("- rl_vs_baselines_report.txt: This comprehensive report\n\n")
            
            f.write("Note: This analysis is based on a short 0.001s validation experiment.\n")
            f.write("For comprehensive evaluation, longer simulation times are recommended.\n")
        
        print(f"Comprehensive report saved to {report_path}")
        
    def run_comparison(self):
        """Run the complete comparison analysis"""
        print("Starting RL vs Baseline Comparison Analysis")
        print("=" * 60)
        
        try:
            # Load data
            if not self.load_rl_results():
                print("Failed to load RL results")
                return False
                
            if not self.load_threshold_baselines():
                print("Failed to load threshold baseline results")
                return False
            
            # Run analysis
            self.create_fct_comparison()
            self.create_performance_summary()
            self.create_comprehensive_report()
            
            print("\n" + "=" * 60)
            print("Comparison Analysis Completed Successfully!")
            print(f"Results saved in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error during comparison analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Compare RL Policy vs Threshold-based Deflection')
    parser.add_argument('--rl-dir', default='rl_analysis',
                        help='Directory containing RL analysis results')
    parser.add_argument('--output-dir', default='comparison_analysis',
                        help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    # Check if RL analysis directory exists
    if not os.path.exists(args.rl_dir):
        print(f"Error: RL analysis directory '{args.rl_dir}' does not exist")
        print("Please run analyze_rl_policy.py first")
        sys.exit(1)
    
    comparator = RLBaselineComparator(args.rl_dir, args.output_dir)
    success = comparator.run_comparison()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
