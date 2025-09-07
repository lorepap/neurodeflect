#!/usr/bin/env python3
"""
RL Policy Analysis Script

This script analyzes the extracted data from RL policy simulations following 
the same methodology as the threshold experiments. It processes flow completion 
times, network metrics, and deflection patterns to evaluate RL policy performance.

Usage:
    python analyze_rl_policy.py [--data-dir extracted_results] [--output-dir rl_analysis]
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

class RLPolicyAnalyzer:
    def __init__(self, data_dir="extracted_results", output_dir="rl_analysis"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Store analysis results
        self.results = {}
        
    def load_csv_data(self, category):
        """Load CSV data for a specific category"""
        category_dir = self.data_dir / category
        if not category_dir.exists():
            print(f"Warning: Directory {category_dir} does not exist")
            return None
            
        csv_files = list(category_dir.glob("*.csv"))
        if not csv_files:
            print(f"Warning: No CSV files found in {category_dir}")
            return None
            
        print(f"Loading {len(csv_files)} files from {category}")
        
        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                if not df.empty:
                    df['source_file'] = csv_file.name
                    dfs.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {e}")
                
        if not dfs:
            print(f"Warning: No valid data found in {category}")
            return None
            
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(combined_df)} records from {category}")
        return combined_df
        
    def analyze_flow_completion_times(self):
        """Analyze Flow Completion Times (FCT)"""
        print("\n=== Analyzing Flow Completion Times ===")
        
        # Load flow ended data
        flow_ended_df = self.load_csv_data("FLOW_ENDED")
        if flow_ended_df is None:
            print("No flow completion data available")
            return
            
        # Process flow completion times
        # The CSV format has time series data - extract completion times
        fcts = []
        requester_ids = []
        
        for _, row in flow_ended_df.iterrows():
            # Parse the time series values (last column should contain completion times)
            time_values = []
            requester_values = []
            
            # Find numeric columns (timestamps and requester IDs)
            numeric_cols = flow_ended_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                val = row[col]
                if pd.notna(val) and val > 0:
                    if 'time' in col.lower() or col.isdigit():
                        time_values.append(val)
                    else:
                        requester_values.append(val)
            
            fcts.extend(time_values)
            requester_ids.extend(requester_values)
        
        if not fcts:
            print("No valid FCT data found")
            return
            
        fcts = np.array(fcts)
        print(f"Analyzed {len(fcts)} flow completion times")
        
        # Calculate FCT statistics
        fct_stats = {
            'mean_fct': np.mean(fcts),
            'median_fct': np.median(fcts),
            'p95_fct': np.percentile(fcts, 95),
            'p99_fct': np.percentile(fcts, 99),
            'std_fct': np.std(fcts),
            'min_fct': np.min(fcts),
            'max_fct': np.max(fcts),
            'total_flows': len(fcts)
        }
        
        self.results['fct_stats'] = fct_stats
        
        # Print FCT statistics
        print(f"FCT Statistics:")
        print(f"  Total flows: {fct_stats['total_flows']}")
        print(f"  Mean FCT: {fct_stats['mean_fct']:.6f}s")
        print(f"  Median FCT: {fct_stats['median_fct']:.6f}s")
        print(f"  95th percentile: {fct_stats['p95_fct']:.6f}s")
        print(f"  99th percentile: {fct_stats['p99_fct']:.6f}s")
        print(f"  Standard deviation: {fct_stats['std_fct']:.6f}s")
        
        # Create FCT distribution plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(fcts, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Flow Completion Time (s)')
        plt.ylabel('Frequency')
        plt.title('FCT Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.hist(fcts, bins=50, alpha=0.7, edgecolor='black', cumulative=True, density=True)
        plt.xlabel('Flow Completion Time (s)')
        plt.ylabel('Cumulative Probability')
        plt.title('FCT CDF')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.boxplot(fcts)
        plt.ylabel('Flow Completion Time (s)')
        plt.title('FCT Box Plot')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        # Log scale for better visibility of tail
        plt.hist(fcts, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Flow Completion Time (s)')
        plt.ylabel('Frequency')
        plt.yscale('log')
        plt.title('FCT Distribution (Log Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fct_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save FCT data to CSV
        fct_df = pd.DataFrame({'fct': fcts})
        fct_df.to_csv(self.output_dir / 'fct_data.csv', index=False)
        
        print(f"FCT analysis saved to {self.output_dir}/fct_analysis.png")
        
    def analyze_network_metrics(self):
        """Analyze network-level metrics"""
        print("\n=== Analyzing Network Metrics ===")
        
        # Analyze request patterns
        request_df = self.load_csv_data("REQUEST_SENT")
        if request_df is not None:
            print(f"Request analysis: {len(request_df)} request records")
            self.results['total_requests'] = len(request_df)
        
        # Analyze reply patterns  
        reply_df = self.load_csv_data("REPLY_LENGTH_ASKED")
        if reply_df is not None:
            print(f"Reply analysis: {len(reply_df)} reply records")
            self.results['total_replies'] = len(reply_df)
        
        # Analyze requester distribution
        requester_df = self.load_csv_data("REQUESTER_ID")
        if requester_df is not None:
            print(f"Requester analysis: {len(requester_df)} requester records")
            
            # Extract unique requesters
            requesters = set()
            for _, row in requester_df.iterrows():
                numeric_cols = requester_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    val = row[col]
                    if pd.notna(val) and val > 0:
                        requesters.add(int(val))
            
            self.results['unique_requesters'] = len(requesters)
            print(f"  Unique requesters: {len(requesters)}")
            
        # Create network metrics summary
        plt.figure(figsize=(10, 6))
        
        metrics = []
        values = []
        
        if 'total_requests' in self.results:
            metrics.append('Total Requests')
            values.append(self.results['total_requests'])
            
        if 'total_replies' in self.results:
            metrics.append('Total Replies')
            values.append(self.results['total_replies'])
            
        if 'unique_requesters' in self.results:
            metrics.append('Unique Requesters')
            values.append(self.results['unique_requesters'])
        
        if metrics:
            plt.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'orange'][:len(metrics)])
            plt.ylabel('Count')
            plt.title('Network Metrics Summary')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for i, v in enumerate(values):
                plt.text(i, v + max(values) * 0.01, str(v), ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'network_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Network metrics saved to {self.output_dir}/network_metrics.png")
        
    def analyze_flow_patterns(self):
        """Analyze flow start/end patterns"""
        print("\n=== Analyzing Flow Patterns ===")
        
        # Load flow started data
        flow_started_df = self.load_csv_data("FLOW_STARTED")
        flow_ended_df = self.load_csv_data("FLOW_ENDED")
        
        if flow_started_df is None and flow_ended_df is None:
            print("No flow pattern data available")
            return
            
        plt.figure(figsize=(12, 8))
        
        if flow_started_df is not None:
            plt.subplot(2, 1, 1)
            # Extract flow start times
            start_times = []
            for _, row in flow_started_df.iterrows():
                numeric_cols = flow_started_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    val = row[col]
                    if pd.notna(val) and val > 0 and val < 1.0:  # Within simulation time
                        start_times.append(val)
            
            if start_times:
                plt.hist(start_times, bins=30, alpha=0.7, color='green', edgecolor='black')
                plt.xlabel('Time (s)')
                plt.ylabel('Flow Starts')
                plt.title(f'Flow Start Pattern ({len(start_times)} flows)')
                plt.grid(True, alpha=0.3)
                
                self.results['flows_started'] = len(start_times)
                print(f"  Flows started: {len(start_times)}")
        
        if flow_ended_df is not None:
            plt.subplot(2, 1, 2)
            # Extract flow end times (already processed in FCT analysis)
            end_times = []
            for _, row in flow_ended_df.iterrows():
                numeric_cols = flow_ended_df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    val = row[col]
                    if pd.notna(val) and val > 0 and val < 1.0:  # Within simulation time
                        end_times.append(val)
            
            if end_times:
                plt.hist(end_times, bins=30, alpha=0.7, color='red', edgecolor='black')
                plt.xlabel('Time (s)')
                plt.ylabel('Flow Ends')
                plt.title(f'Flow End Pattern ({len(end_times)} flows)')
                plt.grid(True, alpha=0.3)
                
                self.results['flows_ended'] = len(end_times)
                print(f"  Flows ended: {len(end_times)}")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'flow_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Flow pattern analysis saved to {self.output_dir}/flow_patterns.png")
        
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("\n=== Creating Summary Report ===")
        
        report_path = self.output_dir / 'rl_policy_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("RL Policy Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Simulation Configuration:\n")
            f.write("- Policy: Reinforcement Learning (IQL)\n")
            f.write("- Simulation Time: 0.001s\n")
            f.write("- Network: LeafSpine 1G (4 spines, 8 aggregates, 40 servers)\n")
            f.write("- Protocol: DCTCP with RL deflection decisions\n\n")
            
            if 'fct_stats' in self.results:
                stats = self.results['fct_stats']
                f.write("Flow Completion Time Analysis:\n")
                f.write(f"- Total flows completed: {stats['total_flows']}\n")
                f.write(f"- Mean FCT: {stats['mean_fct']:.6f}s\n")
                f.write(f"- Median FCT: {stats['median_fct']:.6f}s\n")
                f.write(f"- 95th percentile FCT: {stats['p95_fct']:.6f}s\n")
                f.write(f"- 99th percentile FCT: {stats['p99_fct']:.6f}s\n")
                f.write(f"- FCT standard deviation: {stats['std_fct']:.6f}s\n")
                f.write(f"- Min FCT: {stats['min_fct']:.6f}s\n")
                f.write(f"- Max FCT: {stats['max_fct']:.6f}s\n\n")
            
            f.write("Network Activity Summary:\n")
            if 'total_requests' in self.results:
                f.write(f"- Total requests sent: {self.results['total_requests']}\n")
            if 'total_replies' in self.results:
                f.write(f"- Total replies: {self.results['total_replies']}\n")
            if 'unique_requesters' in self.results:
                f.write(f"- Unique requesting nodes: {self.results['unique_requesters']}\n")
            if 'flows_started' in self.results:
                f.write(f"- Flows started: {self.results['flows_started']}\n")
            if 'flows_ended' in self.results:
                f.write(f"- Flows ended: {self.results['flows_ended']}\n")
                
            f.write("\nGenerated Analysis Files:\n")
            f.write("- fct_analysis.png: Flow completion time distributions\n")
            f.write("- network_metrics.png: Network activity summary\n")
            f.write("- flow_patterns.png: Flow start/end temporal patterns\n")
            f.write("- fct_data.csv: Raw FCT data for further analysis\n")
            f.write("- rl_policy_analysis_report.txt: This summary report\n\n")
            
            f.write("Analysis completed successfully!\n")
        
        print(f"Summary report saved to {report_path}")
        
    def run_full_analysis(self):
        """Run the complete RL policy analysis pipeline"""
        print("Starting RL Policy Analysis")
        print("=" * 50)
        
        try:
            self.analyze_flow_completion_times()
            self.analyze_network_metrics()
            self.analyze_flow_patterns()
            self.create_summary_report()
            
            print("\n" + "=" * 50)
            print("RL Policy Analysis Completed Successfully!")
            print(f"Results saved in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description='Analyze RL Policy simulation results')
    parser.add_argument('--data-dir', default='extracted_results',
                        help='Directory containing extracted CSV files')
    parser.add_argument('--output-dir', default='rl_analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        print("Please run the RL extraction first or specify correct path")
        sys.exit(1)
    
    analyzer = RLPolicyAnalyzer(args.data_dir, args.output_dir)
    success = analyzer.run_full_analysis()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
