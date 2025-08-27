"""
RL Training Utility Script

This script provides utilities for running, monitoring, and managing
RL training experiments for datacenter deflection optimization.
"""

import os
import sys
import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import psutil

# Add the RL_Training directory to the path
sys.path.append('/home/ubuntu/practical_deflection/RL_Training')


class ExperimentManager:
    """
    Manager for RL training experiments.
    """
    
    def __init__(self, 
                 base_dir: str = "/home/ubuntu/practical_deflection/RL_Training",
                 dataset_path: str = "/home/ubuntu/practical_deflection/threshold_combined_dataset.csv"):
        """
        Initialize the experiment manager.
        
        Args:
            base_dir: Base directory for RL training
            dataset_path: Path to the dataset
        """
        self.base_dir = Path(base_dir)
        self.dataset_path = dataset_path
        self.experiments_dir = self.base_dir / "experiments"
        self.logs_dir = self.base_dir / "logs"
        self.models_dir = self.base_dir / "models"
        
        # Create directories
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment manager initialized")
        print(f"Base directory: {self.base_dir}")
        print(f"Dataset: {self.dataset_path}")
    
    def create_experiment_config(self,
                               experiment_name: str,
                               epochs: int = 100,
                               episodes_per_epoch: int = 10,
                               max_steps: int = 100,
                               lr_policy: float = 3e-4,
                               lr_value: float = 3e-4,
                               gamma: float = 0.99,
                               clip_epsilon: float = 0.2,
                               **kwargs) -> Dict:
        """
        Create an experiment configuration.
        
        Args:
            experiment_name: Name of the experiment
            epochs: Number of training epochs
            episodes_per_epoch: Episodes per epoch
            max_steps: Maximum steps per episode
            lr_policy: Policy network learning rate
            lr_value: Value network learning rate
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            **kwargs: Additional parameters
            
        Returns:
            Experiment configuration dictionary
        """
        config = {
            'experiment_name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'dataset_path': self.dataset_path,
            'training_params': {
                'epochs': epochs,
                'episodes_per_epoch': episodes_per_epoch,
                'max_steps_per_episode': max_steps,
                'lr_policy': lr_policy,
                'lr_value': lr_value,
                'gamma': gamma,
                'clip_epsilon': clip_epsilon,
                **kwargs
            },
            'paths': {
                'log_dir': str(self.logs_dir / experiment_name),
                'model_dir': str(self.models_dir / experiment_name),
                'experiment_dir': str(self.experiments_dir / experiment_name)
            }
        }
        
        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Experiment config created: {config_path}")
        return config
    
    def run_experiment(self, experiment_name: str, config: Optional[Dict] = None) -> subprocess.Popen:
        """
        Run an RL training experiment.
        
        Args:
            experiment_name: Name of the experiment
            config: Optional config dictionary. If None, loads from file.
            
        Returns:
            Process object for the running experiment
        """
        exp_dir = self.experiments_dir / experiment_name
        
        if config is None:
            config_path = exp_dir / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Prepare training command
        train_script = self.base_dir / "train_offline_rl.py"
        
        cmd = [
            "python", str(train_script),
            "--dataset", config['dataset_path'],
            "--epochs", str(config['training_params']['epochs']),
            "--episodes-per-epoch", str(config['training_params']['episodes_per_epoch']),
            "--max-steps", str(config['training_params']['max_steps_per_episode']),
            "--lr-policy", str(config['training_params']['lr_policy']),
            "--lr-value", str(config['training_params']['lr_value']),
            "--log-dir", config['paths']['log_dir'],
            "--model-dir", config['paths']['model_dir']
        ]
        
        # Add optional parameters
        if 'device' in config['training_params']:
            cmd.extend(["--device", config['training_params']['device']])
        
        print(f"Starting experiment: {experiment_name}")
        print(f"Command: {' '.join(cmd)}")
        
        # Create log and model directories
        Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['paths']['model_dir']).mkdir(parents=True, exist_ok=True)
        
        # Start the process
        log_file = exp_dir / "training.log"
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(self.base_dir)
            )
        
        # Save process info
        process_info = {
            'pid': process.pid,
            'started_at': datetime.now().isoformat(),
            'command': cmd,
            'log_file': str(log_file)
        }
        
        process_file = exp_dir / "process.json"
        with open(process_file, 'w') as f:
            json.dump(process_info, f, indent=2)
        
        print(f"Experiment started with PID: {process.pid}")
        print(f"Log file: {log_file}")
        
        return process
    
    def check_experiment_status(self, experiment_name: str) -> Dict:
        """
        Check the status of a running experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Status information dictionary
        """
        exp_dir = self.experiments_dir / experiment_name
        process_file = exp_dir / "process.json"
        
        if not process_file.exists():
            return {'status': 'not_started', 'message': 'No process file found'}
        
        with open(process_file, 'r') as f:
            process_info = json.load(f)
        
        pid = process_info['pid']
        
        # Check if process is still running
        try:
            process = psutil.Process(pid)
            if process.is_running():
                status = {
                    'status': 'running',
                    'pid': pid,
                    'started_at': process_info['started_at'],
                    'cpu_percent': process.cpu_percent(),
                    'memory_percent': process.memory_percent(),
                    'runtime': self._calculate_runtime(process_info['started_at'])
                }
            else:
                status = {
                    'status': 'completed',
                    'pid': pid,
                    'started_at': process_info['started_at'],
                    'runtime': self._calculate_runtime(process_info['started_at'])
                }
        except psutil.NoSuchProcess:
            status = {
                'status': 'completed',
                'pid': pid,
                'started_at': process_info['started_at'],
                'runtime': self._calculate_runtime(process_info['started_at'])
            }
        
        # Check for output files
        log_dir = Path(process_info.get('log_file', '')).parent
        if log_dir.exists():
            status['log_files'] = list(log_dir.glob('*.log'))
        
        model_dir = exp_dir.parent.parent / "models" / experiment_name
        if model_dir.exists():
            status['model_files'] = list(model_dir.glob('*.pth'))
        
        return status
    
    def _calculate_runtime(self, start_time: str) -> str:
        """Calculate runtime from start time."""
        start = datetime.fromisoformat(start_time)
        runtime = datetime.now() - start
        return str(runtime).split('.')[0]  # Remove microseconds
    
    def list_experiments(self) -> List[Dict]:
        """
        List all experiments.
        
        Returns:
            List of experiment information dictionaries
        """
        experiments = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                exp_info = {
                    'name': exp_dir.name,
                    'path': str(exp_dir)
                }
                
                # Load config if available
                config_file = exp_dir / "config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    exp_info['created_at'] = config.get('created_at', 'unknown')
                    exp_info['epochs'] = config.get('training_params', {}).get('epochs', 'unknown')
                
                # Get status
                status = self.check_experiment_status(exp_dir.name)
                exp_info['status'] = status['status']
                
                experiments.append(exp_info)
        
        return sorted(experiments, key=lambda x: x.get('created_at', ''))
    
    def stop_experiment(self, experiment_name: str) -> bool:
        """
        Stop a running experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            True if successfully stopped, False otherwise
        """
        status = self.check_experiment_status(experiment_name)
        
        if status['status'] != 'running':
            print(f"Experiment {experiment_name} is not running")
            return False
        
        try:
            process = psutil.Process(status['pid'])
            process.terminate()
            
            # Wait for graceful termination
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()
            
            print(f"Experiment {experiment_name} stopped")
            return True
            
        except psutil.NoSuchProcess:
            print(f"Process {status['pid']} not found")
            return False
    
    def evaluate_experiment(self, 
                          experiment_name: str,
                          model_name: str = "best_model.pth",
                          num_episodes: int = 100) -> str:
        """
        Evaluate a trained model from an experiment.
        
        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model file
            num_episodes: Number of episodes for evaluation
            
        Returns:
            Path to the evaluation report
        """
        model_dir = self.models_dir / experiment_name
        model_path = model_dir / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Prepare evaluation command
        eval_script = self.base_dir / "evaluate_model.py"
        output_dir = self.experiments_dir / experiment_name / "evaluation"
        
        cmd = [
            "python", str(eval_script),
            "--model", str(model_path),
            "--dataset", self.dataset_path,
            "--episodes", str(num_episodes),
            "--output-dir", str(output_dir)
        ]
        
        print(f"Evaluating model: {model_path}")
        print(f"Command: {' '.join(cmd)}")
        
        # Run evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(self.base_dir))
        
        if result.returncode == 0:
            report_path = output_dir / "evaluation_report.md"
            print(f"Evaluation completed successfully")
            print(f"Report: {report_path}")
            return str(report_path)
        else:
            print(f"Evaluation failed:")
            print(result.stderr)
            raise RuntimeError("Evaluation failed")
    
    def clean_experiment(self, experiment_name: str, keep_best_model: bool = True) -> bool:
        """
        Clean up experiment files.
        
        Args:
            experiment_name: Name of the experiment
            keep_best_model: Whether to keep the best model
            
        Returns:
            True if successful, False otherwise
        """
        # Stop experiment if running
        status = self.check_experiment_status(experiment_name)
        if status['status'] == 'running':
            print(f"Stopping running experiment: {experiment_name}")
            self.stop_experiment(experiment_name)
        
        exp_dir = self.experiments_dir / experiment_name
        model_dir = self.models_dir / experiment_name
        log_dir = self.logs_dir / experiment_name
        
        try:
            # Clean log files
            if log_dir.exists():
                for log_file in log_dir.glob('*.log'):
                    log_file.unlink()
                print(f"Cleaned log files for {experiment_name}")
            
            # Clean model files (except best model if requested)
            if model_dir.exists():
                for model_file in model_dir.glob('*.pth'):
                    if keep_best_model and model_file.name == 'best_model.pth':
                        continue
                    model_file.unlink()
                print(f"Cleaned model files for {experiment_name}")
            
            # Clean experiment directory (except config)
            if exp_dir.exists():
                for file in exp_dir.iterdir():
                    if file.name not in ['config.json']:
                        if file.is_file():
                            file.unlink()
                        elif file.is_dir():
                            import shutil
                            shutil.rmtree(file)
                print(f"Cleaned experiment directory for {experiment_name}")
            
            return True
            
        except Exception as e:
            print(f"Error cleaning experiment {experiment_name}: {e}")
            return False


def main():
    """Main function for experiment management."""
    parser = argparse.ArgumentParser(description='RL Training Experiment Manager')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create experiment
    create_parser = subparsers.add_parser('create', help='Create a new experiment')
    create_parser.add_argument('name', type=str, help='Experiment name')
    create_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    create_parser.add_argument('--episodes-per-epoch', type=int, default=10, help='Episodes per epoch')
    create_parser.add_argument('--max-steps', type=int, default=100, help='Max steps per episode')
    create_parser.add_argument('--lr-policy', type=float, default=3e-4, help='Policy learning rate')
    create_parser.add_argument('--lr-value', type=float, default=3e-4, help='Value learning rate')
    create_parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    
    # Run experiment
    run_parser = subparsers.add_parser('run', help='Run an experiment')
    run_parser.add_argument('name', type=str, help='Experiment name')
    
    # Check status
    status_parser = subparsers.add_parser('status', help='Check experiment status')
    status_parser.add_argument('name', type=str, help='Experiment name')
    
    # List experiments
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Stop experiment
    stop_parser = subparsers.add_parser('stop', help='Stop a running experiment')
    stop_parser.add_argument('name', type=str, help='Experiment name')
    
    # Evaluate experiment
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('name', type=str, help='Experiment name')
    eval_parser.add_argument('--model', type=str, default='best_model.pth', help='Model file name')
    eval_parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    
    # Clean experiment
    clean_parser = subparsers.add_parser('clean', help='Clean experiment files')
    clean_parser.add_argument('name', type=str, help='Experiment name')
    clean_parser.add_argument('--keep-best', action='store_true', help='Keep best model')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize manager
    manager = ExperimentManager()
    
    if args.command == 'create':
        config = manager.create_experiment_config(
            experiment_name=args.name,
            epochs=args.epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            max_steps=args.max_steps,
            lr_policy=args.lr_policy,
            lr_value=args.lr_value,
            device=args.device
        )
        print(f"Experiment '{args.name}' created successfully")
    
    elif args.command == 'run':
        try:
            process = manager.run_experiment(args.name)
            print(f"Experiment '{args.name}' started with PID {process.pid}")
        except Exception as e:
            print(f"Failed to start experiment: {e}")
    
    elif args.command == 'status':
        status = manager.check_experiment_status(args.name)
        print(f"\nExperiment: {args.name}")
        print(f"Status: {status['status']}")
        
        if 'pid' in status:
            print(f"PID: {status['pid']}")
        if 'started_at' in status:
            print(f"Started: {status['started_at']}")
        if 'runtime' in status:
            print(f"Runtime: {status['runtime']}")
        if 'cpu_percent' in status:
            print(f"CPU: {status['cpu_percent']:.1f}%")
        if 'memory_percent' in status:
            print(f"Memory: {status['memory_percent']:.1f}%")
    
    elif args.command == 'list':
        experiments = manager.list_experiments()
        
        if not experiments:
            print("No experiments found")
        else:
            print(f"\n{'Name':<20} {'Status':<10} {'Epochs':<8} {'Created':<20}")
            print("-" * 65)
            for exp in experiments:
                name = exp['name'][:19]
                status = exp['status']
                epochs = str(exp.get('epochs', 'N/A'))
                created = exp.get('created_at', 'unknown')[:19]
                print(f"{name:<20} {status:<10} {epochs:<8} {created:<20}")
    
    elif args.command == 'stop':
        success = manager.stop_experiment(args.name)
        if success:
            print(f"Experiment '{args.name}' stopped successfully")
        else:
            print(f"Failed to stop experiment '{args.name}'")
    
    elif args.command == 'evaluate':
        try:
            report_path = manager.evaluate_experiment(
                args.name, 
                args.model, 
                args.episodes
            )
            print(f"Evaluation report saved to: {report_path}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
    
    elif args.command == 'clean':
        success = manager.clean_experiment(args.name, keep_best_model=args.keep_best)
        if success:
            print(f"Experiment '{args.name}' cleaned successfully")
        else:
            print(f"Failed to clean experiment '{args.name}'")


if __name__ == "__main__":
    main()
