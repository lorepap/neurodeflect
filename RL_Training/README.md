# Offline Reinforcement Learning Framework for Datacenter Network Deflection

This framework implements an offline RL system using Proximal Policy Optimization (PPO) to learn optimal packet deflection policies in datacenter networks. The system learns from collected simulation data to optimize network performance through intelligent packet deflection decisions.

## Overview

The framework consists of several key components:

- **Environment**: Gymnasium-compatible environment for datacenter deflection simulation
- **Agent**: PPO-based RL agent for learning deflection policies
- **Training**: Offline training pipeline using collected threshold experiment data
- **Evaluation**: Comprehensive model evaluation and comparison framework
- **Utilities**: Experiment management and monitoring tools

## Directory Structure

```
RL_Training/
├── environments/
│   └── deflection_env.py          # Datacenter deflection environment
├── agents/
│   └── ppo_agent.py               # PPO agent implementation
├── utils/
│   └── experiment_manager.py      # Experiment management utilities
├── models/                        # Saved trained models
├── logs/                          # Training logs
├── experiments/                   # Experiment configurations and results
├── train_offline_rl.py           # Main training script
├── evaluate_model.py             # Model evaluation script
└── README.md                     # This file
```

## Dependencies

The framework requires the following Python packages:

- PyTorch 1.13.1+
- Gymnasium 0.29.1+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- psutil (for experiment management)

Install dependencies with:
```bash
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install gymnasium==0.29.1 pandas numpy matplotlib seaborn scikit-learn psutil
```

## Quick Start

### 1. Basic Training

Train a PPO model on the threshold dataset:

```bash
python train_offline_rl.py \
    --dataset /path/to/threshold_combined_dataset.csv \
    --epochs 100 \
    --episodes-per-epoch 10 \
    --lr-policy 3e-4
```

### 2. Using Experiment Manager

Create and run managed experiments:

```bash
# Create a new experiment
python utils/experiment_manager.py create my_experiment \
    --epochs 100 \
    --episodes-per-epoch 10 \
    --lr-policy 3e-4

# Run the experiment
python utils/experiment_manager.py run my_experiment

# Check status
python utils/experiment_manager.py status my_experiment

# List all experiments
python utils/experiment_manager.py list
```

### 3. Model Evaluation

Evaluate a trained model:

```bash
python evaluate_model.py \
    --model models/my_experiment/best_model.pth \
    --dataset /path/to/threshold_combined_dataset.csv \
    --episodes 100 \
    --output-dir evaluation_results
```

Or using the experiment manager:

```bash
python utils/experiment_manager.py evaluate my_experiment \
    --episodes 100
```

## Components

### Environment (deflection_env.py)

The `DatacenterDeflectionEnv` class implements a Gymnasium environment that:

- **State Space**: 6-dimensional continuous space representing network conditions
  - Queue length, arrival rate, service rate, utilization, threshold, load
- **Action Space**: Discrete 3-action space for packet decisions
  - 0: Forward packet normally
  - 1: Deflect packet to alternative path
  - 2: Drop packet
- **Reward Function**: Multi-objective reward considering:
  - Queue length minimization
  - Throughput maximization
  - Load balancing
  - Deflection cost

### Agent (ppo_agent.py)

The `PPOAgent` class implements:

- **Policy Network**: Neural network for action probability estimation
- **Value Network**: Neural network for state value estimation
- **PPO Training**: Clipped objective function with GAE advantages
- **Features**:
  - Gradient clipping for stable training
  - Entropy regularization for exploration
  - Learning rate scheduling
  - Model checkpointing

### Training Pipeline (train_offline_rl.py)

The `OfflineRLTrainer` class provides:

- **Data Management**: Automatic dataset loading and preprocessing
- **Training Loop**: Configurable epoch-based training with evaluation
- **Monitoring**: Real-time training statistics and logging
- **Checkpointing**: Automatic model saving and restoration
- **Visualization**: Training progress plots

### Evaluation Framework (evaluate_model.py)

The `ModelEvaluator` class offers:

- **Performance Metrics**: Comprehensive evaluation statistics
- **Baseline Comparison**: Comparison with simple baseline policies
- **Action Analysis**: Detailed action pattern investigation
- **Visualization**: Rich plots and charts for analysis
- **Report Generation**: Automated markdown and JSON reports

## Configuration

### Training Parameters

Key training parameters and their recommended values:

```python
{
    "epochs": 100,                 # Number of training epochs
    "episodes_per_epoch": 10,      # Episodes per epoch
    "max_steps_per_episode": 100,  # Maximum steps per episode
    "lr_policy": 3e-4,             # Policy network learning rate
    "lr_value": 3e-4,              # Value network learning rate
    "gamma": 0.99,                 # Discount factor
    "lambda_gae": 0.95,            # GAE lambda parameter
    "clip_epsilon": 0.2,           # PPO clipping parameter
    "entropy_coeff": 0.01,         # Entropy regularization
    "value_loss_coeff": 0.5,       # Value loss coefficient
    "max_grad_norm": 0.5           # Gradient clipping norm
}
```

### Network Architecture

Default network architectures:

- **Policy Network**: [state_dim] → 128 → 64 → [action_dim]
- **Value Network**: [state_dim] → 128 → 64 → 1

## Experiment Management

The experiment manager provides a comprehensive system for managing training experiments:

### Commands

- `create`: Create new experiment configuration
- `run`: Start training experiment
- `status`: Check experiment status and resource usage
- `list`: List all experiments with status
- `stop`: Stop running experiment
- `evaluate`: Evaluate trained model
- `clean`: Clean up experiment files

### Example Workflow

```bash
# Create multiple experiments with different hyperparameters
python utils/experiment_manager.py create exp_lr_high --lr-policy 1e-3
python utils/experiment_manager.py create exp_lr_low --lr-policy 1e-4
python utils/experiment_manager.py create exp_long --epochs 200

# Run experiments
python utils/experiment_manager.py run exp_lr_high
python utils/experiment_manager.py run exp_lr_low

# Monitor progress
python utils/experiment_manager.py list
python utils/experiment_manager.py status exp_lr_high

# Evaluate best performing experiment
python utils/experiment_manager.py evaluate exp_lr_high
```

## Data Format

The framework expects a CSV dataset with the following columns:

- `queue_length`: Current queue length
- `arrival_rate`: Packet arrival rate
- `service_rate`: Service rate
- `utilization`: Server utilization
- `threshold`: Deflection threshold value
- `load`: Current load
- Additional columns are ignored

The dataset should contain data from multiple threshold values to enable comprehensive learning.

## Output Files

### Training Outputs

- `models/`: Trained model checkpoints
  - `best_model.pth`: Best performing model
  - `final_model.pth`: Final model after training
  - `checkpoint_epoch_*.pth`: Periodic checkpoints
- `logs/`: Training logs and history
  - `training_*.log`: Detailed training logs
  - `training_history.json`: Training metrics history
  - `training_progress.png`: Training progress plots

### Evaluation Outputs

- `evaluation_report.md`: Human-readable evaluation report
- `detailed_results.json`: Comprehensive evaluation results
- `policy_comparison.png`: Comparison with baseline policies
- `action_analysis.png`: Action pattern analysis plots

## Performance Tips

### Training Optimization

1. **Batch Size**: Increase batch size for stable gradients
2. **Learning Rate**: Start with 3e-4, adjust based on convergence
3. **Episode Length**: Limit episodes to prevent overly long sequences
4. **Evaluation Frequency**: Balance between monitoring and training speed

### Hardware Considerations

- **CPU Training**: Framework optimized for CPU training
- **Memory Usage**: Monitor memory with large datasets
- **Parallel Training**: Use multiple processes for data collection

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or episode length
2. **Slow Convergence**: Adjust learning rate or network architecture
3. **Unstable Training**: Increase gradient clipping or reduce learning rate
4. **Poor Performance**: Check reward function and state normalization

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Monitor training with real-time plots:

```bash
# View training progress
python -c "
import matplotlib.pyplot as plt
import json
with open('logs/training_history.json') as f:
    history = json.load(f)
plt.plot(history['episode_rewards'])
plt.show()
"
```

## Extension Points

The framework is designed for extensibility:

### Custom Environments

Extend `DatacenterDeflectionEnv` for different network scenarios:

```python
class CustomNetworkEnv(DatacenterDeflectionEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Custom initialization
    
    def _calculate_reward(self, state, action, next_state):
        # Custom reward function
        return custom_reward
```

### Custom Agents

Implement alternative RL algorithms:

```python
class CustomAgent:
    def get_action(self, state):
        # Custom action selection
        pass
    
    def update(self, experiences):
        # Custom learning update
        pass
```

### Custom Metrics

Add custom evaluation metrics:

```python
def custom_evaluation_metric(states, actions, rewards):
    # Custom metric calculation
    return metric_value
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{datacenter_deflection_rl,
    title={Offline Reinforcement Learning for Datacenter Network Deflection Optimization},
    author={Your Name},
    year={2024},
    note={Implementation of PPO-based offline RL for network packet deflection}
}
```

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions, issues, or contributions, please contact [your-email@domain.com] or open an issue on the project repository.
   - `PPOAgent`: Complete PPO implementation with policy and value networks
   - `NetworkRLTrainer`: High-level training orchestration
   - Generalized Advantage Estimation (GAE)
   - Policy gradient with clipping
   - Value function learning

3. **`train_network_rl.py`** - Main training script
   - Command-line interface for training
   - Hyperparameter configuration
   - Experiment management
   - Dataset analysis

4. **`evaluate_model.py`** - Model evaluation and analysis
   - Performance metrics calculation
   - Threshold-based analysis
   - Visualization and reporting
   - Confusion matrix generation

## Features

### Algorithm Features
- **PPO Implementation**: Proximal Policy Optimization with clipping
- **GAE**: Generalized Advantage Estimation for variance reduction
- **Offline Learning**: Learns from pre-collected simulation data
- **Multi-threshold Analysis**: Evaluates performance across different deflection thresholds

### Network Architecture
- **Policy Network**: Multi-layer perceptron for action probability prediction
- **Value Network**: Multi-layer perceptron for state value estimation
- **Configurable Architecture**: Adjustable hidden layer dimensions
- **Gradient Clipping**: Prevents gradient explosion during training

### Training Infrastructure
- **Tensorboard Integration**: Real-time training monitoring
- **Checkpoint Management**: Model saving and resuming
- **Hyperparameter Configuration**: Flexible training configuration
- **Batch Processing**: Efficient mini-batch training

### Evaluation Tools
- **Performance Metrics**: Accuracy, reward analysis, value function quality
- **Threshold Analysis**: Performance breakdown by deflection threshold
- **Visualization**: Training curves, confusion matrices, action distributions
- **Comprehensive Reporting**: JSON reports with detailed statistics

## Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

### 2. Prepare Dataset

First, generate the threshold dataset using the simulation pipeline:

```bash
# Navigate to simulation directory
cd ../Omnet_Sims/dc_simulations/simulations/sims

# Run threshold experiments and create dataset
python3 threshold_pipeline.py --thresholds 0.3,0.5,0.7,0.9 --experiment-name test_rl

# Return to RL training directory
cd ../../../../RL_Training
```

### 3. Analyze Dataset

```bash
# Analyze the dataset before training
python3 train_network_rl.py --analyze-only --dataset ../Omnet_Sims/dc_simulations/simulations/sims/dataset_output/combined_threshold_dataset.csv
```

### 4. Train Model

```bash
# Basic training
python3 train_network_rl.py --experiment-name my_first_experiment

# Training with custom parameters
python3 train_network_rl.py \
    --experiment-name advanced_training \
    --learning-rate 1e-4 \
    --batch-size 128 \
    --total-timesteps 200000 \
    --hidden-dims 256,128,64
```

### 5. Evaluate Model

```bash
# Evaluate trained model
python3 evaluate_model.py \
    --model experiments/my_first_experiment/final_model.pt \
    --output-dir evaluation_results
```

## Configuration

### PPO Hyperparameters

Key hyperparameters for tuning:

- **`learning_rate`** (default: 3e-4): Learning rate for neural networks
- **`gamma`** (default: 0.99): Discount factor for future rewards
- **`clip_epsilon`** (default: 0.2): PPO clipping parameter
- **`batch_size`** (default: 64): Training batch size
- **`epochs_per_update`** (default: 4): Number of gradient steps per batch

### Network Architecture

- **`hidden_dims`** (default: [128, 64]): Hidden layer sizes
- **`state_dim`** (default: 7): Input state dimension
- **`action_dim`** (default: 3): Number of possible actions

### Training Schedule

- **`total_timesteps`** (default: 100000): Total training samples
- **`eval_freq`** (default: 1000): Evaluation frequency
- **`save_freq`** (default: 5000): Model checkpoint frequency

## Command Line Examples

### Training Examples

```bash
# Quick test run
python3 train_network_rl.py --total-timesteps 10000 --experiment-name quick_test

# High-performance training
python3 train_network_rl.py \
    --experiment-name production_run \
    --learning-rate 1e-4 \
    --batch-size 256 \
    --total-timesteps 500000 \
    --hidden-dims 512,256,128 \
    --device cuda

# Resume training
python3 train_network_rl.py \
    --resume experiments/production_run/checkpoint_100000.pt \
    --total-timesteps 200000
```

### Evaluation Examples

```bash
# Basic evaluation
python3 evaluate_model.py --model experiments/quick_test/final_model.pt

# Detailed evaluation with custom output
python3 evaluate_model.py \
    --model experiments/production_run/final_model.pt \
    --output-dir detailed_analysis \
    --dataset custom_dataset.csv

# Evaluation without plots (for automated pipelines)
python3 evaluate_model.py \
    --model final_model.pt \
    --no-plots \
    --output-dir automated_eval
```

## Data Format

### Input Dataset

The framework expects a CSV dataset with the following columns:

- **State features** (7 columns): Network state representation
  - `queue_length`: Current queue length
  - `arrival_rate`: Packet arrival rate
  - `service_rate`: Packet service rate
  - `utilization`: Link utilization
  - `deflection_rate`: Current deflection rate
  - `buffer_occupancy`: Buffer occupancy ratio
  - `threshold`: Deflection threshold value

- **Action**: Deflection decision (0: no deflection, 1: light deflection, 2: heavy deflection)
- **Reward**: Performance reward signal
- **Done**: Episode termination flag

### Output Format

Training produces:
- **Model checkpoints**: `.pt` files with model weights and training state
- **Tensorboard logs**: Real-time training monitoring
- **Evaluation reports**: JSON files with detailed performance metrics
- **Visualization plots**: Training curves and performance analysis

## Monitoring Training

### Tensorboard

Monitor training in real-time:

```bash
# Start tensorboard
tensorboard --logdir experiments/

# Open browser to http://localhost:6006
```

Key metrics to watch:
- **Policy Loss**: Should generally decrease
- **Value Loss**: Should decrease and stabilize
- **Entropy**: Should decrease gradually (exploration vs exploitation)
- **Explained Variance**: Should increase (value function quality)

### Log Files

Training logs are saved to `training.log` with detailed information about:
- Training progress
- Evaluation results
- Error messages
- Performance statistics

## Advanced Usage

### Custom Reward Engineering

Modify the reward function in `network_rl_framework.py`:

```python
def calculate_reward(self, state: NetworkState, action: int) -> float:
    # Custom reward logic
    throughput_reward = state.arrival_rate * (1 - state.deflection_rate)
    latency_penalty = -state.queue_length * 0.1
    deflection_penalty = -action * 0.05
    
    return throughput_reward + latency_penalty + deflection_penalty
```

### Multi-Objective Optimization

Extend the framework for multi-objective optimization:

```python
# In PolicyNetwork
def forward(self, state):
    # Multi-head outputs for different objectives
    shared_features = self.shared_layers(state)
    throughput_logits = self.throughput_head(shared_features)
    latency_logits = self.latency_head(shared_features)
    return torch.cat([throughput_logits, latency_logits], dim=-1)
```

### Curriculum Learning

Implement curriculum learning by gradually increasing threshold difficulty:

```python
# In training loop
def get_curriculum_thresholds(epoch):
    if epoch < 100:
        return [0.5]  # Easy threshold
    elif epoch < 200:
        return [0.3, 0.7]  # Medium difficulty
    else:
        return [0.3, 0.5, 0.7, 0.9]  # Full difficulty
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python3 train_network_rl.py --batch-size 32 --mini-batch-size 8
   ```

2. **Poor Convergence**
   ```bash
   # Try different learning rate
   python3 train_network_rl.py --learning-rate 1e-5
   
   # Increase training time
   python3 train_network_rl.py --total-timesteps 500000
   ```

3. **Overfitting**
   ```bash
   # Increase regularization
   python3 train_network_rl.py --entropy-coef 0.05
   
   # Use smaller network
   python3 train_network_rl.py --hidden-dims 64,32
   ```

4. **Dataset Issues**
   ```bash
   # Check dataset format
   python3 train_network_rl.py --analyze-only
   
   # Verify data quality
   python3 -c "import pandas as pd; df = pd.read_csv('dataset.csv'); print(df.describe())"
   ```

### Performance Optimization

1. **Use GPU acceleration** (if available):
   ```bash
   python3 train_network_rl.py --device cuda
   ```

2. **Increase batch size** for better GPU utilization:
   ```bash
   python3 train_network_rl.py --batch-size 256 --mini-batch-size 64
   ```

3. **Parallel data loading**:
   ```python
   # Modify DataLoader in trainer
   DataLoader(dataset, batch_size=batch_size, num_workers=4)
   ```

## Integration with Simulation Environment

### Real-time Policy Testing

To test trained policies in the simulation environment:

```python
# Load trained model
agent = PPOAgent(config, device)
agent.load_checkpoint('experiments/final_model.pt')

# Use in simulation
def get_deflection_decision(network_state):
    state_tensor = torch.tensor(network_state).float()
    _, action = agent.policy_net.get_action(state_tensor, deterministic=True)
    return action.item()
```

### Continuous Learning

Implement online learning for continuous improvement:

```python
# Collect new experiences
new_experiences = collect_simulation_data()

# Fine-tune existing model
agent.load_checkpoint('pretrained_model.pt')
agent.train_on_new_data(new_experiences)
```

## Performance Benchmarks

### Expected Performance

On the threshold dataset:
- **Training time**: ~5-15 minutes for 100k timesteps (CPU)
- **Accuracy**: 70-90% depending on threshold complexity
- **Memory usage**: ~1-2 GB RAM for typical configurations

### Scaling Guidelines

- **Small dataset** (<10k samples): Use default parameters
- **Medium dataset** (10k-100k samples): Increase batch size to 128-256
- **Large dataset** (>100k samples): Use GPU acceleration and larger networks

## Contributing

To extend the framework:

1. **Add new network architectures** in `network_rl_framework.py`
2. **Implement new algorithms** alongside PPO in `ppo_trainer.py`
3. **Add evaluation metrics** in `evaluate_model.py`
4. **Create specialized datasets** for different network scenarios

## References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [PyTorch Documentation](https://pytorch.org/docs/)

## License

This framework is part of the practical deflection project and follows the same license terms.
