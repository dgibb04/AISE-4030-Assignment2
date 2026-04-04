# AISE 4030 Assignment 02: Policy Gradient Methods

Policy gradient implementation on BipedalWalker-v3 environment.

## Overview

This project implements and compares two policy gradient algorithms:

- **REINFORCE with Baseline**: A Monte Carlo policy gradient method using a learned value function to reduce variance.
- **Proximal Policy Optimization (PPO)**: A modern actor-critic method with clipped surrogate objective for stable training.

Both algorithms are trained on the BipedalWalker-v3 continuous control task.

## Installation

### Prerequisites

- Python 3.10+
- Conda or Miniconda

### Setup

1. Create the conda environment:
```bash
conda create -n AISE4030_A2 python=3.10 -y
conda activate AISE4030_A2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note on Box2D**: If you encounter build issues with Box2D on Windows:
```bash
conda install swig -y
pip install gymnasium[box2d]
```

3. Verify the installation:
```bash
python -c "
import gymnasium as gym
env = gym.make('BipedalWalker-v3')
obs, info = env.reset(seed=42)
print('Environment created successfully!')
print('Observation shape:', obs.shape)
print('Action space:', env.action_space)
print('Action range:', env.action_space.low, 'to', env.action_space.high)
env.close()
print('Setup is complete!')
"
```

## Project Structure

```
BipedalWalker_PolicyGradient/
├── config.yaml           # All hyperparameters and settings
├── environment.py        # BipedalWalker environment setup
├── policy_network.py     # Gaussian policy network (actor, shared)
├── value_network.py      # Value function network (critic, shared)
├── reinforce_agent.py    # REINFORCE with Baseline agent (Task 1)
├── ppo_agent.py          # PPO agent with clipped objective (Task 2)
├── rollout_buffer.py     # On-policy trajectory storage for PPO
├── training_script.py    # Main entry point: training and evaluation
├── utils.py              # Config loading, plotting, logging helpers
├── generate_plots.py     # Plot generation (Section 4.4.1)
├── check_cuda.py         # CUDA availability checker
├── demo.py               # Visualize trained agent
├── README.md             # This file
├── requirements.txt      # Python dependencies
├── reinforce_results/    # Task 1 outputs (models, history, plots)
└── ppo_results/          # Task 2 outputs (models, history, plots)
```

### File Responsibilities (Per Assignment Spec Section 5.1)

| File | Responsibility |
|------|-----------------|
| `config.yaml` | All hyperparameters, environment settings, agent selection, file paths, execution modes |
| `environment.py` | Environment creation, observation/action space setup |
| `policy_network.py` | Gaussian policy (actor): mean/std outputs, action sampling, log-prob computation |
| `value_network.py` | Value function (critic/baseline): state-value V(s) estimation |
| `reinforce_agent.py` | REINFORCE algorithm: episode collection, returns, baseline subtraction, policy updates |
| `ppo_agent.py` | PPO algorithm: rollout collection, GAE, clipped objective, multi-epoch updates |
| `rollout_buffer.py` | On-policy trajectory storage: states, actions, log-probs, rewards, values, done flags |
| `training_script.py` | Main entry point: config loading, agent instantiation, training loop, logging, results saving |
| `utils.py` | Shared utilities: config loading, plotting, logging, seed management, comparative analysis plots |
| `generate_plots.py` | Plot generation for Section 4.4.1 comparative analysis |
| `check_cuda.py` | CUDA availability verification and setup diagnostics |
| `demo.py` | Evaluate and visualize trained agents |
| `requirements.txt` | Python package dependencies with versions |
| `reinforce_results/` | REINFORCE outputs: models, training history (CSV), plots |
| `ppo_results/` | PPO outputs: models, training history (CSV), plots |

## Configuration

All hyperparameters are controlled via `config.yaml`. Key options:

```yaml
agent_type: "ppo"  # Choose "reinforce" or "ppo"

shared:
  gamma: 0.99
  hidden_sizes: [64, 64]
  activation: "tanh"
  max_grad_norm: 0.5
  device: "cpu"  # or "cuda"

reinforce:
  policy_lr: 0.0003
  value_lr: 0.001
  max_episodes: 3000

ppo:
  lr: 0.0003
  rollout_length: 2048
  mini_batch_size: 64
  epochs: 10
  clip_epsilon: 0.2
  gae_lambda: 0.95
  value_loss_coef: 0.5
  entropy_coef: 0.01
  max_episodes: 2000
```

## Running Training

To train the default agent (set in `config.yaml`):

```bash
python training_script.py
```

To switch agents, edit `config.yaml` and change the `agent_type` field:

```yaml
agent_type: "reinforce"  # Switch between "reinforce" and "ppo"
```

Then run:
```bash
python training_script.py
```

## Visualization

### Option 1: Visualize During Training (Slower)

Enable rendering in `config.yaml`:

```yaml
reinforce:
  render: true   # or false to disable

ppo:
  render: true   # or false to disable
```

This will render the environment during evaluation intervals (every 100 episodes by default), but **slows down training**.

### Option 2: Visualize After Training (Recommended)

After training completes, use the demo script to see the trained agent:

```bash
# Visualize PPO agent (default)
python demo.py --episodes 5

# Visualize REINFORCE agent
python demo.py --agent reinforce --episodes 3

# Visualize with custom model path
python demo.py --agent ppo --model ppo_results/checkpoint_1000 --episodes 5
```

This is **faster and cleaner** because rendering doesn't slow down training.

## Network Architecture

### Policy Network (Actor)
- Input: observation (24-dim for BipedalWalker)
- Hidden layers: 2 layers of 64 units with tanh activation
- Output: mean and std of Gaussian action distribution
- Action bounds: [-1, 1]

### Value Network (Critic)
- Input: observation (24-dim)
- Hidden layers: 2 layers of 64 units with tanh activation
- Output: scalar value estimate V(s)

## REINFORCE with Baseline

**Key Features:**
- Collects complete episodes before updates
- Uses Monte Carlo discounted returns
- Baseline subtraction for variance reduction
- Single policy gradient update per episode

**Training Characteristics:**
- Higher variance gradients → noisier learning curves
- Sample inefficient (discards data after one use)
- Simpler algorithm, useful for small problems

## PPO (Proximal Policy Optimization)

**Key Features:**
- Collects fixed-length rollouts (2048 transitions)
- Generalized Advantage Estimation (GAE) for variance reduction
- Clipped surrogate objective prevents large policy changes
- Multiple epochs (10) of mini-batch updates per rollout

**Training Characteristics:**
- Lower variance gradients → smoother learning
- Sample efficient (reuses data multiple times)
- More stable learning due to clipping mechanism
- More hyperparameters to tune

## Results

Training results are saved to `{agent}_results/` directory:

- `final_model_policy.pt` - Trained policy network weights
- `final_model_value.pt` - Trained value network weights
- `rewards.csv` - Episode rewards
- `policy_loss.csv`, `value_loss.csv` - Loss trajectories
- `training_curves.png` - Visualization of training

## Performance Expectations

- **REINFORCE**: Slow, noisy learning. Expect convergence around episode 1000-3000 to average reward > 200.
- **PPO**: Faster, smoother learning. Expect convergence around episode 500-1500 to average reward > 200.

Optimal performance (reward > 300) requires careful hyperparameter tuning.

## Generating Plots for Analysis

After training both agents, automatically generated metrics are saved as CSV files. Generate all required plots for your report (Section 4.4.1):

```bash
python generate_plots.py
```

### Output Plots

Creates a `plots/` directory with 5 plots:

1. **1_reward_curve_overlaid.png** - Both agents' reward curves on same plot (moving average)
2. **2_loss_curves_overlaid.png** - Policy and value loss comparison
3. **3_detailed_reinforce.png** - REINFORCE detailed: raw rewards + moving avg + losses
4. **3_detailed_ppo.png** - PPO detailed: raw rewards + moving avg + losses  
5. **4_entropy_ppo.png** - PPO entropy coefficient decay (exploration → exploitation)

### Data Automatically Saved During Training

**REINFORCE Results** (`reinforce_results/`):
```
rewards.csv              # Episode rewards (one per line)
policy_loss.csv          # Policy loss per update
value_loss.csv           # Value loss per update
final_model_policy.pt    # Trained policy network
final_model_value.pt     # Trained value network
```

**PPO Results** (`ppo_results/`):
```
rewards.csv              # Episode rewards (one per line)
policy_loss.csv          # Policy loss per update
value_loss.csv           # Value loss per update
entropy_loss.csv         # Entropy bonus loss per update
entropy_coef.csv         # Entropy coefficient decay (0.01 → 0.001)
final_model_policy.pt    # Trained policy network
final_model_value.pt     # Trained value network
```

These CSV files are preserved for reproducibility and can be regenerated into plots anytime.

## Troubleshooting

**Issue**: Box2D installation fails on Windows
**Solution**: Install SWIG before installing gymnasium[box2d]
```bash
conda install swig -y
pip install gymnasium[box2d]
```

**Issue**: CUDA out of memory
**Solution**: Change `device` to "cpu" in config.yaml

**Issue**: Training is very slow
**Solution**: Reduce `max_episodes` or `rollout_length` for quick testing

## References

- REINFORCE: Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning.
- PPO: Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms.
- GAE: Schulman, J., et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation.

## Team Collaboration

This is a team-based assignment. Team contributions are documented in the final report.
