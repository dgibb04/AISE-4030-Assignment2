"""
REINFORCE with Baseline agent implementation.

This module implements the REINFORCE algorithm with a learned baseline (value function)
to reduce variance in gradient estimates. It's a Monte Carlo policy gradient method
that collects complete episodes before performing parameter updates.
"""

import torch
import torch.optim as optim
import numpy as np
from policy_network import GaussianPolicyNetwork
from value_network import ValueNetwork


class REINFORCEAgent:
    """
    REINFORCE with Baseline agent for continuous control.

    The agent collects complete episodes, computes discounted Monte Carlo returns,
    subtracts a learned baseline (value function), and performs a single policy
    gradient update per episode.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        policy_lr: float = 0.0003,
        value_lr: float = 0.001,
        gamma: float = 0.99,
        hidden_sizes: list = None,
        activation: str = "tanh",
        max_grad_norm: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_decay: float = 0.9995,
        device: str = "cpu",
    ):
        """
        Initialize the REINFORCE agent.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            policy_lr (float, optional): Learning rate for policy network. Defaults to 0.0003.
            value_lr (float, optional): Learning rate for value network. Defaults to 0.001.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            hidden_sizes (list, optional): Sizes of hidden layers. Defaults to [64, 64].
            activation (str, optional): Activation function. Defaults to "tanh".
            max_grad_norm (float, optional): Maximum gradient norm for clipping. Defaults to 0.5.
            device (str, optional): Device to use ("cpu" or "cuda"). Defaults to "cpu".
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.device = device

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        # Create policy and value networks
        self.policy = GaussianPolicyNetwork(
            obs_dim, action_dim, hidden_sizes, activation
        ).to(device)
        self.value = ValueNetwork(obs_dim, hidden_sizes, activation).to(device)

        # Create optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)

        # Storage for episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple:
        """
        Select an action given an observation.

        Args:
            obs (np.ndarray): Observation from the environment, shape (obs_dim,).
            deterministic (bool, optional): If True, return mean action. Defaults to False.

        Returns:
            tuple: (action, log_prob, value)
                - action (np.ndarray): Action to take, shape (action_dim,).
                - log_prob (float): Log probability of the action.
                - value (float): Value estimate V(s) (not used in REINFORCE, for compatibility).
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action, log_prob = self.policy.sample_action(obs_tensor, deterministic)

        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().item()

        return action_np, log_prob_np, None

    def store_transition(
        self, obs: np.ndarray, action: np.ndarray, log_prob: float, reward: float
    ):
        """
        Store a transition in the episode buffer.

        Args:
            obs (np.ndarray): Observation, shape (obs_dim,).
            action (np.ndarray): Action taken, shape (action_dim,).
            log_prob (float): Log probability of the action.
            reward (float): Reward received.
        """
        self.episode_states.append(obs)
        self.episode_actions.append(action)
        self.episode_log_probs.append(log_prob)
        self.episode_rewards.append(reward)

    def compute_discounted_returns(self) -> np.ndarray:
        """
        Compute discounted returns from stored episode rewards.

        Returns:
            np.ndarray: Discounted returns for each timestep, shape (episode_length,).
        """
        returns = np.zeros(len(self.episode_rewards))
        cumulative_return = 0.0

        for t in reversed(range(len(self.episode_rewards))):
            cumulative_return = self.episode_rewards[t] + self.gamma * cumulative_return
            returns[t] = cumulative_return

        return returns

    def update(self):
        """
        Update policy and value networks after an episode.

        Computes:
        1. Discounted returns for all timesteps
        2. Value function loss (MSE between predicted values and returns)
        3. Advantages (returns - predicted values)
        4. Policy gradient loss
        """
        if not self.episode_rewards:
            return

        # Convert episode data to tensors
        states = torch.tensor(
            np.array(self.episode_states), dtype=torch.float32, device=self.device
        )
        actions = torch.tensor(
            np.array(self.episode_actions), dtype=torch.float32, device=self.device
        )

        # Compute discounted returns
        returns = self.compute_discounted_returns()
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize returns for stability
        returns_normalized = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std() + 1e-8
        )

        # Compute value predictions and advantages
        values = self.value(states)
        advantages = (returns_normalized - values).detach()

        # Recompute log probabilities with current policy (for gradient flow)
        log_probs = self.policy.get_log_prob(states, actions)

        # Compute entropy for exploration bonus
        entropy = self.policy.get_entropy(states)

        # Policy loss: negative expected return (gradient ascent)
        policy_loss = -(log_probs * advantages).mean()

        # Entropy loss: encourages exploration
        entropy_loss = -entropy.mean()

        # Value loss: MSE between predicted and actual returns
        value_loss = ((values - returns_normalized) ** 2).mean()

        # Total policy loss with entropy bonus
        total_policy_loss = policy_loss + self.entropy_coef * entropy_loss

        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        # Clear episode buffers
        self.episode_states = []
        self.episode_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []

        # Decay entropy coefficient
        self.entropy_coef *= self.entropy_decay

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    def save_model(self, path: str):
        """
        Save policy and value network weights to disk.

        Args:
            path (str): Path to save the model (without extension).
        """
        torch.save(self.policy.state_dict(), f"{path}_policy.pt")
        torch.save(self.value.state_dict(), f"{path}_value.pt")

    def load_model(self, path: str):
        """
        Load policy and value network weights from disk.

        Args:
            path (str): Path to load the model (without extension).
        """
        self.policy.load_state_dict(torch.load(f"{path}_policy.pt", map_location=self.device))
        self.value.load_state_dict(torch.load(f"{path}_value.pt", map_location=self.device))
