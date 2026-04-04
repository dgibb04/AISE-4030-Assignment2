"""
Proximal Policy Optimization (PPO) agent implementation.

This module implements the PPO algorithm with the clipped surrogate objective.
PPO is an actor-critic method that collects fixed-length trajectories, computes
advantages using Generalized Advantage Estimation (GAE), and performs multiple
epochs of mini-batch updates with a clipped objective for stability.
"""

import torch
import torch.optim as optim
import numpy as np
from policy_network import GaussianPolicyNetwork
from value_network import ValueNetwork
from rollout_buffer import RolloutBuffer


class PPOAgent:
    """
    Proximal Policy Optimization agent for continuous control.

    Collects fixed-length rollouts, computes GAE advantages, and performs
    multiple epochs of mini-batch gradient updates using a clipped surrogate
    objective to prevent destructive policy changes.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rollout_length: int = 2048,
        mini_batch_size: int = 64,
        epochs: int = 10,
        clip_epsilon: float = 0.2,
        gae_lambda: float = 0.95,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        entropy_decay: float = 1.0,
        lr: float = 0.0003,
        gamma: float = 0.99,
        hidden_sizes: list = None,
        activation: str = "tanh",
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        """
        Initialize the PPO agent.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            rollout_length (int, optional): Number of transitions per rollout. Defaults to 2048.
            mini_batch_size (int, optional): Size of mini-batches. Defaults to 64.
            epochs (int, optional): Number of update epochs per rollout. Defaults to 10.
            clip_epsilon (float, optional): Clipping range. Defaults to 0.2.
            gae_lambda (float, optional): GAE lambda parameter. Defaults to 0.95.
            value_loss_coef (float, optional): Coefficient for value loss. Defaults to 0.5.
            entropy_coef (float, optional): Coefficient for entropy bonus. Defaults to 0.01.
            entropy_decay (float, optional): Decay factor for entropy coefficient per update.
                Defaults to 1.0 (no decay). Use 0.9995 for slow decay over 2000 updates.
            lr (float, optional): Learning rate. Defaults to 0.0003.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            hidden_sizes (list, optional): Sizes of hidden layers. Defaults to [64, 64].
            activation (str, optional): Activation function. Defaults to "tanh".
            max_grad_norm (float, optional): Maximum gradient norm. Defaults to 0.5.
            device (str, optional): Device to use. Defaults to "cpu".
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.rollout_length = rollout_length
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.clip_epsilon = clip_epsilon
        self.gae_lambda = gae_lambda
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.entropy_decay = entropy_decay
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.device = device

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        # Create policy and value networks
        self.policy = GaussianPolicyNetwork(
            obs_dim, action_dim, hidden_sizes, activation
        ).to(device)
        self.value = ValueNetwork(obs_dim, hidden_sizes, activation).to(device)

        # Create shared optimizer for both actor and critic
        params = list(self.policy.parameters()) + list(self.value.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        # Create rollout buffer
        self.buffer = RolloutBuffer(rollout_length, obs_dim, action_dim)

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> tuple:
        """
        Select an action given an observation.

        Args:
            obs (np.ndarray): Observation from environment, shape (obs_dim,).
            deterministic (bool, optional): If True, use mean action. Defaults to False.

        Returns:
            tuple: (action, log_prob, value)
                - action (np.ndarray): Action to take, shape (action_dim,).
                - log_prob (float): Log probability of the action.
                - value (float): Value estimate V(s).
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action, log_prob = self.policy.sample_action(obs_tensor, deterministic)
            value = self.value(obs_tensor)

        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().item()
        value_np = value.cpu().item()

        return action_np, log_prob_np, value_np

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """
        Store a transition in the rollout buffer.

        Args:
            obs (np.ndarray): Observation, shape (obs_dim,).
            action (np.ndarray): Action taken, shape (action_dim,).
            log_prob (float): Log probability under current policy.
            reward (float): Reward received.
            value (float): Value estimate V(s).
            done (bool): Whether episode terminated.
        """
        self.buffer.store(obs, action, log_prob, reward, value, done)

    def compute_gae(self, next_value: float) -> tuple:
        """
        Compute Generalized Advantage Estimation (GAE).

        Computes advantages and return targets using GAE with the stored
        transitions. GAE trades off bias and variance using the lambda parameter.

        Args:
            next_value (float): Bootstrap value for the last state (V(s_next)).

        Returns:
            tuple: (advantages, returns)
                - advantages (np.ndarray): GAE advantage estimates, shape (rollout_length,).
                - returns (np.ndarray): Return targets for value network, shape (rollout_length,).
        """
        advantages = np.zeros(self.buffer.size(), dtype=np.float32)
        gae = 0.0

        # Compute GAE in reverse order for efficiency
        for t in reversed(range(self.buffer.size())):
            if t == self.buffer.size() - 1:
                next_value_t = next_value
            else:
                next_value_t = self.buffer.values[t + 1]

            # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = (
                self.buffer.rewards[t]
                + self.gamma * next_value_t * (1 - self.buffer.dones[t])
                - self.buffer.values[t]
            )

            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            gae = delta + self.gamma * self.gae_lambda * (1 - self.buffer.dones[t]) * gae
            advantages[t] = gae

        # Compute returns: G_t = A_t + V(s_t)
        returns = advantages + self.buffer.values[: self.buffer.size()]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, next_value: float) -> dict:
        """
        Perform PPO update after a rollout.

        Computes GAE advantages, then performs K epochs of mini-batch updates
        using the clipped surrogate objective.

        Args:
            next_value (float): Bootstrap value for computing GAE.

        Returns:
            dict: Training metrics including loss components.
        """
        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(next_value)
        self.buffer.set_returns_and_advantages(returns, advantages)

        # Store old log probabilities for ratio computation
        old_log_probs = self.buffer.log_probs[: self.buffer.size()].copy()

        # Multi-epoch mini-batch updates
        total_losses = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "total_loss": 0.0,
        }
        num_updates = 0

        for epoch in range(self.epochs):
            for mini_batch in self.buffer.get_mini_batches(
                self.mini_batch_size, self.device
            ):
                # Forward pass
                action_log_probs = self.policy.get_log_prob(
                    mini_batch["observations"], mini_batch["actions"]
                )
                values = self.value(mini_batch["observations"])
                entropy = self.policy.get_entropy(mini_batch["observations"])

                # Probability ratio for clipped surrogate objective
                # r_t = log π(a|s) - log π_old(a|s)
                log_ratio = action_log_probs - mini_batch["log_probs_old"]
                ratio = torch.exp(log_ratio)

                # Clipped surrogate objective
                surr1 = ratio * mini_batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * mini_batch["advantages"]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = ((values - mini_batch["returns"]) ** 2).mean()

                # Entropy bonus (negative for gradient descent)
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Accumulate metrics
                total_losses["policy_loss"] += policy_loss.item()
                total_losses["value_loss"] += value_loss.item()
                total_losses["entropy_loss"] += entropy_loss.item()
                total_losses["total_loss"] += total_loss.item()
                num_updates += 1

        # Average metrics
        for key in total_losses:
            total_losses[key] /= num_updates

        # Reset buffer
        self.buffer.reset()

        # Decay entropy coefficient
        self.entropy_coef *= self.entropy_decay

        return total_losses

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
        self.policy.load_state_dict(
            torch.load(f"{path}_policy.pt", map_location=self.device)
        )
        self.value.load_state_dict(
            torch.load(f"{path}_value.pt", map_location=self.device)
        )
