"""
Gaussian Policy Network (Actor) for continuous action spaces.

This module implements a neural network that outputs parameters of a multivariate
Gaussian distribution over actions. The network provides action sampling via the
reparameterization trick and log-probability computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class GaussianPolicyNetwork(nn.Module):
    """
    Gaussian policy network for continuous control.

    The network takes a state as input and outputs:
    - Mean vector μ(s) for the action distribution (bounded to [-1, 1] via tanh)
    - Standard deviation σ (state-independent, learnable parameter)

    Architecture:
    - Input: observation (state vector)
    - Hidden layers: configurable (default [64, 64] with tanh activation)
    - Mean head: outputs action_dim values, applies tanh to bound to [-1, 1]
    - Log-std: learnable parameter vector, initialized to 0.0 (σ = 1.0)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list = None,
        activation: str = "tanh",
        log_std_init: float = 0.0,
    ):
        """
        Initialize the Gaussian policy network.

        Args:
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
            hidden_sizes (list, optional): Sizes of hidden layers. Defaults to [64, 64].
            activation (str, optional): Activation function. Defaults to "tanh".
            log_std_init (float, optional): Initial value for log standard deviation.
                Defaults to 0.0, corresponding to σ = 1.0.
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.activation_name = activation

        # Select activation function
        if activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "relu":
            activation_fn = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build the shared body (feature extraction)
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size

        self.body = nn.Sequential(*layers)

        # Mean head: outputs action_dim values
        self.mean_head = nn.Linear(prev_size, action_dim)

        # Log standard deviation: learnable parameter (not state-dependent)
        self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

    def forward(self, obs: torch.Tensor) -> tuple:
        """
        Forward pass through the policy network.

        Args:
            obs (torch.Tensor): Observation tensor of shape (batch_size, obs_dim)
                or (obs_dim,) for a single observation.

        Returns:
            tuple: (mean, std)
                - mean (torch.Tensor): Mean of the action distribution,
                  shape (batch_size, action_dim) or (action_dim,).
                - std (torch.Tensor): Standard deviation of the action distribution,
                  shape (batch_size, action_dim) or (action_dim,).
        """
        # Ensure obs is at least 2D
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Pass through body
        features = self.body(obs)

        # Mean head with tanh activation to bound to [-1, 1]
        mean = torch.tanh(self.mean_head(features))

        # Standard deviation (exponentiate log_std for numerical stability)
        std = torch.exp(self.log_std).expand_as(mean)

        if squeeze_output:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std

    def sample_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple:
        """
        Sample an action from the policy distribution.

        Uses the reparameterization trick for differentiable sampling.
        Clamps the sampled action to [-1, 1] to match the action space bounds.

        Args:
            obs (torch.Tensor): Observation tensor of shape (obs_dim,) or
                (batch_size, obs_dim).
            deterministic (bool, optional): If True, return the mean action without
                sampling. Useful for evaluation. Defaults to False.

        Returns:
            tuple: (action, log_prob)
                - action (torch.Tensor): Sampled action(s), shape (action_dim,) or
                  (batch_size, action_dim), clamped to [-1, 1].
                - log_prob (torch.Tensor): Log probability of the sampled action(s),
                  shape () or (batch_size,).
        """
        mean, std = self.forward(obs)

        if deterministic:
            # For evaluation: use mean action without sampling
            action = mean
        else:
            # Sample from Gaussian distribution using reparameterization trick
            dist = Normal(mean, std)
            action = dist.rsample()

        # Clamp action to [-1, 1] to match action space bounds
        action = torch.clamp(action, min=-1.0, max=1.0)

        # Compute log probability
        # Note: We compute log prob before clamping for proper gradient flow.
        # This is an approximation commonly used in practice.
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action)

        # Sum log probs across action dimensions if we have multiple actions
        if log_prob.dim() > 0 and log_prob.shape[-1] == self.action_dim:
            log_prob = log_prob.sum(dim=-1)

        return action, log_prob

    def get_log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the log probability of a given action under the current policy.

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch_size, obs_dim)
                or (obs_dim,).
            action (torch.Tensor): Action tensor, shape (batch_size, action_dim)
                or (action_dim,).

        Returns:
            torch.Tensor: Log probability of the action(s), shape () or (batch_size,).
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action)

        # Sum across action dimensions
        if log_prob.dim() > 0 and log_prob.shape[-1] == self.action_dim:
            log_prob = log_prob.sum(dim=-1)

        return log_prob

    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute the entropy of the policy distribution.

        For a multivariate Gaussian, entropy is:
        H = 0.5 * ln(2π * e * σ^2)^d = 0.5 * ln(2πeσ^2) * d

        Args:
            obs (torch.Tensor): Observation tensor, shape (batch_size, obs_dim)
                or (obs_dim,).

        Returns:
            torch.Tensor: Entropy of the policy, shape () or (batch_size,).
        """
        mean, std = self.forward(obs)
        dist = Normal(mean, std)
        entropy = dist.entropy()

        # Sum across action dimensions
        if entropy.dim() > 0 and entropy.shape[-1] == self.action_dim:
            entropy = entropy.sum(dim=-1)

        return entropy
