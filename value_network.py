"""
Value Network (Critic / Baseline) for policy gradient methods.

This module implements a neural network that estimates the state-value function V(s),
which represents the expected return from a given state. The value function is used
as a baseline in REINFORCE to reduce variance and as a critic in PPO.
"""

import torch
import torch.nn as nn


class ValueNetwork(nn.Module):
    """
    State-value function network (critic / baseline).

    The network takes a state as input and outputs a single scalar estimate V(s),
    representing the expected return from that state.

    Architecture:
    - Input: observation (state vector)
    - Hidden layers: configurable (default [64, 64] with tanh activation)
    - Output head: single scalar value with no activation
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list = None,
        activation: str = "tanh",
    ):
        """
        Initialize the value network.

        Args:
            obs_dim (int): Dimension of the observation space.
            hidden_sizes (list, optional): Sizes of hidden layers. Defaults to [64, 64].
            activation (str, optional): Activation function. Defaults to "tanh".
        """
        super().__init__()

        if hidden_sizes is None:
            hidden_sizes = [64, 64]

        self.obs_dim = obs_dim
        self.activation_name = activation

        # Select activation function
        if activation == "tanh":
            activation_fn = nn.Tanh
        elif activation == "relu":
            activation_fn = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build the network
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(activation_fn())
            prev_size = hidden_size

        self.body = nn.Sequential(*layers)

        # Value head: outputs a single scalar
        self.value_head = nn.Linear(prev_size, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.

        Args:
            obs (torch.Tensor): Observation tensor of shape (batch_size, obs_dim)
                or (obs_dim,) for a single observation.

        Returns:
            torch.Tensor: Value estimate(s), shape (batch_size, 1) or (1,).
                If input was 1D, output will be 0D (scalar).
        """
        # Ensure obs is at least 2D
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Pass through body
        features = self.body(obs)

        # Value head
        value = self.value_head(features)

        if squeeze_output:
            value = value.squeeze(0).squeeze(-1)
        else:
            value = value.squeeze(-1)

        return value
