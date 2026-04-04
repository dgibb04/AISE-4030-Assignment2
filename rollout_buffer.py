"""
Rollout buffer for on-policy trajectory storage.

This module provides a buffer class for storing trajectories collected during
PPO rollouts. It supports efficient mini-batch sampling for multi-epoch updates.
"""

import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer for storing on-policy trajectories collected during PPO rollouts.

    Stores states, actions, log-probabilities, rewards, values, returns, and
    advantages for a fixed-length rollout. Provides mini-batch sampling for
    multi-epoch gradient updates.
    """

    def __init__(self, rollout_length: int, obs_dim: int, action_dim: int):
        """
        Initialize the rollout buffer.

        Args:
            rollout_length (int): Maximum number of transitions to store.
            obs_dim (int): Dimension of the observation space.
            action_dim (int): Dimension of the action space.
        """
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0  # Current position in the buffer

        # Initialize storage arrays
        self.observations = np.zeros((rollout_length, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_length, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(rollout_length, dtype=np.float32)
        self.rewards = np.zeros(rollout_length, dtype=np.float32)
        self.values = np.zeros(rollout_length, dtype=np.float32)
        self.dones = np.zeros(rollout_length, dtype=np.bool_)
        self.returns = np.zeros(rollout_length, dtype=np.float32)
        self.advantages = np.zeros(rollout_length, dtype=np.float32)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ):
        """
        Store a single transition in the buffer.

        Args:
            obs (np.ndarray): Observation, shape (obs_dim,).
            action (np.ndarray): Action taken, shape (action_dim,).
            log_prob (float): Log probability of the action under the policy.
            reward (float): Reward received.
            value (float): Value estimate V(s).
            done (bool): Whether the episode terminated.
        """
        if self.ptr >= self.rollout_length:
            raise RuntimeError("Rollout buffer is full. Call reset() to clear.")

        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done

        self.ptr += 1

    def set_returns_and_advantages(self, returns: np.ndarray, advantages: np.ndarray):
        """
        Set the computed returns and advantages for all transitions.

        Args:
            returns (np.ndarray): Discounted return targets, shape (rollout_length,).
            advantages (np.ndarray): Advantage estimates, shape (rollout_length,).
        """
        if len(returns) != self.ptr or len(advantages) != self.ptr:
            raise ValueError(
                f"Returns and advantages must match number of stored transitions. "
                f"Expected {self.ptr}, got {len(returns)}, {len(advantages)}"
            )

        self.returns[: self.ptr] = returns
        self.advantages[: self.ptr] = advantages

    def get_mini_batches(self, mini_batch_size: int, device: str = "cpu"):
        """
        Generate mini-batches of data from the buffer for gradient updates.

        Shuffles the data and yields mini-batches as PyTorch tensors.

        Args:
            mini_batch_size (int): Size of each mini-batch.
            device (str, optional): Device to move tensors to. Defaults to "cpu".

        Yields:
            dict: Mini-batch with keys:
                - "observations": shape (mini_batch_size, obs_dim)
                - "actions": shape (mini_batch_size, action_dim)
                - "log_probs_old": shape (mini_batch_size,)
                - "advantages": shape (mini_batch_size,)
                - "returns": shape (mini_batch_size,)
        """
        # Create shuffled indices
        indices = np.random.permutation(self.ptr)

        # Yield mini-batches
        for start_idx in range(0, self.ptr, mini_batch_size):
            end_idx = min(start_idx + mini_batch_size, self.ptr)
            batch_indices = indices[start_idx:end_idx]

            yield {
                "observations": torch.tensor(
                    self.observations[batch_indices], device=device
                ),
                "actions": torch.tensor(self.actions[batch_indices], device=device),
                "log_probs_old": torch.tensor(
                    self.log_probs[batch_indices], device=device
                ),
                "advantages": torch.tensor(
                    self.advantages[batch_indices], device=device
                ),
                "returns": torch.tensor(self.returns[batch_indices], device=device),
            }

    def reset(self):
        """Clear the buffer and reset the pointer."""
        self.ptr = 0

    def is_full(self) -> bool:
        """
        Check if the buffer is full.

        Returns:
            bool: True if the buffer has stored the maximum number of transitions.
        """
        return self.ptr >= self.rollout_length

    def size(self) -> int:
        """
        Get the number of transitions currently stored.

        Returns:
            int: Number of stored transitions.
        """
        return self.ptr
