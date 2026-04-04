"""
BipedalWalker environment setup and utilities.

This module handles environment creation, observation/action space specifications,
and any necessary preprocessing or wrappers.
"""

import gymnasium as gym
import numpy as np


def create_environment(env_name: str, seed: int = 42) -> tuple:
    """
    Create and initialize the BipedalWalker environment.

    Args:
        env_name (str): Name of the environment (e.g., "BipedalWalker-v3").
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (environment, obs_shape, action_shape, action_space)
            - environment: Gymnasium environment instance.
            - obs_shape (tuple): Shape of observation space.
            - action_shape (tuple): Shape of action space.
            - action_space: Action space object with bounds.
    """
    env = gym.make(env_name)
    env.reset(seed=seed)

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_space = env.action_space

    return env, obs_shape, action_shape, action_space


def reset_environment(env: gym.Env, seed: int = None) -> tuple:
    """
    Reset the environment and return initial observation and info.

    Args:
        env (gym.Env): The environment to reset.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: (observation, info)
            - observation (np.ndarray): Initial observation from the environment.
            - info (dict): Additional information from reset.
    """
    return env.reset(seed=seed)


def step_environment(env: gym.Env, action: np.ndarray) -> tuple:
    """
    Take a step in the environment with the given action.

    Args:
        env (gym.Env): The environment to step in.
        action (np.ndarray): Action to take in the environment.

    Returns:
        tuple: (observation, reward, terminated, truncated, info)
            - observation (np.ndarray): Next observation.
            - reward (float): Reward for the action.
            - terminated (bool): Whether the episode ended naturally.
            - truncated (bool): Whether the episode was truncated (time limit).
            - info (dict): Additional information.
    """
    return env.step(action)
