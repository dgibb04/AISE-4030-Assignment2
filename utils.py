"""
Utility functions for configuration, logging, and visualization.

This module provides helpers for loading YAML config files, plotting training
results, managing random seeds, and other common utilities.
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def load_config(config_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the config.yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """
    Set random seeds for reproducibility.

    Sets seeds for numpy and PyTorch (both CPU and CUDA).

    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def create_directories(agent_type: str, result_dirs: list = None) -> str:
    """
    Create output directories for saving results.

    Args:
        agent_type (str): Type of agent ("reinforce" or "ppo").
        result_dirs (list, optional): Additional directories to create.

    Returns:
        str: Path to the main results directory for this agent.
    """
    result_dir = f"{agent_type}_results"
    Path(result_dir).mkdir(exist_ok=True)

    if result_dirs:
        for subdir in result_dirs:
            Path(os.path.join(result_dir, subdir)).mkdir(exist_ok=True)

    return result_dir


def plot_training_curves(
    rewards: np.ndarray,
    losses: dict,
    agent_type: str,
    save_path: str = None,
    window: int = 10,
):
    """
    Plot training curves (reward and loss).

    Args:
        rewards (np.ndarray): Array of episode rewards, shape (num_episodes,).
        losses (dict): Dictionary of loss arrays, with keys like "policy_loss", "value_loss".
        agent_type (str): Name of the agent ("reinforce" or "ppo") for title.
        save_path (str, optional): Path to save the figure. If None, displays the plot.
        window (int, optional): Window size for moving average. Defaults to 10.
    """
    # Compute moving average of rewards
    rewards_ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
    episodes_ma = np.arange(window - 1, len(rewards))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot reward
    axes[0].plot(rewards, alpha=0.3, label="Raw")
    axes[0].plot(episodes_ma, rewards_ma, linewidth=2, label=f"Moving Avg (window={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title(f"{agent_type.upper()} - Episode Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot losses
    if losses:
        for loss_name, loss_values in losses.items():
            if loss_values is not None and len(loss_values) > 0:
                axes[1].plot(loss_values, label=loss_name, linewidth=1.5)
        axes[1].set_xlabel("Update Step")
        axes[1].set_ylabel("Loss")
        axes[1].set_title(f"{agent_type.upper()} - Training Losses")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_comparison(
    reinforce_rewards: np.ndarray,
    ppo_rewards: np.ndarray,
    save_path: str = None,
    window: int = 10,
):
    """
    Plot comparison of two agents' training curves.

    Args:
        reinforce_rewards (np.ndarray): REINFORCE episode rewards.
        ppo_rewards (np.ndarray): PPO episode rewards.
        save_path (str, optional): Path to save the figure.
        window (int, optional): Window size for moving average. Defaults to 10.
    """
    # Compute moving averages
    reinforce_ma = np.convolve(
        reinforce_rewards, np.ones(window) / window, mode="valid"
    )
    ppo_ma = np.convolve(ppo_rewards, np.ones(window) / window, mode="valid")

    episodes_reinforce = np.arange(window - 1, len(reinforce_rewards))
    episodes_ppo = np.arange(window - 1, len(ppo_rewards))

    plt.figure(figsize=(12, 6))
    plt.plot(episodes_reinforce, reinforce_ma, label="REINFORCE", linewidth=2)
    plt.plot(episodes_ppo, ppo_ma, label="PPO", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Moving Average)")
    plt.title("Agent Comparison - Training Curves")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison figure saved to {save_path}")
    else:
        plt.show()

    plt.close()


def save_training_history(
    rewards: np.ndarray,
    losses: dict,
    save_path: str,
):
    """
    Save training history to CSV files.

    Args:
        rewards (np.ndarray): Array of episode rewards.
        losses (dict): Dictionary of loss arrays.
        save_path (str): Directory to save CSV files.
    """
    Path(save_path).mkdir(exist_ok=True)

    # Save rewards
    np.savetxt(
        os.path.join(save_path, "rewards.csv"), rewards, delimiter=",", fmt="%.4f"
    )

    # Save losses
    if losses:
        for loss_name, loss_values in losses.items():
            if loss_values is not None and len(loss_values) > 0:
                np.savetxt(
                    os.path.join(save_path, f"{loss_name}.csv"),
                    loss_values,
                    delimiter=",",
                    fmt="%.6f",
                )


def compute_moving_average(data: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Compute moving average of data.

    Args:
        data (np.ndarray): 1D array of data.
        window (int, optional): Window size. Defaults to 10.

    Returns:
        np.ndarray: Moving average.
    """
    return np.convolve(data, np.ones(window) / window, mode="valid")


def compute_episode_statistics(rewards: np.ndarray, window: int = 100) -> dict:
    """
    Compute statistics about episode rewards.

    Args:
        rewards (np.ndarray): Array of episode rewards.
        window (int, optional): Window for averaging. Defaults to 100.

    Returns:
        dict: Statistics including mean, std, max, min over windows.
    """
    num_windows = len(rewards) // window
    if num_windows == 0:
        return {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "max": float(np.max(rewards)),
            "min": float(np.min(rewards)),
        }

    window_means = []
    for i in range(num_windows):
        window_data = rewards[i * window : (i + 1) * window]
        window_means.append(np.mean(window_data))

    return {
        "mean": float(np.mean(window_means)),
        "std": float(np.std(window_means)),
        "max": float(np.max(rewards)),
        "min": float(np.min(rewards)),
        "window_means": window_means,
    }


def load_agent_data(agent_type: str, result_dir: str = None) -> dict:
    """
    Load training data for an agent from saved CSV files.

    Args:
        agent_type (str): "reinforce" or "ppo"
        result_dir (str, optional): Custom results directory. Defaults to "{agent_type}_results"

    Returns:
        dict: Dictionary with keys for rewards, losses, etc. Returns None if data not found.
    """
    if result_dir is None:
        result_dir = f"{agent_type}_results"

    data = {"agent": agent_type}

    # Load rewards
    rewards_file = os.path.join(result_dir, "rewards.csv")
    if os.path.exists(rewards_file):
        data["rewards"] = np.loadtxt(rewards_file, delimiter=",")
    else:
        print(f"⚠️  Warning: {rewards_file} not found")
        return None

    # Load losses
    loss_files = {
        "policy_loss": "policy_loss.csv",
        "value_loss": "value_loss.csv",
        "entropy_loss": "entropy_loss.csv",
    }

    for key, filename in loss_files.items():
        filepath = os.path.join(result_dir, filename)
        if os.path.exists(filepath):
            data[key] = np.loadtxt(filepath, delimiter=",")

    # Load PPO-specific data
    if agent_type == "ppo":
        entropy_file = os.path.join(result_dir, "entropy_coef.csv")
        if os.path.exists(entropy_file):
            data["entropy_coef"] = np.loadtxt(entropy_file, delimiter=",")

    return data


def generate_comparative_plots(output_dir: str = "plots") -> bool:
    """
    Generate all required comparative analysis plots (Section 4.4.1).

    Loads data from both agents and creates 5 plots:
    1. Reward curve (both agents overlaid)
    2. Loss curves (both agents overlaid)
    3. Per-agent detailed plots (2 plots)
    4. Entropy plot (PPO only)

    Args:
        output_dir (str, optional): Directory to save plots. Defaults to "plots"

    Returns:
        bool: True if all plots generated successfully, False otherwise.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Load data
    print("\nLoading training data...")
    reinforce_data = load_agent_data("reinforce")
    ppo_data = load_agent_data("ppo")

    if reinforce_data is None or ppo_data is None:
        print("✗ Error: Could not load training data from both agents")
        return False

    print(f"✓ REINFORCE data loaded ({len(reinforce_data['rewards'])} episodes)")
    print(f"✓ PPO data loaded ({len(ppo_data['rewards'])} episodes)")

    print(f"\nGenerating plots in '{output_dir}/' directory...\n")

    # Plot 1: Reward curve (overlaid)
    _plot_reward_curve_overlaid(reinforce_data, ppo_data, output_dir)

    # Plot 2: Loss curves (overlaid)
    _plot_loss_curves_overlaid(reinforce_data, ppo_data, output_dir)

    # Plot 3: Per-agent detailed
    _plot_per_agent_detailed(reinforce_data, "reinforce", output_dir)
    _plot_per_agent_detailed(ppo_data, "ppo", output_dir)

    # Plot 4: Entropy (PPO only)
    _plot_entropy_ppo(ppo_data, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("✓ All plots generated successfully!")
    print("=" * 70)
    print(f"\nPlots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. 1_reward_curve_overlaid.png")
    print("  2. 2_loss_curves_overlaid.png")
    print("  3. 3_detailed_reinforce.png")
    print("  4. 3_detailed_ppo.png")
    print("  5. 4_entropy_ppo.png")
    print("\nUse these plots in your final report for Section 4.4\n")

    return True


def _plot_reward_curve_overlaid(reinforce_data: dict, ppo_data: dict, save_dir: str):
    """Plot 1: Reward curve (both agents overlaid)."""
    fig, ax = plt.subplots(figsize=(12, 6))

    window = 10
    reinforce_rewards = reinforce_data["rewards"]
    ppo_rewards = ppo_data["rewards"]

    reinforce_ma = compute_moving_average(reinforce_rewards, window)
    ppo_ma = compute_moving_average(ppo_rewards, window)

    episodes_reinforce = np.arange(window - 1, len(reinforce_rewards))
    episodes_ppo = np.arange(window - 1, len(ppo_rewards))

    ax.plot(episodes_reinforce, reinforce_ma, label="REINFORCE", linewidth=2, color="blue")
    ax.plot(episodes_ppo, ppo_ma, label="PPO", linewidth=2, color="red")

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (Moving Average)", fontsize=12)
    ax.set_title("Reward Curves - Both Agents", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "1_reward_curve_overlaid.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()


def _plot_loss_curves_overlaid(reinforce_data: dict, ppo_data: dict, save_dir: str):
    """Plot 2: Loss curves (both agents overlaid)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    if "policy_loss" in reinforce_data and "policy_loss" in ppo_data:
        ax1.plot(
            reinforce_data["policy_loss"],
            label="REINFORCE",
            linewidth=1.5,
            color="blue",
            alpha=0.7,
        )
        ax1.plot(
            ppo_data["policy_loss"],
            label="PPO",
            linewidth=1.5,
            color="red",
            alpha=0.7,
        )
        ax1.set_ylabel("Policy Loss", fontsize=11)
        ax1.set_title("Policy Loss Over Training", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

    if "value_loss" in reinforce_data and "value_loss" in ppo_data:
        ax2.plot(
            reinforce_data["value_loss"],
            label="REINFORCE",
            linewidth=1.5,
            color="blue",
            alpha=0.7,
        )
        ax2.plot(
            ppo_data["value_loss"],
            label="PPO",
            linewidth=1.5,
            color="red",
            alpha=0.7,
        )
        ax2.set_xlabel("Update Step", fontsize=11)
        ax2.set_ylabel("Value Loss", fontsize=11)
        ax2.set_title("Value Loss Over Training", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, "2_loss_curves_overlaid.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()


def _plot_per_agent_detailed(agent_data: dict, agent_type: str, save_dir: str):
    """Plot 3: Per-agent detailed plots."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.3)

    rewards = agent_data["rewards"]
    window = 10

    ax1 = fig.add_subplot(gs[0])
    episodes = np.arange(len(rewards))
    ax1.plot(episodes, rewards, alpha=0.3, label="Raw Episode Rewards", color="gray")

    rewards_ma = compute_moving_average(rewards, window)
    episodes_ma = np.arange(window - 1, len(rewards))
    ax1.plot(episodes_ma, rewards_ma, label=f"Moving Average (window={window})", linewidth=2, color="blue")

    ax1.set_ylabel("Episode Reward", fontsize=11)
    ax1.set_title(f"{agent_type.upper()} - Episode Rewards", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    if "policy_loss" in agent_data:
        ax2.plot(agent_data["policy_loss"], linewidth=1.5, color="red", alpha=0.8)
        ax2.set_ylabel("Policy Loss", fontsize=11)
        ax2.set_title(f"{agent_type.upper()} - Policy Loss", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    if "value_loss" in agent_data:
        ax3.plot(agent_data["value_loss"], linewidth=1.5, color="green", alpha=0.8)
        ax3.set_xlabel("Update Step", fontsize=11)
        ax3.set_ylabel("Value Loss", fontsize=11)
        ax3.set_title(f"{agent_type.upper()} - Value Loss", fontsize=12, fontweight="bold")
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f"3_detailed_{agent_type}.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()


def _plot_entropy_ppo(ppo_data: dict, save_dir: str):
    """Plot 4: Entropy plot (PPO only)."""
    if "entropy_coef" not in ppo_data:
        print("⚠️  Entropy coefficient data not found for PPO")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    entropy_coef = ppo_data["entropy_coef"]
    updates = np.arange(len(entropy_coef))

    ax.plot(updates, entropy_coef, linewidth=2, color="purple")
    ax.fill_between(updates, entropy_coef, alpha=0.3, color="purple")

    ax.set_xlabel("PPO Update", fontsize=12)
    ax.set_ylabel("Entropy Coefficient", fontsize=12)
    ax.set_title("PPO - Entropy Coefficient Decay", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    if len(entropy_coef) > 1:
        start_val = entropy_coef[0]
        end_val = entropy_coef[-1]
        ax.text(
            0.02,
            0.98,
            f"Start: {start_val:.6f}\nEnd: {end_val:.6f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    filepath = os.path.join(save_dir, "4_entropy_ppo.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filepath}")
    plt.close()
