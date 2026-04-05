"""
Demo script to visualize a trained agent.

Run this after training to see the agent walk visually with rendering enabled.
This is separate from training to avoid slowing down the training loop.

Usage:
    python demo.py --agent reinforce --episodes 3
    python demo.py --agent ppo --episodes 5
"""

import argparse
import torch
import numpy as np
from pathlib import Path

import environment
import utils
from reinforce_agent import REINFORCEAgent
from ppo_agent import PPOAgent


def load_and_demo(agent_type: str, num_episodes: int = 3, model_path: str = None):
    """
    Load a trained agent and run with rendering.

    Args:
        agent_type (str): "reinforce" or "ppo"
        num_episodes (int): Number of episodes to render
        model_path (str, optional): Path to model (without extension).
            If None, uses latest model from results directory.
    """
    # Load config
    config = utils.load_config("config.yaml")
    device = config["shared"]["device"]

    # Check CUDA availability
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Create environment with rendering enabled
    env, obs_shape, action_shape, action_space = environment.create_environment(
        config["environment"]["name"], config["environment"]["seed"], render_mode="human"
    )
    obs_dim = obs_shape[0]
    action_dim = action_shape[0]

    print(f"\n{'=' * 60}")
    print(f"Visualizing {agent_type.upper()} Agent")
    print(f"{'=' * 60}")
    print(f"Episodes to render: {num_episodes}")
    print(f"Device: {device}")

    # Instantiate agent
    if agent_type == "reinforce":
        agent = REINFORCEAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            policy_lr=config["reinforce"]["policy_lr"],
            value_lr=config["reinforce"]["value_lr"],
            gamma=config["shared"]["gamma"],
            hidden_sizes=config["shared"]["hidden_sizes"],
            activation=config["shared"]["activation"],
            max_grad_norm=config["shared"]["max_grad_norm"],
            device=device,
        )
        result_dir = "reinforce_results"

    elif agent_type == "ppo":
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            rollout_length=config["ppo"]["rollout_length"],
            mini_batch_size=config["ppo"]["mini_batch_size"],
            epochs=config["ppo"]["epochs"],
            clip_epsilon=config["ppo"]["clip_epsilon"],
            gae_lambda=config["ppo"]["gae_lambda"],
            value_loss_coef=config["ppo"]["value_loss_coef"],
            entropy_coef=config["ppo"]["entropy_coef"],
            entropy_decay=config["ppo"].get("entropy_decay", 1.0),
            lr=config["ppo"]["lr"],
            gamma=config["shared"]["gamma"],
            hidden_sizes=config["shared"]["hidden_sizes"],
            activation=config["shared"]["activation"],
            max_grad_norm=config["shared"]["max_grad_norm"],
            device=device,
        )
        result_dir = "ppo_results"

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Load model
    if model_path is None:
        model_path = str(Path(result_dir) / "final_model")

    if not Path(f"{model_path}_policy.pt").exists():
        print(f"✗ Model not found at {model_path}")
        print(f"  Make sure you've trained the {agent_type} agent first!")
        return

    try:
        agent.load_model(model_path)
        print(f"✓ Loaded model from {model_path}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Run episodes with rendering
    print(f"\nRunning {num_episodes} episodes with rendering...\n")

    total_reward = 0.0

    for episode in range(num_episodes):
        obs, info = environment.reset_environment(env)
        episode_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Render the environment
            env.render()

            # Select action (deterministic for demo)
            action, _, _ = agent.select_action(obs, deterministic=True)

            # Step environment
            next_obs, reward, terminated, truncated, info = environment.step_environment(
                env, action
            )
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs
            steps += 1

        total_reward += episode_reward
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}, Steps: {steps}")

    env.close()

    avg_reward = total_reward / num_episodes
    print(f"\n{'=' * 60}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"{'=' * 60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize a trained agent running in BipedalWalker"
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["reinforce", "ppo"],
        default="ppo",
        help="Agent type to visualize (default: ppo)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to render (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model (without extension). If not provided, uses latest from results.",
    )

    args = parser.parse_args()

    load_and_demo(
        agent_type=args.agent, num_episodes=args.episodes, model_path=args.model
    )


if __name__ == "__main__":
    main()
