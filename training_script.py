"""
Main training script for policy gradient agents.

This script serves as the entry point for training REINFORCE or PPO agents.
It reads configuration from config.yaml, instantiates the appropriate agent,
runs the training loop, logs metrics, and saves results.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

import environment
import utils
from reinforce_agent import REINFORCEAgent
from ppo_agent import PPOAgent


def train_reinforce(config: dict, env, agent, result_dir: str):
    """
    Training loop for REINFORCE with Baseline.

    Args:
        config (dict): Configuration dictionary.
        env: Gymnasium environment.
        agent: REINFORCEAgent instance.
        result_dir (str): Directory to save results.
    """
    max_episodes = config["reinforce"]["max_episodes"]
    eval_interval = config["reinforce"]["eval_interval"]
    eval_episodes = config["reinforce"]["eval_episodes"]

    episode_rewards = []
    policy_losses = []
    value_losses = []

    print(f"Training REINFORCE for {max_episodes} episodes...")

    for episode in range(max_episodes):
        obs, info = environment.reset_environment(env)
        episode_reward = 0.0

        # Collect episode
        done = False
        while not done:
            # Select action
            action, log_prob, _ = agent.select_action(obs, deterministic=False)

            # Step environment
            next_obs, reward, terminated, truncated, info = environment.step_environment(
                env, action
            )
            done = terminated or truncated

            # Store transition
            agent.store_transition(obs, action, log_prob, reward)
            episode_reward += reward
            obs = next_obs

        # Update policy
        losses = agent.update()
        episode_rewards.append(episode_reward)
        if losses:
            policy_losses.append(losses["policy_loss"])
            value_losses.append(losses["value_loss"])

        # Logging
        if (episode + 1) % config["training"]["log_interval"] == 0:
            avg_reward = np.mean(episode_rewards[-config["training"]["log_interval"] :])
            print(
                f"Episode {episode + 1}/{max_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg Reward: {avg_reward:.2f}"
            )

        # Evaluation
        if (episode + 1) % eval_interval == 0:
            render = config["reinforce"].get("render", False)
            eval_reward = evaluate_agent(
                env, agent, eval_episodes, deterministic=True, render=render
            )
            print(f"  Evaluation: {eval_reward:.2f}")

        # Save checkpoint
        if (episode + 1) % config["training"]["save_interval"] == 0:
            checkpoint_path = os.path.join(
                result_dir, f"checkpoint_{episode + 1}"
            )
            agent.save_model(checkpoint_path)

    # Save final model and results
    agent.save_model(os.path.join(result_dir, "final_model"))
    utils.save_training_history(
        np.array(episode_rewards),
        {"policy_loss": np.array(policy_losses), "value_loss": np.array(value_losses)},
        result_dir,
    )
    utils.plot_training_curves(
        np.array(episode_rewards),
        {"policy_loss": np.array(policy_losses), "value_loss": np.array(value_losses)},
        "REINFORCE",
        save_path=os.path.join(result_dir, "training_curves.png"),
    )

    print(f"Training complete! Results saved to {result_dir}")

    return np.array(episode_rewards)


def train_ppo(config: dict, env, agent, result_dir: str):
    """
    Training loop for Proximal Policy Optimization.

    Args:
        config (dict): Configuration dictionary.
        env: Gymnasium environment.
        agent: PPOAgent instance.
        result_dir (str): Directory to save results.
    """
    max_episodes = config["ppo"]["max_episodes"]
    rollout_length = config["ppo"]["rollout_length"]
    eval_interval = config["ppo"]["eval_interval"]
    eval_episodes = config["ppo"]["eval_episodes"]

    episode_rewards = []
    episode_count = 0
    policy_losses = []
    value_losses = []
    entropy_losses = []
    entropy_coefs = []  # Track entropy coefficient over time

    print(f"Training PPO for {max_episodes} episodes...")

    while episode_count < max_episodes:
        # Collect rollout
        obs, info = environment.reset_environment(env)
        rollout_episode_reward = 0.0
        steps_in_rollout = 0

        while steps_in_rollout < rollout_length and episode_count < max_episodes:
            # Select action
            action, log_prob, value = agent.select_action(obs, deterministic=False)

            # Step environment
            next_obs, reward, terminated, truncated, info = environment.step_environment(
                env, action
            )
            done = terminated or truncated

            # Store transition
            agent.store_transition(obs, action, log_prob, reward, value, done)
            rollout_episode_reward += reward
            steps_in_rollout += 1

            if done:
                episode_rewards.append(rollout_episode_reward)
                episode_count += 1

                if (episode_count) % config["training"]["log_interval"] == 0:
                    avg_reward = np.mean(
                        episode_rewards[-config["training"]["log_interval"] :]
                    )
                    print(
                        f"Episode {episode_count}/{max_episodes} | "
                        f"Reward: {rollout_episode_reward:.2f} | "
                        f"Avg Reward: {avg_reward:.2f}"
                    )

                # Reset for new episode
                obs, info = environment.reset_environment(env)
                rollout_episode_reward = 0.0
            else:
                obs = next_obs

        # Bootstrap value for GAE
        with torch.no_grad():
            next_obs_tensor = torch.tensor(
                next_obs, dtype=torch.float32, device=agent.device
            )
            next_value = agent.value(next_obs_tensor).item()

        # Update policy
        losses = agent.update(next_value)
        policy_losses.append(losses["policy_loss"])
        value_losses.append(losses["value_loss"])
        entropy_losses.append(losses["entropy_loss"])
        entropy_coefs.append(agent.entropy_coef)  # Track entropy coefficient decay

        # Evaluation
        if episode_count % eval_interval < config["training"]["log_interval"]:
            render = config["ppo"].get("render", False)
            eval_reward = evaluate_agent(
                env, agent, eval_episodes, deterministic=True, render=render
            )
            print(f"  Evaluation: {eval_reward:.2f}")

        # Save checkpoint
        if episode_count % config["training"]["save_interval"] < config["training"][
            "log_interval"
        ]:
            checkpoint_path = os.path.join(result_dir, f"checkpoint_{episode_count}")
            agent.save_model(checkpoint_path)

    # Save final model and results
    agent.save_model(os.path.join(result_dir, "final_model"))
    utils.save_training_history(
        np.array(episode_rewards),
        {
            "policy_loss": np.array(policy_losses),
            "value_loss": np.array(value_losses),
            "entropy_loss": np.array(entropy_losses),
            "entropy_coef": np.array(entropy_coefs),
        },
        result_dir,
    )
    utils.plot_training_curves(
        np.array(episode_rewards),
        {
            "policy_loss": np.array(policy_losses),
            "value_loss": np.array(value_losses),
            "entropy_loss": np.array(entropy_losses),
            "entropy_coef": np.array(entropy_coefs),
        },
        "PPO",
        save_path=os.path.join(result_dir, "training_curves.png"),
    )

    print(f"Training complete! Results saved to {result_dir}")

    return np.array(episode_rewards)


def evaluate_agent(
    env, agent, num_episodes: int, deterministic: bool = True, render: bool = False
) -> float:
    """
    Evaluate a trained agent.

    Args:
        env: Gymnasium environment.
        agent: Agent to evaluate.
        num_episodes (int): Number of episodes to run.
        deterministic (bool, optional): If True, use deterministic policy. Defaults to True.
        render (bool, optional): If True, render the environment. Defaults to False.

    Returns:
        float: Average reward over episodes.
    """
    total_reward = 0.0

    for _ in range(num_episodes):
        obs, info = environment.reset_environment(env)
        episode_reward = 0.0
        done = False

        while not done:
            if render:
                env.render()

            action, _, _ = agent.select_action(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, info = environment.step_environment(
                env, action
            )
            done = terminated or truncated
            episode_reward += reward
            obs = next_obs

        total_reward += episode_reward

    return total_reward / num_episodes


def main():
    """Main entry point."""
    # Load configuration
    config = utils.load_config("config.yaml")
    utils.set_seed(config["environment"]["seed"])

    # Check CUDA availability and auto-fallback
    device = config["shared"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print(
            "⚠️  CUDA requested but not available. Falling back to CPU.\n"
            f"   CUDA available: {torch.cuda.is_available()}"
        )
        device = "cpu"
        config["shared"]["device"] = "cpu"
    elif device == "cuda":
        print(f"✓ Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("✓ Using CPU device")

    # Create environment
    env, obs_shape, action_shape, action_space = environment.create_environment(
        config["environment"]["name"], config["environment"]["seed"]
    )
    obs_dim = obs_shape[0]
    action_dim = action_shape[0]

    print(f"Observation space: {obs_shape}")
    print(f"Action space: {action_shape}")

    # Create result directory
    result_dir = utils.create_directories(config["agent_type"])

    # Instantiate agent
    agent_type = config["agent_type"]
    device = config["shared"]["device"]

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
        rewards = train_reinforce(config, env, agent, result_dir)

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
        rewards = train_ppo(config, env, agent, result_dir)

    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    env.close()
    print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")


if __name__ == "__main__":
    main()
