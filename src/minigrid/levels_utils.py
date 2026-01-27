import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from minigrid_levels_env import MiniGridLevelsEnv
from stable_baselines3.ppo import PPO


def get_level_env_frame(level_id: int) -> np.ndarray:
    """Get level environment frame."""
    env = MiniGridLevelsEnv(level_id=level_id)
    env.reset()

    return env.get_frame(highlight=False)


def display_single_level(level_id: int = 1) -> None:
    """Display single level."""
    img = get_level_env_frame(level_id)

    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Level {level_id}")

    plt.show()


def display_all_levels(max_cols: int = 5) -> None:
    """Display all levels."""
    levels = MiniGridLevelsEnv.get_levels()
    n_levels = len(levels)

    ncols = min(max_cols, n_levels)
    nrows = math.ceil(n_levels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    axes = axes.flatten() if n_levels > 1 else [axes]

    for i, level_id in enumerate(levels):
        img = get_level_env_frame(level_id)

        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(f"Level {level_id}")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def run_random_actions_on_level(level_id: int, debug: bool = False) -> None:
    """Run a MiniGrid level by repeatedly sampling actions from the action space."""
    env = MiniGridLevelsEnv(level_id=level_id, render_mode="human")
    obs, info = env.reset()

    done = False
    steps_counter = 0
    log_records = []

    while not done:
        action = env.action_space.sample()

        agent_x = env.agent_pos[0]  # agent_x before action
        agent_y = env.agent_pos[1]  # agent_y before action
        agent_dir = env.agent_dir  # agent_dir before action

        obs, reward, terminated, truncated, info = env.step(action)
        steps_counter += 1

        step_data = {
            "steps_counter": steps_counter,
            "agent_x": agent_x,
            "agent_y": agent_y,
            "agent_dir": agent_dir,
            "chosen_action": action,
        }
        log_records.append(step_data)

        if terminated:
            if debug:
                print("\nMiniGrid terminated successfully!")
            done = True

        if truncated:
            if debug:
                print("\nMiniGrid truncated (time limit reached)!")
            done = True

    env.close()

    actions_log_df = pd.DataFrame(log_records)
    actions_log_df = actions_log_df.set_index("steps_counter")

    agent_actions_logs = Path("agent_actions_logs/")
    agent_actions_logs.mkdir(parents=True, exist_ok=True)

    actions_log_df.to_csv(agent_actions_logs / f"log_{time.time()}.csv")

    return actions_log_df


def evaluate_random_actions_on_level(level_id: int, n_episodes: int = 100) -> float:
    """Evaluate random actions on level."""
    successes = 0

    for _ in range(n_episodes):
        env = MiniGridLevelsEnv(level_id=level_id, render_mode=None)

        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            done = terminated or truncated

        env.close()

        # If reward > 0 it is a success
        if total_reward > 0:
            successes += 1

    # Return success rate
    return successes / n_episodes


def evaluate_all_levels(level_ids: list, n_episodes: int = 100) -> dict:
    """Evaluate all levels."""
    results = {}

    for level_id in level_ids:
        print(f"Running level {level_id}...")

        rate = evaluate_random_actions_on_level(
            level_id=level_id, n_episodes=n_episodes
        )

        results[level_id] = rate

    return results


def plot_results(results: dict) -> None:
    """Plot results in bar plot."""
    levels = list(results.keys())
    success_rates = list(results.values())

    plt.figure()
    plt.bar(levels, success_rates)

    plt.xlabel("Level")
    plt.ylabel("Success Rate")
    plt.title("Agent Success Rate per Level")

    plt.ylim(0, 1)

    plt.show()


def test_model_on_level(model: PPO, level_id: int) -> None:
    """Test model on level."""
