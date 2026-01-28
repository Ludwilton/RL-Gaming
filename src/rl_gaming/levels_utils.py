import math
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.core import ObsType
from minigrid.minigrid_env import MiniGridEnv
from minigrid_levels_env import MiniGridLevelsEnv
from model_utils import wrap_env_for_model
from procedural_level import ProceduralLevel
from sb3_contrib import RecurrentPPO
from stable_baselines3.ppo import PPO

GYM_STEP_RESULT_LEN = 5
GYM_RESET_RESULT_LEN = 2


def get_level_env_frame(level_id: int) -> np.ndarray:
    """Return the environment frame for a given level."""
    env = MiniGridLevelsEnv(level_id=level_id)
    env.reset()
    frame = env.get_frame(highlight=False)
    env.close()
    return frame


def display_single_level(level_id: int = 1) -> None:
    """Display a single level."""
    frame = get_level_env_frame(level_id)

    plt.imshow(frame)
    plt.axis("off")
    plt.title(f"Level {level_id}")
    plt.show()


def display_all_levels(max_cols: int = 5) -> None:
    """Display all levels in a grid."""
    levels = MiniGridLevelsEnv.get_levels()
    n_levels = len(levels)

    ncols = min(max_cols, n_levels)
    nrows = math.ceil(n_levels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    # Flatten axes array for easy indexing
    axes = axes.flatten() if n_levels > 1 else [axes]

    for i, level_id in enumerate(levels):
        frame = get_level_env_frame(level_id)

        axes[i].imshow(frame)
        axes[i].axis("off")
        axes[i].set_title(f"Level {level_id}")

    # Turn off any remaining unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def run_random_actions_on_level(level_id: int, debug: bool = False) -> None:
    """Execute a MiniGrid level using random actions and log steps.

    Note:
        This is an experimental utility function. It is primarily intended for testing and debugging, and is not optimized for use.

    """
    env = MiniGridLevelsEnv(level_id=level_id, render_mode="human")
    obs = _get_obs_from_reset(env)

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

        done = terminated or truncated

        if debug and done:
            if terminated:
                print("\nMiniGrid terminated successfully!")
            elif truncated:
                print("\nMiniGrid truncated (time limit reached)!")

    env.close()

    # Save logs
    actions_log_df = pd.DataFrame(log_records)
    actions_log_df = actions_log_df.set_index("steps_counter")
    agent_actions_logs = Path("agent_actions_logs/")
    agent_actions_logs.mkdir(parents=True, exist_ok=True)
    actions_log_df.to_csv(agent_actions_logs / f"log_{time.time()}.csv")

    return actions_log_df


def evaluate_levels_with_recurrent_ppo(
    model_path: Path, level_ids: list | None = None, n_episodes: int = 20
) -> dict:
    """Evaluate all levels with RecurrentPPO model."""
    return _evaluate_all_levels(
        eval_fn=_evaluate_recurrent_ppo,
        level_ids=level_ids,
        n_episodes=n_episodes,
        model_path=model_path,
        description="RecurrentPPO model",
    )


def evaluate_levels_with_ppo(
    model_path: Path, level_ids: list | None = None, n_episodes: int = 20
) -> dict:
    """Evaluate all levels with PPO model."""
    return _evaluate_all_levels(
        eval_fn=_evaluate_ppo,
        level_ids=level_ids,
        n_episodes=n_episodes,
        model_path=model_path,
        description="PPO model",
    )


def evaluate_levels_with_random_actions(
    level_ids: list | None = None, n_episodes: int = 20
) -> dict:
    """Evaluate all levels with random actions."""
    return _evaluate_all_levels(
        eval_fn=_evaluate_random,
        level_ids=level_ids,
        n_episodes=n_episodes,
        model_path=None,
        description="random actions",
    )


def _evaluate_all_levels(
    eval_fn: Callable,
    level_ids: list | None = None,
    n_episodes: int = 20,
    model_path: Path | None = None,
    description: str | None = None,
) -> dict:
    """Evaluate multiple levels using the given evaluation function."""
    results = {}

    if level_ids is None:
        level_ids = MiniGridLevelsEnv.get_levels()

    for level_id in level_ids:
        if description is not None:
            print(f"Running level {level_id} with {description}...")

        if model_path is not None:
            rate = eval_fn(level_id=level_id, n_episodes=n_episodes, model_path=model_path)
        else:
            rate = eval_fn(level_id=level_id, n_episodes=n_episodes)

        results[level_id] = rate

    return results


def bar_plot_levels_success_rate(results: dict) -> None:
    """Plot a bar chart of success rates per level.

    Args:
        results (dict): Dictionary mapping level IDs (int) to success rates (float between 0 and 1).

    """
    # Sort levels for consistent plotting
    levels = sorted(results.keys())
    success_rates = [results[level_id] for level_id in levels]
    levels_str = [str(level_id) for level_id in levels]

    plt.figure(figsize=(4, 3))
    plt.bar(levels_str, success_rates)

    plt.xlabel("Level")
    plt.ylabel("Success Rate")
    plt.title("Agent Success Rate per Level")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def get_procedural_level_env_frame(difficulty: int = 1000) -> np.ndarray:
    """Generate a single frame from a procedural level environment.

    Args:
        difficulty (int): Difficulty parameter for the procedural level.

    Returns:
        A numpy array representing the environment frame.

    """
    env = ProceduralLevel(difficulty=difficulty, max_steps=200)
    env.reset()
    frame = env.get_frame(highlight=False)
    env.close()
    return frame


def display_procedural_levels(
    n_levels: int = 10, max_cols: int = 5, difficulty: int = 1000
) -> None:
    """Display multiple procedural levels in a grid.

    Args:
        n_levels (int): Number of levels to display.
        max_cols (int): Maximum number of columns in the plot grid.
        difficulty (int): Difficulty parameter for the procedural level.

    """
    ncols = min(max_cols, n_levels)
    nrows = math.ceil(n_levels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))

    # Flatten axes array for easy indexing
    axes = axes.flatten() if n_levels > 1 else [axes]

    for i in range(n_levels):
        frame = get_procedural_level_env_frame(difficulty=difficulty)
        axes[i].imshow(frame)
        axes[i].axis("off")

    # Turn off any remaining unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def _evaluate_random(level_id: int, n_episodes: int = 20) -> float:
    """Evaluate a level using random actions."""
    return _evaluate_level(level_id=level_id, n_episodes=n_episodes)


def _evaluate_recurrent_ppo(level_id: int, model_path: Path, n_episodes: int = 20) -> float:
    """Evaluate a level using a trained PPO model."""
    model = RecurrentPPO.load(model_path)
    return _evaluate_level(
        level_id=level_id,
        n_episodes=n_episodes,
        model=model,
        use_recurrent=True,
    )


def _evaluate_ppo(level_id: int, model_path: Path, n_episodes: int = 20) -> float:
    """Evaluate a level using trained RecurrentPPO model with LSTM."""
    model = PPO.load(model_path)
    return _evaluate_level(level_id=level_id, n_episodes=n_episodes, model=model)


def _get_obs_from_reset(env: MiniGridEnv) -> ObsType:
    """Reset the environment and return only observation.

    Handles both Gymnasium/Gym API (returns (obs, info))
    and older SB3 VecEnv API (returns obs).
    """
    result = env.reset()
    if isinstance(result, tuple) and len(result) == GYM_RESET_RESULT_LEN:
        obs, _info = result
    else:
        obs = result
    return obs


def _evaluate_level(
    level_id: int,
    n_episodes: int = 20,
    model: PPO | RecurrentPPO | None = None,
    use_recurrent: bool = False,
) -> float:
    """Evaluate a model (or random actions) on a MiniGrid level."""
    successes = 0

    base_env = MiniGridLevelsEnv(level_id=level_id)
    for _ in range(n_episodes):
        env = (
            wrap_env_for_model(base_env, use_recurrent=use_recurrent)
            if model is not None
            else base_env
        )

        obs = _get_obs_from_reset(env)
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        total_reward = 0
        done = False

        while not done:
            if model is None:
                action = env.action_space.sample()
            elif use_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                )
            else:
                action, _state = model.predict(obs, deterministic=True)

            step_result = env.step(action)
            # Handle different return formats
            if len(step_result) == GYM_STEP_RESULT_LEN:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, rewards, dones, info = step_result
                done = dones[0]
                reward = rewards[0]

            total_reward += reward

            if use_recurrent and episode_starts is not None:
                episode_starts = np.array(done, dtype=bool)

        env.close()

        # If reward > 0 it is a success
        if total_reward > 0:
            successes += 1

    # Return success rate
    return successes / n_episodes
