import math

import matplotlib.pyplot as plt
import numpy as np
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

    if debug:
        print("env.action_space", env.action_space)

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps_counter += 1

        if debug:
            print(f"Step: {steps_counter}", f"Action: {action}")

        if terminated:
            if debug:
                print("MiniGrid terminated!")
            done = True

        if truncated:
            if debug:
                print("MiniGrid truncated!")
            done = True

    env.close()


def test_model_on_level(model: PPO, level_id: int) -> None:
    """Test model on level."""
