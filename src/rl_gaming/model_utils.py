from pathlib import Path

import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, OneHotPartialObsWrapper
from minigrid_levels_env import MiniGridLevelsEnv
from procedural_level import ProceduralLevel
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO


def wrap_model_env(base_env: MiniGridEnv, use_recurrent: bool = False) -> MiniGridEnv:
    """Wrap a MiniGrid for environment for PPO or RecurrentPPO."""
    if use_recurrent:
        env = OneHotPartialObsWrapper(base_env)
        env = ImgObsWrapper(env)
        return make_vec_env(lambda: env, n_envs=1, vec_env_cls=DummyVecEnv)

    return ImgObsWrapper(base_env)


def test_recurrent_ppo_on_procedural_level(model_path: Path) -> None:
    """Test RecurrentPPO on procedural level."""
    base_env = ProceduralLevel(render_mode="human", difficulty=1000, max_steps=100)
    test_recurrent_ppo_on_env(model_path, base_env)


def test_recurrent_ppo_on_level(model_path: Path, level_id: int) -> None:
    """Test RecurrentPPO on level."""
    base_env = MiniGridLevelsEnv(level_id=level_id, render_mode="human")
    test_recurrent_ppo_on_env(model_path, base_env)


def test_ppo_model_on_level(model_path: Path, level_id: int) -> None:
    """Test PPO model on level."""
    base_env = MiniGridLevelsEnv(level_id=level_id, render_mode="human")
    test_ppo_model_on_env(model_path, base_env)


def test_ppo_model_on_env(model_path: Path, base_env: MiniGridEnv) -> None:
    """Test PPO model on level."""
    env = wrap_model_env(base_env, use_recurrent=False)

    model = PPO.load(model_path, env=env)

    obs, info = env.reset()

    done = False
    total_reward = 0

    while not done:
        action, _state = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        done = terminated or truncated

    env.close()


def test_recurrent_ppo_on_env(model_path: Path, base_env: MiniGridEnv) -> None:
    """Test RecurrentPPO on environment."""
    env = wrap_model_env(base_env, use_recurrent=True)

    model = RecurrentPPO.load(model_path, env=env)
    obs = env.reset()
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)

    while True:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
        )

        obs, rewards, dones, info = env.step(action)
        episode_starts = dones

        if dones[0]:
            break

    env.close()
