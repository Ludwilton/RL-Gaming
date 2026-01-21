from collections.abc import Callable
from pathlib import Path

from feature_extractor import MinigridFeaturesExtractor
from save_on_best_training_reward_callback import SaveOnBestTrainingRewardCallback
from simple_env import SimpleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper


def make_env(rank: int, seed: int = 0) -> Callable:
    """Make environment."""

    def _init() -> MiniGridEnv:
        """Init method."""
        env = SimpleEnv()
        env = ImgObsWrapper(env)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":
    # Create log dir
    log_dir = "tmp/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    env = VecMonitor(env, "tmp/TestMonitor")

    policy_kwargs = {
        "features_extractor_class": MinigridFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }

    model = PPO(
        "CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.00003
    )

    print("------------- Start Learning -------------")
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=20000, callback=callback)
    model.save("test-save")
    print("------------- Done Learning -------------")
