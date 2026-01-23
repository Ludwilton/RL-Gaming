from collections.abc import Callable
from pathlib import Path

from feature_extractor import MinigridFeaturesExtractor
from simple_env import SimpleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, Wrapper


class EnvOptimiser:
    """Create multiple vectorised environments for parallel processing with any number of compatible wrappers."""

    def __init__(
        self,
        env_cls: MiniGridEnv,
        n_envs: int,
        wrapper_cls: list[Wrapper],
        vec_env_cls: VecEnv = SubprocVecEnv,
        log_dir: Path = Path("logs/"),  # creates dir in root by default
    ) -> None:
        """Init method."""
        self.env_cls = env_cls
        self.n_envs = n_envs
        self.wrapper_cls = wrapper_cls
        self.vec_env_cls = vec_env_cls
        self.log_dir = log_dir

    def _build_env(self, idx: int = 0) -> Callable:
        """Make environment."""

        def _init() -> MiniGridEnv:
            """Init method."""
            env = self.env_cls()

            # Wrappers
            if self.wrapper_cls:
                for wrapper in self.wrapper_cls:
                    env = wrapper(env)

            env.reset(seed=idx)
            return env

        return _init

    def build_vec_env(self) -> VecEnv:
        """Make vectorized environment."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        vec_env = self.vec_env_cls([self._build_env(i) for i in range(self.n_envs)])
        return VecMonitor(
            vec_env, str(self.log_dir)
        )  # record episode reward, length, time to csv


if __name__ == "__main__":
    # Test EnvOptimiser
    builder = EnvOptimiser(
        env_cls=SimpleEnv,
        n_envs=4,
        wrapper_cls=[ImgObsWrapper],
    )
    vec_env = builder.build_vec_env()
    policy_kwargs = {
        "features_extractor_class": MinigridFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }
    model = PPO(
        "CnnPolicy", vec_env, policy_kwargs=policy_kwargs, learning_rate=0.00003
    )
    print("------------- Start Learning -------------")
    model.learn(total_timesteps=20_000, progress_bar=True)
    model.save("test_save")
    print("------------- Done Learning -------------")
