from collections.abc import Callable
from itertools import count
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import Wrapper


class EnvOptimiser:
    """Create multiple vectorised environments for parallel processing with any number of compatible wrappers."""

    def __init__(
        self,
        env: MiniGridEnv,
        n_envs: int,
        wrapper_cls: list[type[Wrapper]],
        vec_env_cls: type[VecEnv] = SubprocVecEnv,
        save_dir: Path = Path("models/"),  # creates dir in root by default
    ) -> None:
        """Init method."""
        self.env = env
        self.n_envs = n_envs
        self.wrapper_cls = wrapper_cls
        self.vec_env_cls = vec_env_cls
        self.save_dir = save_dir
        self.full_path = self._no_overwrite()

    def _no_overwrite(self) -> Path:
        filename = "monitor.csv"
        for counter in count(0):
            file_path = self.save_dir / f"run_{counter}"
            if not file_path.exists():
                break
        return file_path / filename

    def _build_env(self, idx: int = 0) -> Callable:
        """Make environment."""

        def _init() -> MiniGridEnv:
            """Init method."""
            env = self.env
            if self.wrapper_cls:
                for wrapper in self.wrapper_cls:
                    env = wrapper(env)
            env.reset(seed=idx)
            return env

        return _init

    def build_vec_env(self) -> VecEnv:
        """Make vectorized environment."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        vec_env = self.vec_env_cls([self._build_env(i) for i in range(self.n_envs)])

        # Record episode reward, length, time to csv
        return VecMonitor(vec_env, str(self.full_path))


def test_optimiser() -> None:
    """Test the environment optimiser."""
    from feature_extractor import MinigridFeaturesExtractor
    from simple_env import SimpleEnv

    from minigrid.wrappers import ImgObsWrapper

    n_envs = 8
    n_timesteps = 5_000

    save_dir = Path("models/")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = SimpleEnv()
    optimiser = EnvOptimiser(env=env, n_envs=n_envs, wrapper_cls=[ImgObsWrapper], save_dir=save_dir)
    vec_env_train = optimiser.build_vec_env()
    policy_kwargs = {
        "features_extractor_class": MinigridFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }
    model = PPO(policy="CnnPolicy", env=vec_env_train, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=n_timesteps, progress_bar=True)


if __name__ == "__main__":
    test_optimiser()
