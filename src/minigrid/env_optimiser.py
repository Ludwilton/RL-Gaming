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
        env_cls: type[MiniGridEnv],
        n_envs: int,
        wrapper_cls: list[type[Wrapper]],
        vec_env_cls: type[VecEnv] = SubprocVecEnv,
        log_dir: Path = Path("logs/"),  # creates dir in root by default
    ) -> None:
        """Init method."""
        self.env_cls = env_cls
        self.n_envs = n_envs
        self.wrapper_cls = wrapper_cls
        self.vec_env_cls = vec_env_cls
        self.log_dir = log_dir

    def _no_overwrite(self) -> Path:
        """Enforce a filename prefix to ensure that the filename does not overwrite an existing file."""
        suffix = "monitor.csv"  # the hardcoded default suffix used by Monitor/VecMonitor
        for counter in count(0):
            filename = self.log_dir / f"{counter}_{suffix}"
            if not filename.exists():
                break
        return filename

    def _build_env(self, idx: int = 0) -> Callable:
        """Make environment."""

        def _init() -> MiniGridEnv:
            """Init method."""
            env = self.env_cls()
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

        # Record episode reward, length, time to csv
        return VecMonitor(vec_env, str(self._no_overwrite()))


def test_optimiser() -> None:
    """Test the environment optimiser."""
    from feature_extractor import MinigridFeaturesExtractor
    from simple_env import SimpleEnv

    from minigrid.wrappers import ImgObsWrapper

    n_envs = 8
    n_timesteps = 5_000

    log_dir = Path("logs/")
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path("models/")
    save_dir.mkdir(parents=True, exist_ok=True)

    optimiser = EnvOptimiser(
        env_cls=SimpleEnv, n_envs=n_envs, wrapper_cls=[ImgObsWrapper], log_dir=log_dir
    )
    vec_env_train = optimiser.build_vec_env()
    policy_kwargs = {
        "features_extractor_class": MinigridFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }
    model = PPO(policy="CnnPolicy", env=vec_env_train, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=n_timesteps, progress_bar=True)
    model.save(save_dir / "test_model")  # this will overwrite existing model


if __name__ == "__main__":
    test_optimiser()
