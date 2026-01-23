from collections.abc import Callable
from pathlib import Path

from feature_extractor import MinigridFeaturesExtractor
from simple_env import SimpleEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecMonitor

from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import ImgObsWrapper, Wrapper


class EnvOptimiser:
    """Create multiple vectorised environments for parallel processing with any number of compatible wrappers."""

    def __init__(
        self,
        env_cls: MiniGridEnv,
        n_envs: int = 1,
        vec_env_cls: VecEnv = SubprocVecEnv,
        wrapper_cls: list[Wrapper] | None = None,
        log_dir: Path = Path("logs/"),  # creates dir in root by default
    ) -> None:
        """Init method."""
        self.env_cls = env_cls
        self.n_envs = n_envs
        self.vec_env_cls = vec_env_cls
        self.wrapper_cls = wrapper_cls
        self.log_dir = log_dir

    def _build_env(self, idx: int = 0) -> Callable:
        """Make environment."""

        def _init() -> MiniGridEnv:
            """Init method."""
            env = self.env_cls()

            # Wrappers
            if self.wrapper_cls:  # and isinstance(self.wrapper_cls, list):
                for wrapper in self.wrapper_cls:  # if list of wrappers
                    env = wrapper(env)
            # elif self.wrapper_cls:  # if single wrapper
            #     env = self.wrapper_cls(env)

            # Monitor
            self.log_dir.mkdir(parents=True, exist_ok=True)
            # if self.n_envs == 1:
            env = Monitor(
                env, filename=str(self.log_dir / f"{idx}_")
            )  # record episode reward, length, time to csv
            env.reset(seed=idx)

            return env

        return _init

    def build_vec_env(self) -> VecEnv:
        """Make vectorized environment."""
        vec_env = self.vec_env_cls([self._build_env(i) for i in range(self.n_envs)])
        return VecMonitor(
            vec_env, str(self.log_dir)
        )  # record episode reward, length, time to csv


if __name__ == "__main__":
    # Test EnvOptimiser
    log_dir = Path("tmp/")
    builder = EnvOptimiser(
        env_cls=SimpleEnv,
        n_envs=4,
        wrapper_cls=[ImgObsWrapper],
        # log_dir=log_dir,
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
