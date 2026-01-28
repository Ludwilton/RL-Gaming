from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class CustomCallback(BaseCallback):
    """Monitor training and save the model with the highest mean reward."""

    def __init__(
        self,
        check_freq: int,
        save_dir: Path = Path("models/"),
        verbose: int = 0,
    ) -> None:
        """Initialise callback."""
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.best_mean_reward = -np.inf

    def _eval_output(self, mean_reward: float, saved: bool) -> None:
        if self.verbose > 0:
            if not getattr(self, "header_printed", False):
                print("------------------------------------")
                print("| Timesteps | Rew/Ep | Model Saved |")
                print("------------------------------------")
                self.header_printed = True
            print(f"| {self.num_timesteps:9} | {mean_reward:6.2f} | {str(saved):^11} |")
            print("------------------------------------")

    def _on_step(self) -> bool:
        saved = False
        if self.n_calls % self.check_freq == 0:
            df = load_results(self.save_dir)
            x, y = ts2xy(df, "timesteps")  # x=timesteps, y=rewards
            if len(x) > 0:
                mean_reward = np.mean(y[-100:])  # mean reward over the last 100 episodes
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_dir / "model.zip")
                    saved = True
                self._eval_output(mean_reward, saved)
        return True


def test_callback() -> None:
    """Test the custom callback."""
    from env_optimiser import EnvOptimiser
    from feature_extractor import MinigridFeaturesExtractor
    from minigrid.wrappers import ImgObsWrapper
    from simple_env import SimpleEnv
    from stable_baselines3 import PPO

    n_envs = 8
    n_timesteps = 5_000
    freq = 100
    eval_freq = max(freq // n_envs, 1)  # accounting for multiple environments

    save_dir = Path("models/")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = SimpleEnv()
    optimiser = EnvOptimiser(env=env, n_envs=n_envs, wrapper_cls=[ImgObsWrapper], save_dir=save_dir)
    vec_env_train = optimiser.build_vec_env()
    policy_kwargs = {
        "features_extractor_class": MinigridFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }
    model_dir = optimiser.file_path.parent  # makes use of the same folder
    callback = CustomCallback(check_freq=eval_freq, save_dir=model_dir, verbose=1)
    model = PPO(policy="CnnPolicy", env=vec_env_train, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=n_timesteps, callback=callback, progress_bar=True)


if __name__ == "__main__":
    test_callback()
