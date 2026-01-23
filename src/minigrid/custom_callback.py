from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy


class CustomCallback(BaseCallback):
    """Monitor training and save the model with the highest recent mean reward."""

    def __init__(
        self, check_freq: int, log_dir: str, save_dir: str, verbose: int = 1
    ) -> None:
        """Initialise callback."""
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = Path(save_dir) / "best_model"
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        saved = False
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(
                load_results(self.log_dir), "timesteps"
            )  # load training rewards from log
            if len(x) > 0:
                mean_reward = np.mean(
                    y[-100:]
                )  # mean training reward over the last 100 episodes
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    saved = True

                # Evaluation output
                if self.verbose > 0:
                    if not getattr(self, "header_printed", False):
                        print("------------------------------------")
                        print("| Timesteps | Rew/Ep | Model Saved |")
                        print("------------------------------------")
                        self.header_printed = True
                    print(
                        f"| {self.num_timesteps:10} | {mean_reward:7.2f} | {str(saved):^12} |"
                    )
                    print("------------------------------------")

        return True


if __name__ == "__main__":
    pass
