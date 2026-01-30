from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, window_func


def plot_training_results(log_dir: Path, window_size: int = 50) -> None:
    """Plot training results from a directory."""
    df = load_results(log_dir)
    x, y = ts2xy(df, "timesteps")

    # Plot raw data
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.scatter(x, y, s=2, alpha=0.6, zorder=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.ylim(-0.1, 1.1)
    plt.grid(zorder=1)
    plt.title("Plotting all data points")

    # Plot smoothed data
    plt.subplot(2, 1, 2)
    x_smooth, y_smooth = window_func(x, y, window_size, np.mean)
    plt.plot(x_smooth, y_smooth, linewidth=2, zorder=2)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.ylim(-0.1, 1.1)
    plt.grid(zorder=1)
    plt.title("Plotting smoothed data")

    plt.tight_layout()
    plt.show()
