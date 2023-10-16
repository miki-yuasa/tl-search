import os
import pickle

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import seaborn as sns

from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.callbacks import BaseCallback


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            try:
                x, y = ts2xy(load_results(self.log_dir), "timesteps")
                x.astype("float64")
                y.astype("float64")
                if len(x) > 0:
                    # Mean training reward over the last 100 episodes
                    mean_reward = np.mean(y[-100:])
                    if self.verbose > 0:
                        print(f"Num timesteps: {self.num_timesteps}")
                        print(
                            f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                        )

                    # New best model, you could save the agent here
                    if mean_reward > self.best_mean_reward:
                        self.best_mean_reward = mean_reward
                        # Example for saving best model
                        if self.verbose > 0:
                            print(f"Saving new best model to {self.save_path}.zip")
                        self.model.save(self.save_path)
            except:
                print("No results found")
                print(self.log_dir)
                return True

        return True


def moving_average(values: NDArray, window: int) -> NDArray:
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_ablation(
    lcs: list[tuple[NDArray, NDArray]], plot_save_path: str, max_x: int, window: int
) -> None:
    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"
    xvals: NDArray = np.arange(0, max_x, 2)
    Y_interp: NDArray = np.array(
        [moving_average(np.interp(xvals, x, y), window=window) for x, y in lcs]
    )
    y_min: NDArray = np.amin(Y_interp, axis=0)
    y_max: NDArray = np.amax(Y_interp, axis=0)
    y_mean: NDArray = np.mean(Y_interp, axis=0)

    min_y_len = np.min([len(y_min), len(y_max), len(y_mean)])
    max_y_len = np.max([len(y_min), len(y_max), len(y_mean)])
    assert min_y_len == max_y_len
    y_min = y_min[len(y_min) - min_y_len :]
    y_max = y_max[len(y_max) - min_y_len :]
    y_mean = y_mean[len(y_mean) - min_y_len :]
    xvals = xvals[len(xvals) - min_y_len :]

    fig, ax = plt.subplots()

    ax.fill_between(
        xvals,
        y_min,
        y_max,
        alpha=0.2,
        color="b",
    )
    ax.plot(
        xvals,
        y_mean,
        color="b",
    )

    ax.set_xlabel("Number of Timesteps")
    ax.set_ylabel("Rewards")
    plt.savefig(plot_save_path, dpi=600)
    plt.clf()
    plt.close()


def plot_results(
    log_folder: str, learning_curve_path: str, max_x: int, window: int = 10
) -> tuple[NDArray, NDArray]:
    """
    plot the results

    Parameters
    -----------
    log_folder: str
        the save location of the results to plot
    learning_curve_path: str
        the save location of the learning curve
    window: int
        the moving average window

    Returns
    -------
    x: NDArray
        the x values
    y: NDArray
        the y values
    """
    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "Times New Roman"
    try:
        x_orig, y_orig = ts2xy(load_results(log_folder), "timesteps")
        x_orig = x_orig.astype("float64")
        y_orig = y_orig.astype("float64")
        xvals: NDArray = np.arange(0, max_x, 2)
        y_interp: NDArray = np.interp(xvals, x_orig, y_orig)
        y_ave = moving_average(y_interp, window=window)
        xvals = xvals[len(xvals) - len(y_ave) :]

        fig = plt.figure()
        plt.plot(xvals, y_ave)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
        plt.savefig(learning_curve_path)
    except:
        print(log_folder)
        print(learning_curve_path)
        raise Exception("Results not read.")

    with open(
        learning_curve_path.replace(".png", "") + "_reward.pickle",
        mode="wb",
    ) as f:
        pickle.dump((x_orig, y_orig), f)

    return x_orig, y_orig
