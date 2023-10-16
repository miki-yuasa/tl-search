import unittest

from matplotlib import pyplot as plt

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy

from tl_search.common.plotter import moving_average


class TestLoadResults(unittest.TestCase):
    def test_load_results(self):
        log_path: str = "./tmp/log/search/fight/20230519191706_F((psi_ba_bt_or_!psi_ra_bf)_and_(!psi_ba_rf_or_psi_ba_bf))_and_G(psi_ba_ra)/1/"
        max_x: int = 500_000
        loaded_results = load_results(log_path)
        x_orig, y_orig = ts2xy(loaded_results, "timesteps")
        x_orig = x_orig.astype("float64")
        xvals: NDArray = np.arange(0, max_x, 2)
        y_interp: NDArray = np.interp(xvals, x_orig, y_orig)
        y_ave = moving_average(y_interp, window=1000)
        xvals = xvals[len(xvals) - len(y_ave) :]

        fig = plt.figure()
        plt.plot(xvals, y_ave)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Rewards")
