import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import ticker
import seaborn as sns

from tl_search.common.typing import KLDivReport

file_prefix: str = "out/data/search/parking/parking_exp1_extended_sac"
file_suffix: str = "sorted.json"

kl_div_savename: str = "out/data/search/oarking_exp1_single_kl_divs.json"


fig_name: str = "out/plots/rank/rank_parking_exp1.png"

multi_start_kl_divs: list[float] = [
    0.00e-08,
    7.35e-04,
    7.35e-04,
    6.40e-04,
    0.00e-08,
    4.61e-04,
    0.00,
    4.62e-04,
]

nums_searched_specs: list[int] = [36, 27, 26, 29, 28, 43, 36, 32]
num_total_specs: int = 96

max_search_steps: int = 10

rw_kl_div: float | None = None

num_start: int = len(multi_start_kl_divs)

if os.path.exists(kl_div_savename):
    with open(kl_div_savename, mode="r") as f:
        spec_kl_div_dict: dict[str, float] = json.load(f)
else:
    spec_kl_div_dict: dict[str, float] = {}

    for i in range(num_start):
        for j in range(max_search_steps):
            file_name: str = f"{file_prefix}.{i+1}.0.{j}_{file_suffix}"

            if os.path.exists(file_name):
                with open(file_name, mode="r") as f:
                    data = json.load(f)

                specs: list[str] = data["specs"]
                kl_divs: list[float] = data["kl_divs"]

                for spec, kl_div in zip(specs, kl_divs):
                    spec_kl_div_dict[spec] = kl_div

        else:
            break

    with open(kl_div_savename, mode="w") as f:
        json.dump(spec_kl_div_dict, f)

kl_divs_all: list[float] = list(spec_kl_div_dict.values())

kl_divs_all.sort()
num_kl_divs: int = len(kl_divs_all)

percentage_thresholds: list[float] = [0.01, 0.05, 0.10]

threshold_values: list[float] = [
    kl_divs_all[round(percent * num_kl_divs)] for percent in percentage_thresholds
]

x_min: int = 0
x_max: int = 100

font_dict = {"color": "#d62728"}

sns.set(style="whitegrid", font_scale=1.3)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.subplot.bottom"] = 0.13

ax: Axes
fig, ax = plt.subplots(figsize=(7, 3))
# Change the size of the figure

print(np.array(nums_searched_specs) / num_total_specs * 100)
ax.scatter(
    np.array(multi_start_kl_divs),
    np.array(nums_searched_specs) / num_total_specs * 100,
    zorder=2,
)
ax.vlines(threshold_values, x_min, x_max, "#d62728", linestyles="dashed", zorder=1)
if rw_kl_div is not None:
    ax.vlines(rw_kl_div, x_min, x_max, "#2ca02c", linestyles="dashed", zorder=1)
    ax.text(0.55e-5, 18, "RW", color="#2ca02c")
else:
    pass
ax.text(0.1e-4, 22, "1 %", font_dict)
ax.text(4.4e-4, 22, "5 %", font_dict)
ax.text(5.6e-4, 22, "10 %", font_dict)
ax.hlines(66.5, -0.5e-4, 1, linestyles="dashed", zorder=1, colors="#2ca02c")
ax.text(0.2e-4, 61, "Random Search", color="#2ca02c")
ax.set_ylabel("Evaluated specifications [%]")
ax.set_xlabel("Weighted KL divergence [-]")
ax.ticklabel_format(axis="x", style="sci", scilimits=(1, 4), useMathText=True)

ax.set_xlim(-0.00005, 0.0008)
ax.set_ylim(20, 70)
# ax.set_xscale("log")
ax.grid(True)
plt.savefig(fig_name, dpi=600, bbox_inches="tight")
