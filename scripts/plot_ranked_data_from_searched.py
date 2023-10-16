import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tl_search.common.typing import KLDivReport

file_prefix: str = (
    "out/data/search/heuristic/patrol/exp1_single_extended_patrol_enemy_ppo"
)
# file_name: str = "out/data/search/heuristic/patrol/exp1_single_extended_patrol_enemy_ppo.1.0.0_sorted.json"
file_suffix: str = "sorted.json"

kl_div_savename: str = "out/data/search/exp1_single_kl_divs.json"


fig_name: str = "out/plots/rank/rank_exp1.png"

multi_start_kl_divs: list[float] = [
    8.00e-08,
    5.95e-06,
    8.52e-05,
    1.57e-06,
    8.00e-08,
    7.42e-07,
    1.18e-05,
    1.46e-06,
    1.18e-05,
    1.62e-05,
]

nums_searched_specs: list[int] = [52, 58, 42, 42, 65, 54, 55, 42, 73, 46]
num_total_specs: int = 640

max_search_steps: int = 10

rw_kl_div: float = 1.252e-05

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

fig, ax = plt.subplots()

ax.scatter(
    np.array(multi_start_kl_divs), np.array(nums_searched_specs) / 640 * 100, zorder=2
)
ax.vlines(threshold_values, x_min, x_max, "#d62728", linestyles="dashed", zorder=1)
ax.vlines(rw_kl_div, x_min, x_max, "#2ca02c", linestyles="dashed", zorder=1)
ax.text(1e-7, 1, "1 %", font_dict)
ax.text(0.11e-5, 1, "5 %", font_dict)
ax.text(2.3e-5, 1, "10 %", font_dict)
ax.text(0.55e-5, 18, "RW", color="#2ca02c")
ax.set_ylabel("Evaluated specifications [%]")
ax.set_xlabel("Weighted KL divergence [-]")
ax.set_ylim(0, 20)
ax.set_xscale("log")
ax.grid(True)
plt.savefig(fig_name, dpi=600)
