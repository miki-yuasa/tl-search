import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tl_search.common.typing import KLDivReport

file_name: str = "out/data/search/policy_random_ppo_7_data.pickle"
fig_name: str = "out/plots/rank/kl_div_multistart_rank_11_2.png"

multi_start_kl_divs: list[float] = [
    0.016525672748684883,
    0.032976116985082626,
    0.032976116985082626,
    1.5090737342834473,
    0.032976116985082626,
    0.016525672748684883,
    0.8919214606285095,
    0.016525672748684883,
    0.7952090501785278,
    0.34849908476190483,
]

nums_searched_specs: list[int] = [70, 65, 131, 80, 177, 116, 69, 83, 121, 106]
num_total_specs: int = 640

sns.set(style="whitegrid")

num_start: int = len(multi_start_kl_divs)

with open(file_name, mode="rb") as f:
    exp_data = pickle.load(f)

kl_div_report_model_all: list[KLDivReport]
(
    kl_div_report_model_all,
    kl_div_reports_model_all,
    kl_div_mean_rank,
    reward_mean_all,
    reward_std_all,
) = exp_data

kl_divs_all: list[float] = [
    kl_div_report.kl_div_mean for kl_div_report in kl_div_report_model_all
]

kl_divs_all.sort()
num_kl_divs: int = len(kl_divs_all)

percentage_thresholds: list[float] = [0.01, 0.03, 0.05, 0.1]

threshold_values: list[float] = [
    kl_divs_all[round(percent * num_kl_divs)] for percent in percentage_thresholds
]

x_min: int = 0
x_max: int = num_start + 1

font_dict = {"color": "red"}

plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots()
ax.scatter(np.array(multi_start_kl_divs), np.array(nums_searched_specs) / 640, zorder=2)
ax.vlines(threshold_values, x_min, x_max, "red", linestyles="dashed", zorder=1)
ax.text(0.08, 0.02, "1 %", font_dict)
ax.text(0.3, 0.02, "3 %", font_dict)
ax.text(0.73, 0.02, "5 %", font_dict)
ax.text(1.35, 0.02, "10 %", font_dict)
ax.set_ylabel("Fraction of possible specs evaluated [-]")
ax.set_xlabel("KL divergence between target spec and local minima [-]")
ax.set_ylim(0, 0.4)
ax.grid(True)
plt.savefig(fig_name, dpi=600)
