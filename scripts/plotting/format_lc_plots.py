import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ptick
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

from tl_search.common.utils import moving_average

plot_font_name: str = "Times New Roman"
lc_data_filename: str = (
    "out/plots/reward_curve/search/heuristic/patrol/patrol_enemy_ppo_F((psi_ba_rf)_and_(!psi_ra_bf))_and_G(!psi_ba_ra_or_psi_ba_bt)_target_ent_coef_0.1_1_reward.pickle"  # "out/plots/reward_curve/search/heuristic/fight/fight_enemy_ppo_F(psi_ba_rf_and_!psi_ba_bf)_and_G(!psi_ra_bf_and_(!psi_ba_ra_or_psi_ba_bt))_ws_ent_coef_0.0.pkl"
)
savename: str = (
    "out/plots/reward_curve/exp1.png"  # lc_data_filename.replace(".pkl", ".png")
)
moving_average_window: int = 10000
is_bad_one_plotted: bool = True

sns.set(style="whitegrid", font_scale=1.3)
plt.rcParams["font.family"] = plot_font_name
# plt.rcParams["figure.subplot.bottom"] = 0.14
# plt.rcParams["figure.subplot.left"] = 0.15

fig, ax = plt.subplots(figsize=(8, 3))

with open(lc_data_filename, "rb") as f:
    # X: list[NDArray] = pickle.load(f)
    # Y: list[NDArray] = pickle.load(f)
    data = pickle.load(f)

# X: list[NDArray] = []
# Y: list[NDArray] = []

# for x, y in data:
#     X.append(x)
#     Y.append(y)

X = [data[0]]
Y = [data[1]]

xvals: NDArray = np.arange(0, 500000, 4)
Y_iterp: NDArray = np.array(
    [
        moving_average(np.interp(xvals, x, y), moving_average_window)
        for x, y in zip(X, Y)
    ]
)
y_min = np.amin(Y_iterp, axis=0)
y_max = np.amax(Y_iterp, axis=0)
y_mean = np.mean(Y_iterp, axis=0)

ax.fill_between(
    xvals,
    y_min,
    y_max,
    alpha=0.2,
    color="b",
)
line1 = ax.plot(
    xvals,
    y_mean,
    color="b",
    label="Solution specification",
)

if is_bad_one_plotted:
    bad_savename: str = (
        "out/plots/reward_curve/search/heuristic/patrol/patrol_enemy_ppo_F(!psi_ba_bt_and_!psi_ra_bf)_and_G(psi_ba_ra_and_psi_ba_rf)_2_reward.pickle"  # "out/plots/reward_curve/search/policy_random_ppo_7_291_rewards.pickle"
    )
    with open(bad_savename, "rb") as f:
        # X_bad: list[NDArray] = pickle.load(f)
        # Y_bad: list[NDArray] = pickle.load(f)
        bad_data = pickle.load(f)

    X_bad = [bad_data[0]]
    Y_bad = [bad_data[1]]
    Y_bad_interp: NDArray = np.array(
        [np.interp(xvals, x, y) for x, y in zip(X_bad, Y_bad)]
    )
    y_bad_min = np.amin(Y_bad_interp, axis=0)
    y_bad_max = np.amax(Y_bad_interp, axis=0)
    y_bad_mean = np.mean(Y_bad_interp, axis=0)
    ax.tick_params(axis="y", labelcolor="b")
    ax.grid(False)
    ax2 = ax.twinx()
    ax2.fill_between(
        xvals,
        moving_average(y_bad_min, moving_average_window),
        moving_average(y_bad_max, moving_average_window),
        alpha=0.2,
        color="r",
    )
    line2 = ax2.plot(
        xvals,
        moving_average(y_bad_mean, moving_average_window),
        color="r",
        label="Nonsensical specification",
    )
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.grid(False)

    # ax.set_yticks(np.linspace(ax.get_ybound()[0], ax.get_ybound()[1], 4))
    # ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 4))

    lines = line1 + line2
    labels = [l.get_label() for l in lines]

    plt.legend(lines, labels, loc="lower right")


else:
    pass

# for i, (x, y) in enumerate(zip(X, Y)):
#     if x.size == y.size:
#         plt.plot(x, y, label="Replicate {}".format(i))
#     else:
#         print(
#             "Skipping plotting the graph because the dimensions of x and y are different."
#         )

ax.yaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
ax.ticklabel_format(
    style="sci", axis="x", scilimits=(5, 5)
)  # 10^3単位の指数で表示する。
# ax.set_xticks([0, 100000, 200000, 300000, 400000])
# ax.set_xlim([0, 500000])
plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
ax.set_xlabel("Number of Timesteps [-]")
ax.set_ylabel("Returns [-]")
plt.savefig(savename, dpi=600, bbox_inches="tight")
plt.clf()
plt.close()
