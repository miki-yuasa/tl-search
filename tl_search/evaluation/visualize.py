import pickle
from typing import cast, Union

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
from matplotlib.patches import Rectangle, Circle
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from tl_search.tl.environment import Environment
from tl_search.tl.constants import fight_range
from tl_search.map.utils import parse_map
from tl_search.common.io import get_file_path
from tl_search.common.typing import MapObjectIDs, Location, FixedMapLocations, SaveMode


def simulate_model(
    model: Union[PPO, BaseAlgorithm],
    env: Environment,
    blue_agent_init_loc: Union[Location, None] = None,
    red_agent_init_loc: Union[Location, None] = None,
) -> Environment:
    print("Simulating model for: {}".format(env.aut.tl_spec))

    done: bool = False
    obs = (
        env.reset_with_agent_locs(blue_agent_init_loc, red_agent_init_loc)
        if blue_agent_init_loc and red_agent_init_loc
        else env.reset()
    )

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

    return env


def create_animation(
    blue_path: list[Location],
    red_path: list[Location],
    init_map: NDArray,
    map_object_ids: MapObjectIDs,
    video_save_name: str,
    is_gui_shown: bool = False,
) -> None:
    print("Writing out an animation")
    blue_flag, red_flag, blue_territory, red_territory, obstacles, walls = cast(
        FixedMapLocations, parse_map(init_map, map_object_ids)
    )

    fig, ax = plt.subplots()
    artists = []
    h, w = init_map.shape

    markersize = 24
    ax.set_xlim(-0.5, w - 1)
    ax.set_ylim(-0.5, h - 1)
    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(0.5, w + 0.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, h + 0.5, 1), minor=True)
    ax.set_xticks(np.arange(0, w, 1))
    ax.set_yticks(np.arange(0, h, 1))
    ax.grid(which="minor")
    ax.tick_params(which="minor", length=0)
    ax.plot(
        blue_flag.x,
        blue_flag.y,
        marker=">",
        color="mediumblue",
        markersize=markersize,
    )
    ax.plot(
        red_flag.x,
        red_flag.y,
        marker=">",
        color="firebrick",
        markersize=markersize,
    )
    for obs in obstacles:
        obs_rec = Rectangle((obs.x - 0.5, obs.y - 0.5), 1, 1, color="black")
        ax.add_patch(obs_rec)

    for bb in blue_territory:
        bb_rec = Rectangle((bb.x - 0.5, bb.y - 0.5), 1, 1, color="aliceblue")
        ax.add_patch(bb_rec)

    for rb in red_territory:
        rf_rec = Rectangle((rb.x - 0.5, rb.y - 0.5), 1, 1, color="mistyrose")
        ax.add_patch(rf_rec)

    if len(blue_path) > len(red_path):
        max_cnt = len(red_path)
    else:
        max_cnt = len(blue_path)

    for i in range(max_cnt):
        blue_pos = blue_path[i]
        red_pos = red_path[i]
        # blue_agent_fight_region = Circle(
        #     (blue_pos.x, blue_pos.y),
        #     fight_range,
        #     fill=False,
        #     ec="grey",
        #     label="Fight region",
        # )
        # ax.add_artist(blue_agent_fight_region)
        (blue_agent_line,) = ax.plot(
            blue_pos.x,
            (blue_pos.y),
            marker="o",
            color="royalblue",
            markersize=markersize,
            label="Blue agent",
        )
        (red_agent_line,) = ax.plot(
            red_pos.x,
            (red_pos.y),
            marker="o",
            color="crimson",
            markersize=markersize,
        )
        artists.append([blue_agent_line, red_agent_line])

    # Animate
    anim = ArtistAnimation(fig, artists, interval=200)
    anim.save(video_save_name)

    if is_gui_shown:
        plt.show()
    else:
        pass

    plt.clf()
    plt.close()


def plot_reward_curves(
    X: list[NDArray], Y: list[NDArray], spec: str, savename: str, is_gui_shown: bool
):
    for x, y in zip(X, Y):
        if x.size == y.size:
            plt.plot(x, y)
        else:
            print(
                "Skipping plotting the graph because the dimensions of x and y are different."
            )

    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title("Rewards for " + spec)
    if is_gui_shown:
        plt.show(block=False)
    else:
        pass
    plt.savefig("out/plots/reward_curve/{}.png".format(savename))
    plt.clf()
    plt.close()
    with open(
        "out/plots/reward_curve/{}_rewards.pickle".format(savename),
        mode="wb",
    ) as f:
        pickle.dump(X, f)
        pickle.dump(Y, f)


def plot_kl_divs(kl_div_means_plot: list[list[float]], savename: str) -> None:
    plt.boxplot(kl_div_means_plot)
    plt.xlabel("TL Spec ID")
    plt.ylabel("KL Divergence [-]")
    plt.title("KL Divergences of the Given TL Specs")
    plt.grid()
    plt.savefig("out/" + savename)
    plt.show(block=False)
    plt.clf()
    plt.close()


def is_reward_curve_plotted(
    plot_mode: SaveMode, spec_ind: int, suppress_rate: int = 50
) -> bool:
    plot_reward_curve: bool = (
        True
        if plot_mode == "enabled"
        else (
            False
            if plot_mode == "disabled"
            else (
                True
                if plot_mode == "suppressed" and spec_ind % suppress_rate == 0
                else False
            )
        )
    )

    return plot_reward_curve


def is_animation_plotted(plot_mode: SaveMode, rep_ind: int) -> bool:
    plot_animation: bool = (
        True
        if plot_mode == "enabled"
        else (
            False
            if plot_mode == "disabled"
            else (True if plot_mode == "suppressed" and rep_ind == 0 else False)
        )
    )

    return plot_animation
