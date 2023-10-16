import random
from typing import Callable, Final

from stable_baselines3 import PPO
import torch

from tl_search.common.io import spec2title
from tl_search.common.typing import ObsProp
from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.envs.typing import EnemyPolicyMode
from tl_search.map.utils import distance_area_point, distance_points
from tl_search.envs.train import simulate_model


if __name__ == "__main__":
    enemy_policy_mode: Final[EnemyPolicyMode] = "fight"
    num_replicates: Final[int] = 3

    tl_spec: str = "F((psi_ba_rf)&(!psi_ra_bf)) & G(!psi_ba_ra|psi_ba_bt)"

    gpu: Final[int] = 2

    map_path: Final[str] = "tl_search/map/maps/board_0002_obj.txt"

    filename_common: Final[
        str
    ] = f"search/heuristic/{enemy_policy_mode}/fight_enemy_ppo_ws_{spec2title(tl_spec)}_ws"

    model_save_path: Final[str] = f"out/models/{filename_common}.zip"
    animation_save_path: Final[str] = f"out/plots/animation/{filename_common}.gif"

    global_threshold: float = 0.5
    atom_prep_dict: dict[str, str] = {
        "psi_ba_ra": "d_ba_ra < {}".format(1),
        "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
        "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
        "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
        "psi_ba_bt": "d_ba_bt < {}".format(global_threshold),
    }

    obs_info: list[tuple[str, list[str], Callable]] = [
        ("d_ba_ra", ["blue_agent", "red_agent"], distance_points),
        ("d_ba_rf", ["blue_agent", "red_flag"], distance_points),
        (
            "d_ba_bt",
            ["blue_agent", "blue_background"],
            distance_area_point,
        ),
        ("d_ba_bf", ["blue_agent", "blue_flag"], distance_points),
        ("d_ra_bf", ["red_agent", "blue_flag"], distance_points),
    ]
    obs_props: list[ObsProp] = [ObsProp(*obs) for obs in obs_info]

    seeds: Final[list[int]] = [random.randint(0, 10000) for _ in range(num_replicates)]

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    env = TLMultigrid(tl_spec, obs_props, atom_prep_dict, enemy_policy_mode, map_path)
    print(
        f"Simulate the agent against {enemy_policy_mode} enemy policy for spec {tl_spec}"
    )

    for i in range(num_replicates):
        model = PPO.load(
            model_save_path.replace(".zip", f"_{i}.zip"), env, device=device
        )
        simulate_model(
            model, env, animation_save_path.replace(".gif", f"_{i}.gif"), seed=seeds[i]
        )
