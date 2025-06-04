from math import ceil
import multiprocessing as mp
import random
from typing import Callable, Final

from tl_search.common.typing import Exclusion, ObsProp, SpecNode
from tl_search.search.search import train_exh_mp
from tl_search.envs.typing import EnemyPolicyMode, FieldObj
from tl_search.map.utils import distance_area_point, distance_points
from tl_search.search.neighbor import create_all_nodes, nodes2specs

if __name__ == "__main__":
    enemy_policy_mode: EnemyPolicyMode = "patrol"
    warm_start: bool = True
    gpu: int = 1
    num_process: int = 10

    num_samples: int = 8000

    n_envs: Final[int] = 25  # 50  # 20
    total_timesteps: Final[int] = 500_000
    num_replicates: Final[int] = 3
    window: Final[int] = ceil(round(total_timesteps / 100))
    map_path: Final[str] = "tl_search/map/maps/board_0002_obj.txt"

    gpus: tuple[int, ...] = (0, 1, 2, 3)

    predicates: tuple[str, ...] = (
        "psi_ba_ra",
        "psi_ba_bt",
        "psi_ba_rf",
        "psi_ra_bf",
    )

    tuned_param_name: Final[str | None] = "ent_coef"
    tuned_param_value: Final[float] = 0.1
    tuned_param: Final[dict[str, float]] = {tuned_param_name: tuned_param_value}

    exclusions: list[Exclusion] = ["group"]

    suffix: str = "_ws" if warm_start else ""
    target_model_path: str = (
        f"out/models/heuristic/{enemy_policy_mode}_enemy_ppo_curr_ent_coef_0.01.zip"
    )
    model_save_path: str = f"out/models/search/heuristic/{enemy_policy_mode}/{enemy_policy_mode}_enemy_ppo{suffix}.zip"
    learning_curve_path: str = f"out/plots/reward_curve/search/heuristic/{enemy_policy_mode}/{enemy_policy_mode}_enemy_ppo{suffix}.png"
    animation_save_path: str | None = None
    data_save_path: str = f"out/data/kl_div/heuristic/{enemy_policy_mode}/kl_div_{enemy_policy_mode}_enemy_ppo{suffix}.json"

    global_threshold: float = 0.5
    atom_prep_dict: dict[str, str] = {
        "psi_ba_ra": "d_ba_ra < {}".format(1),
        "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
        "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
        "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
        "psi_ba_bt": "d_ba_bt < {}".format(global_threshold),
    }
    obs_info: list[tuple[str, list[str], Callable]] = [
        (
            "d_ba_ra",
            ["blue_agent", "red_agent", "is_red_agent_defeated"],
            distance_points,
        ),
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

    seeds: list[int] = [random.randint(0, 10000) for _ in range(num_replicates)]

    mp.set_start_method("spawn")

    neighbor_nodes: list[SpecNode] = create_all_nodes(predicates, exclusions)
    neighbor_specs: list[str] = list(set(nodes2specs(neighbor_nodes)))

    searching_specs: list[str] = neighbor_specs[
        gpu
        * len(neighbor_nodes)
        // len(gpus) : (gpu + 1)
        * len(neighbor_nodes)
        // len(gpus)
    ]

    train_exh_mp(
        gpu,
        num_process,
        num_replicates,
        n_envs,
        seeds,
        total_timesteps,
        model_save_path,
        learning_curve_path,
        animation_save_path,
        window,
        data_save_path,
        neighbor_specs,
        obs_props,
        atom_prep_dict,
        enemy_policy_mode,
        map_path,
        tuned_param,
        None,
        warm_start_path=None if not warm_start else target_model_path,
    )
