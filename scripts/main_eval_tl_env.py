from typing import Callable
from tl_search.common.io import spec2title
from tl_search.common.typing import ObsProp
from tl_search.evaluation.eval import (
    evaluate_replicate_tl_models,
    evaluate_tuned_param_replicate_tl_models,
)
from tl_search.envs.typing import EnemyPolicyMode
from tl_search.map.utils import distance_area_point, distance_points


enemy_policy_modes: list[EnemyPolicyMode] = [
    "fight",
    # "patrol",
]  # ["fight", "capture", "patrol"]
num_episodes: int = 200
num_replicates: int = 3
map_path: str = "tl_search/map/maps/board_0002_obj.txt"
tuned_param_name: str = "ent_coef"
tuned_param_values: list[float] = [0.0, 0.01, 0.05, 0.1]
tl_spec: str = "F(psi_ba_rf & !psi_ba_bf) & G(!psi_ra_bf & (!psi_ba_ra | psi_ba_bt))"

global_threshold: float = 0.5
atom_prep_dict: dict[str, str] = {
    "psi_ba_ra": "d_ba_ra < {}".format(1.5),
    "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
    "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
    "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
    "psi_ba_bt": "d_ba_bt < {}".format(global_threshold),
}

obs_info: list[tuple[str, list[str], Callable]] = [
    ("d_ba_ra", ["blue_agent", "red_agent"], lambda x, y: distance_points(x, y)),
    ("d_ba_rf", ["blue_agent", "red_flag"], lambda x, y: distance_points(x, y)),
    (
        "d_ba_bt",
        ["blue_agent", "blue_background"],
        lambda x, y: distance_area_point(x, y),
    ),
    ("d_ba_bf", ["blue_agent", "blue_flag"], lambda x, y: distance_points(x, y)),
    ("d_ra_bf", ["red_agent", "blue_flag"], lambda x, y: distance_points(x, y)),
]
obs_props: list[ObsProp] = [ObsProp(*obs) for obs in obs_info]

for enemy_policy_mode in enemy_policy_modes:
    stats_path: str = f"out/data/search/stats/stats_{enemy_policy_mode}_enemy_ppo_{spec2title(tl_spec)}.json"
    model_path: str = f"out/models/search/test/{enemy_policy_mode}_enemy_ppo_{spec2title(tl_spec)}.zip"

    evaluate_tuned_param_replicate_tl_models(
        tuned_param_name,
        tuned_param_values,
        num_replicates,
        tl_spec,
        obs_props,
        atom_prep_dict,
        enemy_policy_mode,
        model_path,
        map_path,
        num_episodes,
        stats_path,
    )
