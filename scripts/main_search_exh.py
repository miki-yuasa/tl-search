from tl_search.tl.constants import (
    obs_props_all,
    atom_prop_dict_all,
    known_policy_map_object_ids,
)
from tl_search.evaluation.ranking import sort_spec
from tl_search.map.utils import map2props
from tl_search.search.search import search_exh
from tl_search.common.typing import (
    EnemyPolicyMode,
    Exclusion,
    ModelProps,
    MapProps,
    PositiveRewardLogger,
    SortedLogger,
)
from tl_search.common.io import data_saver
from tl_search.common.seed import torch_fix_seed

gpus: tuple[int, ...] = (1, 1)
num_process: int = 16

experiment: int = 5
description: str = "exhaustive search for learned rl policy"
num_sampling: int = 8000
num_replicates: int = 3
total_timesteps: int = 50_0000
seed: float = 3406

exlclusions: list[Exclusion] = ["group"]

vars: tuple[str, ...] = ("psi_ba_ra", "psi_ba_bf", "psi_ba_rf", "psi_ba_obs")
atom_prop_dict: dict[str, str] = {var: atom_prop_dict_all[var] for var in vars}
enemy_policy: EnemyPolicyMode = "random"

target_spec: str = "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)"
model_name: str = "policies/learned/learned_policy_search_random_12_0_0"
map_filename: str = "maps/board_0002.txt"

map_props = MapProps(*map2props(map_filename, known_policy_map_object_ids))
target_props = ModelProps(
    model_name, target_spec, obs_props_all, atom_prop_dict, map_props
)

torch_fix_seed(seed)
replicate_seeds: list[int] = [406 + i * 1000 for i in range(num_replicates)]

pos_reward_data: PositiveRewardLogger = search_exh(
    num_process,
    vars,
    description,
    experiment,
    gpus,
    target_props,
    enemy_policy,
    num_sampling,
    num_replicates,
    total_timesteps,
    atom_prop_dict,
    obs_props_all,
    exlclusions,
    replicate_seeds,
)

sorted_specs, sorted_min_kl_divs = sort_spec(
    pos_reward_data.tl_specs_pos, pos_reward_data.kl_divs_pos
)

multi_start_result = SortedLogger(experiment, sorted_specs, sorted_min_kl_divs)

multi_start_savename: str = "data/search/exh_sorted_{}.json".format(experiment)

data_saver(multi_start_result, multi_start_savename)
