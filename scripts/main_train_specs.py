import pickle, argparse

from tl_search.common.seed import torch_fix_seed
from tl_search.map.utils import map2props
from tl_search.search.neighbor import (
    create_all_nodes,
    eliminate_searched_specs,
    nodes2specs,
)
from tl_search.tl.constants import (
    obs_props_all,
    atom_prop_dict_all,
    known_policy_map_object_ids,
)
from tl_search.common.typing import (
    EnemyPolicyMode,
    EnvProps,
    Exclusion,
    MapProps,
    ModelProps,
    RLAlgorithm,
    SaveMode,
    SpecNode,
)
from tl_search.tl.tl_parser import get_used_tl_props
from tl_search.train.wrapper import train_multipocesss

parser = argparse.ArgumentParser(description="Run RLTL training code.")
parser.add_argument("cuda", type=int, help="GPU to use i.e. cuda:0, cuda:1")
args = parser.parse_args()
gpu: int = args.cuda

run: int = 7
num_replicates: int = 3
total_timesteps: int = 50_0000
seed: float = 3406


nums_process: list[int] = [10]  # , 16, 13]

exclusions: list[Exclusion] = ["group"]

vars: tuple[str, ...] = ("psi_ba_ra", "psi_ba_bf", "psi_ba_rf", "psi_ba_obs")
rl_algorithm: RLAlgorithm = "ppo"
enemy_policy: EnemyPolicyMode = "random"
map_filename: str = "maps/board_0002.txt"


log_dir: str = "./tmp/gym/search/exh/{}/".format(run)
suffix: str = "policy_{}_{}_{}".format(enemy_policy, rl_algorithm, run)
model_savename: str = "models/search/{}".format(suffix)
model_props_savename: str = "models/search/{}_{}_data.pickle".format(suffix, gpu)
reward_curve_savename: str = "search/{}".format(suffix)
animation_savename: str = "plots/animation/{}".format(suffix)

reward_curve_plot_mode: SaveMode = "enabled"
animation_mode: SaveMode = "disabled"
is_gui_shown: bool = False
is_reward_calculated: bool = True
is_initialized_in_both_territories: bool = True

torch_fix_seed(seed)
replicate_seeds: list[int] = [406 + i * 1000 for i in range(num_replicates)]

map_props = MapProps(*map2props(map_filename, known_policy_map_object_ids))

neighbor_nodes: list[SpecNode] = create_all_nodes(vars, exclusions)
neighbor_specs: list[str] = list(set(nodes2specs(neighbor_nodes)))
neighbor_specs.sort()

searching_specs, searching_inds_all = eliminate_searched_specs(
    neighbor_specs, "out/plots/reward_curve/search/", suffix, "png"
)

env_props_list_all: list[EnvProps] = [
    EnvProps(
        spec,
        *get_used_tl_props(spec, atom_prop_dict_all, obs_props_all),
        enemy_policy,
        map_props,
    )
    for spec in searching_specs
]

num_envs: int = len(env_props_list_all)
num_gpus: int = len(nums_process)

env_start_ind: int = round(sum(nums_process[0:gpu]) / sum(nums_process) * num_envs)
env_end_ind: int = round(
    (sum(nums_process[0 : (gpu + 1)])) / sum(nums_process) * num_envs
)
env_props_list: list[EnvProps] = (
    env_props_list_all[env_start_ind:env_end_ind]
    if gpu != len(nums_process) - 1
    else env_props_list_all[env_start_ind:]
)
searching_inds: list[int] = (
    searching_inds_all[env_start_ind:env_end_ind]
    if gpu != len(nums_process) - 1
    else searching_inds_all[env_start_ind:]
)

model_props_list: list[list[ModelProps]] = train_multipocesss(
    nums_process[gpu],
    rl_algorithm,
    env_props_list,
    searching_inds,
    gpu,
    num_replicates,
    total_timesteps,
    log_dir,
    model_savename,
    reward_curve_plot_mode,
    reward_curve_savename,
    is_initialized_in_both_territories,
    animation_mode,
    animation_savename,
    is_gui_shown,
    replicate_seeds,
)

print("Saving model properties.")
with open(model_props_savename, mode="wb") as f:
    pickle.dump(model_props_list, f)
