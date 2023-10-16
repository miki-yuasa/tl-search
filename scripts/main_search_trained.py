import itertools
import os

import pickle
from typing import cast

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
import torch

from tl_search.common.io import data_saver
from tl_search.common.seed import torch_fix_seed
from tl_search.evaluation.extractor import get_agent_action_distributions
from tl_search.search.neighbor import (
    create_neighbor_masks,
    initialize_node,
    node2spec,
)
from tl_search.search.wrapper import search_trained_multistart_multiprocess

from tl_search.tl.constants import (
    obs_props_all,
    atom_prop_dict_all,
    known_policy_map_object_ids,
)
from tl_search.common.typing import (
    EnemyPolicyMode,
    Exclusion,
    MapLocations,
    MapProps,
    ModelProps,
    MultiStartSearchLogger,
    ObsProp,
    RLAlgorithm,
    SpecNode,
    ValueTable,
)
from tl_search.map.utils import map2props, sample_agent_locs
from tl_search.tl.environment import restore_model
from tl_search.tl.tl_parser import get_used_tl_props

experiment: int = 12
gpu: int = 2
num_samples: int = 8000
seed: float = 3406
num_start: int = 6

exclusions: list[Exclusion] = ["group"]

rl_algorithm: RLAlgorithm = "ppo"
enemy_policy: EnemyPolicyMode = "random"
map_filename: str = "maps/board_0002.txt"

vars: tuple[str, ...] = ("psi_ba_ra", "psi_ba_bf", "psi_ba_rf", "psi_ba_obs")
ref_spec: str = "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)"
ref_run: int = 17
leanred_run: int = 7
num_replicates: int = 3

excluded_start_specs: list[str] = [
    "F(psi_ba_ra|psi_ba_rf) & G(psi_ba_bf|!psi_ba_obs)",
    "F(psi_ba_obs) & G(!psi_ba_ra|psi_ba_bf|psi_ba_rf)",
    "F(psi_ba_ra&psi_ba_rf) & G(!psi_ba_bf&psi_ba_obs)",
    "F(!psi_ba_ra&psi_ba_bf) & G(!psi_ba_rf&psi_ba_obs)",
]

is_initialized_in_both_territories: bool = True
is_sampled_from_all_territories: bool = False

data_dir: str = "out/data/"
pickle_suffix: str = "_data.pickle"

ref_run_id: str = "{}_{}_{}".format(enemy_policy, rl_algorithm, ref_run)
ref_name_common: str = "policy_{}".format(ref_run_id)
ref_model_data_savename: str = data_dir + "single/" + ref_name_common + pickle_suffix
ref_model_savename: str = "models/single/{}".format(ref_name_common)

learned_run_id: str = "{}_{}_{}".format(enemy_policy, rl_algorithm, leanred_run)
learned_policy_name_common: str = "policy_{}".format(learned_run_id)

exp_filename_common: str = "multi_start"
exp_data_save_dir: str = data_dir + "search/"

exp_var_data_savename: str = exp_data_save_dir + exp_filename_common + pickle_suffix

all_models_props_savename: str = (
    exp_data_save_dir + learned_policy_name_common + "_all" + pickle_suffix
)
multi_start_savename: str = os.path.join(
    exp_data_save_dir,
    "{}_{}.json".format(exp_filename_common, experiment),
)

ref_model_inds: list[int] = list(range(num_replicates))
learned_model_inds: list[int] = list(range(num_replicates))
ind_combs: list[tuple[int, int]] = list(
    itertools.product(ref_model_inds, learned_model_inds)
)

ref_labels: list[str] = ["ref_{}".format(i) for i in ref_model_inds]
learned_labels: list[str] = ["learned_{}".format(i) for i in learned_model_inds]


num_vars: int = len(vars)
init_nodes: list[SpecNode]

if excluded_start_specs:
    init_nodes = []
    for i in range(num_start):
        while True:
            node = initialize_node(vars, exclusions)
            if node2spec(node) not in excluded_start_specs:
                init_nodes.append(node)
                break
else:
    init_nodes = [initialize_node(vars, exclusions) for _ in range(num_start)]

torch_fix_seed(seed)

ref_model_props: list[ModelProps]
learned_models_props_list: list[list[ModelProps]]
with open(all_models_props_savename, mode="rb") as f:
    ref_model_props, learned_models_props_list = pickle.load(f)

map_props = MapProps(*map2props(map_filename, known_policy_map_object_ids))

learned_obs_props_list: list[list[ObsProp]] = []

for models_props in learned_models_props_list:
    obs_props, atom_prop_dict = get_used_tl_props(
        models_props[0].spec, atom_prop_dict_all, obs_props_all
    )
    learned_obs_props_list.append(obs_props)


ref_obs_props, _ = get_used_tl_props(ref_spec, atom_prop_dict_all, obs_props_all)

device = torch.device("cuda", gpu)

ref_models: list[PPO] = [
    cast(
        PPO,
        restore_model(props, ref_obs_props, device, is_initialized_in_both_territories),
    )
    for props in ref_model_props
]


ref_action_probs_list: list[NDArray] = []
ref_trap_masks: list[NDArray] = []

map_locs_list: list[MapLocations] = [
    sample_agent_locs(map_props.fixed_map_locs, is_sampled_from_all_territories)
    for _ in range(num_samples)
]

for i, model in enumerate(ref_models):
    print(f"Sampling reference model {i+1}/{len(ref_models)}...")
    ref_action_probs, ref_trap_mask = get_agent_action_distributions(
        model, map_locs_list
    )
    ref_action_probs_list.append(ref_action_probs)
    ref_trap_masks.append(ref_trap_mask)


device = torch.device("cuda", gpu)
neighbor_masks: tuple[ValueTable, ...] = create_neighbor_masks(num_vars, exclusions)

multi_start_result: MultiStartSearchLogger = search_trained_multistart_multiprocess(
    num_start,
    init_nodes,
    len(learned_models_props_list),
    experiment,
    neighbor_masks,
    ind_combs,
    ref_action_probs_list,
    ref_trap_masks,
    learned_obs_props_list,
    learned_models_props_list,
    map_locs_list,
    device,
    is_initialized_in_both_territories,
    exp_data_save_dir,
    exp_filename_common,
)

data_saver(multi_start_result, multi_start_savename)

with open(exp_var_data_savename, mode="wb") as f:
    pickle.dump(
        (init_nodes, vars),
        f,
    )

print("All searches completed.")
