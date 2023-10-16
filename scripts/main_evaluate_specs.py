import itertools

import pickle
from typing import cast

import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import torch

from tl_search.common.io import data_saver
from tl_search.common.seed import torch_fix_seed
from tl_search.common.utils import find_min_kl_div_tl_spec
from tl_search.evaluation.extractor import get_agent_action_distributions
from tl_search.evaluation.ranking import sort_spec
from tl_search.evaluation.wrapper import evaluate_ppo_models_multiprocess
from tl_search.search.neighbor import create_all_nodes, nodes2specs

from tl_search.tl.constants import (
    obs_props_all,
    atom_prop_dict_all,
    known_policy_map_object_ids,
)
from tl_search.common.typing import (
    EnemyPolicyMode,
    EnvProps,
    Exclusion,
    ExperimentLogger,
    MapLocations,
    MapProps,
    ModelProps,
    ObsProp,
    PositiveRewardLogger,
    RLAlgorithm,
    SortedLogger,
    SpecNode,
)
from tl_search.map.utils import map2props, sample_agent_locs
from tl_search.tl.environment import restore_model
from tl_search.tl.tl_parser import get_used_tl_props

run: int = 7
num_replicates: int = 3
total_timesteps: int = 50_0000
gpu: int = 2
num_process: int = 16
num_samples: int = 8000
seed: float = 3406


exclusions: list[Exclusion] = ["group"]

vars: tuple[str, ...] = ("psi_ba_ra", "psi_ba_bf", "psi_ba_rf", "psi_ba_obs")
rl_algorithm: RLAlgorithm = "ppo"
enemy_policy: EnemyPolicyMode = "random"
map_filename: str = "maps/board_0002.txt"

is_initialized_in_both_territories: bool = True
is_sampled_from_all_territories: bool = False

data_dir: str = "out/data/"
plots_dir: str = "out/plots/"
pickle_suffix: str = "_data.pickle"

ref_spec: str = "F(!psi_ba_bf&psi_ba_rf) & G(!psi_ba_ra&!psi_ba_obs)"
ref_run: int = 17
ref_run_suffix: str = "{}_{}_{}".format(enemy_policy, rl_algorithm, ref_run)
ref_suffix: str = "policy_{}".format(ref_run_suffix)
ref_model_data_savename: str = data_dir + "single/" + ref_suffix + pickle_suffix
ref_model_savename: str = "models/single/{}".format(ref_suffix)

run_suffix: str = "{}_{}_{}".format(enemy_policy, rl_algorithm, run)
policy_suffix: str = "policy_{}".format(run_suffix)
model_data_savename: str = "models/search/" + policy_suffix + pickle_suffix
model_savename: str = "models/search/" + policy_suffix

rep_histogran_save_dir: str = plots_dir + "rank/"
rep_histogram_name: str = "rep_historgram_{}".format(run_suffix)
rep_historgram_savename: str = rep_histogran_save_dir + rep_histogram_name + ".png"
rep_historgram_data_savename: str = (
    rep_histogran_save_dir + rep_histogram_name + pickle_suffix
)

exp_data_save_dir: str = data_dir + "search/"

learned_models_props_savename: str = exp_data_save_dir + policy_suffix + pickle_suffix

exp_savename: str = exp_data_save_dir + "{}_exp.json".format(policy_suffix)
pos_savename: str = exp_data_save_dir + "{}_pos.json".format(policy_suffix)
sorted_savename: str = exp_data_save_dir + "{}_sorted.json".format(policy_suffix)
exp_var_data_savename: str = exp_data_save_dir + policy_suffix + pickle_suffix

all_models_props_savename: str = (
    exp_data_save_dir + policy_suffix + "_all" + pickle_suffix
)

ref_model_inds: list[int] = list(range(num_replicates))
learned_model_inds: list[int] = list(range(num_replicates))

ref_labels: list[str] = ["ref_{}".format(i) for i in ref_model_inds]
learned_labels: list[str] = ["learned_{}".format(i) for i in learned_model_inds]

torch_fix_seed(seed)

map_props = MapProps(*map2props(map_filename, known_policy_map_object_ids))

neighbor_nodes: list[SpecNode] = create_all_nodes(vars, exclusions)  # [0:2]
neighbor_specs: list[str] = list(set(nodes2specs(neighbor_nodes)))
neighbor_specs.sort()

# with open(model_savename, mode="rb") as f:
#     learned_models_props_list: list[list[ModelProps]] = pickle.load(f)
learned_models_props_list: list[list[ModelProps]] = []

# with open(ref_model_data_savename, mode="rb") as f:
#     ref_model_props: list[ModelProps] = pickle.load(f)

obs_props_list: list[list[ObsProp]] = []

for spec_ind, spec in enumerate(neighbor_specs):
    obs_props, atom_prop_dict = get_used_tl_props(
        spec, atom_prop_dict_all, obs_props_all
    )
    env_props = EnvProps(
        spec,
        obs_props,
        atom_prop_dict,
        enemy_policy,
        map_props,
    )

    model_props_rep: list[ModelProps] = []

    for rep_ind in range(num_replicates):
        model_name: str = "{}_{}_{}".format(model_savename, spec_ind, rep_ind)
        model_props_rep.append(
            ModelProps(
                model_name,
                rl_algorithm,
                env_props.spec,
                env_props.atom_prop_dict,
                env_props.enemy_policy,
                env_props.map_props,
            )
        )

    learned_models_props_list.append(model_props_rep)
    obs_props_list.append(obs_props)


ref_obs_props, ref_atom_props = get_used_tl_props(
    ref_spec, atom_prop_dict_all, obs_props_all
)
ref_env_props = EnvProps(
    ref_spec,
    ref_obs_props,
    ref_atom_props,
    enemy_policy,
    map_props,
)

ref_model_props: list[ModelProps] = []

for rep_ind in range(num_replicates):
    ref_model_name: str = "{}_{}_{}".format(ref_model_savename, 0, rep_ind)
    ref_model_props.append(
        ModelProps(
            ref_model_name,
            rl_algorithm,
            ref_env_props.spec,
            ref_env_props.atom_prop_dict,
            ref_env_props.enemy_policy,
            ref_env_props.map_props,
        )
    )

device = torch.device("cuda", gpu)

map_props = MapProps(*map2props(map_filename, known_policy_map_object_ids))

ref_obs_props, _ = get_used_tl_props(ref_spec, atom_prop_dict_all, obs_props_all)
ref_models: list[PPO] = [
    cast(
        PPO,
        restore_model(props, ref_obs_props, device, is_initialized_in_both_territories),
    )
    for props in ref_model_props
]

with open(all_models_props_savename, mode="wb") as f:
    pickle.dump((ref_model_props, learned_models_props_list), f)

ref_action_probs_list: list[NDArray] = []
ref_trap_masks: list[NDArray] = []

map_locs_list: list[MapLocations] = [
    sample_agent_locs(map_props.fixed_map_locs, is_sampled_from_all_territories)
    for _ in range(num_samples)
]

for model in ref_models:
    ref_action_probs, ref_trap_mask = get_agent_action_distributions(
        model, map_locs_list
    )
    ref_action_probs_list.append(ref_action_probs)
    ref_trap_masks.append(ref_trap_mask)

ind_combs: list[tuple[int, int]] = list(
    itertools.product(ref_model_inds, learned_model_inds)
)

device = torch.device("cuda", gpu)

(
    kl_div_report_model_all,
    kl_div_reports_model_all,
    kl_div_mean_rank,
    reward_mean_all,
    reward_std_all,
) = evaluate_ppo_models_multiprocess(
    num_process,
    ind_combs,
    ref_action_probs_list,
    ref_trap_masks,
    learned_models_props_list,
    obs_props_list,
    map_locs_list,
    device,
    is_initialized_in_both_territories,
)

kl_div_means: list[float] = [report.kl_div_mean for report in kl_div_report_model_all]
(
    min_kl_div_spec,
    min_kl_div_mean,
    mean_kl_div_mean,
    max_kl_div_mean,
) = find_min_kl_div_tl_spec(neighbor_specs, kl_div_means)

max_reward_mean: float = float(np.max(reward_mean_all))
mean_reward_mean: float = float(np.mean(reward_mean_all))
min_reward_mean: float = float(np.min(reward_mean_all))
max_reward_spec: str = neighbor_specs[np.argmax(reward_mean_all)]

exp_log = ExperimentLogger(
    "known",
    run,
    "",
    min_kl_div_spec,
    max_reward_spec,
    min_kl_div_mean,
    mean_kl_div_mean,
    max_kl_div_mean,
    min_reward_mean,
    mean_reward_mean,
    max_reward_mean,
    neighbor_specs,
    [],
    [],
    [],
    [],
    [],
    [],
)

reward_means_np = np.array(reward_mean_all)
pos_reward_inds: NDArray = reward_means_np > 0
reward_means_pos: list[float] = reward_means_np[pos_reward_inds].tolist()
tl_specs_pos: list[str] = np.array(neighbor_specs)[pos_reward_inds].tolist()
kl_divs_pos: list[float] = np.array(kl_div_means)[pos_reward_inds].tolist()

pos_reward_log = PositiveRewardLogger(
    tl_specs_pos, reward_means_pos, kl_divs_pos, [], []
)

sorted_specs, sorted_min_kl_divs = sort_spec(
    pos_reward_log.tl_specs_pos, pos_reward_log.kl_divs_pos
)

sorted_log = SortedLogger(run, sorted_specs, sorted_min_kl_divs)

data_saver(exp_log, exp_savename)
data_saver(pos_reward_log, pos_savename)
data_saver(sorted_log, sorted_savename)

# plot
labels: list[str] = [
    "{} & {}".format(ref_labels[comb[0]], learned_labels[comb[1]]) for comb in ind_combs
]

num_combs: int = len(ind_combs)

rank_data: list[NDArray] = [kl_div_mean_rank[:, i].flatten() for i in range(num_combs)]

bin = np.linspace(1, 9, 9)

plt.hist(rank_data, bin, label=labels)
plt.xlabel("Rank")
plt.ylabel("Occurence")
plt.savefig(rep_historgram_savename)

with open(rep_historgram_data_savename, mode="wb") as f:
    pickle.dump(rank_data, f)

with open(exp_var_data_savename, mode="wb") as f:
    pickle.dump(
        (
            kl_div_report_model_all,
            kl_div_reports_model_all,
            kl_div_mean_rank,
            reward_mean_all,
            reward_std_all,
        ),
        f,
    )
