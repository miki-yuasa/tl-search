import itertools
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from tl_search.tl.environment import create_env
from tl_search.tl.constants import (
    known_policy,
    unknown_policy,
    obs_props_all,
)
from tl_search.evaluation.evaluation import evaluate_spec
from tl_search.evaluation.filter import filter_rewards
from tl_search.common.utils import (
    kl_div,
)
from tl_search.common.io import (
    experiment_data_loader_all,
    list_model_names,
    data_saver,
)
from tl_search.common.typing import ActionProbsSpec, PolicyMode, PositiveRewardLogger

policy_mode: PolicyMode = "unknown"
model_policy_mode: PolicyMode = "known"
experiment: int = 4
model_experiment: int = 10
total_processes: int = 16
num_replicates: int = 5
is_batched: bool = True
saved_data_name: str = "data/batch/saved_data_{}_{}.json".format(
    policy_mode, experiment
)

kl_div_steps: int = 1
num_sampling: int = 2000
reward_threshold: float = 0.0

fight_range: float = 2.0
defense_range: float = 1.0
atom_prop_dict_all: dict[str, str] = {
    "psi_ba_ra": "d_ba_ra < {}".format(fight_range),
    # "psi_ba_rf": "d_ba_rf < 0.5",
    "psi_ba_rt": "d_ba_rt < 0.5",
    "psi_ra_bf": "d_ra_bf < {}".format(defense_range),
    "psi_ra_bt": "d_ra_bt < 0.5",
    "psi_ba_ob": "d_ba_ob < 0.5",
    "psi_ba_wa": "d_ba_wa < 0.5",
    # "psi_ba_obwa": "(d_ba_wa < 0.5)|(d_ba_ob < 0.5)",
}

data = experiment_data_loader_all(
    model_policy_mode, model_experiment, is_batched, total_processes
)

filenames: list[str] = list_model_names(
    model_policy_mode,
    model_experiment,
    total_processes,
    len(data.tl_specs),
    num_replicates,
)

filenames_rep: list[list[str]] = (
    np.array(filenames).reshape(-1, num_replicates).tolist()
)

policy = known_policy if policy_mode == "known" else unknown_policy
map_locs_list, ori_action_probs_np = policy.sample(num_sampling)

pos_reward_data: PositiveRewardLogger = filter_rewards(
    reward_threshold,
    data.tl_specs,
    np.mean(np.array(data.kl_divs), axis=1).tolist(),
    data.mean_rewards,
    np.array(filenames_rep)[:, 0].tolist(),
)

tl_specs: list[str] = pos_reward_data.tl_specs_pos
filenames_rep_pos: list[list[str]] = [
    filenames_rep[ind] for ind in pos_reward_data.mean_rewards_pos_inds
]

kl_divs_all: list[list[float]] = []
tl_action_probs_all: list[ActionProbsSpec] = []
ori_action_probs_all: list[ActionProbsSpec] = []

for spec, files in zip(tl_specs, filenames_rep_pos):
    env = create_env(spec, policy, obs_props_all, atom_prop_dict_all)
    env_monitored = Monitor(env)

    kl_divs_spec: list[float] = []
    tl_action_probs_spec: ActionProbsSpec = []
    ori_action_probs_spec: ActionProbsSpec = []

    for file in files:
        model = PPO.load(file, env_monitored)
        tl_action_probs, ori_action_probs, _, _ = evaluate_spec(
            model, map_locs_list, ori_action_probs_np, True, False, True
        )
        kl_divs_spec.append(
            np.mean(
                kl_div(np.array(ori_action_probs), np.array(tl_action_probs))
            ).astype(object)
        )
        tl_action_probs_spec.append(tl_action_probs)
        ori_action_probs_spec.append(ori_action_probs)

    kl_divs_all.append(kl_divs_spec)
    tl_action_probs_all.append(tl_action_probs_spec)
    ori_action_probs_all.append(ori_action_probs_spec)


experiment_data_saver(
    policy_mode,
    experiment,
    "Trap-state KL divs and negative rewards are filtered out",
    pos_reward_data.tl_specs_pos,
    kl_divs_all,
    [],
    [],
    filenames_rep,
    [data.mean_rewards[ind] for ind in pos_reward_data.mean_rewards_pos_inds],
    [data.std_rewards[ind] for ind in pos_reward_data.mean_rewards_pos_inds],
    saved_data_name,
)
