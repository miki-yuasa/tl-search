import numpy as np

from tl_search.evaluation.filter import filter_negative_rewards
from tl_search.common.typing import PolicyMode
from tl_search.common.utils import find_min_kl_div_tl_spec
from tl_search.common.io import (
    experiment_data_loader_all,
    list_model_names,
    data_saver,
)


policy_mode: PolicyMode = "known"
experiment: int = 6
total_processes: int = 16
num_replicates: int = 3
is_batched: bool = True

data = experiment_data_loader_all(policy_mode, experiment, is_batched, total_processes)

model_names_tmp: list[str] = list_model_names(
    policy_mode, experiment, total_processes, len(data.tl_specs), num_replicates
)

model_names = [model_name for model_name in model_names_tmp if model_name[-1] == "0"]

mean_rewards: list[float] = [np.average(rewards) for rewards in data.mean_rewards]

pos_reward_data = filter_negative_rewards(
    data.tl_specs, data.kl_divs, mean_rewards, model_names
)

(
    min_tl_spec,
    min_kl_div_mean,
    mean_kl_div_mean,
    max_kl_div_mean,
) = find_min_kl_div_tl_spec(pos_reward_data.tl_specs_pos, pos_reward_data.kl_divs_pos)

savename: str = "data/batch/positive_reward_{}_{}.json".format(policy_mode, experiment)

data_saver(pos_reward_data, savename)
