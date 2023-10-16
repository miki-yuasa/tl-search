import itertools
import json

import numpy as np
from numpy.typing import NDArray
from scipy.special import entr, rel_entr
import matplotlib.pyplot as plt

from tl_search.common.typing import KLDivReportDict
from tl_search.evaluation.evaluation import collect_kl_div_stats
from tl_search.envs.select import (
    select_max_entropy_spec_replicate,
)

enemy_policy_mode: str = "patrol"

kl_div_save_path: str = (
    "out/data/search/dataset/patrol_kl_div_report_exp1_single_searched_to_searched.json"
)

rw_actions_path: str = "out/data/search/dataset/action_probs_patrol_enemy_ppo_exp1_nested_full_searched_simple"  # f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_rw"  # "out/data/search/dataset/action_probs_patrol_enemy_ppo_exp1_extendedfull_searched_correct"  # f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_rw"
target_actions_path: str = "out/data/search/dataset/action_probs_patrol_enemy_ppo_exp1_nested_full_searched_simple"  # "out/data/search/dataset/action_probs_patrol_enemy_ppo_exp1_extendedfull_correct"  # f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_full"

idx_combs: list[tuple[int, int]] = list(itertools.product(range(3), range(3)))

target_npz = np.load(target_actions_path + ".npz")
target_action_probs_list = target_npz["arr_0"]
target_trap_masks = target_npz["arr_1"]

rw_npz = np.load(rw_actions_path + ".npz")
rep_action_probs_list = rw_npz["arr_0"]
rep_trap_masks = rw_npz["arr_1"]

(
    target_idx,
    target_entropies,
    target_num_non_trap_states,
) = select_max_entropy_spec_replicate(target_action_probs_list, target_trap_masks)
(
    spec_idx,
    spec_entropies,
    num_non_trap_states,
) = select_max_entropy_spec_replicate(rep_action_probs_list, rep_trap_masks)

model_kl_divs: list[NDArray] = []

max_kl_div: float = np.sum(entr([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]))

# for comb in idx_combs:
# target_idx, spec_idx = comb

trap_mask: NDArray = target_trap_masks[2] * rep_trap_masks[0]
target_action_probs_filtered: NDArray = target_action_probs_list[target_idx][
    trap_mask == 1
]
spec_action_probs_filtered: NDArray = rep_action_probs_list[spec_idx][trap_mask == 1]

target_entropy: NDArray = np.sum(entr(target_action_probs_filtered), axis=1)
normalized_entropy: NDArray = (1 - target_entropy) / max_kl_div
weight: NDArray = normalized_entropy / np.sum(normalized_entropy)

kl_divs: NDArray = (
    np.sum(
        rel_entr(target_action_probs_filtered, spec_action_probs_filtered),
        axis=1,
    )
    * weight
)

model_kl_divs.append(kl_divs.flatten())

model_kl_divs_concat: NDArray = np.concatenate(model_kl_divs)

kl_div_report: KLDivReportDict = collect_kl_div_stats(
    model_kl_divs_concat, in_dict=True
)

with open(kl_div_save_path, "w") as f:
    json.dump(kl_div_report, f, indent=4)

plt.hist(model_kl_divs_concat, bins=100, range=(0, 7e-5))
plt.savefig(kl_div_save_path.replace(".json", ".png"))
