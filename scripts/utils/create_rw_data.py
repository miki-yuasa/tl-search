import numpy as np
from numpy.typing import NDArray

num_actions: int = 5
num_data: int = 5000
num_replicates: int = 3

enemy_policy_mode: str = "patrol"

rw_actions_path: str = (
    f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_rw"
)
target_actions_path: str = (
    f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_full"
)


npz = np.load(target_actions_path + ".npz")
target_action_probs_list = npz["arr_0"]
target_trap_masks = npz["arr_1"]

rw_prob_list: NDArray = (
    np.ones((num_replicates, num_data, num_actions)) * 1 / num_actions
)

np.savez(rw_actions_path, rw_prob_list, target_trap_masks)
