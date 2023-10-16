from tl_search.evaluation.ranking import rank_by
from tl_search.common.typing import PolicyMode
from tl_search.common.io import experiment_data_loader_all

policy_mode: PolicyMode = "known"
experiment: int = 5
total_process: int = 16
batched: bool = True
num_rank: int = 20

data = experiment_data_loader_all(policy_mode, experiment, batched, total_process)

ranked_data_kl_div = rank_by(data, "kl_div", num_rank)
ranked_data_reward = rank_by(data, "reward", num_rank)

print(ranked_data_kl_div)
print(ranked_data_reward)
pass
