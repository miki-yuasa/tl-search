from tl_search.evaluation.conversion import (
    kl_div2accuracy,
    kl_div2probability,
)
from tl_search.common.io import kl_div_data_loader_all, kl_div_data_saver_all
from tl_search.common.typing import PolicyMode, Machines

prefix: str = "saved_data"
policy_mode: PolicyMode = "known"
experiment: int = 1
machine_names: list[Machines] = ["dl", "ti"]
processes = [None]

num_sampling: int = 1000

data = kl_div_data_loader_all(prefix, policy_mode, experiment, machine_names, processes)

kl_div_data_saver_all(data, prefix, policy_mode, experiment)

accuracy_metrics_action, acc_data = kl_div2accuracy(
    data, prefix, policy_mode, experiment, num_sampling, False
)

prob_metrics_action, prob_data = kl_div2probability(
    data, prefix, policy_mode, experiment, False
)
pass
