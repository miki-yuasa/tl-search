import numpy as np

from tl_search.common.typing import ActionProbsSpec, ExperimentLogger, PolicyMode
from tl_search.common.utils import find_max_reward_spec, find_min_kl_div_tl_spec


def log_experiment_data(
    policy_mode: str,
    experiment: int,
    description: str,
    tl_specs: list[str],
    kl_divs: list[list[float]],
    tl_action_probs: list[ActionProbsSpec],
    ori_action_probs: list[ActionProbsSpec],
    model_names: list[list[str]],
    mean_rewards: list[list[float]],
    std_rewards: list[list[float]],
) -> ExperimentLogger:
    (
        min_kl_div_spec,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
    ) = find_min_kl_div_tl_spec(
        tl_specs, np.mean(np.array(kl_divs), axis=1).astype(object)
    )

    (
        max_reward_spec,
        min_reward_mean,
        mean_reward_mean,
        max_reward_mean,
    ) = find_max_reward_spec(tl_specs, mean_rewards)

    saved_data = ExperimentLogger(
        policy_mode,
        experiment,
        description,
        min_kl_div_spec,
        max_reward_spec,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
        min_reward_mean,
        mean_reward_mean,
        max_reward_mean,
        tl_specs,
        kl_divs,
        tl_action_probs,
        ori_action_probs,
        model_names,
        mean_rewards,
        std_rewards,
    )

    return saved_data
