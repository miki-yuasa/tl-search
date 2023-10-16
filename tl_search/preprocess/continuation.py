from numpy import result_type
from tl_search.common.utils import experiment_data_loader, kl_div_data_loader
from tl_search.common.typing import ActionProbsSpec, ExperimentLogger, KLDivLogger


def continue_from_data(
    tl_specs_all: list[str], data_filename: str
) -> tuple[
    list[str], list[str], list[float], list[ActionProbsSpec], list[ActionProbsSpec]
]:
    results: KLDivLogger = kl_div_data_loader(data_filename)

    tl_specs_done: list[str] = results.tl_specs
    kl_divs: list[float] = results.kl_divs
    tl_action_probs: list[ActionProbsSpec] = results.tl_action_probs
    ori_action_probs: list[ActionProbsSpec] = results.ori_action_probs

    tl_specs_all_set: set[str] = set(tl_specs_all)
    tl_specs_done_set: set[str] = set(tl_specs_done)

    tl_specs_undone: list[str] = list(tl_specs_all_set - tl_specs_done_set)

    return (tl_specs_undone, tl_specs_done, kl_divs, tl_action_probs, ori_action_probs)


def continue_from_experiment_data(
    tl_specs_all: list[str], data_filename: str
) -> tuple[
    list[str],
    list[str],
    list[list[float]],
    list[ActionProbsSpec],
    list[ActionProbsSpec],
    list[list[float]],
    list[list[float]],
]:
    results: ExperimentLogger = experiment_data_loader(data_filename)

    tl_specs_done = results.tl_specs
    kl_divs = results.kl_divs
    tl_action_probs = results.tl_action_probs
    ori_action_probs = results.ori_action_probs
    mean_rewards = results.mean_rewards
    std_rewards = results.std_rewards

    tl_specs_all_set: set[str] = set(tl_specs_all)
    tl_specs_done_set: set[str] = set(tl_specs_done)

    tl_specs_undone: list[str] = list(tl_specs_all_set - tl_specs_done_set)

    return (
        tl_specs_undone,
        tl_specs_done,
        kl_divs,
        tl_action_probs,
        ori_action_probs,
        mean_rewards,
        std_rewards,
    )
