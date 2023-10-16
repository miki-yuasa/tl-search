from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from numpy.typing import NDArray
import torch

from tl_search.common.typing import KLDivReport, MapLocations, ModelProps, ObsProp
from tl_search.evaluation.evaluation import evaluate_ppo_models


def evaluate_ppo_models_multiprocess(
    num_process: int,
    ind_combs: list[tuple[int, int]],
    ref_action_probs_list: list[NDArray],
    ref_trap_masks: list[NDArray],
    learned_models_props_list: list[list[ModelProps]],
    obs_props_list: list[list[ObsProp]],
    map_locs_list: list[MapLocations],
    device: torch.device,
    is_initialized_in_both_territories: bool,
) -> tuple[
    list[KLDivReport],
    list[list[KLDivReport]],
    NDArray,
    list[float],
    list[float],
]:
    print("multiprocess")
    inputs: list[
        tuple[
            list[tuple[int, int]],
            list[NDArray],
            list[NDArray],
            list[ModelProps],
            list[ObsProp],
            list[MapLocations],
            torch.device,
            bool,
        ]
    ] = [
        (
            ind_combs,
            ref_action_probs_list,
            ref_trap_masks,
            model_props,
            obs_props,
            map_locs_list,
            device,
            is_initialized_in_both_territories,
        )
        for model_props, obs_props in zip(learned_models_props_list, obs_props_list)
    ]

    # with Pool(num_process) as p:
    # results: list[
    # tuple[KLDivReport, list[KLDivReport], NDArray, float, float]
    # ] = p.map(evaluate_ppo_models_wrapper, inputs)
    results: list[tuple[KLDivReport, list[KLDivReport], NDArray, float, float]] = [
        evaluate_ppo_models_wrapper(input) for input in inputs
    ]

    kl_div_report_model_all: list[KLDivReport] = []
    kl_div_reports_model_all: list[list[KLDivReport]] = []
    kl_div_mean_rank_list: list[NDArray] = []
    reward_mean_all: list[float] = []
    reward_std_all: list[float] = []

    for result in results:
        (
            kl_div_report_model,
            kl_div_reports,
            kl_div_mean_rank,
            reward_mean,
            reward_std,
        ) = result

        kl_div_report_model_all.append(kl_div_report_model)
        kl_div_reports_model_all.append(kl_div_reports)
        kl_div_mean_rank_list.append(kl_div_mean_rank)
        reward_mean_all.append(reward_mean)
        reward_std_all.append(reward_std)

    kl_div_mean_rank = np.stack(kl_div_mean_rank_list)

    return (
        kl_div_report_model_all,
        kl_div_reports_model_all,
        kl_div_mean_rank,
        reward_mean_all,
        reward_std_all,
    )


def evaluate_ppo_models_wrapper(
    input: tuple[
        list[tuple[int, int]],
        list[NDArray],
        list[NDArray],
        list[ModelProps],
        list[ObsProp],
        list[MapLocations],
        torch.device,
        bool,
    ]
) -> tuple[KLDivReport, list[KLDivReport], NDArray, float, float]:
    print("wrapper")
    return evaluate_ppo_models(*input)
