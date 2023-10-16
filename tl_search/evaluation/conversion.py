import numpy as np
from numpy.typing import NDArray

from tl_search.common.io import data_saver
from tl_search.common.typing import (
    KLDivLogger,
    AccuracyLogger,
    ProbabilityLogger,
    PolicyMode,
)


def kl_div2accuracy(
    data: KLDivLogger,
    prefix: str,
    policy_mode: PolicyMode,
    experiment: int,
    num_sampling: int,
    save_file: bool = True,
) -> tuple[list[list[NDArray]], AccuracyLogger]:
    accuracy_metrics_action: list[list[NDArray]] = [
        [
            np.array(
                [
                    1
                    if tl_action.index(max(tl_action))
                    == ori_action.index(max(ori_action))
                    else 0
                    for tl_action, ori_action in zip(tl_action_rep, ori_action_rep)
                ]
            )
            for tl_action_rep, ori_action_rep in zip(tl_action_spec, ori_action_spec)
        ]
        for tl_action_spec, ori_action_spec in zip(
            data.tl_action_probs, data.ori_action_probs
        )
    ]

    accuracy_means: NDArray = (
        np.array(
            [
                np.mean(np.array([np.sum(accuracy) for accuracy in accuracy_rep]))
                for accuracy_rep in accuracy_metrics_action
            ]
        )
        / num_sampling
    )

    max_accuracy_mean: float = np.max(accuracy_means)
    mean_accuracy_mean: float = np.mean(accuracy_means)
    min_accuracy_mean: float = np.min(accuracy_means)

    max_acc_spec: str = data.tl_specs[np.argmax(accuracy_means)]

    saved_data = AccuracyLogger(
        max_acc_spec,
        max_accuracy_mean,
        mean_accuracy_mean,
        min_accuracy_mean,
        data.tl_specs,
        accuracy_means.astype(object).tolist(),
    )

    savename: str = "{}_{}_{}_accuracy.json".format(prefix, policy_mode, experiment)

    if save_file:
        data_saver(saved_data, savename)
    else:
        pass

    return (accuracy_metrics_action, saved_data)


def kl_div2probability(
    data: KLDivLogger,
    prefix: str,
    policy_mode: PolicyMode,
    experiment: int,
    save_file: bool = True,
) -> tuple[list[list[NDArray]], ProbabilityLogger]:
    prob_metrics_action: list[list[NDArray]] = [
        [
            np.array(
                [
                    tl_action[int(np.argmax(ori_action))]
                    for tl_action, ori_action in zip(tl_action_rep, ori_action_rep)
                ]
            )
            for tl_action_rep, ori_action_rep in zip(tl_action_spec, ori_action_spec)
        ]
        for tl_action_spec, ori_action_spec in zip(
            data.tl_action_probs, data.ori_action_probs
        )
    ]

    prob_means: NDArray = np.array(
        [
            np.mean(np.array([np.mean(prob) for prob in prob_rep]))
            for prob_rep in prob_metrics_action
        ]
    )

    max_prob_mean: float = np.max(prob_means)
    mean_prob_mean: float = np.mean(prob_means)
    min_prob_mean: float = np.min(prob_means)

    max_prob_spec: str = data.tl_specs[np.argmax(prob_means)]

    saved_data = ProbabilityLogger(
        max_prob_spec,
        max_prob_mean,
        mean_prob_mean,
        min_prob_mean,
        data.tl_specs,
        prob_means.astype(object).tolist(),
    )

    savename: str = "{}_{}_{}_probability.json".format(prefix, policy_mode, experiment)

    if save_file:
        data_saver(saved_data, savename)
    else:
        pass

    return (prob_metrics_action, saved_data)
