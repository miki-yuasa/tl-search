import os, json, heapq, itertools
from statistics import mean
from typing import Union

import numpy as np

from tl_search.common.json import json_zip, json_unzip
from tl_search.common.typing import (
    AccuracyLogger,
    KLDivLogger,
    ExperimentLogger,
    ActionProbsSpec,
    MultiStartSearchLogger,
    SortedLogger,
    MultiStepKLDivLogger,
    PolicyMode,
    Machines,
    ProbabilityLogger,
    PositiveRewardLogger,
    HyperparameterTuningLogger,
)
from tl_search.common.utils import find_max_reward_spec, find_min_kl_div_tl_spec


def save_json(
    config: str,
    path: str,
) -> None:
    filepath = get_file_path(path)
    with open(filepath, "w") as f:
        f.write(config)


def data_saver(
    saved_data: KLDivLogger
    | AccuracyLogger
    | ProbabilityLogger
    | PositiveRewardLogger
    | ExperimentLogger
    | MultiStepKLDivLogger
    | HyperparameterTuningLogger
    | SortedLogger
    | MultiStartSearchLogger,
    savename: str,
    zip_file: bool = False,
) -> None:
    filepath = get_file_path(savename)
    with open(filepath, "w") as f:
        if zip_file:
            json.dump(json_zip(saved_data), f)
        else:
            json.dump(saved_data._asdict(), f, indent=4)


def kl_div_data_saver(
    tl_specs: list[str],
    kl_divs: list[float],
    tl_action_probs: list[ActionProbsSpec],
    ori_action_probs: list[ActionProbsSpec],
    savename: str,
) -> None:
    print("Saving the results in {}.".format(savename))

    (
        min_tl_spec,
        min_kl_ind,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
    ) = find_min_kl_div_tl_spec(tl_specs, kl_divs)

    saved_data: KLDivLogger = KLDivLogger(
        min_tl_spec,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
        tl_specs,
        kl_divs,
        tl_action_probs,
        ori_action_probs,
    )

    data_saver(saved_data, savename)


def experiment_data_saver(
    policy_mode: PolicyMode,
    experiment: int,
    description: str,
    tl_specs: list[str],
    kl_divs: list[list[float]],
    tl_action_probs: list[ActionProbsSpec],
    ori_action_probs: list[ActionProbsSpec],
    model_names: list[list[str]],
    mean_rewards: list[list[float]],
    std_rewards: list[list[float]],
    savename: str,
):
    (
        min_kl_div_spec,
        min_kl_ind,
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

    data_saver(saved_data, "data/" + savename)


def load_zipped_json(filepath: str) -> dict:
    j: dict = {}
    with open(filepath, "r") as f:
        j = json.load(f)

    d: dict = json_unzip(j)

    return d


def load_json_as_dict(savename: str) -> dict:
    filepath: str = get_file_path(savename)
    with open(filepath, "r") as f:
        d: dict = json.load(f)

    return d


def get_file_path(savename: str) -> str:
    filepath: str = os.path.join(os.path.abspath(os.curdir), *os.path.split(savename))

    return filepath


def list_model_names(
    policy_mode: PolicyMode,
    experiment: int,
    total_processes: int,
    total_specs: int,
    num_replicates: int,
) -> list[str]:
    model_names: list[str] = []
    for process in range(1, total_processes + 1):
        for spec_ind in range(int(total_specs / total_processes)):
            for rep_ind in range(num_replicates):
                model_names.append(
                    "models/model_{}_{}_{}_{}_{}".format(
                        policy_mode, experiment, process, spec_ind, rep_ind
                    )
                )
    return model_names


def load_rewards(
    policy_mode: PolicyMode, experiment: int
) -> tuple[list[float], list[float]]:
    savename: str = "rewards_{}_{}.json".format(policy_mode, experiment)
    d: dict[str, list[float]] = load_json_as_dict(savename)

    return (d["mean_rewards"], d["std_rewards"])


def kl_div_data_saver_all(
    data: KLDivLogger, prefix: str, policy_mode: PolicyMode, experiment: int
):
    savename: str = "{}_{}_{}.json".format(prefix, policy_mode, experiment)

    data_saver(data, savename, True)


def kl_div_data_loader(filename: str) -> KLDivLogger:
    d: dict = load_json_as_dict(filename)

    formatted_data: KLDivLogger = KLDivLogger(**d)
    return formatted_data


def experiment_data_loader(filename: str) -> ExperimentLogger:
    d: dict = load_json_as_dict(filename)

    formatted_data = ExperimentLogger(**d)
    return formatted_data


def experiment_data_loader_all(
    policy_mode: PolicyMode, experiment: int, is_batched: bool, total_process: int
) -> ExperimentLogger:
    dir: str = "data/" + ("batch" if is_batched else "single")
    filenames: list[str] = [
        dir + "/saved_data_{}_{}_{}.json".format(policy_mode, experiment, process)
        for process in range(1, total_process + 1)
    ]

    data_pair: list[ExperimentLogger] = [
        experiment_data_loader(filename) for filename in filenames
    ]

    tl_specs: list[str] = []
    kl_divs: list[list[float]] = []
    tl_action_probs: list[ActionProbsSpec] = []
    ori_action_probs: list[ActionProbsSpec] = []
    mean_rewards: list[list[float]] = []
    std_rewards: list[list[float]] = []
    model_names: list[list[str]] = []

    for data in data_pair:
        tl_specs += data.tl_specs
        kl_divs += data.kl_divs
        tl_action_probs += data.tl_action_probs
        ori_action_probs += data.ori_action_probs
        mean_rewards += data.mean_rewards
        std_rewards += data.std_rewards
        model_names += data.model_names

    (
        min_kl_div_spec,
        min_kl_ind,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
    ) = find_min_kl_div_tl_spec(tl_specs, np.mean(np.array(kl_divs), axis=1).tolist())

    (
        max_reward_spec,
        min_reward_mean,
        mean_reward_mean,
        max_reward_mean,
    ) = find_max_reward_spec(tl_specs, mean_rewards)

    all_data = ExperimentLogger(
        policy_mode,
        experiment,
        data_pair[0].description,
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

    return all_data


def kl_div_data_loader_all(
    prefix: str,
    policy_mode: PolicyMode,
    experiment: int,
    machine_names: list[Machines],
    processes: list[Union[int, None]],
) -> KLDivLogger:
    filenames: list[str] = list(
        itertools.chain.from_iterable(
            [
                [
                    "{}_{}_{}_{}_{}.json".format(
                        prefix, policy_mode, experiment, machine_name, process
                    )
                    if process
                    else "{}_{}_{}_{}.json".format(
                        prefix, policy_mode, experiment, machine_name
                    )
                    for process in processes
                ]
                for machine_name in machine_names
            ]
        )
    )

    data_pair: list[KLDivLogger] = [
        kl_div_data_loader(filename) for filename in filenames
    ]

    min_tl_spec: str
    min_kl_div_mean: float
    mean_kl_div_mean: float
    max_kl_div_mean: float
    tl_specs: list[str] = []
    kl_divs: list[float] = []
    tl_action_probs: list[ActionProbsSpec] = []
    ori_action_probs: list[ActionProbsSpec] = []

    min_pairs = [(data.min_kl_div, data.min_tl_spec) for data in data_pair]
    heapq.heapify(min_pairs)
    min_kl_div_mean, min_tl_spec = heapq.heappop(min_pairs)

    mean_kl_div_mean = mean([data.mean_kl_div_mean for data in data_pair])
    max_kl_div_mean = max([data.max_kl_div_mean for data in data_pair])

    for data in data_pair:
        tl_specs += data.tl_specs
        kl_divs += data.kl_divs
        tl_action_probs += data.tl_action_probs
        ori_action_probs += data.ori_action_probs

    all_data: KLDivLogger = KLDivLogger(
        min_tl_spec,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
        tl_specs,
        kl_divs,
        tl_action_probs,
        ori_action_probs,
    )

    return all_data


def spec2title(spec: str) -> str:
    """
    Convert a spec to a snakecased str title

    Parameters
    ----------
    spec: str
        spec to convert

    Returns
    -------
    title: str
        title converted from the given spec
    """

    title: str = (
        spec.replace(" ", "_")
        .replace("&", "_and_")
        .replace("|", "_or_")
        .replace("__", "_")
    )

    return title
