import os, json, heapq, itertools
from statistics import mean
from typing import Union

import numpy as np

from tl_search.common.json import json_zip, json_unzip
from tl_search.common.typing import KLDivLogger, ExperimentLogger, ActionProbsSpec
from tl_search.common.utils import find_max_reward_spec, find_min_kl_div_tl_spec


def save_json(
    config: str,
    path: str,
) -> None:
    filepath = get_file_path(path)
    with open(filepath, "w") as f:
        f.write(config)


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


def kl_div_data_loader(filename: str) -> KLDivLogger:
    d: dict = load_json_as_dict(filename)

    formatted_data: KLDivLogger = KLDivLogger(**d)
    return formatted_data


def experiment_data_loader(filename: str) -> ExperimentLogger:
    d: dict = load_json_as_dict(filename)

    formatted_data = ExperimentLogger(**d)
    return formatted_data


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
