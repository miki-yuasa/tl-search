from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
from numpy.typing import NDArray

import torch

from tl_search.common.typing import (
    MapLocations,
    ModelProps,
    MultiStartSearchLogger,
    ObsProp,
    SortedLogger,
    SpecNode,
    ValueTable,
)
from tl_search.evaluation.ranking import sort_spec
from tl_search.search.search import search_trained


def search_trained_multistart_multiprocess(
    num_start: int,
    init_nodes: list[SpecNode],
    num_max_iter: int,
    experiment: int,
    neighbor_masks: tuple[ValueTable, ...],
    ind_combs: list[tuple[int, int]],
    ref_action_probs_list: list[NDArray],
    ref_trap_masks: list[NDArray],
    learned_obs_props_list: list[list[ObsProp]],
    learned_models_props_list: list[list[ModelProps]],
    map_locs_list: list[MapLocations],
    device: torch.device,
    is_initialized_in_both_territories: bool,
    data_dir: str,
    filename_common: str,
) -> MultiStartSearchLogger:
    inputs: list[
        tuple[
            SpecNode,
            int,
            int,
            int,
            tuple[ValueTable, ...],
            list[tuple[int, int]],
            list[NDArray],
            list[NDArray],
            list[list[ObsProp]],
            list[list[ModelProps]],
            list[MapLocations],
            torch.device,
            bool,
            str,
            str,
        ]
    ] = [
        (
            node,
            num_max_iter,
            experiment,
            i,
            neighbor_masks,
            ind_combs,
            ref_action_probs_list,
            ref_trap_masks,
            learned_obs_props_list,
            learned_models_props_list,
            map_locs_list,
            device,
            is_initialized_in_both_territories,
            data_dir,
            filename_common,
        )
        for i, node in enumerate(init_nodes)
    ]

    # with Pool(num_start) as p:
    #     results: list[tuple[list[SpecNode], list[str], list[float]]] = p.map(
    #         search_trained_wrapper, inputs
    #     )

    results: list[tuple[list[SpecNode], list[str], list[float]]] = [
        search_trained_wrapper(input) for input in inputs
    ]

    local_optimum_nodes: list[SpecNode] = []
    local_optimum_specs: list[str] = []
    local_oprimum_kl_divs: list[float] = []
    searches_traces: list[tuple[list[str], list[float]]] = []

    for result in results:
        local_optimum_nodes.append(result[0][-1])
        local_optimum_specs.append(result[1][-1])
        local_oprimum_kl_divs.append(result[2][-1])
        searches_traces.append((result[1], result[2]))

    sorted_specs, sorted_min_kl_divs = sort_spec(
        local_optimum_specs, local_oprimum_kl_divs
    )

    multi_start_result = MultiStartSearchLogger(
        experiment, sorted_specs, sorted_min_kl_divs, searches_traces
    )

    return multi_start_result


def search_trained_wrapper(
    input: tuple[
        SpecNode,
        int,
        int,
        int,
        tuple[ValueTable, ...],
        list[tuple[int, int]],
        list[NDArray],
        list[NDArray],
        list[list[ObsProp]],
        list[list[ModelProps]],
        list[MapLocations],
        torch.device,
        bool,
        str,
        str,
    ]
) -> tuple[list[SpecNode], list[str], list[float]]:
    return search_trained(*input)
