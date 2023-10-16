import os
import pickle
from typing import Union

import numpy as np
from numpy.typing import NDArray
import torch
from tl_search.common.utils import find_min_kl_div_tl_spec
from tl_search.evaluation.ranking import sort_spec
from tl_search.evaluation.wrapper import evaluate_ppo_models_multiprocess

from policies.known_policy import KnownPolicy
from policies.unknown_policy import UnknownPolicy
from tl_search.search.neighbor import (
    compare_nodes,
    create_all_nodes,
    create_neighbor_masks,
    find_additional_neighbors,
    find_neighbors,
    initialize_node,
    node2spec,
    nodes2specs,
    table2spec,
)
from train.wrapper import train_evaluate_multiprocess
from tl_search.common.typing import (
    ActionProbsSpec,
    EnemyPolicyMode,
    Exclusion,
    ExperimentLogger,
    MapLocations,
    ModelProps,
    ObsProp,
    PositiveRewardLogger,
    SaveMode,
    SortedLogger,
    SpecNode,
    ValueTable,
)
from tl_search.common.io import data_saver


def search_single_start(
    search_depth: int,
    search_ind: int,
    num_process: int,
    vars: tuple[str, ...],
    phi_task: str,
    description: str,
    experiment: int,
    gpus: tuple[int, ...],
    policy: Union[KnownPolicy, UnknownPolicy],
    num_sampling: int,
    num_replicates: int,
    total_timesteps: int,
    atom_prop_dict_all: dict[str, str],
    obs_props_all: list[ObsProp],
    is_from_both_terriotries: bool,
) -> tuple[SpecNode, float]:
    node = initialize_node(vars)
    neighbor_masks: tuple[NDArray, ...] = create_neighbor_masks(vars)

    searched_nodes: list[SpecNode] = []

    kl_div: float = 0.0
    for i in range(search_depth):
        neighbor_nodes: list[SpecNode] = find_neighbors(node, neighbor_masks)
        new_node, kl_div = search_neighbor(
            num_process,
            vars,
            neighbor_nodes,
            i,
            phi_task,
            description,
            experiment,
            gpus,
            policy,
            num_sampling,
            num_replicates,
            total_timesteps,
            atom_prop_dict_all,
            obs_props_all,
            is_from_both_terriotries,
            search_ind,
        )

        if compare_nodes(new_node, searched_nodes):
            searched_nodes += neighbor_nodes
            node = new_node
            break
        else:
            searched_nodes += neighbor_nodes
            node = new_node

    print(node)
    print(kl_div)
    print(table2spec(vars, node))

    return node, kl_div


def search_neighbor(
    num_process: int,
    vars: tuple[str, ...],
    neighbor_nodes: list[SpecNode],
    loop_ind: int,
    phi_task: str,
    description: str,
    experiment: int,
    gpus: tuple[int, ...],
    policy: Union[KnownPolicy, UnknownPolicy],
    num_sampling: int,
    num_replicates: int,
    total_timesteps: int,
    atom_prop_dict_all: dict[str, str],
    obs_props_all: list[ObsProp],
    is_from_both_terriotries: bool,
    search_ind: Union[int, None] = None,
) -> tuple[SpecNode, float]:
    indeces_dir: str = (
        "{}".format(loop_ind)
        if search_ind is None
        else "{}/{}".format(search_ind, loop_ind)
    )
    indeces_file: str = (
        "{}".format(loop_ind)
        if search_ind is None
        else "{}_{}".format(search_ind, loop_ind)
    )

    log_dir: str = "./tmp/gym/{}/search/{}/{}/".format(
        policy.policy_mode, experiment, indeces_dir
    )
    suffix: str = "search_{}_{}".format(experiment, indeces_file)
    model_savename: str = "models/search/{}".format(suffix)
    exp_savename: str = "data/search/{}_exp.json".format(suffix)
    pos_savename: str = "data/search/{}_pos.json".format(suffix)

    neighbor_specs: list[str] = nodes2specs(neighbor_nodes, phi_task, vars)

    experiment_data, positive_reward_data = train_evaluate_multiprocess(
        num_process,
        description,
        neighbor_specs,
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        experiment,
        gpus,
        policy,
        num_sampling,
        num_replicates,
        total_timesteps,
        atom_prop_dict_all,
        obs_props_all,
        log_dir,
        model_savename,
        "disabled",
        "disabled",
        "",
        "disabled",
        "",
        False,
        True,
        True,
        is_initialized_in_both_territories=is_from_both_terriotries,
    )

    min_kl_div_spec: str = (
        positive_reward_data.tl_specs_pos[np.argmin(positive_reward_data.kl_divs_pos)]
        if positive_reward_data.tl_specs_pos
        else experiment_data.min_kl_div_spec
    )

    min_node_idx: int = experiment_data.tl_specs.index(min_kl_div_spec)
    min_kl_div_node: SpecNode = neighbor_nodes[min_node_idx]

    if compare_nodes(min_kl_div_node, neighbor_nodes[0:1]):
        additional_neighbor_nodes = find_additional_neighbors(neighbor_nodes)
        additional_neighbor_specs: list[str] = nodes2specs(
            additional_neighbor_nodes, phi_task, vars
        )

        experiment_data, positive_reward_data = train_evaluate_multiprocess(
            num_process,
            description,
            additional_neighbor_specs,
            experiment_data.tl_specs,
            experiment_data.kl_divs,
            experiment_data.tl_action_probs,
            experiment_data.ori_action_probs,
            experiment_data.model_names,
            experiment_data.mean_rewards,
            experiment_data.std_rewards,
            experiment,
            gpus,
            policy,
            num_sampling,
            num_replicates,
            total_timesteps,
            atom_prop_dict_all,
            obs_props_all,
            log_dir,
            model_savename,
            "disabled",
            "disabled",
            "",
            "disabled",
            "",
            False,
            True,
            True,
            is_initialized_in_both_territories=is_from_both_terriotries,
            spec_ind_start=len(neighbor_nodes),
        )

        min_kl_div_spec = (
            positive_reward_data.tl_specs_pos[
                np.argmin(positive_reward_data.kl_divs_pos)
            ]
            if positive_reward_data.tl_specs_pos
            else experiment_data.min_kl_div_spec
        )

        neighbor_nodes += additional_neighbor_nodes
        min_node_idx = experiment_data.tl_specs.index(min_kl_div_spec)
        min_kl_div_node = neighbor_nodes[min_node_idx]
    else:
        pass

    data_saver(experiment_data, exp_savename)
    data_saver(positive_reward_data, pos_savename)

    local_optimum_idx: int = experiment_data.tl_specs.index(min_kl_div_spec)
    node_kl_div: float = float(np.mean(experiment_data.kl_divs[local_optimum_idx]))

    return min_kl_div_node, node_kl_div


def search_exh(
    num_process: int,
    vars: tuple[str, ...],
    description: str,
    experiment: int,
    gpus: tuple[int, ...],
    target: KnownPolicy | UnknownPolicy | ModelProps,
    enemy_policy: EnemyPolicyMode,
    num_sampling: int,
    num_replicates: int,
    total_timesteps: int,
    atom_prop_dict_all: dict[str, str],
    obs_props_all: list[ObsProp],
    exclusions: list[Exclusion],
    replicate_seeds: list[int],
):
    log_dir: str = "./tmp/gym/search/exh/{}/".format(experiment)
    suffix: str = "exh_{}".format(experiment)
    model_savename: str = "models/search/{}".format(suffix)
    reward_curve_savename: str = "search/{}".format(suffix)
    exp_savename: str = "data/search/{}_exp.json".format(suffix)
    pos_savename: str = "data/search/{}_pos.json".format(suffix)

    neighbor_nodes: list[SpecNode] = create_all_nodes(vars, exclusions)
    neighbor_specs: list[str] = list(set(nodes2specs(neighbor_nodes)))

    tl_specs_done: list[str] = []
    kl_divs_all: list[list[float]] = []
    tl_action_probs_all: list[ActionProbsSpec] = []
    ori_action_probs_all: list[ActionProbsSpec] = []
    model_names_all: list[list[str]] = []
    mean_rewards_all: list[list[float]] = []
    std_rewards_all: list[list[float]] = []

    model_save_mode: SaveMode = "disabled"
    reward_curve_plot_mode: SaveMode = "enabled"
    animation_mode: SaveMode = "disabled"
    animation_savename: str = ""

    is_gui_shown: bool = False
    is_trap_state_filtered: bool = True
    is_reward_calculated: bool = True
    is_initialized_in_both_territories: bool = True

    experiment_data, positive_reward_data = train_evaluate_multiprocess(
        num_process,
        description,
        neighbor_specs,
        tl_specs_done,
        kl_divs_all,
        tl_action_probs_all,
        ori_action_probs_all,
        model_names_all,
        mean_rewards_all,
        std_rewards_all,
        experiment,
        gpus,
        target,
        enemy_policy,
        num_sampling,
        num_replicates,
        total_timesteps,
        atom_prop_dict_all,
        obs_props_all,
        log_dir,
        model_savename,
        model_save_mode,
        reward_curve_plot_mode,
        reward_curve_savename,
        animation_mode,
        animation_savename,
        is_gui_shown,
        is_trap_state_filtered,
        is_reward_calculated,
        is_initialized_in_both_territories,
        replicate_seeds,
    )

    data_saver(experiment_data, exp_savename)
    data_saver(positive_reward_data, pos_savename)

    return positive_reward_data


def search_trained(
    init_node: SpecNode,
    num_max_iter: int,
    experiment: int,
    search_ind: int,
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
) -> tuple[list[SpecNode], list[str], list[float]]:
    print("Started a search with initial node {}.".format(search_ind))
    print("Initial spec: " + node2spec(init_node))

    node: SpecNode = init_node
    searched_specs: list[str] = []
    node_trace: list[SpecNode] = [node]
    spec_trace: list[str] = [node2spec(init_node)]
    metrics_trace: list[float] = [np.nan]

    is_trained: bool = True
    for i in range(num_max_iter):
        print("Search {}, iteration {}".format(search_ind, i))
        (
            new_spec,
            new_node,
            neighbor_specs,
            _,
            min_kl_div,
        ) = search_neighborhood(
            node,
            experiment,
            search_ind,
            i,
            neighbor_masks,
            is_trained,
            ind_combs,
            ref_action_probs_list,
            ref_trap_masks,
            learned_obs_props_list,
            map_locs_list,
            device,
            is_initialized_in_both_territories,
            data_dir,
            filename_common,
            learned_models_props_list,
        )

        node_trace.append(new_node)
        spec_trace.append(new_spec)
        metrics_trace.append(min_kl_div)

        if new_spec in searched_specs:
            print("The new node is already visited. Terminating the search.")
            break
        else:
            searched_specs += neighbor_specs
            node = new_node
            print("Next spec: " + new_spec)

        print("Min of the neighborhood {}:".format(i))
        print(node)
        print(min_kl_div)
        print(node2spec(node))

    return node_trace, spec_trace, metrics_trace


def search_neighborhood(
    node: SpecNode,
    experiment: int,
    search_ind: int,
    num_iter: int,
    neighbor_masks: tuple[ValueTable, ...],
    is_trained: bool,
    ind_combs: list[tuple[int, int]],
    ref_action_probs_list: list[NDArray],
    ref_trap_masks: list[NDArray],
    obs_props_list: list[list[ObsProp]],
    map_locs_list: list[MapLocations],
    device: torch.device,
    is_initialized_in_both_territories: bool,
    data_dir: str,
    filename_common: str,
    models_props_list: list[list[ModelProps]] = [],
) -> tuple[str, SpecNode, list[str], list[SpecNode], float]:
    run: str = "{}.{}.{}".format(experiment, search_ind, num_iter)

    neighbor_nodes: list[SpecNode] = find_neighbors(node, neighbor_masks)
    neighbor_specs: list[str] = nodes2specs(neighbor_nodes)

    kl_div_means, reward_mean_all = search_evaluation(
        neighbor_specs,
        is_trained,
        ind_combs,
        ref_action_probs_list,
        ref_trap_masks,
        obs_props_list,
        map_locs_list,
        device,
        is_initialized_in_both_territories,
        models_props_list,
    )

    (
        min_kl_div_spec,
        min_kl_div_ind,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
    ) = find_min_kl_div_tl_spec(neighbor_specs, kl_div_means)

    if min_kl_div_spec == node2spec(node):
        additional_neighbors: list[SpecNode] = find_additional_neighbors(neighbor_nodes)
        additional_neighbor_specs: list[str] = nodes2specs(additional_neighbors)

        kl_div_means_additional, reward_mean_all_additional = search_evaluation(
            additional_neighbor_specs,
            is_trained,
            ind_combs,
            ref_action_probs_list,
            ref_trap_masks,
            obs_props_list,
            map_locs_list,
            device,
            is_initialized_in_both_territories,
            models_props_list,
        )

        neighbor_nodes += additional_neighbors
        neighbor_specs += additional_neighbor_specs
        kl_div_means += kl_div_means_additional
        reward_mean_all += reward_mean_all_additional

        (
            min_kl_div_spec,
            min_kl_div_ind,
            min_kl_div_mean,
            mean_kl_div_mean,
            max_kl_div_mean,
        ) = find_min_kl_div_tl_spec(neighbor_specs, kl_div_means)

    else:
        pass

    max_reward_mean: float = float(np.max(reward_mean_all))
    mean_reward_mean: float = float(np.mean(reward_mean_all))
    min_reward_mean: float = float(np.min(reward_mean_all))
    max_reward_spec: str = neighbor_specs[np.argmax(reward_mean_all)]

    exp_log = ExperimentLogger(
        "known",
        run,
        "",
        min_kl_div_spec,
        max_reward_spec,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
        min_reward_mean,
        mean_reward_mean,
        max_reward_mean,
        neighbor_specs,
        [],
        [],
        [],
        [],
        [],
        [],
    )

    reward_means_np = np.array(reward_mean_all)
    pos_reward_inds: NDArray = reward_means_np > 0
    reward_means_pos: list[float] = reward_means_np[pos_reward_inds].tolist()
    tl_specs_pos: list[str] = np.array(neighbor_specs)[pos_reward_inds].tolist()
    kl_divs_pos: list[float] = np.array(kl_div_means)[pos_reward_inds].tolist()

    pos_reward_log = PositiveRewardLogger(
        tl_specs_pos, reward_means_pos, kl_divs_pos, [], []
    )

    sorted_specs, sorted_kl_divs = sort_spec(
        pos_reward_log.tl_specs_pos, pos_reward_log.kl_divs_pos
    )

    sorted_log = SortedLogger(experiment, sorted_specs, sorted_kl_divs)

    file_path_common: str = os.path.join(
        data_dir,
        "{}_{}_start_{}_iter_{}".format(
            filename_common, experiment, search_ind, num_iter
        ),
    )
    exp_savename: str = file_path_common + "_exp.json"
    pos_savename: str = file_path_common + "_pos.json"
    sorted_savename: str = file_path_common + "_sorted.json"
    nodes_savename: str = file_path_common + "_nodes_data.pickle"

    data_saver(exp_log, exp_savename)
    data_saver(pos_reward_log, pos_savename)
    data_saver(sorted_log, sorted_savename)

    with open(nodes_savename, mode="wb") as f:
        pickle.dump(neighbor_nodes, f)

    min_kl_div_node: SpecNode = neighbor_nodes[min_kl_div_ind]

    return (
        min_kl_div_spec,
        min_kl_div_node,
        neighbor_specs,
        neighbor_nodes,
        min_kl_div_mean,
    )


def search_evaluation(
    neighbor_specs: list[str],
    is_trained: bool,
    ind_combs: list[tuple[int, int]],
    ref_action_probs_list: list[NDArray],
    ref_trap_masks: list[NDArray],
    obs_props_list: list[list[ObsProp]],
    map_locs_list: list[MapLocations],
    device: torch.device,
    is_initialized_in_both_territories: bool,
    models_props_list: list[list[ModelProps]] = [],
) -> tuple[list[float], list[float]]:
    neighbor_models_props_list: list[list[ModelProps]] = []

    if is_trained:
        all_specs_dict: dict[str, int] = {
            models_props[0].spec: i for i, models_props in enumerate(models_props_list)
        }
        neighbor_inds: list[int] = [all_specs_dict[spec] for spec in neighbor_specs]
        neighbor_models_props_list = [models_props_list[ind] for ind in neighbor_inds]
    else:
        pass

    (
        kl_div_report_model_all,
        kl_div_reports_model_all,
        kl_div_mean_rank,
        reward_mean_all,
        reward_std_all,
    ) = evaluate_ppo_models_multiprocess(
        0,
        ind_combs,
        ref_action_probs_list,
        ref_trap_masks,
        neighbor_models_props_list,
        obs_props_list,
        map_locs_list,
        device,
        is_initialized_in_both_territories,
    )

    kl_div_means: list[float] = [
        report.kl_div_mean for report in kl_div_report_model_all
    ]

    return kl_div_means, reward_mean_all
