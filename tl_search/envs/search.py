import json, os, pickle
from multiprocessing import Pool
import multiprocessing as mp
import random

import numpy as np
from numpy.typing import NDArray
from scipy.special import rel_entr, entr
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from gymnasium import spaces

from tl_search.common.io import spec2title
from tl_search.common.typing import (
    EntropyReportDict,
    EpisodeLengthReport,
    FilteredLog,
    KLDivReportDict,
    ObsProp,
    PositiveRewardLog,
    RewardReportDict,
    SearchLog,
    SortedLog,
    SpecNode,
    ValueTable,
)
from tl_search.common.utils import find_min_kl_div_tl_spec
from tl_search.envs.extractor import generate_possible_states, get_action_distributions
from tl_search.envs.select import (
    select_max_entropy_spec_replicate,
)
from tl_search.envs.tl_multigrid import TLMultigrid, TLMultigridDefaultArgs
from tl_search.envs.train import simulate_model
from tl_search.envs.typing import EnemyPolicyMode, FieldObj
from tl_search.evaluation.count import (
    evaluate_episode_lengths_tl,
    get_episode_length_report,
)
from tl_search.evaluation.evaluation import collect_kl_div_stats
from tl_search.evaluation.filter import apply_filter
from tl_search.evaluation.ranking import sort_spec
from tl_search.search.neighbor import (
    find_additional_neighbors,
    find_neighbor_nodes_specs,
    find_neighbors,
    node2spec,
    nodes2specs,
)
from tl_search.train.tl_train import train_replicate_tl_agent


def search_train_evaluate(
    init_node: SpecNode,
    num_max_search_steps: int,
    run: int,
    start_idx: int,
    neighbor_masks: tuple[ValueTable, ...],
    log_save_path: str,
    num_processes: int,
    num_replicates: int,
    n_envs: int,
    seeds: list[int],
    total_timesteps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    target_action_probs_list: list[NDArray],
    target_trap_masks: list[NDArray],
    field_obj_list: list[FieldObj],
    data_save_path: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    tuned_param: dict[str, float],
    default_env_args: TLMultigridDefaultArgs | None = None,
    search_start_iter: int = 0,
    episode_length_report: EpisodeLengthReport | None = None,
    reward_threshold: float = 0.0,
    episode_length_sigma: float | None = 1,
    warm_start_path: str | None = None,
    kl_div_suffix: str | None = None,
) -> tuple[list[SpecNode], list[str], list[float], list[str]]:
    print("Searching for a spec. Start index: ", start_idx)
    print("Initial spec: ", node2spec(init_node))

    node: SpecNode = init_node
    searched_specs: list[str] = []
    node_trace: list[SpecNode] = [node]
    spec_trace: list[str] = [node2spec(init_node)]
    metrics_trace: list[float] = [np.nan]

    log_save_path_orig: str = log_save_path

    hit_local_minimum: bool = False
    extended_step_count: int = 0
    max_extended_steps: int = 3
    hit_step: int = 0

    term_cond: bool = False

    extension_counter: int = 0
    local_minimum_spec: str = ""

    for step in range(search_start_iter, num_max_search_steps):
        print(f"Start {start_idx}, Step {step}")
        print(f"Current spec: {node2spec(node)}")

        run_label: str = f"{run}.{start_idx}.{step}"

        log_save_path = log_save_path_orig.replace(".json", f".{run_label}.json")

        neighbor_nodes: list[SpecNode]
        neighbor_specs: list[str]

        neighbor_nodes, neighbor_specs = find_neighbor_nodes_specs(
            node, neighbor_masks, True
        )

        for n, s in zip(neighbor_nodes, neighbor_specs):
            node_path: str = f"out/nodes/{enemy_policy_mode}/{spec2title(s)}.pkl"

            if os.path.exists(node_path):
                pass
            else:
                with open(node_path, "wb") as f:
                    pickle.dump(n, f)

        nodes_save_path: str = log_save_path.replace(".json", "_nodes.pkl")

        with open(nodes_save_path, "wb") as f:
            pickle.dump(neighbor_nodes, f)

        search_log: SearchLog = {
            "run": run_label,
            "enemy_policy": enemy_policy_mode,
            "seeds": seeds,
            "num_searched_specs": len(neighbor_specs),
            "min_kl_div_spec": "N/A",
            "max_reward_spec": "N/A",
            "min_kl_div": 0,
            "mean_kl_div": 0,
            "max_kl_div": 0,
            "min_reward": 0,
            "mean_reward": 0,
            "max_reward": 0,
            "searched_specs": neighbor_specs,
        }

        search_log_path: str = log_save_path.replace(".json", "_search.json")

        with open(search_log_path, "w") as f:
            json.dump(search_log, f, indent=4)

        kl_div_means, reward_means, mean_episode_lengths = train_evaluate_multiprocess(
            num_processes,
            num_replicates,
            n_envs,
            seeds,
            total_timesteps,
            model_save_path,
            learning_curve_path,
            animation_save_path,
            device,
            window,
            target_action_probs_list,
            target_trap_masks,
            field_obj_list,
            data_save_path,
            neighbor_specs,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            tuned_param,
            default_env_args,
            warm_start_path,
            kl_div_suffix,
        )

        filtered_nodes: list[SpecNode]
        filtered_specs: list[str]
        filtered_kl_divs: list[float]
        filtered_rewards: list[float]
        filtered_episode_lengths: list[float]

        (
            filtered_nodes,
            filtered_specs,
            filtered_kl_divs,
            filtered_rewards,
            filtered_episode_lengths,
        ) = apply_filter(
            neighbor_nodes,
            neighbor_specs,
            kl_div_means,
            reward_means,
            mean_episode_lengths,
            episode_length_report["mean"],
            episode_length_report["std"],
            reward_threshold,
            episode_length_sigma,
        )

        if len(filtered_specs) == 0:
            break
        else:
            pass

        (
            min_kl_div_spec,
            min_kl_div_idx,
            min_kl_div_mean,
            mean_kl_div_mean,
            max_kl_div_mean,
        ) = find_min_kl_div_tl_spec(filtered_specs, filtered_kl_divs)

        if min_kl_div_spec == node2spec(node):
            print("The min KL-divergence spec is the same as the current spec.")
            print("Expand the search.")
            additional_neighbors: list[SpecNode] = find_additional_neighbors(
                neighbor_nodes
            )
            additional_neighbor_specs: list[str] = nodes2specs(additional_neighbors)

            (
                additional_kl_div_means,
                additional_reward_means,
                additional_mean_episode_lengths,
            ) = train_evaluate_multiprocess(
                num_processes,
                num_replicates,
                n_envs,
                seeds,
                total_timesteps,
                model_save_path,
                learning_curve_path,
                animation_save_path,
                device,
                window,
                target_action_probs_list,
                target_trap_masks,
                field_obj_list,
                data_save_path,
                additional_neighbor_specs,
                obs_props,
                atom_prep_dict,
                enemy_policy_mode,
                map_path,
                tuned_param,
                default_env_args,
                warm_start_path,
                kl_div_suffix,
            )

            neighbor_nodes += additional_neighbors
            neighbor_specs += additional_neighbor_specs
            kl_div_means += additional_kl_div_means
            reward_means += additional_reward_means
            mean_episode_lengths += additional_mean_episode_lengths

            (
                filtered_nodes,
                filtered_specs,
                filtered_kl_divs,
                filtered_rewards,
                filtered_episode_lengths,
            ) = apply_filter(
                neighbor_nodes,
                neighbor_specs,
                kl_div_means,
                reward_means,
                mean_episode_lengths,
                episode_length_report["mean"],
                episode_length_report["std"],
                reward_threshold,
                episode_length_sigma,
            )

            (
                min_kl_div_spec,
                min_kl_div_idx,
                min_kl_div_mean,
                mean_kl_div_mean,
                max_kl_div_mean,
            ) = find_min_kl_div_tl_spec(filtered_specs, filtered_kl_divs)

            if min_kl_div_spec == node2spec(node) and not hit_local_minimum:
                local_minimum_spec = min_kl_div_spec
                extension_counter += 1
                (
                    min_kl_div_spec,
                    min_kl_div_idx,
                    min_kl_div_mean,
                    mean_kl_div_mean,
                    max_kl_div_mean,
                ) = find_min_kl_div_tl_spec(
                    filtered_specs, filtered_kl_divs, k_smallest=extension_counter
                )
                hit_local_minimum = True
                hit_step = step

            elif hit_local_minimum and (hit_step - step) % 2 == 0:
                if local_minimum_spec == min_kl_div_spec:
                    extension_counter += 1
                    (
                        min_kl_div_spec,
                        min_kl_div_idx,
                        min_kl_div_mean,
                        mean_kl_div_mean,
                        max_kl_div_mean,
                    ) = find_min_kl_div_tl_spec(
                        filtered_specs,
                        filtered_kl_divs,
                        k_smallest=extension_counter,
                    )
                else:
                    local_minimum_spec = ""
                    extension_counter = 0
                    hit_local_minimum = False
                    hit_step = 0
            else:
                pass

        else:
            pass

        if hit_local_minimum:
            if extension_counter > max_extended_steps:
                print(
                    "Hit local minimum after extending the search. Terminating the search."
                )
                term_cond = True
            else:
                print("Next spec: " + min_kl_div_spec)

        # elif (
        #     not hit_local_minimum
        #     and spec_trace[-1] == min_kl_div_spec
        #     or min_kl_div_spec in searched_specs
        # ):
        #     print("The new node is already visited. Extending the search.")
        #     print(f"Input: {spec_trace[-1]}")
        #     print(node)
        #     print(f"Output: {min_kl_div_spec}")
        #     print(filtered_nodes[min_kl_div_idx])
        #     (
        #         min_kl_div_spec,
        #         min_kl_div_idx,
        #         min_kl_div_mean,
        #         mean_kl_div_mean,
        #         max_kl_div_mean,
        #     ) = find_min_kl_div_tl_spec(filtered_specs, filtered_kl_divs, k_smallest=1)
        #     hit_local_minimum = True
        else:
            print("Next spec: " + min_kl_div_spec)

        min_kl_div_node: SpecNode = filtered_nodes[min_kl_div_idx]

        max_reward_mean: float = float(np.max(filtered_rewards))
        mean_reward_mean: float = float(np.mean(filtered_rewards))
        min_reward_mean: float = float(np.min(filtered_rewards))
        max_reward_spec: str = neighbor_specs[np.argmax(filtered_rewards)]

        search_log = {
            "run": run_label,
            "enemy_policy": enemy_policy_mode,
            "seeds": seeds,
            "num_searched_specs": len(neighbor_specs),
            "min_kl_div_spec": min_kl_div_spec,
            "max_reward_spec": max_reward_spec,
            "min_kl_div": min_kl_div_mean,
            "mean_kl_div": mean_kl_div_mean,
            "max_kl_div": max_kl_div_mean,
            "min_reward": min_reward_mean,
            "mean_reward": mean_reward_mean,
            "max_reward": max_reward_mean,
            "searched_specs": neighbor_specs,
            "kl_div_means": kl_div_means,
        }

        filtered_log: FilteredLog = {
            "run": run_label,
            "specs": filtered_specs,
            "rewards": filtered_rewards,
            "kl_divs": filtered_kl_divs,
            "episode_lengths": filtered_episode_lengths,
        }

        sorted_specs, sorted_kl_divs = sort_spec(filtered_specs, filtered_kl_divs)

        sorted_log: SortedLog = {
            "run": run_label,
            "specs": sorted_specs,
            "kl_divs": sorted_kl_divs,
        }

        pos_reward_log_path: str = log_save_path.replace(".json", "_pos_reward.json")
        sorted_log_path: str = log_save_path.replace(".json", "_sorted.json")

        with open(search_log_path, "w") as f:
            json.dump(search_log, f, indent=4)

        with open(pos_reward_log_path, "w") as f:
            json.dump(filtered_log, f, indent=4)

        with open(sorted_log_path, "w") as f:
            json.dump(sorted_log, f, indent=4)

        with open(nodes_save_path, "wb") as f:
            pickle.dump(neighbor_nodes, f)

        new_spec, new_node, neighbor_specs, min_kl_div = (
            min_kl_div_spec,
            min_kl_div_node,
            neighbor_specs,
            min_kl_div_mean,
        )

        searched_specs += neighbor_specs
        node = new_node

        node_trace.append(new_node)
        spec_trace.append(new_spec)
        metrics_trace.append(min_kl_div)

        print("Min of the neighborhood {}:".format(step))
        print(node)
        print(min_kl_div)
        print(node2spec(node))

        if term_cond:
            break
        else:
            pass

    return node_trace, spec_trace, metrics_trace, searched_specs


def search_neighborhood(
    run: str,
    node: SpecNode,
    neighbor_masks: tuple[ValueTable, ...],
    log_save_path: str,
    num_processes: int,
    num_replicates: int,
    n_envs: int,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    idx_combs: list[tuple[int, int]],
    target_action_probs_list: list[NDArray],
    target_trap_masks: list[NDArray],
    field_obj_list: list[FieldObj],
    data_save_path: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    tuned_param: dict[str, float],
    default_env_args: TLMultigridDefaultArgs | None = None,
    episode_length_report: EpisodeLengthReport | None = None,
    reward_threshold: float = 0.0,
    episode_length_sigma: float = 2.16,
) -> tuple[str, SpecNode, list[str], float]:
    neighbor_nodes: list[SpecNode]
    neighbor_specs: list[str]

    neighbor_nodes, neighbor_specs = find_neighbor_nodes_specs(
        node, neighbor_masks, True
    )

    for node, spec in zip(neighbor_nodes, neighbor_specs):
        node_path: str = f"out/nodes/{enemy_policy_mode}/{spec2title(spec)}.pkl"

        if os.path.exists(node_path):
            pass
        else:
            with open(node_path, "wb") as f:
                pickle.dump(node, f)

    nodes_save_path: str = log_save_path.replace(".json", "_nodes.pkl")

    with open(nodes_save_path, "wb") as f:
        pickle.dump(neighbor_nodes, f)

    search_log: SearchLog = {
        "run": run,
        "enemy_policy": enemy_policy_mode,
        "seeds": seeds,
        "num_searched_specs": len(neighbor_specs),
        "min_kl_div_spec": "N/A",
        "max_reward_spec": "N/A",
        "min_kl_div": 0,
        "mean_kl_div": 0,
        "max_kl_div": 0,
        "min_reward": 0,
        "mean_reward": 0,
        "max_reward": 0,
        "searched_specs": neighbor_specs,
    }

    search_log_path: str = log_save_path.replace(".json", "_search.json")

    with open(search_log_path, "w") as f:
        json.dump(search_log, f, indent=4)

    kl_div_means, reward_means, mean_episode_lengths = train_evaluate_multiprocess(
        num_processes,
        num_replicates,
        n_envs,
        seeds,
        total_time_steps,
        model_save_path,
        learning_curve_path,
        animation_save_path,
        device,
        window,
        idx_combs,
        target_action_probs_list,
        target_trap_masks,
        field_obj_list,
        data_save_path,
        neighbor_specs,
        obs_props,
        atom_prep_dict,
        enemy_policy_mode,
        map_path,
        tuned_param,
        default_env_args,
    )

    filtered_nodes: list[SpecNode]
    filtered_specs: list[str]
    filtered_kl_divs: list[float]
    filtered_rewards: list[float]
    filtered_episode_lengths: list[float]

    (
        filtered_nodes,
        filtered_specs,
        filtered_kl_divs,
        filtered_rewards,
        filtered_episode_lengths,
    ) = apply_filter(
        neighbor_nodes,
        neighbor_specs,
        kl_div_means,
        reward_means,
        mean_episode_lengths,
        episode_length_report["mean"],
        episode_length_report["std"],
        reward_threshold,
        episode_length_sigma,
    )

    (
        min_kl_div_spec,
        min_kl_div_idx,
        min_kl_div_mean,
        mean_kl_div_mean,
        max_kl_div_mean,
    ) = find_min_kl_div_tl_spec(filtered_specs, filtered_kl_divs)

    if min_kl_div_spec == node2spec(node):
        print("The min KL-divergence spec is the same as the current spec.")
        print("Expand the search.")
        additional_neighbors: list[SpecNode] = find_additional_neighbors(neighbor_nodes)
        additional_neighbor_specs: list[str] = nodes2specs(additional_neighbors)

        (
            additional_kl_div_means,
            additional_reward_means,
            additional_mean_episode_lengths,
        ) = train_evaluate_multiprocess(
            num_processes,
            num_replicates,
            n_envs,
            seeds,
            total_time_steps,
            model_save_path,
            learning_curve_path,
            animation_save_path,
            device,
            window,
            idx_combs,
            target_action_probs_list,
            target_trap_masks,
            field_obj_list,
            data_save_path,
            additional_neighbor_specs,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            tuned_param,
            default_env_args,
        )

        neighbor_nodes += additional_neighbors
        neighbor_specs += additional_neighbor_specs
        kl_div_means += additional_kl_div_means
        reward_means += additional_reward_means
        mean_episode_lengths += additional_mean_episode_lengths

        (
            filtered_nodes,
            filtered_specs,
            filtered_kl_divs,
            filtered_rewards,
            filtered_episode_lengths,
        ) = apply_filter(
            neighbor_nodes,
            neighbor_specs,
            kl_div_means,
            reward_means,
            mean_episode_lengths,
            episode_length_report["mean"],
            episode_length_report["std"],
            reward_threshold,
            episode_length_sigma,
        )

        (
            min_kl_div_spec,
            min_kl_div_idx,
            min_kl_div_mean,
            mean_kl_div_mean,
            max_kl_div_mean,
        ) = find_min_kl_div_tl_spec(filtered_specs, filtered_kl_divs)

    else:
        pass

    min_kl_div_node: SpecNode = filtered_nodes[min_kl_div_idx]

    max_reward_mean: float = float(np.max(filtered_rewards))
    mean_reward_mean: float = float(np.mean(filtered_rewards))
    min_reward_mean: float = float(np.min(filtered_rewards))
    max_reward_spec: str = neighbor_specs[np.argmax(filtered_rewards)]

    search_log = {
        "run": run,
        "enemy_policy": enemy_policy_mode,
        "seeds": seeds,
        "num_searched_specs": len(neighbor_specs),
        "min_kl_div_spec": min_kl_div_spec,
        "max_reward_spec": max_reward_spec,
        "min_kl_div": min_kl_div_mean,
        "mean_kl_div": mean_kl_div_mean,
        "max_kl_div": max_kl_div_mean,
        "min_reward": min_reward_mean,
        "mean_reward": mean_reward_mean,
        "max_reward": max_reward_mean,
        "searched_specs": neighbor_specs,
        "kl_div_means": kl_div_means,
    }

    filtered_log: FilteredLog = {
        "run": run,
        "specs": filtered_specs,
        "rewards": filtered_rewards,
        "kl_divs": filtered_kl_divs,
        "episode_lengths": filtered_episode_lengths,
    }

    sorted_specs, sorted_kl_divs = sort_spec(filtered_specs, filtered_kl_divs)

    sorted_log: SortedLog = {
        "run": run,
        "specs": sorted_specs,
        "kl_divs": sorted_kl_divs,
    }

    pos_reward_log_path: str = log_save_path.replace(".json", "_pos_reward.json")
    sorted_log_path: str = log_save_path.replace(".json", "_sorted.json")

    with open(search_log_path, "w") as f:
        json.dump(search_log, f, indent=4)

    with open(pos_reward_log_path, "w") as f:
        json.dump(filtered_log, f, indent=4)

    with open(sorted_log_path, "w") as f:
        json.dump(sorted_log, f, indent=4)

    with open(nodes_save_path, "wb") as f:
        pickle.dump(neighbor_nodes, f)

    return (
        min_kl_div_spec,
        min_kl_div_node,
        neighbor_specs,
        min_kl_div_mean,
    )


def train_evaluate_multiprocess(
    num_processes: int,
    num_replicates: int,
    n_envs: int,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    target_action_probs_list: list[NDArray],
    target_trap_masks: list[NDArray],
    field_obj_list: list[FieldObj],
    data_save_path: str,
    tl_specs: list[str],
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    tuned_param: dict[str, float],
    default_env_args: TLMultigridDefaultArgs | None = None,
    warm_start_path: str | None = None,
    kl_div_suffix: str | None = None,
) -> tuple[list[float], list[float], list[float]]:
    inputs = [
        (
            num_replicates,
            n_envs,
            seeds,
            total_time_steps,
            model_save_path,
            learning_curve_path,
            animation_save_path,
            device,
            window,
            target_action_probs_list,
            target_trap_masks,
            field_obj_list,
            data_save_path,
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            tuned_param,
            default_env_args,
            warm_start_path,
            kl_div_suffix,
        )
        for tl_spec in tl_specs
    ]

    with Pool(num_processes) as p:
        results = p.starmap(train_evaluate, inputs)

    p.close()
    p.join()

    # results = [train_evaluate(*input_) for input_ in inputs]

    kl_div_means: list[float] = []
    reward_means: list[float] = []
    episode_lengths: list[float] = []

    for kl_div_report, reward_mean, episode_length in results:
        kl_div_means.append(kl_div_report["kl_div_mean"])
        reward_means.append(reward_mean)
        episode_lengths.append(episode_length)

    return kl_div_means, reward_means, episode_lengths


def train_evaluate(
    num_replicates: int,
    n_envs: int,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    target_action_probs_list: list[NDArray],
    target_trap_masks: list[NDArray],
    field_obj_list: list[FieldObj],
    data_save_path: str,
    tl_spec: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    tuned_param: dict[str, float],
    default_env_args: TLMultigridDefaultArgs | None = None,
    warm_start_path: str | None = None,
    kl_div_suffix: str | None = None,
) -> tuple[KLDivReportDict, float, float]:
    env = (
        TLMultigrid(
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            **default_env_args,
        )
        if default_env_args
        else TLMultigrid(
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
        )
    )

    warm_start_suffix: str = "_ws" if warm_start_path is not None else ""
    seeds_tmp: list[int] = [random.randint(0, 10000) for _ in range(num_replicates)]

    try:
        spec_models: list[PPO] = train_replicate_tl_agent(
            num_replicates,
            n_envs,
            env,
            seeds_tmp,
            total_time_steps,
            model_save_path.replace(
                ".zip", f"_{spec2title(tl_spec)}{warm_start_suffix}.zip"
            ),
            learning_curve_path.replace(
                ".png", f"_{spec2title(tl_spec)}{warm_start_suffix}.png"
            ),
            animation_save_path.replace(
                ".gif", f"_{spec2title(tl_spec)}{warm_start_suffix}.gif"
            )
            if animation_save_path is not None
            else None,
            device,
            window,
            tuned_param,
            warm_start_path,
        )

        kl_div_report, reward_mean, episode_length = evaluate_models(
            env,
            target_action_probs_list,
            target_trap_masks,
            spec_models,
            field_obj_list,
            data_save_path.replace(".json", f"_{spec2title(tl_spec)}.json"),
            kl_div_suffix=kl_div_suffix,
        )

    except:
        kl_div_report: KLDivReportDict = {
            "kl_div_mean": 999,
            "kl_div_std": 0,
            "kl_div_max": 999,
            "kl_div_min": 999,
            "kl_ci": (999, 999),
        }
        reward_mean: float = 0
        episode_length: float = 0

    return kl_div_report, reward_mean, episode_length


def evaluate_models(
    env: TLMultigrid,
    target_action_probs_list: list[NDArray],
    target_trap_masks: list[NDArray],
    spec_models: list[PPO],
    field_obj_list: list[FieldObj],
    data_save_path: str,
    num_episodes: int = 200,
    kl_div_suffix: str | None = None,
) -> tuple[KLDivReportDict, float, float]:
    print(f"Evaluating models for {env._tl_spec}...")

    kl_div_save_path: str = data_save_path.replace(
        ".json",
        "_kl_div.json" if kl_div_suffix is None else f"_kl_div_{kl_div_suffix}.json",
    )
    if kl_div_suffix is not None:
        kl_div_save_path.replace("/kl_div", "/kl_div/kl_div")
    else:
        pass
    reward_save_path: str = data_save_path.replace(".json", "_reward.json")
    episode_length_save_path: str = data_save_path.replace(".json", "_episode.json")

    entropy_save_path: str = data_save_path.replace(".json", "_entropy.json")

    kl_div_report: KLDivReportDict
    reward_mean: float

    if os.path.exists(reward_save_path):
        with open(reward_save_path, "r") as f:
            reward_report: RewardReportDict = json.load(f)

        reward_mean = reward_report["mean"]
    else:
        model_rewards: list[float] = []
        for i, model in enumerate(spec_models):
            print(
                f"Getting model rewards for {env._tl_spec} for rep {i + 1}/{len(spec_models)}...."
            )

            rewards: list[float] = []

            for _ in range(200):
                simulate_model(model, env)
                rewards.append(env._episodic_reward)

            model_rewards.append(float(np.mean(rewards)))

        reward_mean = float(np.mean(model_rewards))
        reward_std: float = float(np.std(model_rewards))
        reward_report: RewardReportDict = {"mean": reward_mean, "std": reward_std}
        with open(reward_save_path, "w") as f:
            json.dump(reward_report, f, indent=4)

    if os.path.exists(kl_div_save_path) and os.path.exists(entropy_save_path):
        print(f"Loading saved reward and KL div for {env._tl_spec}...")
        with open(kl_div_save_path, "r") as f:
            kl_div_report = json.load(f)

    else:
        rep_action_probs_list: list[NDArray] = []
        rep_trap_masks: list[NDArray] = []

        for i, model in enumerate(spec_models):
            print(
                f"Getting action distributions for {env._tl_spec} for rep {i + 1}/{len(spec_models)}..."
            )
            field_list_rep = random.sample(
                generate_possible_states(env), len(field_obj_list)
            )

            rep_action_probs, rep_trap_mask = get_action_distributions(
                model, env, field_list_rep
            )
            rep_action_probs_list.append(rep_action_probs)
            rep_trap_masks.append(rep_trap_mask)

        (
            max_entropy_idx,
            entropies,
            num_non_trap_states,
        ) = select_max_entropy_spec_replicate(rep_action_probs_list, rep_trap_masks)

        entropy_report: EntropyReportDict = {
            "max_entropy_idx": max_entropy_idx,
            "entropies": entropies,
            "num_non_trap_states": num_non_trap_states,
        }

        with open(entropy_save_path, "w") as f:
            json.dump(entropy_report, f, indent=4)

        model_kl_divs: list[NDArray] = []
        # kl_div_reports: list[KLDivReportDict] = []

        print(f"Getting KL divergences for {env._tl_spec}...")

        max_kl_div: float = np.sum(entr([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]))

        # for comb in idx_combs:
        target_idx = 0
        spec_idx = max_entropy_idx

        trap_mask: NDArray = target_trap_masks[target_idx] * rep_trap_masks[spec_idx]
        target_action_probs_filtered: NDArray = target_action_probs_list[target_idx][
            trap_mask == 1
        ]
        spec_action_probs_filtered: NDArray = rep_action_probs_list[spec_idx][
            trap_mask == 1
        ]

        target_entropy: NDArray = np.sum(entr(target_action_probs_filtered), axis=1)
        normalized_entropy: NDArray = 1 - target_entropy / max_kl_div
        weight: NDArray = normalized_entropy / np.sum(normalized_entropy)

        kl_divs: NDArray = (
            np.sum(
                rel_entr(target_action_probs_filtered, spec_action_probs_filtered),
                axis=1,
            )
            * weight
        )

        model_kl_divs.append(kl_divs.flatten())
        # kl_div_reports.append(kl_div_report)

        model_kl_divs_concat: NDArray = np.concatenate(model_kl_divs)

        kl_div_report: KLDivReportDict = (
            {
                "kl_div_mean": 999,
                "kl_div_std": 0,
                "kl_div_max": 999,
                "kl_div_min": 999,
                "kl_ci": (0, 0),
            }
            if model_kl_divs_concat.size == 0
            else collect_kl_div_stats(model_kl_divs_concat, in_dict=True)
        )

        with open(kl_div_save_path, "w") as f:
            json.dump(kl_div_report, f, indent=4)

    episode_length_report: EpisodeLengthReport

    if os.path.exists(episode_length_save_path):
        with open(episode_length_save_path, "r") as f:
            episode_length_report = json.load(f)

    else:
        episode_lengths: list[int] = []
        for model in spec_models:
            episode_lengths += evaluate_episode_lengths_tl(num_episodes, model, env)

        episode_length_report = get_episode_length_report(episode_lengths)

        with open(episode_length_save_path, "w") as f:
            json.dump(episode_length_report, f, indent=4)

    mean_episode_length = episode_length_report["mean"]

    return kl_div_report, reward_mean, mean_episode_length


def train_exh_mp(
    gpu: int,
    num_process: int,
    num_replicates: int,
    n_envs: int,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    window: int,
    data_save_path: str,
    tl_specs: list[str],
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    tuned_param: dict[str, float],
    default_env_args: TLMultigridDefaultArgs | None = None,
    warm_start_path: str | None = None,
) -> None:
    input = [
        (
            num_replicates,
            n_envs,
            seeds,
            total_time_steps,
            model_save_path,
            learning_curve_path,
            animation_save_path,
            torch.device(f"cuda:{gpu}"),
            window,
            data_save_path,
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            tuned_param,
            default_env_args,
            warm_start_path,
        )
        for tl_spec in tl_specs
    ]

    with Pool(num_process) as p:
        p.starmap(train_exh, input)

    p.close()
    p.join()


def train_exh(
    num_replicates: int,
    n_envs: int,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    data_save_path: str,
    tl_spec: str,
    obs_props: list[ObsProp],
    atom_prep_dict: dict[str, str],
    enemy_policy_mode: EnemyPolicyMode,
    map_path: str,
    tuned_param: dict[str, float],
    default_env_args: TLMultigridDefaultArgs | None = None,
    warm_start_path: str | None = None,
) -> None:
    env = (
        TLMultigrid(
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            **default_env_args,
        )
        if default_env_args
        else TLMultigrid(
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
        )
    )

    warm_start_suffix: str = "_ws" if warm_start_path is not None else ""

    try:
        spec_models: list[PPO] = train_replicate_tl_agent(
            num_replicates,
            n_envs,
            env,
            seeds,
            total_time_steps,
            model_save_path.replace(
                ".zip", f"_{spec2title(tl_spec)}{warm_start_suffix}.zip"
            ),
            learning_curve_path.replace(
                ".png", f"_{spec2title(tl_spec)}{warm_start_suffix}.png"
            ),
            animation_save_path.replace(
                ".gif", f"_{spec2title(tl_spec)}{warm_start_suffix}.gif"
            )
            if animation_save_path is not None
            else None,
            device,
            window,
            tuned_param,
            warm_start_path,
        )

        evaluate_reward_episode(env, spec_models, data_save_path)

    except:
        print(f"Failed to train {tl_spec}.")


def evaluate_reward_episode(
    env: TLMultigrid,
    spec_models: list[PPO],
    data_save_path: str,
    num_episodes: int = 200,
) -> None:
    print(f"Evaluating models for {env._tl_spec}...")

    reward_save_path: str = data_save_path.replace(".json", "_reward.json")
    episode_length_save_path: str = data_save_path.replace(".json", "_episode.json")

    reward_mean: float

    if os.path.exists(reward_save_path):
        print(f"Skipping reward evaluation for {env._tl_spec}...")

    else:
        model_rewards: list[float] = []

        for i, model in enumerate(spec_models):
            print(
                f"Getting model rewards for {env._tl_spec} for rep {i + 1}/{len(spec_models)}...."
            )

            rewards: list[float] = []

            for _ in range(200):
                simulate_model(model, env)
                rewards.append(env._episodic_reward)

            model_rewards.append(float(np.mean(rewards)))

        reward_mean = float(np.mean(model_rewards))
        reward_std: float = float(np.std(model_rewards))
        reward_report: RewardReportDict = {"mean": reward_mean, "std": reward_std}

        with open(reward_save_path, "w") as f:
            json.dump(reward_report, f, indent=4)

    episode_length_report: EpisodeLengthReport

    if os.path.exists(episode_length_save_path):
        print(f"Skipping episode length evaluation for {env._tl_spec}...")

    else:
        episode_lengths: list[int] = []
        for model in spec_models:
            episode_lengths += evaluate_episode_lengths_tl(num_episodes, model, env)

        episode_length_report = get_episode_length_report(episode_lengths)

        with open(episode_length_save_path, "w") as f:
            json.dump(episode_length_report, f, indent=4)
