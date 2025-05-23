import json, os, pickle
from multiprocessing import Pool
import random
from typing import Any, Literal, overload

import numpy as np
from numpy.typing import NDArray
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution

from tl_search.common.io import spec2title
from tl_search.common.typing import (
    EntropyReportDict,
    EpisodeLengthReport,
    FilteredLog,
    KLDivReportDict,
    ObsProp,
    RewardReportDict,
    SearchLog,
    SortedLog,
    SpecNode,
    ValueTable,
)
from tl_search.common.utils import find_min_kl_div_tl_spec
from tl_search.envs.tl_safety_builder import CustomBuilder
from tl_search.envs.tl_safety_tasks import atom_tl_ob2rob
from tl_search.evaluation.count import get_episode_length_report
from tl_search.evaluation.eval import collect_kl_div_stats
from tl_search.evaluation.filter import apply_filter
from tl_search.evaluation.ranking import sort_spec
from tl_search.search.neighbor import (
    find_additional_neighbors,
    find_neighbor_nodes_specs,
    node2spec,
    nodes2specs,
)
from tl_search.tl.synthesis import TLAutomaton
from tl_search.tl.transition import find_trap_transitions
from tl_search.train.tl_train_goal import train_replicate_tl_agent


def search_train_evaluate(
    init_node: SpecNode,
    num_max_search_steps: int,
    run: int,
    start_idx: int,
    neighbor_masks: tuple[ValueTable, ...],
    log_save_path: str,
    num_processes: int,
    num_replicates: list[str],
    seeds: list[int],
    total_timesteps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    target_gaus_means_list: list[NDArray[np.float64]],
    target_gaus_stds_list: list[NDArray[np.float64]],
    target_trap_masks: list[NDArray],
    obs_list: list[dict[str, Any]],
    data_save_path: str,
    algo_kwargs: dict[str, Any],
    env_config: dict[str, Any] | None = None,
    search_start_iter: int = 0,
    episode_length_report: EpisodeLengthReport | None = None,
    reward_threshold: float = 0.0,
    episode_length_sigma: float | None = 1,
    warm_start_mode: Literal["target", "parent", None] = None,
    warm_start_path: str | None = None,
    kl_div_suffix: str | None = None,
    max_extended_steps: int = 3,
    expand_search: bool = True,
    continue_from_checkpoint: bool = False,
    suboptimal_ckpt_timestep: int | None = None,
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
            node_path: str = f"out/nodes/goal/{spec2title(s)}.pkl"

            if os.path.exists(node_path):
                pass
            else:
                os.makedirs(os.path.dirname(node_path), exist_ok=True)
                with open(node_path, "wb") as f:
                    pickle.dump(n, f)

        nodes_save_path: str = log_save_path.replace(".json", "_nodes.pkl")

        os.makedirs(os.path.dirname(nodes_save_path), exist_ok=True)
        with open(nodes_save_path, "wb") as f:
            pickle.dump(neighbor_nodes, f)

        search_log: SearchLog = {
            "run": run_label,
            "enemy_policy": "none",
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

        os.makedirs(os.path.dirname(search_log_path), exist_ok=True)
        with open(search_log_path, "w") as f:
            json.dump(search_log, f, indent=4)

        match warm_start_mode:
            case "target":
                assert warm_start_path is not None
            case "parent":
                warm_start_path = model_save_path.replace(
                    ".zip", f"_{spec2title(node2spec(node))}.zip"
                )

        kl_div_means, reward_means, mean_episode_lengths = train_evaluate_multiprocess(
            num_processes,
            num_replicates,
            total_timesteps,
            model_save_path,
            learning_curve_path,
            animation_save_path,
            device,
            window,
            target_gaus_means_list,
            target_gaus_stds_list,
            target_trap_masks,
            obs_list,
            data_save_path,
            neighbor_specs,
            algo_kwargs,
            env_config,
            warm_start_path,
            kl_div_suffix,
            continue_from_checkpoint=continue_from_checkpoint,
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
            if expand_search:
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
                    total_timesteps,
                    model_save_path,
                    learning_curve_path,
                    animation_save_path,
                    device,
                    window,
                    target_gaus_means_list,
                    target_gaus_stds_list,
                    target_trap_masks,
                    obs_list,
                    data_save_path,
                    additional_neighbor_specs,
                    algo_kwargs,
                    env_config,
                    warm_start_path,
                    kl_div_suffix,
                    continue_from_checkpoint=continue_from_checkpoint,
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
                print("The min KL-divergence spec is the same as the current spec.")
                print("Not expanding the search.")

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

        else:
            print("Next spec: " + min_kl_div_spec)

        min_kl_div_node: SpecNode = filtered_nodes[min_kl_div_idx]

        max_reward_mean: float = float(np.max(filtered_rewards))
        mean_reward_mean: float = float(np.mean(filtered_rewards))
        min_reward_mean: float = float(np.min(filtered_rewards))
        max_reward_spec: str = neighbor_specs[np.argmax(filtered_rewards)]

        search_log = {
            "run": run_label,
            "enemy_policy": "none",
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

        os.makedirs(os.path.dirname(search_log_path), exist_ok=True)
        with open(search_log_path, "w") as f:
            json.dump(search_log, f, indent=4)

        os.makedirs(os.path.dirname(pos_reward_log_path), exist_ok=True)
        with open(pos_reward_log_path, "w") as f:
            json.dump(filtered_log, f, indent=4)

        os.makedirs(os.path.dirname(sorted_log_path), exist_ok=True)
        with open(sorted_log_path, "w") as f:
            json.dump(sorted_log, f, indent=4)

        os.makedirs(os.path.dirname(nodes_save_path), exist_ok=True)
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


def train_evaluate_multiprocess(
    num_processes: int,
    num_replicates: list[str],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    target_gaus_means_list: list[NDArray[np.float64]],
    target_gaus_stds_list: list[NDArray[np.float64]],
    target_trap_masks: list[NDArray],
    obs_list: list[dict[str, Any]],
    data_save_path: str,
    tl_specs: list[str],
    algo_kwargs: dict[str, Any],
    env_config: dict[str, Any] | None = None,
    warm_start_path: str | None = None,
    kl_div_suffix: str | None = None,
    kl_div_weighted: bool = True,
    continue_from_checkpoint: bool = False,
) -> tuple[list[float], list[float], list[float]]:
    inputs = [
        (
            num_replicates,
            total_time_steps,
            model_save_path,
            learning_curve_path,
            animation_save_path,
            device,
            window,
            target_gaus_means_list,
            target_gaus_stds_list,
            target_trap_masks,
            obs_list,
            data_save_path,
            tl_spec,
            algo_kwargs,
            env_config,
            warm_start_path,
            kl_div_suffix,
            kl_div_weighted,
            continue_from_checkpoint,
        )
        for tl_spec in tl_specs
    ]

    with Pool(num_processes) as p:
        results = p.starmap(train_evaluate, inputs)

    p.close()
    p.join()

    kl_div_means: list[float] = []
    reward_means: list[float] = []
    episode_lengths: list[float] = []

    for kl_div_report, reward_mean, episode_length in results:
        kl_div_means.append(kl_div_report["kl_div_mean"])
        reward_means.append(reward_mean)
        episode_lengths.append(episode_length)

    return kl_div_means, reward_means, episode_lengths


def train_evaluate(
    num_replicates: list[str],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    window: int,
    target_gaus_means_list: list[NDArray[np.float64]],
    target_gaus_stds_list: list[NDArray[np.float64]],
    target_trap_masks: list[NDArray],
    obs_list: list[dict[str, Any]],
    data_save_path: str,
    tl_spec: str,
    algo_kwargs: dict[str, Any],
    env_config: dict[str, Any] | None = None,
    warm_start_path: str | None = None,
    kl_div_suffix: str | None = None,
    kl_div_weighted: bool = True,
    continue_from_checkpoint: bool = False,
) -> tuple[KLDivReportDict, float, float]:

    env_config["config"]["tl_spec"] = tl_spec
    env = CustomBuilder(**env_config)

    seeds_tmp: list[int] = [random.randint(0, 10000) for _ in num_replicates]

    # try:
    spec_models: list[PPO] = train_replicate_tl_agent(
        num_replicates,
        env,
        seeds_tmp,
        total_time_steps,
        model_save_path.replace(".zip", f"_{spec2title(tl_spec)}.zip"),
        learning_curve_path.replace(".png", f"_{spec2title(tl_spec)}.png"),
        (
            animation_save_path.replace(".gif", f"_{spec2title(tl_spec)}.gif")
            if animation_save_path is not None
            else None
        ),
        device,
        algo_kwargs,
        window,
        warm_start_path,
        continue_from_checkpoint=continue_from_checkpoint,
    )

    kl_div_report, reward_mean, episode_length = evaluate_models(
        env,
        device,
        target_gaus_means_list,
        target_gaus_stds_list,
        target_trap_masks,
        spec_models,
        obs_list,
        data_save_path.replace(".json", f"_{spec2title(tl_spec)}.json"),
        kl_div_suffix=kl_div_suffix,
        kl_div_weighted=kl_div_weighted,
    )

    # except:
    #     kl_div_report: KLDivReportDict = {
    #         "kl_div_mean": 999,
    #         "kl_div_std": 0,
    #         "kl_div_max": 999,
    #         "kl_div_min": 999,
    #         "kl_div_median": 999,
    #         "kl_ci": (999, 999),
    #     }
    #     reward_mean: float = 0
    #     episode_length: float = 0

    return kl_div_report, reward_mean, episode_length


def evaluate_models(
    env: CustomBuilder,
    device: torch.device | str,
    target_gaus_means_list: list[NDArray[np.float64]],
    target_gaus_stds_list: list[NDArray[np.float64]],
    target_trap_masks: list[NDArray],
    spec_models: list[PPO],
    obs_list: list[dict[str, Any]],
    data_save_path: str,
    num_episodes: int = 100,
    kl_div_suffix: str | None = None,
    kl_div_weighted: bool = True,
) -> tuple[KLDivReportDict, float, float]:
    print(f"Evaluating models for {env.task.tl_spec}...")

    kl_div_save_path: str = data_save_path.replace(
        ".json",
        "_kl_div.json" if kl_div_suffix is None else f"_kl_div_{kl_div_suffix}.json",
    )
    # if kl_div_suffix is not None:
    #     kl_div_save_path.replace("/kl_div", "/kl_div/kl_div")
    # else:
    #     pass
    reward_save_path: str = data_save_path.replace(".json", "_reward.json")
    episode_length_save_path: str = data_save_path.replace(".json", "_episode.json")

    entropy_save_path: str = data_save_path.replace(".json", "_entropy.json")

    kl_div_report: KLDivReportDict
    reward_mean: float

    if os.path.exists(reward_save_path) and os.path.exists(episode_length_save_path):
        print(f"Loading saved reward and episode length for {env.task.tl_spec}...")
        with open(reward_save_path, "r") as f:
            reward_report: RewardReportDict = json.load(f)
        reward_mean = reward_report["mean"]

        with open(episode_length_save_path, "r") as f:
            episode_length_report = json.load(f)

    else:
        model_rewards: list[float] = []
        episode_lengths: list[int] = []

        print(f"Evaluating rewards and episode lengths for {env.task.tl_spec}...")

        for i, model in enumerate(spec_models):
            rewards: list[float]
            rewards, lengths = evaluate_policy(model, env, num_episodes)

            model_rewards += rewards
            episode_lengths += lengths

            print(f"Replicate {i + 1}/{len(spec_models)}")

        reward_mean = float(np.mean(model_rewards))
        reward_std: float = float(np.std(model_rewards))
        reward_report: RewardReportDict = {"mean": reward_mean, "std": reward_std}

        print(f"Saved reward and episode length for {env.task.tl_spec}...")
        print(f"- path: {reward_save_path}")
        os.makedirs(os.path.dirname(reward_save_path), exist_ok=True)
        with open(reward_save_path, "w") as f:
            json.dump(reward_report, f, indent=4)

        episode_length_report = get_episode_length_report(episode_lengths)

        print(f"Saved episode length for {env.task.tl_spec}...")
        print(f"- path: {episode_length_save_path}")
        os.makedirs(os.path.dirname(episode_length_save_path), exist_ok=True)
        with open(episode_length_save_path, "w") as f:
            json.dump(episode_length_report, f, indent=4)

    mean_episode_length = episode_length_report["mean"]

    if os.path.exists(kl_div_save_path) and os.path.exists(entropy_save_path):
        print(f"Loading saved reward and KL div for {env.task.tl_spec}...")
        with open(kl_div_save_path, "r") as f:
            kl_div_report = json.load(f)

    else:
        rep_gaus_means_list: list[NDArray[np.float64]] = []
        rep_gaus_stds_list: list[NDArray[np.float64]] = []
        rep_trap_masks: list[NDArray[np.float64]] = []

        for i, model in enumerate(spec_models):
            print(
                f"Getting action distributions for {env.task.tl_spec} for rep {i + 1}/{len(spec_models)}..."
            )

            rep_gaus_means, rep_gaus_stds, rep_trap_mask = get_action_distributions(
                model, env, obs_list
            )
            rep_gaus_means_list.append(rep_gaus_means)
            rep_gaus_stds_list.append(rep_gaus_stds)
            rep_trap_masks.append(rep_trap_mask)

        (
            max_entropy_idx,
            entropies,
            num_non_trap_states,
        ) = select_max_entropy_spec_replicate(rep_gaus_stds_list, rep_trap_masks)

        entropy_report: EntropyReportDict = {
            "max_entropy_idx": max_entropy_idx,
            "entropies": entropies,
            "num_non_trap_states": num_non_trap_states,
        }

        os.makedirs(os.path.dirname(entropy_save_path), exist_ok=True)
        with open(entropy_save_path, "w") as f:
            json.dump(entropy_report, f, indent=4)

        model_kl_divs: list[NDArray] = []

        print(f"Getting KL divergences for {env.task.tl_spec}...")

        max_kl_div: float = gaussian_dist_entropy(10)

        # for comb in idx_combs:
        target_idx = 0
        spec_idx = max_entropy_idx

        print(f"Calculating KL divergence for {env.task.tl_spec}...")
        trap_mask: NDArray = target_trap_masks[target_idx] * rep_trap_masks[spec_idx]
        target_gaus_means_filtered: NDArray = target_gaus_means_list[target_idx][
            trap_mask == 1
        ]
        target_gaus_stds_filtered: NDArray = target_gaus_stds_list[target_idx][
            trap_mask == 1
        ]
        spec_gaus_means_filtered: NDArray = rep_gaus_means_list[spec_idx][
            trap_mask == 1
        ]
        spec_gaus_stds_filtered: NDArray = rep_gaus_stds_list[spec_idx][trap_mask == 1]

        print(f"Calculating entropy for {env.task.tl_spec}...")
        raw_kl_div: NDArray[np.float64] = np.mean(
            gaussian_kl_div(
                target_gaus_means_filtered,
                target_gaus_stds_filtered,
                spec_gaus_means_filtered,
                spec_gaus_stds_filtered,
            ),
            axis=1,
        )
        if kl_div_weighted:
            print("Calculating weighted KL divergence...")
            target_entropy: NDArray = np.mean(
                gaussian_dist_entropy(target_gaus_stds_filtered), axis=1
            )
            normalized_entropy: NDArray = 1 - target_entropy / max_kl_div
            weight: NDArray = normalized_entropy / np.sum(normalized_entropy)
        else:
            print("Calculating unweighted KL divergence...")
            weight: NDArray = np.ones_like(raw_kl_div)

        print(f"Calculating weighted KL divergence for {env.task.tl_spec}...")
        kl_divs: NDArray = raw_kl_div * weight

        model_kl_divs.append(kl_divs.flatten())

        model_kl_divs_concat: NDArray = np.concatenate(model_kl_divs)

        kl_div_report: KLDivReportDict = (
            {
                "kl_div_mean": 999,
                "kl_div_std": 0,
                "kl_div_max": 999,
                "kl_div_min": 999,
                "kl_div_median": 999,
                "kl_ci": (0, 0),
            }
            if model_kl_divs_concat.size == 0
            else collect_kl_div_stats(model_kl_divs_concat, in_dict=True)
        )

        print(f"Saved KL divergence for {env.task.tl_spec}...")
        os.makedirs(os.path.dirname(kl_div_save_path), exist_ok=True)
        with open(kl_div_save_path, "w") as f:
            json.dump(kl_div_report, f, indent=4)

    return kl_div_report, reward_mean, mean_episode_length


def get_action_distributions(
    model: PPO,
    env: CustomBuilder,
    obs_list: list[dict[str, Any]],
) -> tuple[NDArray, NDArray, NDArray]:
    aut: TLAutomaton | None = env.task.aut if hasattr(env.task, "aut") else None
    gaus_means: list[NDArray] = []
    gaus_stds: list[NDArray] = []
    trap_mask: list[int] = []
    for i, obs in enumerate(obs_list):
        model.policy.set_training_mode(False)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        distribution: DiagGaussianDistribution = model.policy.get_distribution(
            obs_tensor
        )
        mean_actions = distribution.distribution.mean
        action_std = distribution.distribution.stddev
        gaus_mean = mean_actions.cpu().detach().numpy()
        gaus_std = action_std.cpu().detach().numpy()
        gaus_means.append(gaus_mean)
        gaus_stds.append(gaus_std)

        if aut is not None:
            kin_dict = obs2kin_dict(obs)
            atom_rob_dict, _ = atom_tl_ob2rob(aut, kin_dict)
            trap_mask.append(0 if find_trap_transitions(atom_rob_dict, aut) else 1)
        else:
            # If no automaton is provided, assume all states are non-trap states
            trap_mask.append(1)

    env.close()

    return np.array(gaus_means), np.array(gaus_stds), np.array(trap_mask)


def obs2kin_dict(obs: NDArray[np.float64]) -> dict[str, np.float64]:
    """
    Observation of size (72,) is flattened version of the following dict structure:
    {
        "accelerometer": NDArray[np.float64] (3,),
        "velocimeter": NDArray[np.float64] (3,),
        "gyro": NDArray[np.float64] (3,),
        "magnetometer": NDArray[np.float64] (3,),
        "ballangvel_rear": NDArray[np.float64] (3,),
        "ballquat_rear": NDArray[np.float64] (3,3),
        "goal_lidar": NDArray[np.float64] (16,),
        "hazards_lidar": NDArray[np.float64] (16,),
        "vase_lidar": NDArray[np.float64] (16,),
    }
    """
    obs_dict: dict[str, NDArray[np.float64]] = {
        "accelerometer": obs[:3],
        "velocimeter": obs[3:6],
        "gyro": obs[6:9],
        "magnetometer": obs[9:12],
        "ballangvel_rear": obs[12:15],
        "ballquat_rear": obs[15:24].reshape(3, 3),
        "goal_lidar": obs[24:40],
        "hazards_lidar": obs[40:56],
        "vase_lidar": obs[56:72],
    }

    lidar_max_dist: float = 3.0

    kin_dict: dict[str, np.float64] = {
        "d_gl": lidar2dist(np.max(obs_dict["goal_lidar"]), lidar_max_dist),
        "d_hz": lidar2dist(np.max(obs_dict["hazards_lidar"]), lidar_max_dist),
        "d_vs": lidar2dist(np.max(obs_dict["vase_lidar"]), lidar_max_dist),
    }

    return kin_dict


def lidar2dist(lidar_read: float, lidar_max_dist: float) -> float:
    if lidar_read == 0:
        return lidar_max_dist
    else:
        return (1 - lidar_read) * lidar_max_dist


def select_max_entropy_spec_replicate(
    rep_gaus_stds_list: list[NDArray[np.float64]],
    rep_trap_masks: list[NDArray],
) -> tuple[int, list[float], list[int]]:
    max_entropy: float = 0
    max_entropy_idx: int = 0
    entropies: list[float] = []

    for idx, (rep_gaus_stds, rep_trap_mask) in enumerate(
        zip(rep_gaus_stds_list, rep_trap_masks)
    ):
        # Gaussian entropy
        rep_entropy: float = float(
            np.sum(gaussian_dist_entropy(rep_gaus_stds[rep_trap_mask.astype(np.int32)]))
        )
        entropies.append(rep_entropy)
        if rep_entropy > max_entropy:
            max_entropy = rep_entropy
            max_entropy_idx = idx

    num_non_trap_states: list[int] = []
    for rep_trap_mask in rep_trap_masks:
        num_non_trap_states.append(int(np.sum(rep_trap_mask == 1)))

    return max_entropy_idx, entropies, num_non_trap_states


@overload
def gaussian_dist_entropy(
    std: NDArray[np.float64],
) -> NDArray[np.float64]: ...
@overload
def gaussian_dist_entropy(
    std: float,
) -> float: ...


def gaussian_dist_entropy(
    std: NDArray[np.float64] | float,
) -> NDArray[np.float64] | float:
    return 1 / 2 * (1 + np.log(2 * np.pi * np.square(std)))


def gaussian_kl_div(
    mean1: NDArray[np.float64],
    std1: NDArray[np.float64],
    mean2: NDArray[np.float64],
    std2: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (
        np.log(std2 / std1)
        + (np.square(std1) + np.square(mean1 - mean2)) / (2 * np.square(std2))
        - 1 / 2
    )


def return_input(x: Any) -> Any:
    return x


def evaluate_policy(
    model: PPO, env: CustomBuilder, num_eval_episodes: int
) -> tuple[list[float], list[int]]:
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []

    for i in range(num_eval_episodes):
        terminated: bool = False
        truncated: bool = False
        obs, _ = env.reset()
        episode_reward: float = 0
        episode_length: int = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

        print(f"Episode {i + 1}/{num_eval_episodes}, Steps {episode_length}")
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    return episode_rewards, episode_lengths


def train_exh_mp(
    gpu: int,
    num_process: int,
    num_replicates: list[str],
    n_envs: int,
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    algo_kwargs: dict[str, Any],
    window: int,
    tl_specs: list[str],
    env_config: dict[str, Any] = {},
    warm_start_path: str | None = None,
    continue_from_checkpoint: bool = False,
) -> None:
    input = [
        (
            num_replicates,
            seeds,
            total_time_steps,
            model_save_path,
            learning_curve_path,
            animation_save_path,
            torch.device(f"cuda:{gpu}"),
            algo_kwargs,
            window,
            tl_spec,
            env_config,
            warm_start_path,
            continue_from_checkpoint,
        )
        for tl_spec in tl_specs
    ]

    with Pool(num_process) as p:
        p.starmap(train_exh, input)

    p.close()
    p.join()


def train_exh(
    num_replicates: list[str],
    seeds: list[int],
    total_time_steps: int,
    model_save_path: str,
    learning_curve_path: str,
    animation_save_path: str | None,
    device: torch.device | str,
    algo_kwargs: dict[str, Any],
    window: int,
    tl_spec: str,
    env_config: dict[str, Any] = {},
    warm_start_path: str | None = None,
    continue_from_checkpoint: bool = False,
) -> None:
    env_config["config"]["tl_spec"] = tl_spec
    env = CustomBuilder(**env_config)

    warm_start_suffix: str = "_ws" if warm_start_path is not None else ""

    # try:
    spec_models: list[PPO] = train_replicate_tl_agent(
        num_replicates,
        env,
        seeds,
        total_time_steps,
        model_save_path.replace(
            ".zip", f"_{spec2title(tl_spec)}{warm_start_suffix}.zip"
        ),
        learning_curve_path.replace(
            ".png", f"_{spec2title(tl_spec)}{warm_start_suffix}.png"
        ),
        (
            animation_save_path.replace(
                ".gif", f"_{spec2title(tl_spec)}{warm_start_suffix}.gif"
            )
            if animation_save_path is not None
            else None
        ),
        device,
        algo_kwargs,
        window,
        warm_start_path,
        continue_from_checkpoint=continue_from_checkpoint,
    )

    # except:
    #     print(f"Failed to train {tl_spec}.")


def sample_obs(
    env: CustomBuilder,
    model: PPO,
    num_samples: int,
    population_scale: float = 1.5,
) -> list[dict[str, Any]]:
    obs_list: list[dict[str, Any]] = []

    num_collected: int = round(num_samples * population_scale)
    sample_counter: int = 0

    while sample_counter < num_collected:
        obs, _ = env.reset()
        while True:
            action = model.predict(obs, deterministic=True)[0]
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                if env._is_success():
                    obs_list.append(obs)
                    sample_counter += 1
                else:
                    pass
                break
            else:
                obs_list.append(obs)
                sample_counter += 1

    obs_list_sampled: list[dict[str, Any]] = random.sample(obs_list, num_samples)

    return obs_list_sampled
