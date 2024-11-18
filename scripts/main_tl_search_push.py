import itertools, json, os, random
import pickle
from math import ceil
import multiprocessing as mp
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import TQC

from tl_search.common.io import spec2title
from tl_search.common.typing import (
    EntropyReportDict,
    EpisodeLengthReport,
    Exclusion,
    MultiStartSearchLog,
    ObsProp,
    SpecNode,
    ValueTable,
)
from tl_search.search.search_push import (
    search_train_evaluate,
    select_max_entropy_spec_replicate,
    get_action_distributions,
    return_input,
)
from tl_search.envs.tl_push import TLBlockedFetchPushEnv
from tl_search.evaluation.ranking import sort_spec
from tl_search.search.neighbor import create_neighbor_masks, initialize_node, spec2node
from tl_search.search.sample import sample_obs

if __name__ == "__main__":
    run: int = 7
    gpu: int = 1  # (run - 1) % 4
    num_samples: int = 5000
    num_start: int = 1
    num_max_search_steps: int = 10
    num_processes: int = 15

    warm_start_mode: Literal["target", "parent", None] = None
    reward_threshold: float = 0.02
    episode_length_sigma: float | None = 2 if warm_start_mode == "target" else None
    kl_div_suffix: str | None = "push_exp1"
    max_extended_steps: int = 3
    expand_search: bool = True
    kl_div_weighted: bool = False

    target_spec: str | None = "F(psi_blk_tar) & G(!psi_obs_moved & !psi_blk_fallen)"

    start_iter: int = 0
    start_specs: list[str | None] = [
        None,
        None,
    ]

    predicates: tuple[str, ...] = (
        "psi_blk_tar",
        "psi_obs_moved",
        "psi_blk_fallen",
    )

    n_envs: Final[int] = 25  # 50  # 20
    total_timesteps: Final[int] = 2_000_000
    num_replicates: Final[int] = 1
    num_episodes: Final[int] = 200
    window: Final[int] = ceil(round(total_timesteps / 100))

    net_arch: list[int] = [512 for _ in range(3)]

    tb_log_dir: str = f"out/logs/push_search/multistart_{kl_div_suffix}_tqc_{run}/"

    policy_kwargs: dict[str, Any] = {
        "net_arch": [512, 512, 512],
        "n_critics": 2,
    }

    tqc_config: dict[str, Any] = {
        "policy": "MultiInputPolicy",
        "buffer_size": int(1e6),
        "batch_size": 2048,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "tau": 0.05,
        "tensorboard_log": tb_log_dir,
        "policy_kwargs": policy_kwargs,
    }

    her_kwargs = {
        "n_sampled_goal": 4,
        "goal_selection_strategy": "future",
        "copy_info_dict": True,
    }

    env_config: dict[str, Any] = {
        "render_mode": "rgb_array",
        "reward_type": "dense",
        "penalty_type": "dense",
        "dense_penalty_coef": 0.01,
        "sparse_penalty_value": 10,
        "max_episode_steps": 100,
    }

    obs_props: list[ObsProp] = [
        ObsProp("d_blk_tar", ["d_blk_tar"], lambda d_blk_tar: d_blk_tar),
        ObsProp("d_obs_moved", ["d_obs_moved"], lambda d_obs_moved: d_obs_moved),
        ObsProp("d_blk_fallen", ["d_blk_fallen"], lambda d_blk_fallen: d_blk_fallen),
    ]

    default_distance_threshold: float = 0.05
    atom_pred_dict: dict[str, str] = {
        "psi_blk_tar": f"d_blk_tar < {default_distance_threshold}",
        "psi_obs_moved": f"d_obs_moved < {default_distance_threshold}",
        "psi_blk_fallen": f"d_blk_fallen > {default_distance_threshold}",
    }

    ref_num_replicates: int = 1

    exclusions: list[Exclusion] = ["group"]

    suffix: str
    match warm_start_mode:
        case "target":
            suffix = "_ws"
        case "parent":
            suffix = "_parent_ws"
        case None:
            suffix = ""

    log_suffix: str = (
        (f"{kl_div_suffix}_extended_" if kl_div_suffix is not None else "extended_")
        + f"filtered_{reward_threshold}_extended_{max_extended_steps}_expanded_{expand_search}_weighted_{kl_div_weighted}_"
    )

    dir_name: str = "push"

    common_dir_path: str = "search/push"
    target_model_path: str = (
        f"out/models/{common_dir_path}/tqc_F(psi_blk_tar)_and_G(!psi_obs_moved_and_!psi_blk_fallen).zip"
    )
    summary_log_path: str = (
        f"out/data/{common_dir_path}/multistart_{log_suffix}_tqc_{run}{suffix}.json"
    )
    log_save_path: str = f"out/data/{common_dir_path}/{log_suffix}tqc{suffix}.json"
    model_save_path: str = f"out/models/{common_dir_path}/tqc{suffix}.zip"
    learning_curve_path: str = (
        f"out/plots/reward_curve/{common_dir_path}/tqc{suffix}.png"
    )
    animation_save_path: str | None = None
    data_save_path: str = f"out/data/kl_div/push/kl_div_tqc{suffix}.json"
    target_actions_path: str = (
        f"out/data/search/dataset/action_probs_tqc_{log_suffix}full.npz"
    )
    target_entropy_path: str = target_actions_path.replace(".npz", "_ent.json")
    obs_list_path: str = f"out/data/search/dataset/obs_list_tqc_full.pkl"
    target_episode_path: str = f"out/data/search/dataset/episode_tqc.json"

    seeds: list[int] = [
        random.randint(0, 10000) for _ in range(num_replicates)
    ]  # not used

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    mp.set_start_method("spawn")

    idx_combs: list[tuple[int, int]] = list(
        itertools.product(range(num_replicates), range(ref_num_replicates))
    )

    init_nodes: list[SpecNode] = [
        initialize_node(predicates, exclusions) for _ in range(num_start)
    ]

    start_spec = start_specs[run - 1]

    if start_spec is not None:
        print(f"Using start spec {start_spec}")
        init_nodes[0] = spec2node(start_spec, dir_name)
    else:
        pass

    sample_env = TLBlockedFetchPushEnv(
        target_spec, obs_props, atom_pred_dict, **env_config
    )

    obs_list: list[dict[str, Any]]

    if os.path.exists(obs_list_path):
        with open(obs_list_path, "rb") as f:
            obs_list = pickle.load(f)
    else:
        model = TQC.load(target_model_path, sample_env)
        obs_list = sample_obs(sample_env, model, num_samples)

        # Save the observations
        with open(obs_list_path, "wb") as f:
            pickle.dump(obs_list, f)

    target_gaus_means_list: list[NDArray]
    target_gaus_stds_list: list[NDArray]
    target_trap_masks: list[NDArray]
    if os.path.exists(target_actions_path):
        print("Loading target actions...")
        npz = np.load(target_actions_path)
        target_gaus_means_list = npz["gaus_means"]
        target_gaus_stds_list = npz["gaus_stds"]
        target_trap_masks = npz["masks"]
    else:
        print("Generating target actions...")
        target_env = TLBlockedFetchPushEnv(
            target_spec, obs_props, atom_pred_dict, **env_config
        )

        target_gaus_means_list = []
        target_gaus_stds_list = []
        target_trap_masks = []
        print("Loading target model...")
        for i in range(num_replicates):
            print(f"Replicate {i}")
            model = TQC.load(target_model_path.replace(".zip", f"_{i}.zip"), target_env)
            action_probs: NDArray
            gaus_means, gaus_stds, trap_mask = get_action_distributions(
                model, target_env, obs_list
            )
            target_gaus_means_list.append(gaus_means)
            target_gaus_stds_list.append(gaus_stds)
            target_trap_masks.append(trap_mask)

            del model

        np.savez(
            target_actions_path,
            gaus_means=target_gaus_means_list,
            gaus_stds=target_gaus_stds_list,
            masks=target_trap_masks,
        )

    if os.path.exists(target_episode_path):
        with open(target_episode_path, "r") as f:
            episode_length_report: EpisodeLengthReport = json.load(f)
    else:
        model = TQC.load(target_model_path.replace(".zip", f"_{0}.zip"), sample_env)
        _, episode_lengths = evaluate_policy(
            model, sample_env, n_eval_episodes=num_episodes, return_episode_rewards=True
        )
        episode_length_report: EpisodeLengthReport = {
            "mean": float(np.mean(episode_lengths)),
            "std": float(np.std(episode_lengths)),
            "max": float(np.max(episode_lengths)),
            "min": float(np.min(episode_lengths)),
        }
        with open(target_episode_path, "w") as f:
            json.dump(episode_length_report, f, indent=4)

    if os.path.exists(target_entropy_path):
        with open(target_entropy_path, "r") as f:
            target_entropy_dict: EntropyReportDict = json.load(f)

        target_max_entropy_idx: int = target_entropy_dict["max_entropy_idx"]
    else:
        (
            target_max_entropy_idx,
            target_entropies,
            num_non_trap_states,
        ) = select_max_entropy_spec_replicate(target_gaus_stds_list, target_trap_masks)

        target_entropy_dict: EntropyReportDict = {
            "max_entropy_idx": target_max_entropy_idx,
            "entropies": target_entropies,
            "num_non_trap_states": num_non_trap_states,
        }

        with open(target_entropy_path, "w") as f:
            json.dump(target_entropy_dict, f, indent=4)

    target_gaus_means_list = [target_gaus_means_list[target_max_entropy_idx]]
    target_gaus_stds_list = [target_gaus_stds_list[target_max_entropy_idx]]
    target_trap_masks = [target_trap_masks[target_max_entropy_idx]]

    neighbor_masks: tuple[ValueTable, ...] = create_neighbor_masks(
        len(predicates), exclusions
    )

    local_optimum_nodes: list[SpecNode] = []
    local_optimum_specs: list[str] = []
    local_optimum_kl_divs: list[float] = []
    searches_traces: list[tuple[list[str], list[float]]] = []
    nums_searched_specs: list[int] = []
    searched_specs_all: list[str] = []

    for i, init_node in enumerate(init_nodes):
        node_trace, spec_trace, metrics_trace, searched_specs = search_train_evaluate(
            init_node,
            num_max_search_steps,
            run,
            i,
            neighbor_masks,
            log_save_path,
            num_processes,
            num_replicates,
            seeds,
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
            obs_props,
            atom_pred_dict,
            tqc_config,
            her_kwargs,
            env_config,
            search_start_iter=start_iter,
            episode_length_report=episode_length_report,
            reward_threshold=reward_threshold,
            episode_length_sigma=episode_length_sigma,
            warm_start_path=None,
            kl_div_suffix=kl_div_suffix,
            max_extended_steps=max_extended_steps,
            expand_search=expand_search,
        )
        local_optimum_nodes.append(node_trace[-1])
        local_optimum_specs.append(spec_trace[-1])
        local_optimum_kl_divs.append(metrics_trace[-1])
        searches_traces.append((spec_trace, metrics_trace))
        nums_searched_specs.append(len(searched_specs))
        searched_specs_all += searched_specs

    sorted_specs, sorted_min_kl_divs = sort_spec(
        local_optimum_specs, local_optimum_kl_divs
    )

    multistart_log: MultiStartSearchLog = {
        "run": run,
        "sorted_local_optimal_specs": sorted_specs,
        "sorted_local_optimal_kl_divs": sorted_min_kl_divs,
        "total_searched_specs": int(len(set(searched_specs_all))),
        "nums_searched_specs": nums_searched_specs,
        "searches_traces": searches_traces,
    }

    with open(summary_log_path, "w") as f:
        json.dump(multistart_log, f, indent=4)
