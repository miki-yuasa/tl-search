import itertools, json, os, random
import pickle
from math import ceil
import multiprocessing as mp
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

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
from tl_search.search.search_parking import (
    search_train_evaluate,
    select_max_entropy_spec_replicate,
    get_action_distributions,
    return_input,
)
from tl_search.envs.tl_parking import TLAdversarialParkingEnv
from tl_search.evaluation.ranking import sort_spec
from tl_search.search.neighbor import create_neighbor_masks, initialize_node, spec2node
from tl_search.search.sample import sample_obs

if __name__ == "__main__":
    run: int = 8
    gpu: int = (run - 1) % 4
    num_samples: int = 5000
    num_start: int = 1
    num_max_search_steps: int = 10
    num_processes: int = 15

    warm_start_mode: Literal["target", "parent", None] = None
    reward_threshold: float = 0.02
    episode_length_sigma: float | None = 2 if warm_start_mode == "target" else None
    kl_div_suffix: str | None = "parking_exp1"

    target_spec: str | None = "F(psi_ego_goal) & G(!psi_ego_adv & !psi_ego_wall)"

    start_iter: int = 0
    start_specs: list[str | None] = [
        "F(!psi_ego_goal) & G(psi_ego_adv|psi_ego_wall)",
        "F(!psi_ego_adv&!psi_ego_wall) & G(psi_ego_goal)",
        "F(psi_ego_wall) & G(!psi_ego_goal|psi_ego_adv)",
        "F(psi_ego_adv|psi_ego_wall) & G(!psi_ego_goal)",
        "F(!psi_ego_goal) & G(!psi_ego_adv&!psi_ego_wall)",
        "F(psi_ego_wall) & G(psi_ego_goal&!psi_ego_adv)",
        "F(!psi_ego_goal) & G(psi_ego_adv|psi_ego_wall)",
        "F(!psi_ego_wall) & G(!psi_ego_goal&psi_ego_adv)",
        None,
        None,
    ]

    predicates: tuple[str, ...] = (
        "psi_ego_goal",
        "psi_ego_adv",
        "psi_ego_wall",
    )

    n_envs: Final[int] = 25  # 50  # 20
    total_timesteps: Final[int] = 100_000
    num_replicates: Final[int] = 1
    num_episodes: Final[int] = 200
    window: Final[int] = ceil(round(total_timesteps / 100))

    net_arch: list[int] = [512 for _ in range(3)]

    tb_log_dir: str = f"out/logs/parking_search/multistart_{kl_div_suffix}_sac_{run}/"

    sac_kwargs: dict[str, Any] = {
        "tensorboard_log": tb_log_dir,
        "buffer_size": int(1e6),
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "batch_size": 1024,
        "tau": 0.05,
        "policy_kwargs": dict(net_arch=net_arch),
        "learning_starts": 1000,
    }

    her_kwargs = dict(
        n_sampled_goal=4, goal_selection_strategy="future", copy_info_dict=True
    )

    config = {
        "observation": {
            "type": "KinematicsGoal",
            "features": ["x", "y", "vx", "vy", "cos_h", "sin_h"],  # "heading"],
            "scales": [1, 1, 5, 5, 1, 1],
            "normalize": False,
        },
        "action": {"type": "ContinuousAction"},
        "reward_weights": [1, 0.2, 0, 0, 0.1, 0.1],
        "success_goal_reward": 0.05,
        "collision_reward": -5,
        "steering_range": np.deg2rad(45),
        "simulation_frequency": 15,
        "policy_frequency": 5,
        "duration": 50,
        "screen_width": 600,
        "screen_height": 300,
        "screen_center": "centering_position",
        "centering_position": [0.5, 0.5],
        "scaling": 7,
        "controlled_vehicles": 1,
        "vehicles_count": 0,
        "adversarial_vehicle": True,
        "add_walls": True,
        "adversarial_vehicle_spawn_config": [
            {"spawn_point": [-30, 4], "heading": 0, "speed": 5},
            {"spawn_point": [-30, -4], "heading": 0, "speed": 5},
            {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
            {"spawn_point": [30, -4], "heading": np.pi, "speed": 5},
        ],
        "dense_reward": True,
    }

    obs_props: list[ObsProp] = [
        ObsProp("d_ego_goal", ["d_ego_goal"], return_input),
        ObsProp("d_ego_adv", ["d_ego_adv"], return_input),
        ObsProp("d_ego_wall", ["d_ego_wall"], return_input),
    ]

    atom_pred_dict: dict[str, str] = {
        "psi_ego_goal": "d_ego_goal < {}".format(2),
        "psi_ego_adv": "d_ego_adv < {}".format(4),
        "psi_ego_wall": "d_ego_wall < {}".format(5),
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
        f"{kl_div_suffix}_extended_" if kl_div_suffix is not None else "extended_"
    )

    dir_name: str = "parking"

    target_model_path: str = (
        f"out/models/search/parking/sac_F(psi_ego_goal)_and_G(!psi_ego_adv_and_!psi_ego_wall).zip"
    )
    summary_log_path: str = (
        f"out/data/search/parking/multistart_{log_suffix}_sac_{run}{suffix}.json"
    )
    log_save_path: str = f"out/data/search/parking/{log_suffix}sac{suffix}.json"
    model_save_path: str = f"out/models/search/parking/sac{suffix}.zip"
    learning_curve_path: str = f"out/plots/reward_curve/search/parking/sac{suffix}.png"
    animation_save_path: str | None = None
    data_save_path: str = f"out/data/kl_div/parking/kl_div_sac{suffix}.json"
    target_actions_path: str = (
        f"out/data/search/dataset/action_probs_sac_{log_suffix}full.npz"
    )
    target_entropy_path: str = target_actions_path.replace(".npz", "_ent.json")
    obs_list_path: str = f"out/data/search/dataset/obs_list_sac_full.pkl"
    target_episode_path: str = f"out/data/search/dataset/episode_sac.json"

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

    sample_env = TLAdversarialParkingEnv(target_spec, config, obs_props, atom_pred_dict)

    obs_list: list[dict[str, Any]]

    if os.path.exists(obs_list_path):
        with open(obs_list_path, "rb") as f:
            obs_list = pickle.load(f)
    else:
        model = SAC.load(target_model_path, sample_env)
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
        target_env = TLAdversarialParkingEnv(
            target_spec, config, obs_props, atom_pred_dict
        )

        target_gaus_means_list = []
        target_gaus_stds_list = []
        target_trap_masks = []
        print("Loading target model...")
        for i in range(num_replicates):
            print(f"Replicate {i}")
            model = SAC.load(target_model_path.replace(".zip", f"_{i}.zip"), target_env)
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
        model = SAC.load(target_model_path.replace(".zip", f"_{0}.zip"), sample_env)
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
            sac_kwargs,
            her_kwargs,
            config,
            search_start_iter=start_iter,
            episode_length_report=episode_length_report,
            reward_threshold=reward_threshold,
            episode_length_sigma=episode_length_sigma,
            warm_start_path=None,
            kl_div_suffix=kl_div_suffix,
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
