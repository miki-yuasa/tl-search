import itertools, json, os, random
import pickle
from math import ceil
import multiprocessing as mp
from typing import Any, Final, Literal

import numpy as np
from numpy.typing import NDArray
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO

from tl_search.common.io import spec2title
from tl_search.common.typing import (
    EntropyReportDict,
    EpisodeLengthReport,
    Exclusion,
    MultiStartSearchLog,
    SpecNode,
    ValueTable,
)
from tl_search.search.search_goal import (
    search_train_evaluate,
    select_max_entropy_spec_replicate,
    get_action_distributions,
    sample_obs,
)
from tl_search.envs.tl_safety_builder import CustomBuilder
from tl_search.evaluation.ranking import sort_spec
from tl_search.search.neighbor import create_neighbor_masks, initialize_node, spec2node

if __name__ == "__main__":
    run: int = 8
    gpu: int = (run - 1) % 4
    num_samples: int = 5000
    num_start: int = 1
    num_max_search_steps: int = 10
    num_processes: int = 15

    warm_start_mode: Literal["target", "parent", None] = None
    continue_from_checkpoint: bool = True
    reward_threshold: float = -4
    episode_length_sigma: float | None = 2 if warm_start_mode == "target" else None
    kl_div_suffix: str | None = "goal_exp2_nr"
    max_extended_steps: int = 3
    expand_search: bool = True
    kl_div_weighted: bool = False

    target_spec: str | Literal["normal_reward"] = (
        "normal_reward"  # "F(psi_gl) & G(!psi_hz & !psi_vs)"
    )

    start_iter: int = 0
    start_specs: list[str | None] = [
        "F(psi_gl)&G(!psi_hz|!psi_vs)",
        "F(psi_vs) & G(psi_gl|!psi_hz)",
        "F(psi_gl|psi_hz) & G(psi_vs)",
        "F(!psi_gl) & G(!psi_hz&psi_vs)",
        "F(psi_vs) & G(!psi_gl&!psi_hz)",
        "F(!psi_gl&!psi_hz) & G(psi_vs)",
        "F(psi_gl) & G(!psi_hz&!psi_vs)",
        "F(psi_gl&!psi_hz) & G(!psi_vs)",
        None,
        None,
    ]

    predicates: tuple[str, ...] = (
        "psi_gl",
        "psi_hz",
        "psi_vs",
    )

    n_envs: Final[int] = 25  # 50  # 20
    total_timesteps: Final[int] = 600_000
    num_replicates: Final[list[str]] = ["0"]
    num_episodes: Final[int] = 200
    window: Final[int] = ceil(round(total_timesteps / 100))

    net_arch: list[int] = [512 for _ in range(3)]

    tb_log_dir: str = f"out/logs/goal_search/multistart_{kl_div_suffix}_ppo_{run}/"

    policy_kwargs: dict[str, Any] = {
        "net_arch": [512, 512],
    }

    ppo_config: dict[str, Any] = {
        "policy": "MlpPolicy",
        "tensorboard_log": tb_log_dir,
        "policy_kwargs": policy_kwargs,
    }

    env_config: dict[str, Any] = {
        "config": {
            "agent_name": "Car",
        },
        "task_id": "SafetyCarTLGoal1-v0",
        "render_mode": "rgb_array",
        "max_episode_steps": 500,
        "width": 512,
        "height": 512,
        "camera_name": "fixedfar",
        "ignore_cost": True,
    }

    sample_env_config: dict[str, Any] = {
        "config": {
            "agent_name": "Car",
        },
        "render_mode": "rgb_array",
        "max_episode_steps": 500,
        "width": 512,
        "height": 512,
        "camera_name": "fixedfar",
    }
    if target_spec == "normal_reward":
        sample_env_config["task_id"] = "SafetyCarGoal1-v0"
        sample_env_config["ignore_cost"] = False
    else:
        sample_env_config["task_id"] = "SafetyCarTLGoal1-v0"
        sample_env_config["config"]["tl_spec"] = target_spec
        sample_env_config["ignore_cost"] = True

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
        + f"filtered_{reward_threshold}_extended_{max_extended_steps}_expanded_{expand_search}_weighted_{kl_div_weighted}"
    )

    dir_name: str = "goal"
    common_dir_path: str = f"search/{dir_name}"
    target_model_path: str = (
        f"out/models/safety_car_goal1/safety_car_goal1_ppo_128.zip"
        if target_spec == "normal_reward"
        else f"out/models/search/goal/ppo_{spec2title(target_spec)}.zip"
    )
    summary_log_path: str = (
        f"out/data/{common_dir_path}/multistart_{log_suffix}_ppo_{run}{suffix}.json"
    )
    log_save_path: str = (
        f"out/data/{common_dir_path}/logs/{log_suffix}_ppo{suffix}.json"
    )
    model_save_path: str = f"out/models/{common_dir_path}/ppo{suffix}.zip"
    learning_curve_path: str = (
        f"out/plots/reward_curve/{common_dir_path}/ppo{suffix}.png"
    )
    animation_save_path: str | None = None
    data_save_path: str = f"out/data/{common_dir_path}/kl_div/ppo{suffix}.json"
    target_actions_path: str = (
        f"out/data/{common_dir_path}/dataset/action_probs_ppo_{kl_div_suffix}_full.npz"
    )
    target_entropy_path: str = target_actions_path.replace(".npz", "_ent.json")
    obs_list_path: str = (
        f"out/data/{common_dir_path}/dataset/obs_list_ppo_{kl_div_suffix}.pkl"
    )
    target_episode_path: str = (
        f"out/data/{common_dir_path}/dataset/episode_ppo._{kl_div_suffix}.json"
    )

    seeds: list[int] = [random.randint(0, 10000) for _ in num_replicates]  # not used

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    if target_spec == "normal_reward":
        assert "nr" in kl_div_suffix
    else:
        assert "tl" in kl_div_suffix

    mp.set_start_method("spawn")

    init_nodes: list[SpecNode] = [
        initialize_node(predicates, exclusions) for _ in range(num_start)
    ]

    start_spec = start_specs[run - 1]

    if start_spec is not None:
        print(f"Using start spec {start_spec}")
        init_nodes[0] = spec2node(start_spec, dir_name)
    else:
        pass

    sample_env = CustomBuilder(**sample_env_config)

    obs_list: list[dict[str, Any]]

    if os.path.exists(obs_list_path):
        with open(obs_list_path, "rb") as f:
            obs_list = pickle.load(f)
    else:
        model = PPO.load(target_model_path.replace(".zip", f"_{0}.zip"), sample_env)
        obs_list = sample_obs(sample_env, model, num_samples)

        # Save the observations
        os.makedirs(os.path.dirname(obs_list_path), exist_ok=True)
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
        target_env = CustomBuilder(**sample_env_config)

        target_gaus_means_list = []
        target_gaus_stds_list = []
        target_trap_masks = []
        print("Loading target model...")
        for i in range(ref_num_replicates):
            print(f"Replicate {i}")
            model = PPO.load(target_model_path.replace(".zip", f"_{i}.zip"), target_env)
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
        model = PPO.load(target_model_path.replace(".zip", f"_{0}.zip"), sample_env)
        _, episode_lengths = evaluate_policy(
            model, sample_env, n_eval_episodes=num_episodes, return_episode_rewards=True
        )
        episode_length_report: EpisodeLengthReport = {
            "mean": float(np.mean(episode_lengths)),
            "std": float(np.std(episode_lengths)),
            "max": float(np.max(episode_lengths)),
            "min": float(np.min(episode_lengths)),
        }
        os.makedirs(os.path.dirname(target_episode_path), exist_ok=True)
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

        os.makedirs(os.path.dirname(target_entropy_path), exist_ok=True)
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
            ppo_config,
            env_config,
            search_start_iter=start_iter,
            episode_length_report=episode_length_report,
            reward_threshold=reward_threshold,
            episode_length_sigma=episode_length_sigma,
            warm_start_path=None,
            kl_div_suffix=kl_div_suffix,
            max_extended_steps=max_extended_steps,
            expand_search=expand_search,
            continue_from_checkpoint=continue_from_checkpoint,
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

    os.makedirs(os.path.dirname(summary_log_path), exist_ok=True)
    with open(summary_log_path, "w") as f:
        json.dump(multistart_log, f, indent=4)
