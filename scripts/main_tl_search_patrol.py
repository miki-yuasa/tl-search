import itertools, json, os, random
from math import ceil
import multiprocessing as mp
from typing import Callable, Final

import numpy as np
from numpy.typing import NDArray
import torch
from stable_baselines3 import PPO

from tl_search.common.typing import (
    EpisodeLengthReport,
    Exclusion,
    Location,
    MultiStartSearchLog,
    ObsProp,
    SpecNode,
    ValueTable,
)
from tl_search.envs.heuristic import HeuristicEnemyEnv
from tl_search.search.search import search_train_evaluate
from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.envs.typing import EnemyPolicyMode, FieldObj
from tl_search.evaluation.count import report_episode_lengths
from tl_search.evaluation.ranking import sort_spec
from tl_search.map.utils import distance_area_point, distance_points
from tl_search.search.neighbor import create_neighbor_masks, initialize_node, spec2node

if __name__ == "__main__":
    run: int = 5
    num_samples: int = 5000
    num_start: int = 1
    num_max_search_steps: int = 50
    num_processes: int = 25
    gpu: int = 1

    start_iter: int = 0
    start_specs: list[str | None] = [
        "F((psi_ba_bt)&(!psi_ba_ra|!psi_ba_bf)) & G((psi_ra_bf)|(!psi_ba_rf))",
        "F((psi_ba_rf)|(psi_ra_bf)) & G((psi_ba_ra&!psi_ba_bt)|(psi_ba_bf))",
        "F((psi_ba_rf)|(!psi_ra_bf)) & G(!psi_ba_ra|!psi_ba_bt|!psi_ba_bf)",
        "F(!psi_ra_bf) & G(!psi_ba_ra&!psi_ba_bt&!psi_ba_rf&psi_ba_bf)",
        "F(!psi_ba_ra&!psi_ba_rf) & G((!psi_ba_bf)&(!psi_ba_bt|!psi_ra_bf))",
    ]

    predicates: tuple[str, ...] = (
        "psi_ba_ra",
        "psi_ba_bt",
        "psi_ba_rf",
        "psi_ba_bf",
        "psi_ra_bf",
    )

    reward_threshold: float = 0.0
    episode_length_sigma: float = 1
    max_extended_steps: int = 0
    expand_search: bool = True

    enemy_policy_mode: EnemyPolicyMode = "patrol"
    n_envs: Final[int] = 50  # 20
    total_timesteps: Final[int] = 300_000
    num_replicates: Final[int] = 3
    num_episodes: Final[int] = 200
    window: Final[int] = ceil(round(total_timesteps / 100))
    map_path: Final[str] = "assets/maps/board_0002_obj.txt"
    warm_start: bool = True

    tuned_param_name: Final[str | None] = "ent_coef"
    tuned_param_value: Final[float] = 0.1
    tuned_param: Final[dict[str, float]] = {tuned_param_name: tuned_param_value}

    ref_num_replicates: int = 3

    exclusions: list[Exclusion] = []

    suffix: str = ""
    log_suffix: str = (
        f"filtered_{reward_threshold}_extended_{max_extended_steps}_expanded_{expand_search}_"
    )

    target_model_path: str = (
        f"out/models/heuristic/{enemy_policy_mode}_enemy_ppo_curr_ent_coef_0.01.zip"
    )
    summary_log_path: str = (
        f"out/data/search/heuristic/multistart_{log_suffix}{enemy_policy_mode}_enemy_ppo_{run}{suffix}.json"
    )
    log_save_path: str = (
        f"out/data/search/heuristic/{enemy_policy_mode}/{log_suffix}{enemy_policy_mode}_enemy_ppo{suffix}.json"
    )
    model_save_path: str = (
        f"out/models/search/heuristic/{enemy_policy_mode}/{enemy_policy_mode}_enemy_ppo{suffix}.zip"
    )
    learning_curve_path: str = (
        f"out/plots/reward_curve/search/heuristic/{enemy_policy_mode}/{enemy_policy_mode}_enemy_ppo{suffix}.png"
    )
    animation_save_path: str | None = None
    data_save_path: str = (
        f"out/data/kl_div/heuristic/{enemy_policy_mode}/kl_div_{enemy_policy_mode}_enemy_ppo{suffix}.json"
    )
    target_actions_path: str = (
        f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo"
    )
    target_episode_path: str = (
        f"out/data/search/dataset/episode_{enemy_policy_mode}_enemy_ppo.json"
    )

    global_threshold: float = 0.5
    atom_prep_dict: dict[str, str] = {
        "psi_ba_ra": "d_ba_ra < {}".format(1.5),
        "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
        "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
        "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
        "psi_ba_bt": "d_ba_bt < {}".format(global_threshold),
    }
    obs_info: list[tuple[str, list[str], Callable]] = [
        ("d_ba_ra", ["blue_agent", "red_agent"], distance_points),
        ("d_ba_rf", ["blue_agent", "red_flag"], distance_points),
        (
            "d_ba_bt",
            ["blue_agent", "blue_background"],
            distance_area_point,
        ),
        ("d_ba_bf", ["blue_agent", "blue_flag"], distance_points),
        ("d_ra_bf", ["red_agent", "blue_flag"], distance_points),
    ]
    obs_props: list[ObsProp] = [ObsProp(*obs) for obs in obs_info]

    seeds: list[int] = [random.randint(0, 10000) for _ in range(num_replicates)]

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
        init_nodes[0] = spec2node(start_spec, enemy_policy_mode)
    else:
        pass

    sample_env = TLMultigrid(
        "F(psi_ba_rf & !psi_ba_bf) & G(!psi_ra_bf & !(psi_ba_ra & !psi_ba_bt))",
        obs_props,
        atom_prep_dict,
        enemy_policy_mode,
        map_path,
    )

    possible_locs: list[Location] = list(
        set(
            [
                Location(*loc)
                for loc in itertools.product(
                    range(sample_env._field_map.shape[0]),
                    range(sample_env._field_map.shape[1]),
                )
            ]
        )
        - set(sample_env._fixed_obj.obstacle)
    )
    sampled_agent_locs: list[tuple[Location, Location]] = random.sample(
        list(itertools.product(possible_locs, possible_locs)),
        num_samples,
    )

    field_list: list[FieldObj] = []
    for blue_agent, red_agent in sampled_agent_locs:
        sample_env.reset(
            blue_agent_loc=Location(*blue_agent), red_agent_loc=Location(*red_agent)
        )
        field_list.append(sample_env._field)

    if os.path.exists(target_actions_path + ".npz"):
        npz = np.load(target_actions_path + ".npz")
        target_action_probs_list = npz["arr_0"]
        target_trap_masks = npz["arr_1"]
    else:
        target_env = HeuristicEnemyEnv(enemy_policy_mode, map_path)

        target_action_probs_list: list[NDArray] = []
        target_trap_masks: list[NDArray] = []
        print("Loading target model...")
        for i in range(num_replicates):
            print(f"Replicate {i}")
            model = PPO.load(target_model_path.replace(".zip", f"_{i}.zip"), target_env)

            action_probs: list[NDArray] = []

            for field_obj in field_list:
                obs, _ = target_env.reset(field_obj.blue_agent, field_obj.red_agent)
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                probs_np: NDArray = (
                    model.policy.get_distribution(obs_tensor)
                    .distribution.probs[0]
                    .cpu()
                    .detach()
                    .numpy()
                )

                action_probs.append(probs_np)

            target_action_probs_list.append(np.array(action_probs))
            target_trap_masks.append(np.ones(num_samples))

            del model

        np.savez(target_actions_path, target_action_probs_list, target_trap_masks)

    if os.path.exists(target_episode_path):
        with open(target_episode_path, "r") as f:
            episode_length_report: EpisodeLengthReport = json.load(f)
    else:
        episode_length_report: EpisodeLengthReport = report_episode_lengths(
            num_episodes,
            enemy_policy_mode,
            map_path,
            num_replicates,
            target_model_path,
            n_envs,
        )
        with open(target_episode_path, "w") as f:
            json.dump(episode_length_report, f, indent=4)

    neighbor_masks: tuple[ValueTable, ...] = create_neighbor_masks(
        len(predicates), exclusions
    )

    local_optimum_nodes: list[SpecNode] = []
    local_optimum_specs: list[str] = []
    local_optimum_kl_divs: list[float] = []
    searches_traces: list[tuple[list[str], list[float]]] = []
    nums_searched_specs: list[int] = []

    for i, init_node in enumerate(init_nodes):
        node_trace, spec_trace, metrics_trace, searched_specs = search_train_evaluate(
            init_node=init_node,
            num_max_search_steps=num_max_search_steps,
            run=run,
            start_idx=i,
            neighbor_masks=neighbor_masks,
            log_save_path=log_save_path,
            num_processes=num_processes,
            num_replicates=num_replicates,
            n_envs=n_envs,
            seeds=seeds,
            total_timesteps=total_timesteps,
            model_save_path=model_save_path,
            learning_curve_path=learning_curve_path,
            animation_save_path=animation_save_path,
            device=device,
            window=window,
            # idx_combs,
            target_action_probs_list=target_action_probs_list,
            target_trap_masks=target_trap_masks,
            field_obj_list=field_list,
            data_save_path=data_save_path,
            obs_props=obs_props,
            atom_prep_dict=atom_prep_dict,
            enemy_policy_mode=enemy_policy_mode,
            map_path=map_path,
            tuned_param=tuned_param,
            search_start_iter=start_iter,
            episode_length_report=episode_length_report,
            reward_threshold=reward_threshold,
            episode_length_sigma=episode_length_sigma,
            warm_start_path=None if not warm_start else target_model_path,
            max_extended_steps=max_extended_steps,
            expand_search=expand_search,
        )
        local_optimum_nodes.append(node_trace[-1])
        local_optimum_specs.append(spec_trace[-1])
        local_optimum_kl_divs.append(metrics_trace[-1])
        searches_traces.append((spec_trace, metrics_trace))
        nums_searched_specs.append(len(searched_specs))

    sorted_specs, sorted_min_kl_divs = sort_spec(
        local_optimum_specs, local_optimum_kl_divs
    )

    multistart_log: MultiStartSearchLog = {
        "run": run,
        "sorted_local_optimal_specs": sorted_specs,
        "sorted_local_optimal_kl_divs": sorted_min_kl_divs,
        "total_searched_specs": int(np.sum(nums_searched_specs)),
        "nums_searched_specs": nums_searched_specs,
        "searches_traces": searches_traces,
    }

    with open(summary_log_path, "w") as f:
        json.dump(multistart_log, f, indent=4)
