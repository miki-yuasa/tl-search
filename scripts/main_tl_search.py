import itertools, json, os, random
import pickle
from math import ceil
import multiprocessing as mp
from typing import Callable, Final, Literal

import numpy as np
from numpy.typing import NDArray
import torch
from stable_baselines3 import PPO

from tl_search.common.typing import (
    EntropyReportDict,
    EpisodeLengthReport,
    Exclusion,
    MultiStartSearchLog,
    ObsProp,
    SpecNode,
    ValueTable,
)
from tl_search.evaluation.extractor import (
    generate_possible_states,
    get_action_distributions,
)
from tl_search.envs.heuristic import HeuristicEnemyEnv
from tl_search.search.search import search_train_evaluate
from tl_search.search.select import select_max_entropy_spec_replicate
from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.envs.typing import EnemyPolicyMode, FieldObj
from tl_search.evaluation.count import report_episode_lengths
from tl_search.evaluation.ranking import sort_spec
from tl_search.map.utils import distance_area_point, distance_points
from tl_search.search.neighbor import create_neighbor_masks, initialize_node, spec2node

if __name__ == "__main__":
    run: int = 10
    gpu: int = 1  # (run - 1) % 4
    num_samples: int = 5000
    num_start: int = 1
    num_max_search_steps: int = 10
    num_processes: int = 25

    enemy_policy_mode: EnemyPolicyMode = "patrol"
    warm_start_mode: Literal["target", "parent", None] = "parent"
    reward_threshold: float = 0.5 * 0.1
    episode_length_sigma: float | None = 2 if warm_start_mode == "target" else None
    kl_div_suffix: str | None = "exp3_single_stoch_ws"

    target_spec: str | None = "F((psi_ba_rf)&(!psi_ra_bf)) & G(!psi_ba_ra|psi_ba_bt)"

    start_iter: int = 0
    start_specs: list[str | None] = [
        "F(psi_ba_rf&!psi_ra_bf) & G(psi_ba_ra|psi_ba_bt)",  # "F((!psi_ba_bt)&(!psi_ra_bf)) & G((!psi_ba_ra)|(psi_ba_rf))",
        "F((!psi_ra_bf)|(!psi_ba_rf)) & G(!psi_ba_ra|!psi_ba_bt)",
        "F(!psi_ra_bf) & G((!psi_ba_ra&!psi_ba_bt)|(psi_ba_rf))",
        "F(!psi_ba_rf&!psi_ra_bf) & G((!psi_ba_bt)&(!psi_ba_ra))",
        "F(!psi_ba_bt&psi_ba_rf) & G(!psi_ba_ra&!psi_ra_bf)",
        "F((!psi_ba_bt&psi_ra_bf)|(psi_ba_rf)) & G(!psi_ba_ra)",  # "F((psi_ba_ra|!psi_ba_rf)&(!psi_ba_bt)) & G(psi_ra_bf))",
        "F(psi_ba_ra) & G((psi_ba_bt)&(!psi_ba_rf|!psi_ra_bf))",  # "F((!psi_ba_bt)&(!psi_ba_rf|!psi_ra_bf)) & G(psi_ba_ra)",
        "F((psi_ba_rf)|(psi_ba_bt&!psi_ra_bf)) & G(!psi_ba_ra)",
        "F((!psi_ba_bt)&(!psi_ra_bf)) & G((!psi_ba_ra)|(psi_ba_rf))",  # "F(psi_ba_rf&!psi_ra_bf) & G(psi_ba_ra|psi_ba_bt)",  # "F((psi_ba_rf)|(!psi_ba_ra)) & G(!psi_ba_bt|!psi_ra_bf)",  # "F(psi_ba_bt&!psi_ra_bf) & G((psi_ba_ra)|(!psi_ba_rf))",
        "F((psi_ra_bf)|(!psi_ba_bt)) & G((!psi_ba_ra)&(!psi_ba_rf))",  # "F(psi_ba_ra) & G((psi_ra_bf)|(!psi_ba_bt&!psi_ba_rf))",
    ]

    predicates: tuple[str, ...] = (
        "psi_ba_ra",
        "psi_ba_bt",
        "psi_ba_rf",
        "psi_ra_bf",
    )

    n_envs: Final[int] = 25  # 50  # 20
    total_timesteps: Final[int] = 500_000
    num_replicates: Final[int] = 3
    num_episodes: Final[int] = 200
    window: Final[int] = ceil(round(total_timesteps / 100))
    map_path: Final[str] = "tl_search/map/maps/board_0002_obj.txt"

    tuned_param_name: Final[str | None] = "ent_coef"
    tuned_param_value: Final[float] = 0.1
    tuned_param: Final[dict[str, float]] = {tuned_param_name: tuned_param_value}

    ref_num_replicates: int = 3

    exclusions: list[Exclusion] = []

    suffix: str
    match warm_start_mode:
        case "target":
            suffix = "_ws"
        case "parent":
            suffix = "_parent_ws"
        case None:
            suffix = "_stoch"

    log_suffix: str = (
        f"{kl_div_suffix}_extended_" if kl_div_suffix is not None else "extended_"
    )

    target_model_path: str = (
        (
            "out/models/search/heuristic/patrol/patrol_enemy_ppo_ws_F(psi_ba_rf_and_!psi_ra_bf)_and_G(!psi_ba_ra_or_psi_ba_bt)_ws.zip"
            if warm_start_mode == "target"
            else "out/models/search/heuristic/patrol/patrol_enemy_ppo_F((psi_ba_rf)_and_(!psi_ra_bf))_and_G(!psi_ba_ra_or_psi_ba_bt).zip"
        )
        if "exp1" in kl_div_suffix
        else (
            f"out/models/heuristic/{enemy_policy_mode}_enemy_ppo_curr_ent_coef_0.01.zip"
        )
    )
    summary_log_path: str = f"out/data/search/heuristic/multistart_{log_suffix}{enemy_policy_mode}_enemy_ppo_{run}{suffix}.json"
    log_save_path: str = f"out/data/search/heuristic/{enemy_policy_mode}/{log_suffix}{enemy_policy_mode}_enemy_ppo{suffix}.json"
    model_save_path: str = f"out/models/search/heuristic/{enemy_policy_mode}/{enemy_policy_mode}_enemy_ppo{suffix}.zip"
    learning_curve_path: str = f"out/plots/reward_curve/search/heuristic/{enemy_policy_mode}/{enemy_policy_mode}_enemy_ppo{suffix}.png"
    animation_save_path: str | None = None
    data_save_path: str = f"out/data/kl_div/heuristic/{enemy_policy_mode}/kl_div/kl_div_{enemy_policy_mode}_enemy_ppo{suffix}.json"
    target_actions_path: str = f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_{log_suffix}full.npz"  # "out/data/search/dataset/action_probs_patrol_enemy_ppo_exp1_nested.npz"  # f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_{log_suffix}full"
    target_entropy_path: str = target_actions_path.replace(".npz", "_ent.json")
    field_list_path: str = (
        f"out/data/search/dataset/field_list_{enemy_policy_mode}_enemy_ppo_full.pkl"
    )
    target_episode_path: str = (
        f"out/data/search/dataset/episode_{enemy_policy_mode}_enemy_ppo.json"
    )

    global_threshold: float = 0.5
    atom_prep_dict: dict[str, str] = {
        "psi_ba_ra": "d_ba_ra < {}".format(1),
        "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
        "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
        "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
        "psi_ba_bt": "d_ba_bt < {}".format(global_threshold),
    }
    obs_info: list[tuple[str, list[str], Callable]] = [
        (
            "d_ba_ra",
            ["blue_agent", "red_agent", "is_red_agent_defeated"],
            distance_points,
        ),
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

    field_list: list[FieldObj]

    if os.path.exists(field_list_path):
        with open(field_list_path, "rb") as f:
            field_list = pickle.load(f)
    else:
        field_list = random.sample(generate_possible_states(sample_env), num_samples)

        with open(field_list_path, "wb") as f:
            pickle.dump(field_list, f)

    if os.path.exists(target_actions_path):
        print("Loading target actions...")
        npz = np.load(target_actions_path)
        target_action_probs_list = npz["arr_0"]
        target_trap_masks = npz["arr_1"]
    else:
        print("Generating target actions...")
        target_env = (
            TLMultigrid(
                target_spec,
                obs_props,
                atom_prep_dict,
                enemy_policy_mode,
                map_path,
            )
            if target_spec is not None
            else HeuristicEnemyEnv(enemy_policy_mode, map_path)
        )

        target_action_probs_list: list[NDArray] = []
        target_trap_masks: list[NDArray] = []
        print("Loading target model...")
        for i in range(num_replicates):
            print(f"Replicate {i}")
            model = PPO.load(target_model_path.replace(".zip", f"_{i}.zip"), target_env)

            if target_spec is None:
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

            else:
                action_probs: NDArray
                action_probs, trap_mask = get_action_distributions(
                    model, target_env, field_list
                )
                target_action_probs_list.append(action_probs)
                target_trap_masks.append(trap_mask)

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

    if os.path.exists(target_entropy_path):
        with open(target_entropy_path, "r") as f:
            target_entropy_dict: EntropyReportDict = json.load(f)

        target_max_entropy_idx: int = target_entropy_dict["max_entropy_idx"]
    else:
        (
            target_max_entropy_idx,
            target_entropies,
            num_non_trap_states,
        ) = select_max_entropy_spec_replicate(
            target_action_probs_list, target_trap_masks
        )

        target_entropy_dict: EntropyReportDict = {
            "max_entropy_idx": target_max_entropy_idx,
            "entropies": target_entropies,
            "num_non_trap_states": num_non_trap_states,
        }

        with open(target_entropy_path, "w") as f:
            json.dump(target_entropy_dict, f, indent=4)

    target_action_probs_list = [target_action_probs_list[target_max_entropy_idx]]
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
            field_list,
            data_save_path,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
            tuned_param,
            search_start_iter=start_iter,
            episode_length_report=episode_length_report,
            reward_threshold=reward_threshold,
            episode_length_sigma=episode_length_sigma,
            warm_start_path=None if warm_start_mode == "target" else target_model_path,
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
