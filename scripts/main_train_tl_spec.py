import itertools
import json
import os
import pickle
import random
import multiprocessing as mp
from typing import Callable, Final

import numpy as np
from numpy.typing import NDArray
from stable_baselines3 import PPO
import torch

from tl_search.common.typing import Location
from tl_search.common.io import spec2title
from tl_search.common.typing import ObsProp
from tl_search.envs.eval import evaluate_tuned_param_replicate_tl_models
from tl_search.envs.extractor import generate_possible_states, get_action_distributions
from tl_search.envs.heuristic import HeuristicEnemyEnv
from tl_search.envs.search import evaluate_models
from tl_search.envs.tl_multigrid import TLMultigrid
from tl_search.envs.typing import EnemyPolicyMode, FieldObj, TLMultigridTrainingConfig
from tl_search.map.utils import distance_area_point, distance_points
from tl_search.train.tl_train import train_replicate_tl_agent

if __name__ == "__main__":
    enemy_policy_mode: Final[EnemyPolicyMode] = "patrol"
    n_envs: Final[int] = 50
    total_timesteps: Final[int] = 500_000
    num_replicates: Final[int] = 3
    window: Final[int] = round(total_timesteps / 100)

    num_samples: int = 5000
    tuned_param_name: Final[str | None] = "ent_coef"
    tuned_param_values: Final[list[float] | None] = [0.1]  # , 0.01, 0.05, 0.1]

    tl_spec: str = "F(psi_ba_rf&!psi_ra_bf) & G(!psi_ba_ra|psi_ba_bt)"  # "F(psi_ba_rf) & G((psi_ba_ra&!psi_ba_bt)|(!psi_ra_bf))"  # "F(psi_ba_rf) & G((!psi_ba_ra)|(psi_ba_bt&!psi_ra_bf))"  # "F(psi_ba_rf) & G((psi_ba_ra&!psi_ba_bt)|(!psi_ra_bf))"  # "F(psi_ba_rf & !psi_ba_bt & !psi_ba_bf) & G(!psi_ba_ra & !psi_ra_bf)"

    gpu: Final[int] = 1

    force_train: Final[bool] = False

    map_path: Final[str] = "tl_search/map/maps/board_0002_obj.txt"

    suffix: Final[str] = "_correction"
    log_suffix: Final[str] = "exp1_nested"
    filename_common: Final[
        str
    ] = f"search/heuristic/{enemy_policy_mode}/{enemy_policy_mode}_enemy_ppo_{spec2title(tl_spec)}{suffix}"
    model_save_path: Final[
        str
    ] = f"out/models/search/heuristic/{enemy_policy_mode}/target/{enemy_policy_mode}_enemy_ppo_{spec2title(tl_spec)}{suffix}.zip"
    learning_curve_path: Final[str] = f"out/plots/reward_curve/{filename_common}.png"
    animation_save_path: Final[str] = f"out/plots/animation/{filename_common}.gif"

    target_model_path: Final[
        str
    ] = f"out/models/heuristic/{enemy_policy_mode}_enemy_ppo_curr_ent_coef_0.01.zip"  # f"out/models/search/heuristic/patrol/patrol_enemy_ppo_{spec2title(tl_spec)}.zip"

    target_actions_path: str = f"out/data/search/dataset/action_probs_{enemy_policy_mode}_enemy_ppo_{log_suffix}_full_searched_simple.npz"
    field_list_path: str = (
        f"out/data/search/dataset/field_list_{enemy_policy_mode}_enemy_ppo_full.pkl"
    )

    log_path: str = f"out/data/{filename_common}_{tuned_param_name}.json"
    stats_path: str = f"out/data/search/stats/stats_{enemy_policy_mode}_enemy_ppo_{spec2title(tl_spec)}{suffix}.json"
    eval_path: str = f"out/data/test_kldiv_{enemy_policy_mode}_enemy_ppo_{spec2title(tl_spec)}{suffix}.json"

    global_threshold: float = 0.5
    atom_prep_dict: dict[str, str] = {
        "psi_ba_ra": "d_ba_ra < {}".format(1),
        "psi_ba_bf": "d_ba_bf < {}".format(global_threshold),
        "psi_ba_rf": "d_ba_rf < {}".format(global_threshold),
        "psi_ra_bf": "d_ra_bf < {}".format(global_threshold),
        "psi_ba_bt": "d_ba_bt < {}".format(global_threshold),
        # "psi_ra_bt": "d_ra_bt < {}".format(global_threshold),
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
        # ("d_ra_bt", ["red_agent", "blue_background"], distance_area_point),
    ]
    obs_props: list[ObsProp] = [ObsProp(*obs) for obs in obs_info]

    seeds: Final[list[int]] = [random.randint(0, 10000) for _ in range(num_replicates)]

    device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    sample_env = TLMultigrid(
        "F(psi_ba_rf & !psi_ba_bf) & G(!psi_ra_bf & !(psi_ba_ra & !psi_ba_bt))",
        obs_props,
        atom_prep_dict,
        enemy_policy_mode,
        map_path,
    )

    field_list: list[FieldObj] = random.sample(
        generate_possible_states(sample_env), num_samples
    )

    if os.path.exists(field_list_path):
        with open(field_list_path, "rb") as f:
            field_list = pickle.load(f)
    else:
        field_list = random.sample(generate_possible_states(sample_env), num_samples)

        with open(field_list_path, "wb") as f:
            pickle.dump(field_list, f)

    print(target_actions_path)

    if os.path.exists(target_actions_path):
        npz = np.load(target_actions_path)
        target_action_probs_list = npz["arr_0"]
        target_trap_masks = npz["arr_1"]
    else:
        print("Generating target actions...")
        target_env = TLMultigrid(
            tl_spec,
            obs_props,
            atom_prep_dict,
            enemy_policy_mode,
            map_path,
        )  # HeuristicEnemyEnv(enemy_policy_mode, map_path)

        target_action_probs_list: list[NDArray] = []
        target_trap_masks: list[NDArray] = []
        print("Loading target model...")
        for i in range(num_replicates):
            print(f"Replicate {i}")
            model = PPO.load(target_model_path.replace(".zip", f"_{i}.zip"), target_env)

            action_probs, trap_mask = get_action_distributions(
                model, target_env, field_list
            )

            target_action_probs_list.append(action_probs)
            target_trap_masks.append(trap_mask)

            del model

        np.savez(target_actions_path, target_action_probs_list, target_trap_masks)

    env = TLMultigrid(tl_spec, obs_props, atom_prep_dict, enemy_policy_mode, map_path)
    print(f"Train the agent against {enemy_policy_mode} enemy policy")

    tuned_params = [
        {tuned_param_name: tuned_param_value}
        for tuned_param_value in tuned_param_values
    ]
    tuned_param_suffixes = [
        f"_{tuned_param_name}_{tuned_param_value}"
        for tuned_param_value in tuned_param_values
    ]
    mp.set_start_method("spawn")
    inputs = [
        (
            num_replicates,
            n_envs,
            env,
            seeds,
            total_timesteps,
            model_save_path,  # .replace(".zip", tuned_param_suffix + ".zip"),
            learning_curve_path,  # .replace(".png", tuned_param_suffix + ".png"),
            animation_save_path,  # .replace(".gif", tuned_param_suffix + ".gif"),
            device,
            window,
            tuned_param,
            target_model_path,
            force_train,
        )
        for tuned_param, tuned_param_suffix in zip(tuned_params, tuned_param_suffixes)
    ]
    # with mp.Pool(len(tuned_param_values)) as pool:
    #     pool.starmap(train_replicate_tl_agent, inputs)
    #     pool.close()

    # models = train_replicate_tl_agent(*inputs[0])

    tuned_param_suffix = tuned_param_suffixes[0]

    models = train_replicate_tl_agent(
        num_replicates,
        n_envs,
        env,
        seeds,
        total_timesteps,
        model_save_path,  # .replace(".zip", tuned_param_suffix + ".zip"),
        learning_curve_path,  # .replace(".png", tuned_param_suffix + ".png"),
        animation_save_path,  # .replace(".gif", tuned_param_suffixes[0] + ".gif"),
        device,
        window,
        tuned_params[0],
        # target_model_path,
    )
    # for tuned_param_value in tuned_param_values:
    #     print('Tuned parameter "{}" = {}'.format(tuned_param_name, tuned_param_value))

    #     tuned_param_suffix: Final[str] = f"_{tuned_param_name}_{tuned_param_value}"
    #     tuned_param: Final[dict[str, float]] = {tuned_param_name: tuned_param_value}

    #     if (
    #         os.path.exists(
    #             learning_curve_path.replace(".png", tuned_param_suffix + ".png")
    #         )
    #         and not force_train
    #     ):
    #         print("Skipping training because model already exists")
    #     else:
    #         models = train_replicate_tl_agent(
    #             num_replicates,
    #             n_envs,
    #             env,
    #             seeds,
    #             total_timesteps,
    #             model_save_path.replace(".zip", tuned_param_suffix + ".zip"),
    #             learning_curve_path.replace(".png", tuned_param_suffix + ".png"),
    #             animation_save_path.replace(".gif", tuned_param_suffix + ".gif"),
    #             device,
    #             window,
    #             tuned_param,
    #         )

    training_config: TLMultigridTrainingConfig = {
        "tl_spec": tl_spec,
        "enemy_policy_mode": enemy_policy_mode,
        "seeds": seeds,
        "num_envs": n_envs,
        "total_timesteps": total_timesteps,
        "map_path": map_path,
        "model_save_path": model_save_path,
        "learning_curve_path": learning_curve_path,
        "animation_save_path": animation_save_path,
        "window": window,
        "tuned_param_name": tuned_param_name,
        "tuned_param_values": tuned_param_values,
    }

    with open(log_path, "w") as f:
        json.dump(training_config, f, indent=4)

    idx_combs: list[tuple[int, int]] = list(
        itertools.product(range(num_replicates), range(3))
    )

    kl_div_report = evaluate_models(
        env,
        target_action_probs_list,
        target_trap_masks,
        models,
        field_list,
        eval_path.replace(".json", "_sub.json"),
    )
    # evaluate_tuned_param_replicate_tl_models(
    #     tuned_param_name,
    #     tuned_param_values,
    #     num_replicates,
    #     tl_spec,
    #     obs_props,
    #     atom_prep_dict,
    #     enemy_policy_mode,
    #     model_save_path,
    #     map_path,
    #     200,
    #     stats_path,
    # )
