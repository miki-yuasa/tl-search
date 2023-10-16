import copy
import os
from typing import Union

import numpy as np
from numpy.typing import NDArray

import torch
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

from tl_search.tl.environment import (
    Environment,
    Environment,
    restore_model,
)
from tl_search.tl.tl_parser import tokenize, get_vars, get_used_obs
from tl_search.tl.synthesis import TLAutomaton
from policies.known_policy import KnownPolicy
from policies.unknown_policy import UnknownPolicy
from tl_search.evaluation.evaluation import evaluate_spec
from tl_search.evaluation.visualize import (
    create_animation,
    is_animation_plotted,
    is_reward_curve_plotted,
    plot_reward_curves,
    simulate_model,
)
from policies.utils import sample_learned_policy
from tl_search.common.seed import torch_fix_seed
from tl_search.common.utils import kl_div_prob, moving_average
from tl_search.common.plotter import (
    SaveOnBestTrainingRewardCallback,
)
from tl_search.common.typing import (
    EnemyPolicyMode,
    EnvProps,
    MapObjectIDs,
    ModelProps,
    ObsProp,
    ActionProb,
    ActionProbsSpec,
    RLAlgorithm,
    SaveMode,
    MapLocations,
)


def spec_train(
    spec_ind: int,
    gpu: int,
    target: KnownPolicy | UnknownPolicy | ModelProps,
    enemy_policy: EnemyPolicyMode,
    num_sampling: int,
    num_replicates: int,
    total_timesteps: int,
    tl_spec: str,
    atom_prop_dict_all: dict[str, str],
    obs_props_all: list[ObsProp],
    log_dir: str,
    model_savename: str,
    model_save_mode: SaveMode,
    reward_curve_plot_mode: SaveMode,
    reward_curve_plot_savename: str,
    is_trap_state_filtered: bool | None,
    is_reward_calculated: bool,
    is_initialized_in_both_territories: bool,
    animation_mode: SaveMode,
    animation_savename: str,
    is_gui_shown: bool,
    replicate_seeds: list[int],
    is_similarity_compared: bool = True,
    is_from_all_territories: bool = False,
    learning_rate: float = 1e-5,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    clip_range_vf: float | None = None,
    ent_coef: float = 0.0,
    vf_coef: float = 1.0,
    max_grad_norm: float = 0.75,
) -> tuple[
    list[float], ActionProbsSpec, ActionProbsSpec, list[float], list[float], list[str]
]:
    print("Training {}".format(tl_spec))

    prop_vars = get_vars(tokenize(tl_spec), atom_prop_dict_all)
    used_obs = get_used_obs(prop_vars, atom_prop_dict_all, obs_props_all)

    n_in: int = len(used_obs)

    atom_prop_dict: dict[str, str] = {
        prop_var: atom_prop_dict_all[prop_var] for prop_var in prop_vars
    }
    obs_props: list[ObsProp] = [
        obs_prop for obs_prop in obs_props_all if obs_prop.name in used_obs
    ]

    aut: TLAutomaton = TLAutomaton(tl_spec, atom_prop_dict, obs_props)

    if is_similarity_compared:
        map_locs_list: list[MapLocations]
        ori_action_probs_np: list[NDArray]

        if type(target) == KnownPolicy or type(target) == UnknownPolicy:
            map_locs_list, ori_action_probs_np = target.sample(
                num_sampling, is_from_all_territories
            )
        else:
            target_model = restore_model_tmp(
                target.model_name,
                target.spec,
                target.obs_props,
                target.atom_prop_dict,
                target.map_props.map,
                target.map_props.map_object_ids,
                enemy_policy,
                gpu,
                is_initialized_in_both_territories,
            )
            map_locs_list, ori_action_probs_np = sample_learned_policy(
                target.map_props.fixed_map_locs,
                target_model,
                num_sampling,
                is_from_all_territories,
            )
    else:
        pass

    map: NDArray = (
        target.map
        if type(target) == KnownPolicy or type(target) == UnknownPolicy
        else target.map_props.map
    )
    map_object_ids: MapObjectIDs = (
        target.map_object_ids
        if type(target) == KnownPolicy or type(target) == UnknownPolicy
        else target.map_props.map_object_ids
    )

    kl_divs_spec: list[float] = []
    tl_action_probs_spec: ActionProbsSpec = []
    ori_action_probs_spec: ActionProbsSpec = []
    mean_rewards_spec: list[float] = []
    std_rewards_spec: list[float] = []

    model_names_spec: list[str] = []

    # for reward curve plots
    X: list[NDArray] = []
    Y: list[NDArray] = []

    env = Environment(
        n_in,
        map,
        map_object_ids,
        aut,
        enemy_policy,
        is_initialized_in_both_territories,
    )

    for rep_ind, seed in zip(range(num_replicates), replicate_seeds):
        print("Replicate {}".format(rep_ind))

        torch_fix_seed(seed)

        log_dir_spec_ind: str = log_dir + "{}/{}/".format(spec_ind, rep_ind)
        os.makedirs(log_dir_spec_ind, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=1000, log_dir=log_dir_spec_ind
        )
        env_copy: Environment = copy.deepcopy(env)
        env.seed(seed)
        env_monitored = Monitor(env_copy, log_dir_spec_ind)

        model = PPO(
            "MlpPolicy",
            env_monitored,
            verbose=0,
            device=torch.device("cuda:{}".format(gpu)),
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)

        model_name: Union[str, None] = (
            ("{}_{}".format(model_savename, spec_ind) if rep_ind == 0 else None)
            if model_save_mode == "suppressed"
            else "{}_{}_{}".format(model_savename, spec_ind, rep_ind)
        )

        if model_save_mode != "disabled" and model_name:
            model.save(model_name)
            model_names_spec.append(model_name)
        else:
            pass

        if is_reward_curve_plotted(reward_curve_plot_mode, spec_ind):
            smoothing_size: int = 500
            x, y = ts2xy(load_results(log_dir_spec_ind), "timesteps")
            y = moving_average(y, smoothing_size)
            # Truncate x
            x = x[len(x) - len(y) :]
            X.append(x)
            Y.append(y)

            if rep_ind == num_replicates - 1:
                plot_reward_curves(
                    X,
                    Y,
                    tl_spec,
                    "{}_{}".format(reward_curve_plot_savename, spec_ind),
                    is_gui_shown,
                )
            else:
                pass
        else:
            pass

        if is_animation_plotted(animation_mode, rep_ind):
            env_done = simulate_model(model, env, tl_spec)
            video_save_name: str = animation_savename + (
                ".gif"
                if animation_mode == "suppresed"
                else "_{}_{}.gif".format(spec_ind, rep_ind)
            )

            create_animation(
                env_done.blue_path,
                env_done.red_path,
                env_done.init_map,
                env_done.map_object_ids,
                video_save_name,
                is_gui_shown,
            )

        else:
            pass

        if is_similarity_compared:
            tl_action_probs: list[ActionProb]
            ori_action_probs: list[ActionProb]

            tl_action_probs, ori_action_probs, mean_reward, std_reward = evaluate_spec(
                model,
                map_locs_list,
                ori_action_probs_np,
                is_trap_state_filtered,
                is_reward_calculated,
            )

            kl_divs_spec.append(kl_div_prob(ori_action_probs, tl_action_probs))
            tl_action_probs_spec.append(tl_action_probs)
            ori_action_probs_spec.append(ori_action_probs)

            mean_rewards_spec.append(mean_reward)
            std_rewards_spec.append(std_reward)
        else:
            pass

    return (
        kl_divs_spec,
        tl_action_probs_spec,
        ori_action_probs_spec,
        mean_rewards_spec,
        std_rewards_spec,
        model_names_spec,
    )


def train_spec(
    target_env_props: EnvProps,
    rl_algorithm: RLAlgorithm,
    spec_ind: int,
    gpu: int,
    num_replicates: int,
    total_timesteps: int,
    log_dir: str,
    model_savename: str,
    reward_curve_plot_mode: SaveMode,
    reward_curve_plot_savename: str,
    is_initialized_in_both_territories: bool,
    animation_mode: SaveMode,
    animation_savename: str,
    is_gui_shown: bool,
    replicate_seeds: list[int],
    learning_rate: float = 1e-5,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    clip_range_vf: float | None = None,
    ent_coef: float = 0.0,
    vf_coef: float = 1.0,
    max_grad_norm: float = 0.75,
) -> list[ModelProps]:
    print("Training {}".format(target_env_props.spec))

    model_props_list: list[ModelProps] = []

    # for reward curve plots
    X: list[NDArray] = []
    Y: list[NDArray] = []

    env = Environment(
        target_env_props,
        is_initialized_in_both_territories,
    )

    for rep_ind, seed in zip(range(num_replicates), replicate_seeds):
        print("Replicate {}".format(rep_ind))

        torch_fix_seed(seed)

        log_dir_spec_ind: str = log_dir + "{}/{}/".format(spec_ind, rep_ind)
        os.makedirs(log_dir_spec_ind, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(
            check_freq=1000, log_dir=log_dir_spec_ind
        )
        env_copy: Environment = copy.deepcopy(env)
        env.seed(seed)
        env_monitored = Monitor(env_copy, log_dir_spec_ind)

        model = (
            PPO(
                "MlpPolicy",
                env_monitored,
                verbose=0,
                device=torch.device("cuda:{}".format(gpu)),
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
            )
            if rl_algorithm == "ppo"
            else DQN("MlpPolicy", env_monitored)
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)

        model_name: str = "{}_{}_{}".format(model_savename, spec_ind, rep_ind)
        model.save(model_name)
        model_props_list.append(
            ModelProps(
                model_name,
                rl_algorithm,
                target_env_props.spec,
                target_env_props.atom_prop_dict,
                target_env_props.enemy_policy,
                target_env_props.map_props,
            )
        )

        if is_reward_curve_plotted(reward_curve_plot_mode, spec_ind):
            print("Plotting reward curve")
            smoothing_size: int = round(total_timesteps / 1000)
            x, y = ts2xy(load_results(log_dir_spec_ind), "timesteps")
            y = moving_average(y, smoothing_size)
            # Truncate x
            x = x[len(x) - len(y) :]
            X.append(x)
            Y.append(y)

            if rep_ind == num_replicates - 1:
                plot_reward_curves(
                    X,
                    Y,
                    target_env_props.spec,
                    "{}_{}".format(reward_curve_plot_savename, spec_ind),
                    is_gui_shown,
                )
            else:
                pass
        else:
            pass

        if is_animation_plotted(animation_mode, rep_ind):
            env_done = simulate_model(model, env)
            video_save_name: str = animation_savename + (
                ".gif"
                if animation_mode == "suppresed"
                else "_{}_{}.gif".format(spec_ind, rep_ind)
            )

            create_animation(
                env_done.blue_path,
                env_done.red_path,
                env_done.init_map,
                env_done.map_object_ids,
                video_save_name,
                is_gui_shown,
            )

        else:
            pass

    return model_props_list
